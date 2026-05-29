#!/usr/bin/env python3
"""
Audio-Curation-Webtool — lokale Web-App zum schnellen Durchhören, Bewerten
und Markieren von TTS/Seed-VC-WAV-Dateien.

Start:  python3 scripts/audio_curation_server.py --root /home/dida-80b/victoria_ds --port 8090
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import math

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("audio_curation")

# ── Constants ────────────────────────────────────────────────────────────
STATUSES = ("unreviewed", "approved", "top", "rerender", "delete_marked")
SPECIAL_CATEGORIES = {"universal"}  # klar markierte Spezial-Kategorie
PROTECTED_STATUSES = {"approved", "top"}  # NIEMALS automatisch auf rerender setzen
ASR_IMAGE = "hokuspokus-asr-checker:latest"  # Docker-Image für Deep-Check

# ── Helper ───────────────────────────────────────────────────────────────
CANDIDATE_ID_RE = re.compile(
    r"^([a-z_]+)_(\d{5})_([\w\-]+)\.wav$"
)


def parse_candidate_id(filename: str) -> Optional[tuple[str, str]]:
    """Extrahiere (candidate_id, category) aus Dateiname."""
    m = CANDIDATE_ID_RE.match(filename)
    if m:
        category = m.group(1)
        cid = f"{category}_{m.group(2)}_{m.group(3)}"
        return cid, category
    # Fallback: Alles vor .wav
    base = filename.rsplit(".wav", 1)[0]
    return (base, "unknown") if base else None


# ── Index & State ────────────────────────────────────────────────────────
class CurationState:
    """Hält den gesamten File-Index + Review-States im Speicher."""

    def __init__(self, root: Path):
        self.root = root
        self.generated_dir = root / "generated"
        self.augmented_texts_dir = root / "intermediate" / "augmented_texts"
        self.manifests_dir = root / "manifests"

        # State & Export-Pfade
        self.state_path = self.manifests_dir / "audio_curation_state.json"
        self.approved_path = self.manifests_dir / "audio_curation_approved.jsonl"
        self.top_path = self.manifests_dir / "audio_curation_top.jsonl"
        self.rerender_path = self.manifests_dir / "audio_curation_rerender.jsonl"
        self.delete_marked_path = self.manifests_dir / "audio_curation_delete_marked.jsonl"

        self.files: list[dict] = []          # sorted file list
        self._file_by_id: dict[str, dict] = {}  # candidate_id -> file dict
        self._text_map: dict[str, str] = {}  # candidate_id -> text
        self.categories: list[str] = []      # alle gefundenen Kategorien

        self._load_texts()
        self._scan_files()
        self._load_state()
        self._reconcile_processed_actions()
        self._persist_state()
        self._export_all()

    # ── Text-Loading (primär: augmented_texts, fallback: generated manifests) ──

    def _load_texts(self):
        """Lade Texte: 1) augmented_texts/*.jsonl (primär), 2) generated/*/manifest.jsonl (Fallback)."""
        # Phase 1: augmented_texts (Primärquelle)
        if self.augmented_texts_dir.is_dir():
            for fpath in sorted(self.augmented_texts_dir.glob("*.jsonl")):
                try:
                    with open(fpath, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            cid = obj.get("candidate_id")
                            text = obj.get("text")
                            if cid and text and cid not in self._text_map:
                                self._text_map[cid] = text
                except Exception as exc:
                    log.warning("Fehler beim Laden von %s: %s", fpath.name, exc)

        log.info("Texte aus augmented_texts geladen: %d Einträge", len(self._text_map))

        # Phase 2: generated/*/manifest.jsonl (Fallback)
        if self.generated_dir.is_dir():
            for cat_dir in sorted(self.generated_dir.iterdir()):
                if not cat_dir.is_dir():
                    continue
                manifest = cat_dir / "manifest.jsonl"
                if not manifest.is_file():
                    continue
                try:
                    with open(manifest, "r", encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            cid = obj.get("candidate_id")
                            text = obj.get("text")
                            if cid and text and cid not in self._text_map:
                                self._text_map[cid] = text
                except Exception as exc:
                    log.warning("Fehler bei manifest %s: %s", manifest, exc)

        log.info("Texte gesamt (inkl. Manifest-Fallback): %d", len(self._text_map))

    # ── File-Scanning ─────────────────────────────────────────────────

    def _scan_files(self):
        """Durchsuche ALLE Kategorie-Ordner unter generated/ die einen wavs/ Unterordner haben."""
        if not self.generated_dir.is_dir():
            log.warning("generated/ Verzeichnis nicht gefunden")
            return

        found_categories: set[str] = set()
        total = 0

        for cat_dir in sorted(self.generated_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            cat_name = cat_dir.name
            wavs_dir = cat_dir / "wavs"
            if not wavs_dir.is_dir():
                # Spezial-Kategorien ohne WAVs trotzdem merken (z.B. universal)
                if cat_name in SPECIAL_CATEGORIES:
                    found_categories.add(cat_name)
                continue

            found_categories.add(cat_name)
            cat_total = 0

            for entry in sorted(wavs_dir.iterdir()):
                if not entry.is_file() or entry.suffix.lower() != ".wav":
                    continue

                parsed = parse_candidate_id(entry.name)
                if parsed:
                    cid, parsed_cat = parsed
                else:
                    cid = entry.name

                # Text primär aus augmented_texts, sonst Fallback
                text = self._text_map.get(cid, "unbekannt")

                # Fallback: Versuche emotion_xxxxx match ohne Suffix
                if text == "unbekannt" and cid:
                    try:
                        parts = cid.rsplit("_", 2)
                        base_id = "_".join(parts[:2])
                        for key in self._text_map:
                            if key.startswith(base_id + "_"):
                                text = self._text_map[key]
                                break
                    except Exception:
                        pass

                try:
                    rel = entry.relative_to(self.root)
                except ValueError:
                    rel = entry

                file_info = {
                    "audio_path": str(entry),
                    "relative_path": str(rel),
                    "filename": entry.name,
                    "candidate_id": cid,
                    "text": text,
                    "status": "unreviewed",
                    "reviewed_at": None,
                    "notes": "",
                    "category": cat_name,
                }
                self.files.append(file_info)
                if cid:
                    self._file_by_id[cid] = file_info
                total += 1
                cat_total += 1

            log.info("  %-20s %4d WAVs", cat_name, cat_total)

        self.categories = sorted(found_categories)
        self.files.sort(key=lambda f: (f["category"], f["filename"]))
        log.info("Index erstellt: %d WAV-Dateien in %d Kategorien", total, len(self.categories))

    # ── State Persistence ─────────────────────────────────────────────

    def _load_state(self):
        """Lade gespeicherten Review-State und merge in Index. Migriert alte Status-Werte."""
        if not self.state_path.exists():
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except Exception as exc:
            log.warning("Konnte state.json nicht laden: %s", exc)
            return

        # Keine Migration: Status getrennt lassen (approved, top, etc.)
        state_map = {r["audio_path"]: r for r in saved.get("records", [])}

        updated = 0
        for f in self.files:
            ap = f["audio_path"]
            if ap in state_map:
                sr = state_map[ap]
                status = sr.get("status", "unreviewed")
                if status in STATUSES:
                    f["status"] = status
                else:
                    f["status"] = "unreviewed"
                f["reviewed_at"] = sr.get("reviewed_at")
                f["notes"] = sr.get("notes", "")
                updated += 1

        log.info("State geladen: %d bestehende Bewertungen übernommen", updated)

    def _persist_state(self):
        """Schreibe kompletten State als JSON."""
        records = []
        for f in self.files:
            records.append({
                "audio_path": f["audio_path"],
                "relative_path": f["relative_path"],
                "candidate_id": f["candidate_id"],
                "text": f["text"],
                "status": f["status"],
                "reviewed_at": f["reviewed_at"],
                "notes": f["notes"],
                "category": f["category"],
            })

        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(records),
            "records": records,
        }

        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        tmp = f"{self.state_path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, str(self.state_path))

    @staticmethod
    def _reviewed_at_to_timestamp(reviewed_at: Optional[str]) -> Optional[float]:
        if not reviewed_at:
            return None
        try:
            return datetime.fromisoformat(reviewed_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None

    def _reconcile_processed_actions(self) -> bool:
        """
        Räume verarbeitete Delete-/Rerender-Einträge weg.

        - `delete_marked`: Wenn die Datei inzwischen fehlt, entferne sie aus dem Live-Index.
        - `rerender`: Wenn die WAV nach dem Markieren neu geschrieben wurde, ist der Auftrag
          abgearbeitet und der Eintrag wird wieder `unreviewed`.
        """
        changed = False
        kept_files: list[dict] = []

        for f in self.files:
            audio_path = Path(f["audio_path"])

            if f["status"] == "delete_marked" and not audio_path.exists():
                changed = True
                continue

            if f["status"] == "rerender" and audio_path.exists():
                reviewed_ts = self._reviewed_at_to_timestamp(f.get("reviewed_at"))
                if reviewed_ts is not None and audio_path.stat().st_mtime > reviewed_ts:
                    f["status"] = "unreviewed"
                    f["reviewed_at"] = None
                    f["notes"] = ""
                    changed = True

            kept_files.append(f)

        if changed:
            self.files = kept_files
            self._file_by_id = {
                f["candidate_id"]: f
                for f in self.files
                if f.get("candidate_id")
            }

        return changed

    def _export_all(self):
        """Exportiere approved, top, rerender, delete_marked als JSONL."""
        exports: dict[str, tuple[Path, list[dict]]] = {
            "approved": (self.approved_path, []),
            "top": (self.top_path, []),
            "rerender": (self.rerender_path, []),
            "delete_marked": (self.delete_marked_path, []),
        }

        for f in self.files:
            rec = {
                "candidate_id": f["candidate_id"],
                "category": f["category"],
                "text": f["text"],
                "audio_path": f["audio_path"],
                "relative_path": f["relative_path"],
                "status": f["status"],
                "reviewed_at": f["reviewed_at"],
                "notes": f["notes"],
            }
            for status_key in exports:
                if f["status"] == status_key:
                    exports[status_key][1].append(rec)

        for status_key, (path, recs) in exports.items():
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                for rec in recs:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp, str(path))

        log.info("Export: approved=%d  top=%d  rerender=%d  delete_marked=%d",
                 len(exports["approved"][1]), len(exports["top"][1]),
                 len(exports["rerender"][1]), len(exports["delete_marked"][1]))

    def update_status(self, idx: int, new_status: str, notes: str = ""):
        """Aktualisiere Status einer Datei und persistiere."""
        if 0 <= idx < len(self.files):
            f = self.files[idx]
            f["status"] = new_status
            f["reviewed_at"] = datetime.now(timezone.utc).isoformat()
            if notes:
                f["notes"] = notes
            self._persist_state()
            self._export_all()

    # ── Vorabprüfung ─────────────────────────────────────────────────

    def precheck(self, status: str = "all", category: str = "all") -> dict:
        """
        Automatische Vorabprüfung aller Dateien im aktuellen Filter-Scope.
        Setzt Verdachtsfälle auf `rerender` — aber NIE wenn Status `approved` oder `top`.
        """
        candidates = self.get_filtered(status=status, category=category)
        checked = 0
        flagged = 0
        skipped_protected = 0
        details: list[dict] = []

        now = datetime.now(timezone.utc).isoformat()

        for f in candidates:
            # ═══ SCHUTZ: approved + top ═══
            if f["status"] in PROTECTED_STATUSES:
                skipped_protected += 1
                details.append({
                    "candidate_id": f["candidate_id"],
                    "filename": f["filename"],
                    "reason": f"skipped ({f['status']})",
                    "action": "none",
                })
                continue

            checked += 1
            reason = self._check_audio_issues(f)

            if reason:
                flags = f.copy()
                flags["status"] = "rerender"
                flags["reviewed_at"] = now
                flags["notes"] = reason

                # Inplace-Update im Haupt-Index
                main_idx = next(
                    (i for i, mf in enumerate(self.files) if mf["audio_path"] == f["audio_path"]),
                    None,
                )
                if main_idx is not None:
                    self.files[main_idx]["status"] = "rerender"
                    self.files[main_idx]["reviewed_at"] = now
                    self.files[main_idx]["notes"] = reason

                flagged += 1
                details.append({
                    "candidate_id": f["candidate_id"],
                    "filename": f["filename"],
                    "reason": reason,
                    "action": "flagged -> rerender",
                })
            else:
                details.append({
                    "candidate_id": f["candidate_id"],
                    "filename": f["filename"],
                    "reason": "ok",
                    "action": "none",
                })

        if flagged > 0:
            self._persist_state()
            self._export_all()

        return {
            "scanned": len(candidates),
            "checked": checked,
            "flagged": flagged,
            "skipped_protected": skipped_protected,
            "details": details,
        }

    @staticmethod
    def _check_audio_issues(f: dict) -> Optional[str]:
        """Prüfe eine Datei auf Audio-/Text-Probleme. Liefert Grund oder None."""
        filename = f.get("filename", "")
        text = f.get("text", "")
        cid = f.get("candidate_id", "")

        reasons: list[str] = []

        # 1) Dateigröße: zu klein oder verdächtig null
        try:
            size = Path(f["audio_path"]).stat().st_size
            if size < 2048:
                reasons.append("Datei zu klein (<2KB, vermutlich kaputt)")
            elif size < 8192:
                reasons.append("Datei sehr klein (<8KB)")
        except OSError:
            reasons.append("Datei nicht lesbar")
            return "; ".join(reasons)

        # 2) Dauer schätzen via WAV-Header (PCM 16-bit mono 22050 angenommen, wie typisch TTS)
        #    44 Bytes Header, Rest PCM 16-bit mono → bytes_total = 44 + samples * 2
        bytes_audio = size - 44
        duration_s = 0.0
        if bytes_audio > 0:
            samples = bytes_audio // 2
            duration_s = samples / 22050  # typische Kokoro SR
            if duration_s < 0.4:
                reasons.append(f"extrem kurz ({duration_s:.1f}s)")
            elif duration_s > 40:
                reasons.append(f"extrem lang ({duration_s:.0f}s)")

        # 3) Text-Verdacht / Kauderwelsch
        text_check = CurationState._check_text_suspicious(text)
        if text_check:
            reasons.append(text_check)

        # 4) Stille-Verdacht via Dateigröße (sehr kleine Datei für lange Dauer schätzen)
        if bytes_audio > 0 and duration_s > 1.5 and size < 20000:
            reasons.append("Verdacht auf überwiegend Stille (kleine Datei trotz Dauer)")

        return "; ".join(reasons) if reasons else None

    @staticmethod
    def _check_text_suspicious(text: str) -> Optional[str]:
        """Heuristiken für Text-/Kauderwelsch-Verdacht."""
        if not text or text == "unbekannt":
            return "kein Text verfügbar"

        t = text.strip()

        # Kauderwelsch: hoher Anteil Nicht-Buchstaben
        alpha = sum(1 for c in t if c.isalpha())
        if len(t) > 0:
            ratio = alpha / len(t)
            if ratio < 0.35:
                return f"Kauderwelsch-Verdacht (nur {ratio:.0%} Buchstaben)"

        # Wiederholungen von 3+ gleichen Zeichen
        if re.search(r'(.)\1{4,}', t):
            return "Wiederholtes Zeichenmuster erkannt"

        # Sehr kurzer Text bei längerem Filename (Mismatch)
        if len(t) < 5 and len(t) > 0:
            return f"Text zu kurz ({len(t)} Zeichen)"

        return None

    # ── Deep-Check (ASR via Docker/ROCm) ────────────────────────────────

    def deepcheck(self, status: str = "all", category: str = "all") -> dict:
        """
        ASR-basierte Tiefenprüfung via Whisper/ROCm im Docker-Container.
        Überspringt `approved` und `top` vollständig.
        """
        import subprocess
        import shutil

        candidates = self.get_filtered(status=status, category=category)
        if not candidates:
            return {
                "scanned": 0,
                "checked": 0,
                "skipped_protected": 0,
                "flagged_rerender": 0,
                "details": [],
                "error": "Keine Dateien im aktuellen Filter",
            }

        # Bereite Input für den ASR-Worker vor
        input_records = []
        for f in candidates:
            input_records.append({
                "audio_path": f["audio_path"],
                "candidate_id": f["candidate_id"],
                "text": f["text"],
                "status": f["status"],
                "category": f["category"],
            })

        input_json = json.dumps(input_records, ensure_ascii=False)

        # Prüfe ob Docker-Image existiert
        result = subprocess.run(
            ["docker", "images", "-q", ASR_IMAGE],
            capture_output=True, text=True,
        )
        if not result.stdout.strip():
            return {
                "scanned": len(candidates),
                "checked": 0,
                "skipped_protected": 0,
                "flagged_rerender": 0,
                "details": [],
                "error": f"Docker-Image {ASR_IMAGE} nicht gefunden — bitte 'docker build -f scripts/Dockerfile.asr -t {ASR_IMAGE} scripts/' ausführen",
            }

        log.info(
            "Starte ASR-Deep-Check: %d Dateien im Scope (Status=%s, Kategorie=%s)",
            len(candidates), status, category,
        )
        t_start = time.time()

        # Rufe ASR-Worker im Docker-Container auf
        asr_script = Path(__file__).parent / "asr_deep_check.py"
        try:
            proc = subprocess.run(
                [
                    "docker", "run", "--rm", "-i",
                    "--device", "/dev/kfd",
                    "--device", "/dev/dri",
                    "--group-add", "video",
                    "--group-add", "render",
                    "-e", "HSA_OVERRIDE_GFX_VERSION=11.0.0",
                    "-e", "HIP_VISIBLE_DEVICES=0",
                    "-v", f"{asr_script}:/asr_check.py:ro",
                    "-v", f"{self.root}:{self.root}:ro",
                    ASR_IMAGE,
                    "python3", "-u", "/asr_check.py",
                ],
                input=input_json,
                capture_output=True, text=True,
                timeout=7200,  # 2h max
            )
        except subprocess.TimeoutExpired:
            return {
                "scanned": len(candidates),
                "checked": 0,
                "skipped_protected": 0,
                "flagged_rerender": 0,
                "details": [],
                "error": "ASR-Deep-Check Timeout (2h)",
            }

        # Log stderr
        if proc.stderr:
            for line in proc.stderr.strip().splitlines():
                log.info("[ASR] %s", line)

        if proc.returncode != 0:
            return {
                "scanned": len(candidates),
                "checked": 0,
                "skipped_protected": 0,
                "flagged_rerender": 0,
                "details": [],
                "error": f"ASR-Container-Fehler (rc={proc.returncode}): {proc.stderr[-500:]}",
            }

        # Parse Worker-Output
        try:
            worker_result = json.loads(proc.stdout.strip())
        except json.JSONDecodeError:
            return {
                "scanned": len(candidates),
                "checked": 0,
                "skipped_protected": 0,
                "flagged_rerender": 0,
                "details": [],
                "error": f"ASR-Worker-Output nicht parsebar: {proc.stdout[:500]}",
            }

        if worker_result.get("error"):
            return {
                "scanned": len(candidates),
                "checked": 0,
                "skipped_protected": 0,
                "flagged_rerender": 0,
                "details": [],
                "error": worker_result["error"],
            }

        # Merge flagged Dateien zurück in den Haupt-Index
        now = datetime.now(timezone.utc).isoformat()
        updated = 0
        for detail in worker_result.get("details", []):
            if detail.get("action") == "flagged -> rerender":
                cid = detail.get("candidate_id", "")
                for f in self.files:
                    if f["candidate_id"] == cid:
                        f["status"] = "rerender"
                        f["reviewed_at"] = now
                        f["notes"] = detail.get("reason", "")
                        updated += 1
                        break

        if updated > 0:
            self._persist_state()
            self._export_all()

        elapsed = time.time() - t_start
        log.info(
            "ASR-Deep-Check abgeschlossen: %d/%d flagged in %.1fs",
            updated, worker_result.get("checked", 0), elapsed,
        )

        return {
            "scanned": worker_result.get("scanned", len(candidates)),
            "checked": worker_result.get("checked", 0),
            "skipped_protected": worker_result.get("skipped_protected", 0),
            "flagged_rerender": worker_result.get("flagged_rerender", 0),
            "asr_model": worker_result.get("asr_model", "unknown"),
            "elapsed_s": round(elapsed, 1),
            "details": worker_result.get("details", []),
        }

    # ── Query ───────────────────────────────────────────────────────────

    def get_filtered(self, status: str = "all", category: str = "all") -> list[dict]:
        """Gefilterte Liste zurückgeben."""
        if self._reconcile_processed_actions():
            self._persist_state()
            self._export_all()

        result = self.files.copy()
        if status != "all":
            result = [f for f in result if f["status"] == status]
        if category != "all":
            result = [f for f in result if f["category"] == category]
        return result


# ── Deep-Check Job Tracker ─────────────────────────────────────────────────
_deepcheck_jobs: dict[str, dict] = {}  # job_id -> {status, result, error, started_at}
_deepcheck_lock = threading.Lock()


def _run_deepcheck_async(job_id: str, status_filter: str, category_filter: str):
    """Hintergrund-Thread: Führt deepcheck() aus und speichert Ergebnis."""
    global _deepcheck_jobs
    try:
        result = state.deepcheck(status=status_filter, category=category_filter)
        with _deepcheck_lock:
            _deepcheck_jobs[job_id]["status"] = "completed"
            _deepcheck_jobs[job_id]["result"] = result
    except Exception as exc:
        log.exception("Deep-Check-Job %s fehlgeschlagen", job_id)
        with _deepcheck_lock:
            _deepcheck_jobs[job_id]["status"] = "failed"
            _deepcheck_jobs[job_id]["error"] = str(exc)


# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Audio Curation Tool", version="2.0")
state: Optional[CurationState] = None
ROOT_PATH: Path = Path("/")

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Audio Curation Tool</title>
<style>
  :root {
    --bg: #0f0f14;
    --bg2: #181820;
    --bg3: #222230;
    --fg: #e0e0e8;
    --fg2: #9090a0;
    --accent: #6c8cff;
    --orange: #ff8c42;
    --green: #4caf50;
    --gold: #ffc107;
    --red: #e53935;
    --purple: #ce93d8;
    --purple-bg: #2a1030;
    --gray: #666;
    --border: #2a2a38;
    --radius: 6px;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg); color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 15px; height: 100vh; display: flex; overflow: hidden;
  }

  /* ── Sidebar ─────────────────────────── */
  #sidebar {
    width: 360px; min-width: 300px; background: var(--bg2);
    border-right: 1px solid var(--border); display: flex;
    flex-direction: column; overflow: hidden;
  }
  #sidebar h2 { padding: 14px 16px 6px; font-size: 16px; color: var(--fg2); }
  #filters { padding: 0 16px 10px; display: flex; gap: 6px; flex-wrap: wrap; }
  #filters select {
    background: var(--bg3); color: var(--fg); border: 1px solid var(--border);
    padding: 5px 8px; border-radius: var(--radius); font-size: 13px;
    cursor: pointer; min-width: 0; flex: 1 1 auto;
  }
  #filters select:focus { border-color: var(--accent); outline: none; }
  #search-wrap { padding: 0 16px 10px; }
  #search {
    width: 100%; background: var(--bg3); color: var(--fg);
    border: 1px solid var(--border); padding: 6px 10px;
    border-radius: var(--radius); font-size: 13px;
  }
  #search:focus { border-color: var(--accent); outline: none; }
  #stats { padding: 6px 16px; font-size: 13px; color: var(--fg2); }
  #filelist {
    flex: 1; overflow-y: auto; padding: 0 8px 8px;
    scrollbar-width: thin; scrollbar-color: var(--border) transparent;
  }
  .fl-item {
    display: flex; align-items: center; padding: 7px 10px; border-radius: var(--radius);
    cursor: pointer; font-size: 13px; gap: 8px;
    border: 1px solid transparent; margin-bottom: 2px; transition: background .1s;
  }
  .fl-item:hover { background: var(--bg3); }
  .fl-item.active { background: var(--bg3); border-color: var(--accent); }
  .fl-item.cat-universal { border-left: 3px solid var(--purple); }
  .fl-dot {
    width: 8px; height: 8px; min-width: 8px; border-radius: 50%;
  }
  .fl-dot.unreviewed { background: var(--gray); }
  .fl-dot.approved { background: var(--green); }
  .fl-dot.top { background: var(--gold); }
  .fl-dot.rerender { background: var(--orange); }
  .fl-dot.delete_marked { background: var(--red); }
  .fl-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* ── Main ─────────────────────────── */
  #main {
    flex: 1; display: flex; flex-direction: column; min-width: 0;
  }
  #topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 20px; border-bottom: 1px solid var(--border); gap: 12px;
  }
  #topbar h1 { font-size: 20px; font-weight: 600; }
  #progress { font-size: 14px; color: var(--fg2); white-space: nowrap; }
  #top-right { display: flex; align-items: center; gap: 12px; }
  .toggle-wrap { display: flex; align-items: center; gap: 6px; font-size: 13px; color: var(--fg2); }
  .toggle-wrap input { accent-color: var(--accent); }

  #detail {
    flex: 1; padding: 20px; overflow-y: auto;
    display: flex; flex-direction: column; gap: 16px;
  }

  .field { font-size: 13px; color: var(--fg2); }
  .field strong { color: var(--fg); margin-right: 4px; }
  .field .val { color: var(--fg); }

  .status-badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px;
  }
  .status-badge.unreviewed { background: #333; color: #aaa; }
  .status-badge.approved { background: #1a301a; color: var(--green); }
  .status-badge.top { background: #3a3010; color: var(--gold); }
  .status-badge.rerender { background: #3a2010; color: var(--orange); }
  .status-badge.delete_marked { background: #3a1010; color: var(--red); }

  .category-badge {
    display: inline-block; padding: 2px 10px; border-radius: 10px;
    font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: .5px;
    background: var(--bg3); color: var(--fg2); border: 1px solid var(--border);
  }
  .category-badge.cat-universal {
    background: var(--purple-bg); color: var(--purple); border-color: var(--purple);
  }

  #audio-block { background: var(--bg2); border-radius: var(--radius); padding: 16px; }
  #audio-block audio { width: 100%; outline: none; }
  #audio-block audio:focus { outline: 2px solid var(--accent); }

  #act-buttons {
    display: flex; gap: 10px; flex-wrap: wrap;
  }
  button {
    padding: 9px 18px; border: 1px solid var(--border); border-radius: var(--radius);
    font-size: 14px; cursor: pointer; background: var(--bg3); color: var(--fg);
    transition: background .15s, border-color .15s;
    display: flex; align-items: center; gap: 6px;
  }
  button:hover { background: var(--border); }
  button:active { background: var(--bg2); }
  button.approved { border-color: var(--green); color: var(--green); }
  button.approved:hover { background: #102010; }
  button.top { border-color: var(--gold); color: var(--gold); }
  button.top:hover { background: #282010; }
  button.rerender { border-color: var(--orange); color: var(--orange); }
  button.rerender:hover { background: #201810; }
  button.delete_marked { border-color: var(--red); color: var(--red); }
  button.delete_marked:hover { background: #201010; }
  button.pn { border-color: var(--accent); color: var(--accent); }

  #kbd-hint {
    font-size: 12px; color: var(--fg2); padding-top: 4px; line-height: 1.8;
  }
  #kbd-hint kbd {
    background: var(--bg3); border: 1px solid var(--border); border-radius: 3px;
    padding: 1px 5px; font-size: 11px; margin: 0 2px;
  }

  .hide { display: none !important; }
</style>
</head>
<body>

<!-- Sidebar -->
<div id="sidebar">
  <h2 id="sidebarTitle">Alle Kategorien</h2>
  <div id="filters">
    <select id="fCategory">
      <option value="all">Kategorie: alle</option>
    </select>
    <select id="fStatus">
      <option value="all">Status: alle</option>
      <option value="unreviewed">unreviewed</option>
      <option value="approved">approved</option>
      <option value="top">top</option>
      <option value="rerender">rerender</option>
      <option value="delete_marked">delete_marked</option>
    </select>
  </div>
  <div id="search-wrap">
    <input type="text" id="search" placeholder="Suche nach Text / ID..." />
  </div>
  <div id="stats">0 Dateien</div>
  <div id="filelist"></div>
</div>

<!-- Main -->
<div id="main">
  <div id="topbar">
    <div>
      <h1 id="curTitle">Audio Curation Tool</h1>
      <div style="font-size:13px;color:var(--fg2);display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <span id="curPath">–</span>
        <span id="categoryBadge" class="category-badge hide">–</span>
      </div>
    </div>
    <div id="top-right">
      <label class="toggle-wrap">
        <input type="checkbox" id="autoplay" checked />
        Autoplay
      </label>
      <div id="progress">–</div>
    </div>
  </div>
  <div id="detail">
    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:center">
      <span id="statusBadge" class="status-badge">unreviewed</span>
      <span class="field"><strong>CID:</strong> <span id="candidateId">–</span></span>
    </div>
    <div class="field" style="font-size:15px"><strong>Text:</strong> <span id="curText" style="color:var(--fg)">–</span></div>
    <div id="audio-block">
      <audio id="player" controls></audio>
    </div>
    <div id="act-buttons">
      <button class="pn" onclick="prevFile()" title="ArrowLeft / B">⏮ Zurück</button>
      <button id="btnPlay" onclick="togglePlay()" title="Space">▶ Play</button>
      <button class="approved" onclick="setApproved()" title="A">✓ Approve</button>
      <button class="top" onclick="setTop()" title="T">★ Top</button>
      <button class="rerender" onclick="setRerender()" title="R">↻ Rerender</button>
      <button class="delete_marked" onclick="setDeleteMarked()" title="Delete">✕ Delete markieren</button>
      <button class="pn" onclick="nextFile()" title="ArrowRight / N">Vor ⏭</button>
      <button id="btnPrecheck" class="pn" onclick="runPrecheck()" title="Auto-Prüfung auf problematische Dateien" style="border-color:var(--purple);color:var(--purple);margin-left:12px">🔍 Vorabprüfen</button>
      <button id="btnDeepcheck" class="pn" onclick="runDeepcheck()" title="ASR-Tiefenprüfung mit Whisper auf ROCm/GPU" style="border-color:#00bcd4;color:#00bcd4;margin-left:4px">🧠 Tiefenprüfung (ASR)</button>
      <span id="precheckStatus" style="font-size:13px;color:var(--fg2);display:flex;align-items:center"></span>
    </div>
    <div id="kbd-hint">
      Tasten: <kbd>A</kbd> Approve
      <kbd>T</kbd> Top
      <kbd>R</kbd> Rerender
      <kbd>Entf</kbd> Delete markieren
      <kbd>N</kbd>/<kbd>→</kbd> Next
      <kbd>B</kbd>/<kbd>←</kbd> Prev
      <kbd>Space</kbd> Play/Pause
    </div>
  </div>
</div>

<script>
// ── Global State ────────────────────────
let files = [];
let currentIdx = -1;
let currentFiltered = [];

// ── Init ────────────────────────────────
async function init() {
  await loadCategories();
  await loadFiles();
  bindKeyboard();
  bindFilters();
  if (currentFiltered.length > 0) selectFile(0, false);
}

async function loadCategories() {
  const res = await fetch('/api/categories');
  const data = await res.json();
  const sel = document.getElementById('fCategory');
  // Keep "alle" option, add category options
  sel.innerHTML = '<option value="all">Kategorie: alle</option>';
  for (const cat of data.categories) {
    const opt = document.createElement('option');
    opt.value = cat;
    opt.textContent = 'Kategorie: ' + cat;
    if (data.special.indexOf(cat) >= 0) {
      opt.textContent += ' ★';
    }
    sel.appendChild(opt);
  }
}

async function loadFiles() {
  const st = document.getElementById('fStatus').value;
  const cat = document.getElementById('fCategory').value;
  const params = new URLSearchParams({status: st});
  if (cat !== 'all') params.set('category', cat);
  const res = await fetch('/api/files?' + params.toString());
  const data = await res.json();
  files = data.files;
  currentFiltered = [...files];
  currentIdx = -1;
  applySearch();
  renderFileList();
  updateStats();
  updateSidebarTitle();
}

function updateSidebarTitle() {
  const cat = document.getElementById('fCategory').value;
  const title = document.getElementById('sidebarTitle');
  if (cat === 'all') {
    title.textContent = 'Alle Kategorien';
  } else {
    title.textContent = 'Kategorie: ' + cat;
  }
}

// ── Filter ──────────────────────────────
function bindFilters() {
  const reloadAndSelect = () => {
    loadFiles().then(() => {
      if (currentFiltered.length > 0 && currentIdx >= currentFiltered.length)
        currentIdx = currentFiltered.length - 1;
      if (currentIdx >= 0) selectFile(currentIdx, false);
      else if (currentFiltered.length > 0) { currentIdx = 0; selectFile(0, false); }
      else clearDetail();
    });
  };

  document.getElementById('fStatus').addEventListener('change', reloadAndSelect);
  document.getElementById('fCategory').addEventListener('change', reloadAndSelect);

  document.getElementById('search').addEventListener('input', () => {
    const term = document.getElementById('search').value.toLowerCase().trim();
    const prevAudioPath = currentIdx >= 0 && currentIdx < currentFiltered.length
      ? currentFiltered[currentIdx].audio_path : null;
    applySearch(term);
    renderFileList();
    if (prevAudioPath) {
      const newIdx = currentFiltered.findIndex(f => f.audio_path === prevAudioPath);
      currentIdx = newIdx >= 0 ? newIdx : (currentFiltered.length > 0 ? 0 : -1);
    } else if (currentFiltered.length > 0) {
      currentIdx = 0;
    } else {
      currentIdx = -1;
    }
    if (currentIdx >= 0) selectFile(currentIdx, false);
    else clearDetail();
  });
}

function applySearch(term) {
  term = term ?? document.getElementById('search').value.toLowerCase().trim();
  if (!term) {
    currentFiltered = [...files];
  } else {
    currentFiltered = files.filter(f =>
      f.text.toLowerCase().includes(term) ||
      (f.candidate_id || '').toLowerCase().includes(term) ||
      f.filename.toLowerCase().includes(term)
    );
  }
}

function updateStats() {
  const st = document.getElementById('fStatus').value;
  let topC = 0, unreviewed = 0, rerender = 0, deleteMarked = 0, approvedC = 0;
  for (const f of files) {
    if (f.status === 'approved') approvedC++;
    else if (f.status === 'top') topC++;
    else if (f.status === 'rerender') rerender++;
    else if (f.status === 'delete_marked') deleteMarked++;
    else unreviewed++;
  }
  document.getElementById('stats').textContent =
    `${currentFiltered.length} / ${files.length} Dateien · ✓${approvedC} ★${topC} ↻${rerender} ✕${deleteMarked} ○${unreviewed}`;

  document.getElementById('progress').textContent =
    currentIdx >= 0 && currentFiltered.length > 0
      ? `${currentIdx + 1} / ${currentFiltered.length}`
      : '–';
}

// ── File List ───────────────────────────
function renderFileList() {
  const el = document.getElementById('filelist');
  el.innerHTML = '';
  const batch = currentFiltered.slice(0, 400);
  for (let i = 0; i < batch.length; i++) {
    const f = batch[i];
    const div = document.createElement('div');
    const isUniversal = f.category === 'universal';
    div.className = 'fl-item' + (i === currentIdx ? ' active' : '') + (isUniversal ? ' cat-universal' : '');
    div.innerHTML = `<span class="fl-dot ${f.status}"></span>
      <span class="fl-name" title="${f.filename}">[${f.category}] ${f.filename}</span>`;
    div.onclick = () => selectFile(i, true);
    el.appendChild(div);
  }
  if (files.length > 400) {
    const hint = document.createElement('div');
    hint.style.cssText = 'padding:8px;font-size:12px;color:var(--fg2)';
    hint.textContent = `… und ${files.length - 400} weitere (Nutze Filter)`;
    el.appendChild(hint);
  }
}

// ── Detail ──────────────────────────────
function selectFile(idx, autoplay) {
  if (idx < 0 || idx >= currentFiltered.length) return;
  currentIdx = idx;
  const f = currentFiltered[idx];
  const isUniversal = f.category === 'universal';

  document.getElementById('curTitle').textContent = f.filename;
  document.getElementById('curPath').textContent = f.relative_path;
  document.getElementById('candidateId').textContent = f.candidate_id || '–';
  document.getElementById('curText').textContent = f.text || 'unbekannt';

  const sb = document.getElementById('statusBadge');
  sb.textContent = f.status;
  sb.className = 'status-badge ' + f.status;

  // Category badge
  const cb = document.getElementById('categoryBadge');
  cb.textContent = f.category;
  cb.className = 'category-badge' + (isUniversal ? ' cat-universal' : '');
  cb.classList.remove('hide');

  // Audio
  const player = document.getElementById('player');
  player.src = '/api/audio?path=' + encodeURIComponent(f.audio_path);
  player.load();
  if (autoplay && document.getElementById('autoplay').checked) {
    player.play().catch(() => {});
  }

  document.getElementById('progress').textContent =
    `${currentIdx + 1} / ${currentFiltered.length}`;

  renderFileList();
  updateStats();
}

function clearDetail() {
  document.getElementById('curTitle').textContent = 'Audio Curation Tool';
  document.getElementById('curPath').textContent = '–';
  document.getElementById('candidateId').textContent = '–';
  document.getElementById('curText').textContent = '–';
  document.getElementById('statusBadge').textContent = '–';
  document.getElementById('statusBadge').className = 'status-badge';
  document.getElementById('categoryBadge').classList.add('hide');
  document.getElementById('player').src = '';
  document.getElementById('progress').textContent = '–';
}

// ── Actions ─────────────────────────────
function updateFileInAllLists(audioPath, newStatus, reviewedAt) {
  for (const fl of files) {
    if (fl.audio_path === audioPath) {
      fl.status = newStatus;
      fl.reviewed_at = reviewedAt;
    }
  }
  const fStatus = document.getElementById('fStatus').value;
  for (const fl of currentFiltered) {
    if (fl.audio_path === audioPath) {
      fl.status = newStatus;
      fl.reviewed_at = reviewedAt;
    }
  }
  // Re-apply filters if status filter is active
  if (fStatus !== 'all') {
    const term = document.getElementById('search').value.toLowerCase().trim();
    applySearch(term);
  }
}

async function setStatus(newStatus) {
  if (currentIdx < 0 || currentIdx >= currentFiltered.length) return;
  const f = currentFiltered[currentIdx];
  const resp = await fetch(
    '/api/files/status?path=' + encodeURIComponent(f.audio_path) + '&status=' + newStatus,
    { method: 'PATCH' }
  );
  if (!resp.ok) { alert('Fehler beim Speichern'); return; }
  const data = await resp.json();

  updateFileInAllLists(f.audio_path, data.status, data.reviewed_at);

  if (currentFiltered.length === 0) {
    currentIdx = -1;
    clearDetail();
  } else {
    // Wenn die Datei noch an gleicher Stelle ist (Filter=all), manuell vorwärts
    if (currentFiltered[currentIdx]?.audio_path === f.audio_path) {
      currentIdx = Math.min(currentIdx + 1, currentFiltered.length - 1);
    } else if (currentIdx >= currentFiltered.length) {
      currentIdx = currentFiltered.length - 1;
    }
    selectFile(currentIdx, true);
  }
  updateStats();
}

function setApproved() { setStatus('approved'); }
function setTop() { setStatus('top'); }
function setRerender() { setStatus('rerender'); }
function setDeleteMarked() { setStatus('delete_marked'); }

function nextFile() {
  if (currentIdx + 1 < currentFiltered.length) {
    selectFile(currentIdx + 1, true);
  }
}
function prevFile() {
  if (currentIdx > 0) {
    selectFile(currentIdx - 1, true);
  }
}
let togglePlay = () => {
  const p = document.getElementById('player');
  p.paused ? p.play() : p.pause();
};

async function runPrecheck() {
  const btn = document.getElementById('btnPrecheck');
  const stat = document.getElementById('precheckStatus');
  btn.disabled = true;
  stat.textContent = ' Prüfe …';
  try {
    const st = document.getElementById('fStatus').value;
    const cat = document.getElementById('fCategory').value;
    const params = new URLSearchParams();
    if (st !== 'all') params.set('status', st);
    if (cat !== 'all') params.set('category', cat);
    const resp = await fetch('/api/precheck?' + params.toString(), { method: 'POST' });
    const data = await resp.json();
    stat.textContent =
      ` ✓ ${data.scanned} gescannt, ${data.flagged} auf rerender, ${data.skipped_protected} geschützt (approved/top) übersprungen`;
    await loadFiles();
    if (currentIdx >= currentFiltered.length) currentIdx = currentFiltered.length - 1;
    if (currentIdx >= 0) selectFile(currentIdx, false);
    else clearDetail();
  } catch (err) {
    stat.textContent = ' Fehler: ' + err.message;
  } finally {
    btn.disabled = false;
  }
}

async function runDeepcheck() {
  const btn = document.getElementById('btnDeepcheck');
  const stat = document.getElementById('precheckStatus');
  btn.disabled = true;
  stat.textContent = ' 🧠 ASR-Tiefenprüfung gestartet …';
  try {
    const st = document.getElementById('fStatus').value;
    const cat = document.getElementById('fCategory').value;
    const params = new URLSearchParams();
    if (st !== 'all') params.set('status', st);
    if (cat !== 'all') params.set('category', cat);

    // 1) Starte Job
    const startResp = await fetch('/api/deepcheck?' + params.toString(), { method: 'POST' });
    const startData = await startResp.json();
    const jobId = startData.job_id;

    // 2) Poll Status alle 5s
    const poll = async () => {
      try {
        const statusResp = await fetch('/api/deepcheck/status?job_id=' + encodeURIComponent(jobId));
        const statusData = await statusResp.json();
        if (statusData.status === 'running') {
          stat.textContent = ' 🧠 ASR-Tiefenprüfung läuft … (Job ' + jobId + ')';
          setTimeout(poll, 5000);
        } else if (statusData.status === 'completed') {
          const r = statusData.result || {};
          if (r.error) {
            stat.textContent = ' ❌ ' + r.error;
          } else {
            stat.textContent =
              ' ✓ ' + (r.scanned || 0) + ' gescannt, ' + (r.checked || 0) + ' ASR-geprüft, ' +
              (r.flagged_rerender || 0) + ' auf rerender, ' + (r.skipped_protected || 0) + ' geschützt';
          }
          await loadFiles();
          if (currentIdx >= currentFiltered.length) currentIdx = currentFiltered.length - 1;
          if (currentIdx >= 0) selectFile(currentIdx, false);
          else clearDetail();
          btn.disabled = false;
        } else if (statusData.status === 'failed') {
          stat.textContent = ' ❌ Job ' + jobId + ' fehlgeschlagen: ' + (statusData.error || 'unbekannter Fehler');
          btn.disabled = false;
        }
      } catch (err) {
        stat.textContent = ' ❌ Status-Abfrage fehlgeschlagen: ' + err.message;
        btn.disabled = false;
      }
    };
    poll();
  } catch (err) {
    stat.textContent = ' ❌ Fehler: ' + err.message;
    btn.disabled = false;
  }
}

// ── Keyboard ────────────────────────────
function bindKeyboard() {
  document.addEventListener('keydown', (e) => {
    const tag = e.target.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;

    switch (e.key) {
      case ' ': e.preventDefault(); togglePlay(); break;
      case 'a':
      case 'A': setApproved(); break;
      case 't':
      case 'T': setTop(); break;
      case 'r':
      case 'R': setRerender(); break;
      case 'Delete': e.preventDefault(); setDeleteMarked(); break;
      case 'ArrowRight':
      case 'n':
      case 'N': e.preventDefault(); nextFile(); break;
      case 'ArrowLeft':
      case 'b':
      case 'B': e.preventDefault(); prevFile(); break;
    }
  });
}

// ── Start ───────────────────────────────
init();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_PAGE


@app.get("/api/categories")
async def api_categories():
    """Liefere alle Kategorien + spezielle Kategorien."""
    return {
        "categories": state.categories,
        "special": sorted(list(SPECIAL_CATEGORIES)),
    }


@app.get("/api/stats")
async def api_stats():
    fs = state.files
    counts = {s: 0 for s in STATUSES}
    for f in fs:
        counts[f["status"]] = counts.get(f["status"], 0) + 1
    return {"total": len(fs), "counts": counts}


@app.get("/api/files")
async def api_files(
    status: str = Query("all"),
    category: str = Query("all"),
):
    filtered = state.get_filtered(status=status, category=category)
    return {"total": len(filtered), "files": filtered}


@app.get("/api/audio")
async def api_audio(path: str):
    """Liefere Audio-Datei aus, nur innerhalb des Root-Verzeichnisses."""
    if not path:
        raise HTTPException(400, "path required")

    p = Path(path).resolve()

    try:
        p.relative_to(ROOT_PATH.resolve())
    except ValueError:
        raise HTTPException(403, "Pfad liegt nicht im erlaubten Verzeichnis")

    if not p.is_file():
        raise HTTPException(404, "Datei nicht gefunden")

    return FileResponse(p, media_type="audio/wav")


@app.patch("/api/files/status")
async def api_update_status(
    path: str = Query(...),
    status: str = Query(...),
    notes: str = Query(""),
):
    if status not in STATUSES:
        raise HTTPException(400, f"Ungültiger Status: {status}")

    # Finde Index
    idx = None
    for i, f in enumerate(state.files):
        if f["audio_path"] == path:
            idx = i
            break

    if idx is None:
        p = str(Path(path).resolve())
        for i, f in enumerate(state.files):
            if str(Path(f["audio_path"]).resolve()) == p:
                idx = i
                break

    if idx is None:
        raise HTTPException(404, "Datei nicht gefunden")

    state.update_status(idx, status, notes)

    f = state.files[idx]
    return {
        "audio_path": f["audio_path"],
        "status": f["status"],
        "reviewed_at": f["reviewed_at"],
    }


@app.get("/api/export/approved")
async def api_export_approved():
    """Lade approved JSONL-Datei herunter."""
    p = state.approved_path
    if not p.exists():
        raise HTTPException(404, "Keine Export-Datei vorhanden")
    return FileResponse(p, media_type="application/jsonl",
                        filename="audio_curation_approved.jsonl")


@app.get("/api/export/top")
async def api_export_top():
    """Lade top JSONL-Datei herunter."""
    p = state.top_path
    if not p.exists():
        raise HTTPException(404, "Keine Export-Datei vorhanden")
    return FileResponse(p, media_type="application/jsonl",
                        filename="audio_curation_top.jsonl")


@app.get("/api/export/rerender")
async def api_export_rerender():
    """Lade rerender JSONL-Datei herunter."""
    p = state.rerender_path
    if not p.exists():
        raise HTTPException(404, "Keine Export-Datei vorhanden")
    return FileResponse(p, media_type="application/jsonl",
                        filename="audio_curation_rerender.jsonl")


@app.post("/api/precheck")
async def api_precheck(
    status: str = Query("all"),
    category: str = Query("all"),
):
    """
    Vorabprüfung: Automatische Analyse aller Dateien im aktuellen Scope.
    Setzt Verdachtsfälle auf `rerender` — überspringt `approved` und `top`.
    """
    result = state.precheck(status=status, category=category)
    return result


@app.post("/api/deepcheck")
async def api_deepcheck(
    status: str = Query("all"),
    category: str = Query("all"),
):
    """
    ASR-Tiefenprüfung: Whisper-basierte Transkription auf ROCm/GPU.
    Startet einen Hintergrund-Job und liefert sofort eine job_id zurück.
    Abfrage per GET /api/deepcheck/status?job_id=...
    """
    job_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat()
    with _deepcheck_lock:
        _deepcheck_jobs[job_id] = {
            "job_id": job_id,
            "status": "running",
            "result": None,
            "error": None,
            "started_at": now,
            "status_filter": status,
            "category_filter": category,
        }
    t = threading.Thread(
        target=_run_deepcheck_async,
        args=(job_id, status, category),
        daemon=True,
    )
    t.start()
    log.info("Deep-Check-Job %s gestartet (status=%s, category=%s)", job_id, status, category)
    return {
        "job_id": job_id,
        "status": "running",
        "message": "Deep-Check läuft im Hintergrund. Status abfragen mit GET /api/deepcheck/status?job_id=" + job_id,
    }


@app.get("/api/deepcheck/status")
async def api_deepcheck_status(job_id: str = Query(...)):
    """
    Status eines laufenden/abgeschlossenen Deep-Check-Jobs abfragen.
    """
    with _deepcheck_lock:
        job = _deepcheck_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"Job {job_id} nicht gefunden")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "result": job.get("result"),
        "error": job.get("error"),
        "started_at": job.get("started_at"),
        "status_filter": job.get("status_filter"),
        "category_filter": job.get("category_filter"),
    }


@app.get("/api/export/delete_marked")
async def api_export_delete_marked():
    """Lade delete_marked JSONL-Datei herunter."""
    p = state.delete_marked_path
    if not p.exists():
        raise HTTPException(404, "Keine Export-Datei vorhanden")
    return FileResponse(p, media_type="application/jsonl",
                        filename="audio_curation_delete_marked.jsonl")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    global ROOT_PATH, state

    parser = argparse.ArgumentParser(description="Audio Curation Server")
    parser.add_argument("--root", default="/home/dida-80b/victoria_ds",
                        help="Root des Datasets")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    ROOT_PATH = Path(args.root).resolve()
    if not ROOT_PATH.is_dir():
        log.error("Root-Verzeichnis nicht gefunden: %s", ROOT_PATH)
        sys.exit(1)

    log.info("Starte Audio-Curation-Server v2 …")
    log.info("  Root:    %s", ROOT_PATH)
    log.info("  Port:    %d", args.port)
    log.info("  URL:     http://%s:%d", args.host, args.port)

    t0 = time.time()
    state = CurationState(ROOT_PATH)
    elapsed = time.time() - t0

    log.info("Initialisierung in %.1fs abgeschlossen", elapsed)
    log.info("  %d WAV-Dateien im Index (%d Kategorien)", len(state.files), len(state.categories))
    log.info("Server bereit: http://%s:%d", args.host, args.port)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
