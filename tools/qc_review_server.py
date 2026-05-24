#!/usr/bin/env python3
"""
Human review server — loads clips from a HuggingFace dataset, no local audio needed.

Workflow:
  1. Each reviewer runs their own instance:
       python3 tools/qc_review_server.py --reviewer dida
  2. Make keep/delete/rerender decisions via the web UI.
  3. Commit reviews/<reviewer>.json and push to GitHub.
  4. Run qc_merge_reviews.py to find conflicts.

Usage:
  python3 tools/qc_review_server.py \
    --reviewer dida \
    --hf-dataset dida-80b/victoria-emotion-de-synthetic

State: reviews/<reviewer>.json  (only this file is written)
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn


DECISIONS = ("keep", "delete", "rerender")

_HTML = r"""<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="utf-8">
<title>QC Review — %%REVIEWER%%</title>
<style>
:root{--bg:#0f0f14;--bg2:#181820;--bg3:#222230;--fg:#e0e0e8;--fg2:#9090a0;
  --accent:#6c8cff;--green:#4caf50;--red:#e53935;--orange:#ff8c42;
  --gray:#555;--border:#2a2a38;--r:6px}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--fg);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  font-size:15px;height:100vh;display:flex;overflow:hidden}
#sidebar{width:320px;min-width:220px;background:var(--bg2);border-right:1px solid var(--border);
  display:flex;flex-direction:column;overflow:hidden}
#filters{padding:10px 10px 4px;display:flex;gap:6px;flex-wrap:wrap}
#filters select{flex:1 1 auto;background:var(--bg3);color:var(--fg);border:1px solid var(--border);
  padding:5px 8px;border-radius:var(--r);font-size:13px;cursor:pointer}
#filters select:focus{border-color:var(--accent);outline:none}
#search-wrap{padding:0 10px 6px}
#search{width:100%;background:var(--bg3);color:var(--fg);border:1px solid var(--border);
  padding:6px 10px;border-radius:var(--r);font-size:13px}
#search:focus{border-color:var(--accent);outline:none}
#stats{padding:0 14px 6px;font-size:12px;color:var(--fg2)}
#filelist{flex:1;overflow-y:auto;padding:0 6px 8px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}
.fl-item{display:flex;align-items:center;padding:6px 8px;border-radius:var(--r);cursor:pointer;
  font-size:12px;gap:7px;border:1px solid transparent;margin-bottom:2px}
.fl-item:hover{background:var(--bg3)}
.fl-item.active{background:var(--bg3);border-color:var(--accent)}
.fl-dot{width:8px;height:8px;min-width:8px;border-radius:50%}
.fl-dot.unreviewed{background:var(--gray)}.fl-dot.keep{background:var(--green)}
.fl-dot.delete{background:var(--red)}.fl-dot.rerender{background:var(--orange)}
.fl-emo{color:var(--accent);white-space:nowrap;font-size:11px}
.fl-name{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--fg2)}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#topbar{display:flex;align-items:center;justify-content:space-between;padding:10px 18px;
  border-bottom:1px solid var(--border);gap:10px;flex-wrap:wrap}
#topbar h1{font-size:15px;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:400px}
#reviewer-label{font-size:12px;color:var(--fg2)}
#reviewer-label b{color:var(--accent)}
#topbar-right{display:flex;align-items:center;gap:12px;flex-shrink:0}
#progress{font-size:13px;color:var(--fg2);white-space:nowrap}
#detail{flex:1;padding:14px 18px;overflow-y:auto;display:flex;flex-direction:column;gap:12px}
.meta-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.badge{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;font-weight:600;text-transform:uppercase;letter-spacing:.4px}
.badge.unreviewed{background:#2a2a38;color:#888}.badge.keep{background:#1a301a;color:var(--green)}
.badge.delete{background:#3a1010;color:var(--red)}.badge.rerender{background:#3a2010;color:var(--orange)}
.badge.emotion{background:var(--bg3);color:var(--accent);border:1px solid var(--border);text-transform:none;font-weight:500}
.badge.conflict{background:#3a1a1a;color:#f88;border:1px solid #7a2a2a}
.badge.agree{background:#1a2a1a;color:#8f8;border:1px solid #2a5a2a}
.other-votes{display:flex;gap:5px;flex-wrap:wrap;align-items:center}
.other-vote{font-size:11px;padding:1px 7px;border-radius:8px;font-weight:600}
.other-vote.keep{background:#1a301a;color:#4caf50}
.other-vote.delete{background:#3a1010;color:#e53935}
.other-vote.rerender{background:#3a2010;color:#ff8c42}
.other-vote.unreviewed{background:#2a2a38;color:#888}
.fl-others{display:flex;gap:3px;margin-left:auto;flex-shrink:0}
.fl-o{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.fl-o.keep{background:#4caf50}.fl-o.delete{background:#e53935}.fl-o.rerender{background:#ff8c42}.fl-o.unreviewed{background:#555}
.field .flabel{font-size:11px;color:var(--fg2);margin-bottom:3px;text-transform:uppercase;letter-spacing:.4px}
.field .fval{color:var(--fg);font-size:14px;line-height:1.4}
#audio-block{background:var(--bg2);border-radius:var(--r);padding:12px}
#player{width:100%;outline:none}
#act-buttons{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
button{padding:8px 16px;border:1px solid var(--border);border-radius:var(--r);font-size:14px;
  cursor:pointer;background:var(--bg3);color:var(--fg);transition:background .12s,border-color .12s}
button:hover{background:var(--border)}
button:active{transform:scale(.97)}
button.keep{border-color:var(--green);color:var(--green)}
button.keep:hover{background:#102010}
button.delete{border-color:var(--red);color:var(--red)}
button.delete:hover{background:#201010}
button.rerender{border-color:var(--orange);color:var(--orange)}
button.rerender:hover{background:#201408}
button.nav{border-color:var(--accent);color:var(--accent)}
button.sync{border-color:#8888ff;color:#aaf}
button.sync:hover{background:#1a1a30}
button.sync:disabled{opacity:.4;cursor:default}
#kbd{font-size:12px;color:var(--fg2);line-height:2}
#kbd kbd{background:var(--bg3);border:1px solid var(--border);border-radius:3px;padding:1px 5px;font-size:11px;margin:0 2px}
#sync-status{font-size:12px;color:var(--fg2);white-space:pre-wrap;background:var(--bg2);border-radius:var(--r);
  padding:8px 12px;max-height:120px;overflow-y:auto;display:none;border:1px solid var(--border)}
</style>
</head>
<body>
<div id="sidebar">
  <div id="filters">
    <select id="fEmotion"><option value="all">Emotion: alle</option></select>
    <select id="fDecision">
      <option value="all">Status: alle</option>
      <option value="unreviewed">unreviewed</option>
      <option value="keep">keep</option>
      <option value="delete">delete</option>
      <option value="rerender">rerender</option>
    </select>
  </div>
  <div id="search-wrap">
    <input type="text" id="search" placeholder="Text / Dateiname suchen…">
  </div>
  <div id="stats">–</div>
  <div id="filelist"></div>
</div>
<div id="main">
  <div id="topbar">
    <div style="min-width:0">
      <h1 id="curTitle">QC Review</h1>
      <div id="reviewer-label">Reviewer: <b>%%REVIEWER%%</b></div>
    </div>
    <div id="topbar-right">
      <label style="display:flex;align-items:center;gap:5px;font-size:13px;color:var(--fg2);cursor:pointer">
        <input type="checkbox" id="autoplay" checked style="accent-color:var(--accent)"> Autoplay
      </label>
      <div id="progress">–</div>
      <button class="sync" id="btnSync" onclick="syncGit()" title="Änderungen auf GitHub pushen">↑ Sync</button>
    </div>
  </div>
  <div id="detail">
    <div class="meta-row">
      <span id="decisionBadge" class="badge unreviewed">unreviewed</span>
      <span id="emotionBadge" class="badge emotion" style="display:none">–</span>
    </div>
    <div id="otherVotes" class="other-votes" style="display:none"></div>
    <div class="field">
      <div class="flabel">Text</div>
      <div class="fval" id="curText">–</div>
    </div>
    <div id="audio-block">
      <audio id="player" controls preload="none" style="width:100%"></audio>
    </div>
    <div id="act-buttons">
      <button class="nav" onclick="prevFile()" title="B / ←">⏮ Zurück</button>
      <button onclick="togglePlay()" title="Leertaste">▶ Play</button>
      <button class="keep"    onclick="decide('keep')"     title="K">✓ Keep</button>
      <button class="delete"  onclick="decide('delete')"   title="D">✕ Delete</button>
      <button class="rerender" onclick="decide('rerender')" title="R">↻ Rerender</button>
      <button class="nav" onclick="nextFile()" title="N / →">Vor ⏭</button>
    </div>
    <div id="kbd">
      <kbd>K</kbd> Keep &nbsp;
      <kbd>D</kbd> Delete &nbsp;
      <kbd>R</kbd> Rerender &nbsp;
      <kbd>N</kbd>/<kbd>→</kbd> Weiter &nbsp;
      <kbd>B</kbd>/<kbd>←</kbd> Zurück &nbsp;
      <kbd>Leerz.</kbd> Play/Pause
    </div>
    <div id="sync-status"></div>
  </div>
</div>
<script>
const HF_BASE = "%%HF_BASE%%";
let clips = [], filtered = [], idx = -1;

async function init() {
  const data = await fetch('/api/emotions').then(r => r.json());
  const sel = document.getElementById('fEmotion');
  for (const e of data) {
    const o = document.createElement('option');
    o.value = e; o.textContent = e; sel.appendChild(o);
  }
  await loadClips();
  document.getElementById('fEmotion').addEventListener('change', () => loadClips());
  document.getElementById('fDecision').addEventListener('change', () => loadClips());
  document.getElementById('search').addEventListener('input', () => { applySearch(); renderList(); });
  document.addEventListener('keydown', onKey);
}

async function loadClips() {
  const emo = document.getElementById('fEmotion').value;
  const dec = document.getElementById('fDecision').value;
  const p = new URLSearchParams({emotion: emo, decision: dec});
  const data = await fetch('/api/clips?' + p).then(r => r.json());
  clips = data.clips; filtered = [...clips];
  applySearch(); renderList(); renderStats(data.stats);
  if (filtered.length > 0 && idx < 0) select(0, false);
}

function applySearch() {
  const t = document.getElementById('search').value.toLowerCase().trim();
  filtered = t ? clips.filter(c =>
    (c.text||'').toLowerCase().includes(t) || c.file_name.toLowerCase().includes(t)
  ) : [...clips];
}

function renderStats(s) {
  if (!s) return;
  document.getElementById('stats').textContent =
    filtered.length + '/' + s.total + ' · ✓' + (s.keep||0) + ' ✕' + (s.delete||0) +
    ' ↻' + (s.rerender||0) + ' ○' + (s.unreviewed||0);
}

function renderList() {
  const el = document.getElementById('filelist');
  el.innerHTML = '';
  for (let i = 0; i < Math.min(filtered.length, 500); i++) {
    const c = filtered[i];
    const div = document.createElement('div');
    div.className = 'fl-item' + (i === idx ? ' active' : '');
    const otherDots = Object.entries(c.others||{}).map(([,d]) =>
      `<span class="fl-o ${d||'unreviewed'}" title="${d}"></span>`).join('');
    div.innerHTML = '<span class="fl-dot ' + (c.decision||'unreviewed') + '"></span>' +
      '<span class="fl-emo">[' + (c.emotion||'?') + ']</span>' +
      '<span class="fl-name" title="' + c.file_name + '">' + c.file_name.split('/').pop() + '</span>' +
      (otherDots ? '<span class="fl-others">' + otherDots + '</span>' : '');
    const ii = i;
    div.onclick = () => select(ii, true);
    el.appendChild(div);
  }
  if (filtered.length > 500) {
    const d = document.createElement('div');
    d.style.cssText = 'padding:8px 12px;font-size:12px;color:var(--fg2)';
    d.textContent = '… ' + (filtered.length - 500) + ' weitere (Filter nutzen)';
    el.appendChild(d);
  }
}

function select(i, play) {
  if (i < 0 || i >= filtered.length) return;
  idx = i; const c = filtered[i];
  document.getElementById('curTitle').textContent = c.file_name.split('/').pop();
  const dec = c.decision || 'unreviewed';
  const db = document.getElementById('decisionBadge');
  db.textContent = dec; db.className = 'badge ' + dec;
  const eb = document.getElementById('emotionBadge');
  if (c.emotion) { eb.textContent = c.emotion; eb.style.display = ''; }
  else eb.style.display = 'none';
  document.getElementById('curText').textContent = c.text || '–';

  // Other reviewer votes
  const othersEl = document.getElementById('otherVotes');
  const others = c.others || {};
  const otherKeys = Object.keys(others);
  if (otherKeys.length > 0) {
    const myDec = c.decision || 'unreviewed';
    const allVotes = otherKeys.map(rv => others[rv]);
    const allSame = allVotes.every(v => v === myDec) && myDec !== 'unreviewed';
    const hasConflict = myDec !== 'unreviewed' && allVotes.some(v => v !== myDec && v);
    let html = '<span style="font-size:11px;color:var(--fg2);margin-right:4px">Andere:</span>';
    for (const [rv, dec] of Object.entries(others)) {
      html += `<span class="other-vote ${dec||'unreviewed'}">${rv}: ${dec||'–'}</span>`;
    }
    if (hasConflict) html += '<span class="badge conflict" style="margin-left:6px">Konflikt</span>';
    else if (allSame) html += '<span class="badge agree" style="margin-left:6px">Einig</span>';
    othersEl.innerHTML = html; othersEl.style.display = 'flex';
  } else {
    othersEl.innerHTML = ''; othersEl.style.display = 'none';
  }

  const url = HF_BASE + '/' + c.file_name.split('/').map(encodeURIComponent).join('/');
  const player = document.getElementById('player');
  player.src = url; player.load();
  if (play && document.getElementById('autoplay').checked) player.play().catch(()=>{});
  document.getElementById('progress').textContent = (i+1) + ' / ' + filtered.length;
  renderList();
  const items = document.getElementById('filelist').querySelectorAll('.fl-item');
  if (items[i]) items[i].scrollIntoView({block:'nearest'});
}

async function decide(decision) {
  if (idx < 0 || idx >= filtered.length) return;
  const c = filtered[idx];
  const res = await fetch('/api/clips/decision', {
    method:'PATCH', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({file_name: c.file_name, decision})
  });
  if (!res.ok) { alert('Fehler'); return; }
  const u = await res.json();
  c.decision = u.decision;
  for (const f of clips) if (f.file_name === c.file_name) f.decision = u.decision;
  const stats = await fetch('/api/stats').then(r=>r.json());
  renderStats(stats); renderList();
  const next = idx + 1 < filtered.length ? idx + 1 : idx;
  select(next, next !== idx);
}

function nextFile() { if (idx+1<filtered.length) select(idx+1,true); }
function prevFile() { if (idx>0) select(idx-1,true); }
function togglePlay() { const p=document.getElementById('player'); p.paused?p.play():p.pause(); }

function onKey(e) {
  if (['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
  if (e.key===' ') { e.preventDefault(); togglePlay(); }
  else if (e.key==='k'||e.key==='K') decide('keep');
  else if (e.key==='d'||e.key==='D') decide('delete');
  else if (e.key==='r'||e.key==='R') decide('rerender');
  else if (e.key==='ArrowRight'||e.key==='n'||e.key==='N') { e.preventDefault(); nextFile(); }
  else if (e.key==='ArrowLeft' ||e.key==='b'||e.key==='B') { e.preventDefault(); prevFile(); }
}

async function syncGit() {
  const btn = document.getElementById('btnSync');
  const box = document.getElementById('sync-status');
  btn.disabled = true; btn.textContent = '↑ …';
  box.style.display = 'block'; box.textContent = 'Pushing…';
  try {
    const res = await fetch('/api/sync', {method:'POST'});
    const d = await res.json();
    box.textContent = d.output || d.error || JSON.stringify(d);
    box.style.color = d.ok ? '#4caf50' : '#e53935';
    btn.textContent = d.ok ? '↑ OK' : '↑ Fehler';
    if (d.ok) await loadClips();
    setTimeout(() => { btn.textContent = '↑ Sync'; btn.disabled = false; }, 3000);
  } catch(e) {
    box.textContent = 'Fehler: ' + e.message;
    box.style.color = '#e53935';
    btn.textContent = '↑ Sync'; btn.disabled = false;
  }
}

init();
</script>
</body>
</html>
"""


class ReviewState:
    def __init__(self, clips: list[dict], reviewer: str, state_path: Path, state_dir: Path):
        self._clips = clips
        self._by_name = {c["file_name"]: i for i, c in enumerate(clips)}
        self.reviewer = reviewer
        self._path = state_path
        self._state_dir = state_dir
        self._decisions: dict[str, dict] = {}
        # {file_name: {other_reviewer: decision}}
        self._others: dict[str, dict[str, str]] = {}
        self._lock = threading.Lock()
        self._load_own()
        self._load_others()
        self._apply_to_clips()

    def _load_own(self):
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._decisions = data.get("decisions", {})
            print(f"Loaded {len(self._decisions)} own decisions", file=sys.stderr)
        except Exception as exc:
            print(f"WARNING: could not load own state: {exc}", file=sys.stderr)

    def _load_others(self):
        self._others = {}
        for p in self._state_dir.glob("*.json"):
            if p == self._path:
                continue
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                name = data.get("reviewer") or p.stem
                for fn, entry in data.get("decisions", {}).items():
                    self._others.setdefault(fn, {})[name] = entry.get("decision", "")
                print(f"Loaded decisions from {name} ({len(data.get('decisions', {}))} clips)", file=sys.stderr)
            except Exception as exc:
                print(f"WARNING: could not load {p.name}: {exc}", file=sys.stderr)

    def _apply_to_clips(self):
        for c in self._clips:
            fn = c["file_name"]
            e = self._decisions.get(fn)
            c["decision"] = e["decision"] if e else "unreviewed"
            c["others"] = self._others.get(fn, {})

    def reload_others(self):
        with self._lock:
            self._load_own()
            self._load_others()
            self._apply_to_clips()

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "reviewer": self.reviewer,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_clips": len(self._clips),
            "decisions": self._decisions,
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    def set_decision(self, file_name: str, decision: str) -> dict:
        if decision not in DECISIONS:
            raise ValueError(decision)
        i = self._by_name.get(file_name)
        if i is None:
            raise KeyError(file_name)
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._decisions[file_name] = {"decision": decision, "reviewed_at": now}
            self._clips[i]["decision"] = decision
            self._save()
        return self._clips[i]

    def get_filtered(self, emotion: str = "all", decision: str = "all") -> list[dict]:
        result = self._clips
        if emotion != "all":
            result = [c for c in result if c.get("emotion") == emotion]
        if decision != "all":
            result = [c for c in result if c.get("decision", "unreviewed") == decision]
        return result

    @property
    def emotions(self) -> list[str]:
        return sorted({c.get("emotion", "") for c in self._clips if c.get("emotion")})

    @property
    def stats(self) -> dict:
        counts = {d: 0 for d in ("keep", "delete", "rerender", "unreviewed")}
        for c in self._clips:
            dec = c.get("decision", "unreviewed")
            counts[dec] = counts.get(dec, 0) + 1
        return {"total": len(self._clips), **counts}


def fetch_metadata(hf_base: str) -> list[dict]:
    url = hf_base + "/metadata.jsonl"
    print(f"Fetching {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "qc-review/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
    except Exception as exc:
        sys.exit(f"ERROR: could not fetch metadata from HuggingFace:\n  {exc}\n  URL: {url}")
    clips = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            clips.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not clips:
        sys.exit(f"ERROR: metadata.jsonl at {url} returned no valid records")
    return clips


def normalize_clips(raw: list[dict]) -> list[dict]:
    """Normalize field names to file_name / text / emotion."""
    clips = []
    for r in raw:
        c = dict(r)
        if "file_name" not in c:
            # HF datasets sometimes store audio as {"path": ..., "bytes": null}
            audio = c.get("audio") or {}
            c["file_name"] = (audio.get("path") or c.get("path") or "").lstrip("/")
        if "text" not in c:
            c["text"] = c.get("transcription") or c.get("transcript") or c.get("sentence") or ""
        if "emotion" not in c:
            # derive from folder name if possible
            fn = c.get("file_name", "")
            c["emotion"] = Path(fn).parent.name if fn else ""
        if not c.get("file_name"):
            continue
        clips.append({k: c[k] for k in ("file_name", "text", "emotion") if k in c})
    return clips


class DecisionUpdate(BaseModel):
    file_name: str
    decision: str


def build_app(state: ReviewState, hf_base: str, state_dir: Path) -> FastAPI:
    app = FastAPI(title="QC Review")

    @app.get("/", response_class=HTMLResponse)
    def root():
        return _HTML.replace("%%REVIEWER%%", state.reviewer).replace("%%HF_BASE%%", hf_base)

    @app.get("/api/emotions")
    def api_emotions():
        return state.emotions

    @app.get("/api/clips")
    def api_clips(emotion: str = "all", decision: str = "all"):
        clips = state.get_filtered(emotion=emotion, decision=decision)
        return {"clips": clips, "stats": state.stats}

    @app.get("/api/stats")
    def api_stats():
        return state.stats

    @app.patch("/api/clips/decision")
    def api_decision(body: DecisionUpdate):
        if body.decision not in DECISIONS:
            raise HTTPException(400, f"Invalid decision '{body.decision}'")
        try:
            clip = state.set_decision(body.file_name, body.decision)
        except KeyError:
            raise HTTPException(404, body.file_name)
        return clip

    @app.post("/api/sync")
    def api_sync():
        s = state.stats
        msg = f"review: {state.reviewer} ({s.get('keep',0)} keep, {s.get('delete',0)} delete, {s.get('rerender',0)} rerender)"
        fname = f"{state.reviewer}.json"
        lines = []
        ok = False
        try:
            def run(cmd):
                r = subprocess.run(cmd, cwd=state_dir, capture_output=True, text=True)
                out = (r.stdout + r.stderr).strip()
                if out:
                    lines.append(out)
                return r.returncode

            run(["git", "add", fname])
            commit_rc = run(["git", "commit", "-m", msg])
            combined = "\n".join(lines)
            if commit_rc != 0 and ("nothing to commit" in combined or "nothing added" in combined):
                lines.append("(keine neuen Entscheidungen)")
            # pull first to get others' changes
            pull_rc = run(["git", "pull", "--rebase"])
            push_rc = run(["git", "push"])
            ok = push_rc == 0
            # reload others from newly pulled files
            state.reload_others()
        except Exception as exc:
            lines.append(f"Fehler: {exc}")
        return {"ok": ok, "output": "\n".join(lines)}

    return app


def main():
    parser = argparse.ArgumentParser(description="QC Review Server — streams audio from HuggingFace")
    parser.add_argument("--reviewer", required=True, help="Reviewer name → reviews/<name>.json")
    parser.add_argument("--hf-dataset", default="dida-80b/victoria-emotion-de-synthetic",
                        help="HuggingFace dataset ID (default: dida-80b/victoria-emotion-de-synthetic)")
    parser.add_argument("--hf-branch", default="main", help="HuggingFace branch (default: main)")
    parser.add_argument("--state-dir", type=Path, default=Path("reviews"),
                        help="Directory for state files (default: reviews/)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8091)
    args = parser.parse_args()

    hf_base = f"https://huggingface.co/datasets/{args.hf_dataset}/resolve/{args.hf_branch}"
    raw = fetch_metadata(hf_base)
    clips = normalize_clips(raw)
    if not clips:
        sys.exit("ERROR: no valid clips found in metadata.jsonl")

    state_dir = args.state_dir if args.state_dir.is_absolute() else Path.cwd() / args.state_dir
    state_path = state_dir / f"{args.reviewer}.json"

    state = ReviewState(clips=clips, reviewer=args.reviewer, state_path=state_path, state_dir=state_dir)
    app = build_app(state, hf_base, state_dir)

    print(f"Clips:    {len(clips)}")
    print(f"Reviewer: {args.reviewer}")
    print(f"State:    {state_path}")
    print(f"Dataset:  {args.hf_dataset} ({args.hf_branch})")
    print(f"URL:      http://{args.host}:{args.port}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
