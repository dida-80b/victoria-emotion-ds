#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import re
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def generate(model: str, prompt: str, audio_path: Path, timeout: int, host: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "images": [base64.b64encode(audio_path.read_bytes()).decode("ascii")],
        "options": {"temperature": 0, "top_p": 0.9, "num_ctx": 4096},
    }
    req = urllib.request.Request(host.rstrip("/") + "/api/generate", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "")


def build_prompt(template: str, row: dict) -> str:
    metrics = {k: row.get(k) for k in ["duration_sec", "peak_dbfs", "integrated_lufs", "true_peak_dbfs", "rms_dbfs", "start_200ms_rms_dbfs", "end_200ms_rms_dbfs", "leading_silence_ms", "trailing_silence_ms", "signal_flags"]}
    return template + "\n\nClip-Kontext:\n" + json.dumps({"file_name": row["file_name"], "expected_text": row["text"], "expected_emotion": row["emotion"], "signal_metrics": metrics}, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Run Ollama/Voxtral audio QC.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-jsonl", type=Path, default=Path("qc/pilot_signal_metrics.jsonl"))
    parser.add_argument("--prompt", type=Path, default=Path("qc/prompt.md"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("qc/pilot_voxtral.jsonl"))
    parser.add_argument("--output-csv", type=Path, default=Path("qc/pilot_voxtral.csv"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--host", default="http://127.0.0.1:11434")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed clips and append to existing output")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel Ollama requests (use with OLLAMA_NUM_PARALLEL)")
    parser.add_argument("--live-review-dir", type=Path, default=None, help="Create symlinks here immediately for every flagged clip")
    args = parser.parse_args()

    root = args.root.resolve()
    input_jsonl = args.input_jsonl if args.input_jsonl.is_absolute() else root / args.input_jsonl
    prompt_path = args.prompt if args.prompt.is_absolute() else root / args.prompt
    out_jsonl = args.output_jsonl if args.output_jsonl.is_absolute() else root / args.output_jsonl
    out_csv = args.output_csv if args.output_csv.is_absolute() else root / args.output_csv
    template = prompt_path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in input_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    already_done: set[str] = set()
    if args.resume and out_jsonl.exists():
        for line in out_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    already_done.add(json.loads(line)["file_name"])
                except Exception:
                    pass
        print(f"resume: skipping {len(already_done)} already-processed clips", file=sys.stderr)

    rows_to_process = [r for r in rows if r["file_name"] not in already_done]
    file_mode = "a" if args.resume and out_jsonl.exists() else "w"

    total = len(rows)
    done_so_far = len(already_done)
    counter = {"n": done_so_far}
    write_lock = threading.Lock()

    live_dir = None
    if args.live_review_dir:
        live_dir = args.live_review_dir if args.live_review_dir.is_absolute() else root / args.live_review_dir
        (live_dir / "clips").mkdir(parents=True, exist_ok=True)

    def maybe_symlink(result: dict):
        if live_dir is None:
            return
        if not (str(result.get("review")).lower() == "true" or str(result.get("reject_candidate")).lower() == "true"):
            return
        src = root / result["file_name"]
        quality = result.get("overall_quality", "na")
        tag = "reject" if str(result.get("reject_candidate")).lower() == "true" else "review"
        emotion = Path(result["file_name"]).parent.name
        fname = Path(result["file_name"]).name
        link_name = f"q{quality}_{tag}_{emotion}_{fname}"
        dst = live_dir / "clips" / link_name
        if not dst.exists():
            dst.symlink_to(src)

    def process_row(row: dict) -> dict:
        raw = generate(args.model, build_prompt(template, row), root / row["file_name"], args.timeout, args.host)
        try:
            review = extract_json(raw)
            parse_error = None
        except Exception as exc:
            review = {}
            parse_error = str(exc)
        return {**row, "voxtral_raw": raw, "parse_error": parse_error, **review}

    results = []
    with out_jsonl.open(file_mode, encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_row, row): row for row in rows_to_process}
            for future in as_completed(futures):
                result = future.result()
                with write_lock:
                    counter["n"] += 1
                    flag = " [REVIEW]" if str(result.get("review")).lower() == "true" else ""
                    flag = " [REJECT]" if str(result.get("reject_candidate")).lower() == "true" else flag
                    print(f"[{counter['n']}/{total}] done {result['file_name']}{flag}", file=sys.stderr)
                    results.append(result)
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    maybe_symlink(result)
    all_results = [json.loads(line) for line in out_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    fields = ["file_name", "emotion", "text", "overall_quality", "audio_quality", "loudness_quality", "text_quality", "emotion_clarity", "review", "reject_candidate", "abrupt_ending", "speech_rate", "artifacts", "leading_silence_ms", "trailing_silence_ms", "hard_cutoff_candidate", "signal_flags", "reason", "parse_error"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_results)
    print(f"wrote {len(all_results)} rows total ({len(results)} new) to {out_jsonl} and {out_csv}")


if __name__ == "__main__":
    main()
