#!/usr/bin/env python3
import argparse
import csv
import html
import json
import shutil
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def truthy(value) -> bool:
    return value is True or str(value).lower() == "true"


def sort_key(row: dict):
    reject = 0 if truthy(row.get("reject_candidate")) else 1
    review = 0 if truthy(row.get("review")) else 1
    quality = row.get("overall_quality")
    quality = quality if isinstance(quality, int) else 99
    abrupt = 0 if truthy(row.get("abrupt_ending")) else 1
    return (reject, review, quality, abrupt, row.get("file_name", ""))


def selected_rows(rows: list[dict], mode: str, limit: int) -> list[dict]:
    if mode == "all":
        picked = rows
    elif mode == "rejects":
        picked = [row for row in rows if truthy(row.get("reject_candidate"))]
    elif mode == "review":
        picked = [row for row in rows if truthy(row.get("review")) or truthy(row.get("reject_candidate"))]
    elif mode == "low-quality":
        picked = [row for row in rows if (row.get("overall_quality") or 99) <= 7 or truthy(row.get("reject_candidate"))]
    else:
        raise ValueError(f"unknown mode: {mode}")

    picked = sorted(picked, key=sort_key)
    if limit > 0:
        picked = picked[:limit]
    return picked


def safe_name(index: int, row: dict) -> str:
    source = Path(row["file_name"])
    quality = row.get("overall_quality", "na")
    review = "review" if truthy(row.get("review")) else "ok"
    reject = "reject" if truthy(row.get("reject_candidate")) else "keep"
    return f"{index:03d}_q{quality}_{review}_{reject}_{source.parent.name}_{source.name}"


def write_csv(path: Path, rows: list[dict], names: list[str]):
    fields = [
        "review_file",
        "file_name",
        "emotion",
        "text",
        "overall_quality",
        "audio_quality",
        "loudness_quality",
        "text_quality",
        "emotion_clarity",
        "review",
        "reject_candidate",
        "abrupt_ending",
        "breath_at_start",
        "noise_at_start",
        "artifacts",
        "integrated_lufs",
        "true_peak_dbfs",
        "reason",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row, name in zip(rows, names):
            writer.writerow({**row, "review_file": name})


def write_html(path: Path, rows: list[dict], names: list[str]):
    parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Victoria QC Review</title>",
        "<style>body{font-family:sans-serif;max-width:1200px;margin:2rem auto;line-height:1.35}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:.45rem;vertical-align:top}th{background:#f3f3f3}.bad{background:#ffecec}.review{background:#fff8dd}.ok{background:#eefbea}audio{width:260px}</style>",
        "<h1>Victoria QC Review</h1>",
        f"<p>{len(rows)} Clips. Audios are symlinks to the dataset files, not copies.</p>",
        "<table>",
        "<tr><th>#</th><th>Audio</th><th>Quality</th><th>Emotion</th><th>Text</th><th>Flags</th><th>Reason</th><th>Source</th></tr>",
    ]
    for idx, (row, name) in enumerate(zip(rows, names), 1):
        cls = "bad" if truthy(row.get("reject_candidate")) else "review" if truthy(row.get("review")) else "ok"
        flags = []
        for key in ["abrupt_ending", "breath_at_start", "noise_at_start"]:
            if truthy(row.get(key)):
                flags.append(key)
        artifacts = row.get("artifacts") or []
        flags.extend(str(item) for item in artifacts)
        parts.append(
            "<tr class='{}'><td>{}</td><td><audio controls src='clips/{}'></audio><br><code>{}</code></td><td>overall {}<br>audio {}<br>emotion {}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td><code>{}</code></td></tr>".format(
                cls,
                idx,
                html.escape(name),
                html.escape(name),
                html.escape(str(row.get("overall_quality"))),
                html.escape(str(row.get("audio_quality"))),
                html.escape(str(row.get("emotion_clarity"))),
                html.escape(str(row.get("emotion"))),
                html.escape(str(row.get("text"))),
                html.escape(", ".join(flags)),
                html.escape(str(row.get("reason"))),
                html.escape(str(row.get("file_name"))),
            )
        )
    parts.append("</table>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Create a symlink-based human QC review folder.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input-jsonl", type=Path, default=Path("qc/pilot_voxtral_30_v2.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("qc/review_trash"))
    parser.add_argument("--mode", choices=["review", "rejects", "low-quality", "all"], default="review")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    input_jsonl = args.input_jsonl if args.input_jsonl.is_absolute() else root / args.input_jsonl
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    clips_dir = output_dir / "clips"

    if output_dir.exists():
        if not args.replace:
            raise SystemExit(f"output dir exists; use --replace to overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    clips_dir.mkdir(parents=True)

    rows = selected_rows(load_rows(input_jsonl), args.mode, args.limit)
    names = []
    for idx, row in enumerate(rows, 1):
        name = safe_name(idx, row)
        names.append(name)
        src = root / row["file_name"]
        dst = clips_dir / name
        dst.symlink_to(src)

    write_csv(output_dir / "review_manifest.csv", rows, names)
    write_html(output_dir / "index.html", rows, names)
    (output_dir / "README.txt").write_text(
        "This folder contains symlinks, not audio copies. Delete this folder when done.\n"
        "Open index.html in a browser or review review_manifest.csv.\n",
        encoding="utf-8",
    )
    print(f"created {len(rows)} symlinks in {clips_dir}")
    print(f"open {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
