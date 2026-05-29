#!/usr/bin/env python3
import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Create a balanced QC pilot sample.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--per-emotion", type=int, default=1)
    parser.add_argument("--extra-random", type=int, default=2)
    parser.add_argument("--seed", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("qc/pilot_sample.csv"))
    args = parser.parse_args()

    root = args.root.resolve()
    rng = random.Random(args.seed)
    rows = [json.loads(line) for line in (root / "metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_emotion = defaultdict(list)
    for row in rows:
        by_emotion[row["emotion"]].append(row)

    sample = []
    seen = set()
    for emotion in sorted(by_emotion):
        for row in rng.sample(by_emotion[emotion], min(args.per_emotion, len(by_emotion[emotion]))):
            sample.append(row)
            seen.add(row["file_name"])
    remaining = [row for row in rows if row["file_name"] not in seen]
    sample.extend(rng.sample(remaining, min(args.extra_random, len(remaining))))

    out = args.output if args.output.is_absolute() else root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_name", "text", "emotion", "candidate_id"])
        writer.writeheader()
        writer.writerows(sample)
    print(f"wrote {len(sample)} rows to {out}")


if __name__ == "__main__":
    main()
