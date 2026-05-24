#!/usr/bin/env python3
"""
Merge multiple reviewer state files (reviews/*.json) into consensus output.

Output files (in --output-dir):
  consensus_keep.jsonl     — all reviewers agreed: keep
  consensus_delete.jsonl   — all reviewers agreed: delete
  consensus_rerender.jsonl — all reviewers agreed: rerender
  conflicts.jsonl          — reviewers disagreed
  partial.jsonl            — not all reviewers have reviewed this clip
  summary.json             — counts + per-reviewer stats
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


DECISIONS = ("keep", "delete", "rerender")


def load_state_files(state_dir: Path) -> dict[str, dict]:
    """Return {reviewer_name: state_dict} for all *.json in state_dir."""
    states = {}
    for p in sorted(state_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"WARNING: could not read {p}: {exc}", file=sys.stderr)
            continue
        reviewer = data.get("reviewer") or p.stem
        states[reviewer] = data
    return states


def merge(states: dict[str, dict], min_reviewers: int) -> dict:
    """
    Collect all file_names from all state files, then for each clip determine:
      - consensus  → all present reviewers agree AND count >= min_reviewers
      - conflict   → at least 2 reviewers disagree
      - partial    → fewer than min_reviewers have reviewed this clip
    """
    # Gather all known clip names
    all_clips: set[str] = set()
    for state in states.values():
        all_clips.update(state.get("decisions", {}).keys())

    results = {
        "keep": [],
        "delete": [],
        "rerender": [],
        "conflicts": [],
        "partial": [],
    }

    reviewers = list(states.keys())

    for file_name in sorted(all_clips):
        clip_decisions: dict[str, str] = {}  # reviewer → decision
        for reviewer, state in states.items():
            entry = state.get("decisions", {}).get(file_name)
            if entry:
                clip_decisions[reviewer] = entry["decision"]

        reviewed_by = list(clip_decisions.keys())
        n = len(reviewed_by)
        unique = set(clip_decisions.values())

        record = {
            "file_name": file_name,
            "decisions": clip_decisions,
            "reviewed_by": reviewed_by,
            "reviewer_count": n,
        }

        if n < min_reviewers:
            results["partial"].append(record)
        elif len(unique) == 1:
            decision = unique.pop()
            if decision in results:
                results[decision].append(record)
            else:
                results["partial"].append(record)
        else:
            results["conflicts"].append(record)

    return results


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Merge reviewer state files into consensus output.")
    parser.add_argument("--state-dir", type=Path, default=Path("reviews"),
                        help="Directory containing reviewer *.json state files (default: reviews/)")
    parser.add_argument("--output-dir", type=Path, default=Path("qc/merge"),
                        help="Directory for output files (default: qc/merge/)")
    parser.add_argument("--min-reviewers", type=int, default=2,
                        help="Minimum number of reviewers required for consensus (default: 2)")
    parser.add_argument("--replace", action="store_true",
                        help="Overwrite existing output files")
    args = parser.parse_args()

    state_dir = args.state_dir if args.state_dir.is_absolute() else Path.cwd() / args.state_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else Path.cwd() / args.output_dir

    if not state_dir.exists():
        sys.exit(f"ERROR: state-dir not found: {state_dir}")

    states = load_state_files(state_dir)
    if not states:
        sys.exit(f"ERROR: no reviewer state files found in {state_dir}")

    print(f"Loaded {len(states)} reviewer(s): {', '.join(sorted(states))}", file=sys.stderr)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.replace:
        sys.exit(f"ERROR: output-dir {output_dir} is not empty — use --replace to overwrite")

    merged = merge(states, args.min_reviewers)

    write_jsonl(output_dir / "consensus_keep.jsonl", merged["keep"])
    write_jsonl(output_dir / "consensus_delete.jsonl", merged["delete"])
    write_jsonl(output_dir / "consensus_rerender.jsonl", merged["rerender"])
    write_jsonl(output_dir / "conflicts.jsonl", merged["conflicts"])
    write_jsonl(output_dir / "partial.jsonl", merged["partial"])

    # Per-reviewer stats
    reviewer_stats = {}
    for reviewer, state in states.items():
        decisions = state.get("decisions", {})
        counts = {d: 0 for d in DECISIONS}
        for entry in decisions.values():
            d = entry.get("decision")
            if d in counts:
                counts[d] += 1
        reviewer_stats[reviewer] = {
            "total_reviewed": len(decisions),
            **counts,
            "updated_at": state.get("updated_at"),
        }

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "min_reviewers": args.min_reviewers,
        "reviewers": sorted(states.keys()),
        "consensus_keep": len(merged["keep"]),
        "consensus_delete": len(merged["delete"]),
        "consensus_rerender": len(merged["rerender"]),
        "conflicts": len(merged["conflicts"]),
        "partial": len(merged["partial"]),
        "total_clips": sum(len(v) for v in merged.values()),
        "per_reviewer": reviewer_stats,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"consensus keep={len(merged['keep'])}  delete={len(merged['delete'])}"
        f"  rerender={len(merged['rerender'])}"
        f"  conflicts={len(merged['conflicts'])}  partial={len(merged['partial'])}",
    )
    print(f"output: {output_dir}/")

    if merged["conflicts"]:
        print(f"\nConflicts ({len(merged['conflicts'])}):")
        for r in merged["conflicts"]:
            votes = ", ".join(f"{rv}={d}" for rv, d in sorted(r["decisions"].items()))
            print(f"  {r['file_name']}  [{votes}]")


if __name__ == "__main__":
    main()
