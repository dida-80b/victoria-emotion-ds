#!/usr/bin/env python3
import argparse
import array
import csv
import json
import math
import re
import subprocess
import wave
from pathlib import Path


def pcm16(data: bytes) -> array.array:
    samples = array.array("h")
    samples.frombytes(data)
    return samples


def rms(samples: array.array) -> float:
    return math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0


def dbfs(value: float) -> float | None:
    return 20 * math.log10(value / 32768.0) if value > 0 else None


def frame_rms_values(samples: array.array, frame_size: int) -> list[float]:
    return [rms(samples[i : i + frame_size]) for i in range(0, len(samples), frame_size)]


def edge_silence_ms(samples: array.array, sample_rate: int, threshold_dbfs: float = -45.0, frame_ms: int = 10) -> tuple[float, float]:
    frame_size = max(1, int(sample_rate * frame_ms / 1000))
    values = frame_rms_values(samples, frame_size)
    threshold = 32768.0 * (10 ** (threshold_dbfs / 20.0))

    lead_frames = 0
    for value in values:
        if value > threshold:
            break
        lead_frames += 1

    trail_frames = 0
    for value in reversed(values):
        if value > threshold:
            break
        trail_frames += 1

    return lead_frames * frame_ms, trail_frames * frame_ms


def loudness(path: Path) -> dict:
    proc = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-filter_complex", "ebur128=peak=true", "-f", "null", "-"], text=True, capture_output=True)
    summary = proc.stderr.split("Summary:")[-1]
    i = re.search(r"Integrated loudness:\s*\n\s*I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", summary)
    p = re.search(r"True peak:\s*\n\s*Peak:\s*(-?\d+(?:\.\d+)?)\s*dBFS", summary)
    return {"integrated_lufs": float(i.group(1)) if i else None, "true_peak_dbfs": float(p.group(1)) if p else None}


def analyze(path: Path, with_lufs: bool) -> dict:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        n = w.getnframes()
        data = w.readframes(n)
    samples = pcm16(data)
    leading_silence_ms, trailing_silence_ms = edge_silence_ms(samples, sr)
    peak = max((abs(s) for s in samples), default=0)
    full_rms = rms(samples)
    win_frames = max(1, int(sr * 0.2))
    bytes_per_frame = sw * ch
    start = pcm16(data[: win_frames * bytes_per_frame])
    end = pcm16(data[-win_frames * bytes_per_frame :]) if data else array.array("h")
    loud = loudness(path) if with_lufs else {"integrated_lufs": None, "true_peak_dbfs": None}

    peak_dbfs = dbfs(peak)
    rms_dbfs = dbfs(full_rms)
    start_dbfs = dbfs(rms(start))
    end_dbfs = dbfs(rms(end))
    flags = []
    if peak_dbfs is not None and peak_dbfs > -0.5:
        flags.append("near_clipping")
    if loud["integrated_lufs"] is not None and loud["integrated_lufs"] < -26:
        flags.append("very_quiet")
    if loud["integrated_lufs"] is not None and loud["integrated_lufs"] > -16:
        flags.append("very_loud")
    if start_dbfs is not None and rms_dbfs is not None and start_dbfs > rms_dbfs - 8:
        flags.append("possible_start_noise")
    if end_dbfs is not None and rms_dbfs is not None and end_dbfs > rms_dbfs - 6:
        flags.append("possible_abrupt_end")
    if leading_silence_ms > 300:
        flags.append("long_leading_silence")
    if trailing_silence_ms < 40 and "possible_abrupt_end" in flags:
        flags.append("hard_cutoff_candidate")
    if trailing_silence_ms < 50:
        flags.append("very_short_trailing_silence")
    elif trailing_silence_ms < 100:
        flags.append("short_trailing_silence")
    elif trailing_silence_ms > 700:
        flags.append("long_trailing_silence")

    return {
        "duration_sec": round(n / sr, 3),
        "sample_rate": sr,
        "channels": ch,
        "sample_width": sw,
        "peak_dbfs": round(peak_dbfs, 2) if peak_dbfs is not None else None,
        "integrated_lufs": round(loud["integrated_lufs"], 2) if loud["integrated_lufs"] is not None else None,
        "true_peak_dbfs": round(loud["true_peak_dbfs"], 2) if loud["true_peak_dbfs"] is not None else None,
        "rms_dbfs": round(rms_dbfs, 2) if rms_dbfs is not None else None,
        "start_200ms_rms_dbfs": round(start_dbfs, 2) if start_dbfs is not None else None,
        "end_200ms_rms_dbfs": round(end_dbfs, 2) if end_dbfs is not None else None,
        "leading_silence_ms": round(leading_silence_ms, 1),
        "trailing_silence_ms": round(trailing_silence_ms, 1),
        "hard_cutoff_candidate": "hard_cutoff_candidate" in flags,
        "signal_flags": flags,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute lightweight signal metrics.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--jsonl-output", type=Path, default=Path("qc/pilot_signal_metrics.jsonl"))
    parser.add_argument("--csv-output", type=Path, default=Path("qc/pilot_signal_metrics.csv"))
    parser.add_argument("--with-lufs", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    input_csv = args.input_csv if args.input_csv.is_absolute() else root / args.input_csv
    rows = list(csv.DictReader(input_csv.open("r", encoding="utf-8", newline="")))
    results = [{**row, **analyze(root / row["file_name"], args.with_lufs)} for row in rows]

    jout = args.jsonl_output if args.jsonl_output.is_absolute() else root / args.jsonl_output
    cout = args.csv_output if args.csv_output.is_absolute() else root / args.csv_output
    jout.parent.mkdir(parents=True, exist_ok=True)
    with jout.open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = ["file_name", "emotion", "text", "candidate_id", "duration_sec", "peak_dbfs", "integrated_lufs", "true_peak_dbfs", "rms_dbfs", "start_200ms_rms_dbfs", "end_200ms_rms_dbfs", "leading_silence_ms", "trailing_silence_ms", "hard_cutoff_candidate", "signal_flags"]
    with cout.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"wrote {len(results)} rows to {jout} and {cout}")


if __name__ == "__main__":
    main()
