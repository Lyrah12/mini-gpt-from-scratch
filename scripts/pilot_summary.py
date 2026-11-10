"""Estimate compute time from measured steady-state steps, excluding setup/eval/I/O."""
import argparse
import statistics
from pathlib import Path

from minigpt.common import jsonl, read_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="runs/cloud/pretrain")
    ap.add_argument("--hourly-price", type=float, help="Actual instance GPU hourly price shown by AutoDL")
    args = ap.parse_args()
    root = Path(args.run)
    info, status = read_json(root / "run.json"), read_json(root / "status.json")
    rows = [r for r in jsonl(root / "metrics.jsonl") if r["event"] == "train" and r["step"] >= 10]
    if len(rows) < 2:
        raise ValueError("Need at least two logged training steps after warm-up; run a 30-step pilot")
    seconds = statistics.median(r["step_seconds"] for r in rows)
    hours = seconds * (status["planned_steps"] - status["step"]) / 3600
    print(f"GPU: {info['environment']['gpu']}; precision: {info['actual_precision']}")
    print(f"Median sampled step: {seconds:.3f} s; remaining pretrain compute: {hours:.2f} h")
    print(f"Peak allocated GPU memory: {max(r['peak_cuda_memory_bytes'] for r in rows) / 2**30:.2f} GiB")
    if args.hourly_price is not None:
        if args.hourly_price < 0:
            raise ValueError("Hourly price cannot be negative")
        print(f"Estimated remaining pretrain compute charge: {hours * args.hourly_price:.2f} yuan")
    print("Estimate excludes validation, checkpoint writes, downloads, SFT, evaluation and storage fees.")


if __name__ == "__main__":
    main()
