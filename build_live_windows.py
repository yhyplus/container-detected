#!/usr/bin/env python3
"""Aggregate recent Pixie exports into unlabeled windows for inference."""

import argparse
import csv
from datetime import datetime, timedelta
from pathlib import Path

from build_dataset import FIELDS, aggregate_scenario, parse_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lookback-seconds", type=int, default=120)
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--generator-ip", default="")
    args = parser.parse_args()

    end = parse_time(args.end_time) if args.end_time else datetime.now()
    start = parse_time(args.start_time) if args.start_time else end - timedelta(seconds=args.lookback_seconds)
    rows = [
        row for row in aggregate_scenario(
            args.raw_dir, "normal", 0, start, end, args.generator_ip
        )
        if row["window_coverage_ratio"] >= 0.5
    ]
    if not rows:
        raise SystemExit("recent Pixie exports did not produce any inference windows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["window_start"], row["pod"])))
    print(f"windows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
