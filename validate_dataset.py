#!/usr/bin/env python3
"""Validate the final numeric training dataset."""

import csv
import sys
from collections import Counter
from pathlib import Path


def main():
    path = Path(sys.argv[1])
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("dataset has no rows")
    required = {"scenario", "label", "window_start", "pod", "cpu_pct", "rss_mib",
                "http_rps", "remote_endpoint_count", "dns_failure_rate"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")
    for index, row in enumerate(rows, 2):
        if row["label"] not in {"0", "1"}:
            raise SystemExit(f"line {index}: invalid label")
        for key, value in row.items():
            if key not in {"scenario", "attack_type", "attack_subtype", "anomaly_type",
                           "window_start", "pod"}:
                float(value)
    print(f"rows={len(rows)} labels={dict(Counter(row['label'] for row in rows))}")
    print(f"scenarios={dict(Counter(row['scenario'] for row in rows))}")


if __name__ == "__main__":
    main()
