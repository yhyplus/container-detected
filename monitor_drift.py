#!/usr/bin/env python3
"""Compare feature distributions between reference and current datasets."""

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


EXCLUDED_COLUMNS = {
    "scenario", "attack_type", "attack_subtype", "anomaly_type",
    "class_id", "label", "is_target", "window_coverage_ratio",
    "window_start", "pod",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"dataset has no rows: {path}")
    return rows


def feature_names(rows):
    return [name for name in rows[0] if name not in EXCLUDED_COLUMNS]


def matrix(rows, features):
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float64)


def safe_ratio(current, reference):
    denominator = abs(reference) if abs(reference) > 1e-9 else 1.0
    return (current - reference) / denominator


def summarize_feature(name, reference_values, current_values):
    ref_mean = float(reference_values.mean())
    cur_mean = float(current_values.mean())
    ref_std = float(reference_values.std())
    cur_std = float(current_values.std())
    pooled = math.sqrt(ref_std ** 2 + cur_std ** 2) or 1.0
    z_delta = abs(cur_mean - ref_mean) / pooled
    return {
        "feature": name,
        "reference_mean": ref_mean,
        "current_mean": cur_mean,
        "mean_relative_change": safe_ratio(cur_mean, ref_mean),
        "reference_std": ref_std,
        "current_std": cur_std,
        "std_relative_change": safe_ratio(cur_std, ref_std),
        "z_delta": z_delta,
    }


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_markdown(path, report):
    top = sorted(report["features"], key=lambda item: item["z_delta"], reverse=True)[:12]
    rows = [
        [
            item["feature"],
            f"{item['reference_mean']:.4f}",
            f"{item['current_mean']:.4f}",
            f"{item['mean_relative_change']:.2f}",
            f"{item['z_delta']:.3f}",
        ]
        for item in top
    ]
    text = f"""# Drift Report

Generated at: `{report['generated_at']}`

Reference dataset: `{report['reference']}`

Current dataset: `{report['current']}`

Reference rows: `{report['reference_rows']}`

Current rows: `{report['current_rows']}`

Overall drift score: `{report['overall_drift_score']:.3f}`

## Top Changed Features

{markdown_table(['Feature', 'Reference Mean', 'Current Mean', 'Relative Change', 'Z Delta'], rows)}

## Interpretation

- `Z Delta` compares the mean shift against pooled standard deviation.
- Larger values suggest stronger distribution shift.
- This lightweight drift check is intended for MLOps monitoring, not as a statistical proof of production drift.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("runs/train-v3-combined-20260603/training_multiclass_windows.csv"))
    parser.add_argument("--current", type=Path, default=Path("runs/eval-v3-combined-20260603/training_multiclass_windows.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/drift_report.json"))
    args = parser.parse_args()

    reference_rows = read_rows(args.reference)
    current_rows = read_rows(args.current)
    features = feature_names(reference_rows)
    missing = set(features) - set(current_rows[0])
    if missing:
        raise SystemExit(f"current dataset is missing features: {sorted(missing)}")
    reference_matrix = matrix(reference_rows, features)
    current_matrix = matrix(current_rows, features)
    feature_reports = [
        summarize_feature(name, reference_matrix[:, index], current_matrix[:, index])
        for index, name in enumerate(features)
    ]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference": str(args.reference),
        "current": str(args.current),
        "reference_rows": len(reference_rows),
        "current_rows": len(current_rows),
        "feature_count": len(features),
        "overall_drift_score": float(np.mean([item["z_delta"] for item in feature_reports])),
        "features": feature_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), report)
    print(f"drift_report={args.output}")
    print(f"overall_drift_score={report['overall_drift_score']:.3f}")


if __name__ == "__main__":
    main()
