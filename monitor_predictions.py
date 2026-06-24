#!/usr/bin/env python3
"""Summarize online prediction outputs for lightweight model monitoring."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_rows(path):
    if not path.exists():
        raise SystemExit(f"predictions file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    try:
        return float(row.get(key, 0) or 0)
    except ValueError:
        return 0.0


def write_markdown(path, report):
    prediction_rows = [
        f"| {label} | {count} |"
        for label, count in sorted(report["prediction_counts"].items())
    ]
    text = f"""# Online Prediction Monitoring Report

Generated at: `{report['generated_at']}`

Predictions file: `{report['predictions']}`

Rows: `{report['rows']}`

Observed rows: `{report['observed_rows']}`

Alerts: `{report['alerts']}`

Alert rate: `{report['alert_rate']:.2%}`

Anomaly probability mean: `{report['anomaly_probability']['mean']:.4f}`

Anomaly probability p95: `{report['anomaly_probability']['p95']:.4f}`

## Prediction Counts

| Predicted class | Count |
| --- | --- |
{chr(10).join(prediction_rows)}
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/prediction_monitor_report.json"))
    args = parser.parse_args()

    rows = read_rows(args.predictions)
    if not rows:
        raise SystemExit("predictions file has no rows")
    anomaly_probabilities = np.asarray([as_float(row, "anomaly_probability") for row in rows])
    alerts = [row for row in rows if row.get("is_anomaly") in {"1", 1}]
    observed = [row for row in rows if row.get("has_signal") in {"1", 1}]
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "predictions": str(args.predictions),
        "rows": len(rows),
        "observed_rows": len(observed),
        "alerts": len(alerts),
        "alert_rate": len(alerts) / len(rows),
        "prediction_counts": dict(Counter(row.get("predicted_class", "") for row in rows)),
        "anomaly_probability": {
            "min": float(anomaly_probabilities.min()),
            "mean": float(anomaly_probabilities.mean()),
            "median": float(np.median(anomaly_probabilities)),
            "p95": float(np.percentile(anomaly_probabilities, 95)),
            "max": float(anomaly_probabilities.max()),
        },
        "latest_alerts": [
            {
                "window_start": row.get("window_start", ""),
                "pod": row.get("pod", ""),
                "predicted_class": row.get("predicted_class", ""),
                "anomaly_probability": as_float(row, "anomaly_probability"),
            }
            for row in alerts[-10:]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), report)
    print(f"prediction_monitor_report={args.output}")
    print(f"alerts={report['alerts']} rows={report['rows']} alert_rate={report['alert_rate']:.2%}")


if __name__ == "__main__":
    main()
