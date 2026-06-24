#!/usr/bin/env python3
"""Predict deviations from normal traffic with a trained Isolation Forest."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent / "models" / "anomaly_isolation_forest.joblib",
    )
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    model = artifact["model"]
    scaler = artifact["scaler"]
    features = artifact["feature_names"]
    score_scale = max(float(artifact["score_scale"]), 1e-6)

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("input dataset has no rows")
    missing = set(features) - set(rows[0])
    if missing:
        raise SystemExit(f"input dataset is missing model features: {sorted(missing)}")

    values = np.asarray(
        [[float(row[name]) for name in features] for row in rows],
        dtype=np.float64,
    )
    scaled_values = scaler.transform(values)
    anomaly_margins = -model.decision_function(scaled_values)
    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(anomaly_margins / score_scale, -60.0, 60.0))
    )
    output_fields = list(rows[0]) + [
        "predicted_class_id", "predicted_class", "confidence",
        "anomaly_probability", "isolation_score", "has_signal", "is_anomaly",
        "probability_normal", "probability_anomaly",
    ]
    alerts = []
    for row, probability, isolation_score in zip(rows, probabilities, anomaly_margins):
        probability = float(probability)
        is_anomaly = probability >= args.threshold
        has_signal = any(float(row[name]) != 0 for name in features)
        row.update({
            "predicted_class_id": int(is_anomaly),
            "predicted_class": "anomaly" if is_anomaly else "normal",
            "confidence": probability if is_anomaly else 1.0 - probability,
            "anomaly_probability": probability,
            "isolation_score": float(isolation_score),
            "has_signal": int(has_signal),
            "is_anomaly": int(has_signal and is_anomaly),
            "probability_normal": 1.0 - probability,
            "probability_anomaly": probability,
        })
        if row["is_anomaly"]:
            alerts.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    with args.output.with_name("alerts.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(alerts)

    counts = Counter(row["predicted_class"] for row in rows)
    summary = {
        "algorithm": "isolation_forest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "target": "label",
        "training_mode": "normal_only",
        "threshold": args.threshold,
        "windows": len(rows),
        "observed_windows": sum(int(row["has_signal"]) for row in rows),
        "alerts": len(alerts),
        "prediction_counts": dict(counts),
        "latest_alerts": [
            {
                "window_start": row.get("window_start", ""),
                "pod": row.get("pod", ""),
                "predicted_class": row["predicted_class"],
                "anomaly_probability": float(row["anomaly_probability"]),
            }
            for row in alerts[-10:]
        ],
    }
    args.output.with_name("summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"windows={len(rows)} alerts={len(alerts)} predictions={dict(counts)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
