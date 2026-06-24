#!/usr/bin/env python3
"""Predict anomaly classes for aggregated Pixie windows."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib


def class_name(class_names, class_id):
    return class_names.get(class_id, class_names.get(str(class_id), str(class_id)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="aggregated window CSV, such as training_windows.csv")
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).parent / "models" / "anomaly_detector.joblib")
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="minimum anomaly probability required to emit an alert")
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    model = artifact["model"]
    features = artifact["feature_names"]
    class_names = artifact["class_names"]
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("input dataset has no rows")
    missing = set(features) - set(rows[0])
    if missing:
        raise SystemExit(f"input dataset is missing model features: {sorted(missing)}")

    matrix = [[float(row[name]) for name in features] for row in rows]
    predictions = model.predict(matrix)
    probabilities = model.predict_proba(matrix)
    classes = [int(value) for value in model.classes_]
    normal_index = classes.index(0) if 0 in classes else None
    probability_fields = [f"probability_{class_name(class_names, item)}" for item in classes]
    output_fields = list(rows[0]) + [
        "predicted_class_id", "predicted_class", "confidence",
        "anomaly_probability", "has_signal", "is_anomaly",
    ] + probability_fields

    alerts = []
    for row, prediction, row_probabilities in zip(rows, predictions, probabilities):
        prediction = int(prediction)
        anomaly_probability = (
            1.0 - row_probabilities[normal_index] if normal_index is not None
            else float(prediction != 0)
        )
        has_signal = any(float(row[name]) != 0 for name in features)
        row.update({
            "predicted_class_id": prediction,
            "predicted_class": class_name(class_names, prediction),
            "confidence": max(row_probabilities),
            "anomaly_probability": anomaly_probability,
            "has_signal": int(has_signal),
            "is_anomaly": int(
                has_signal and anomaly_probability >= args.threshold
                and (artifact["target"] == "label" or prediction != 0)
            ),
        })
        row.update(dict(zip(probability_fields, row_probabilities)))
        if row["is_anomaly"]:
            alerts.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    alerts_output = args.output.with_name("alerts.csv")
    with alerts_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(alerts)
    counts = Counter(row["predicted_class"] for row in rows)
    summary = {
        "algorithm": "random_forest",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": str(args.model),
        "target": artifact["target"],
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
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"windows={len(rows)} alerts={len(alerts)} predictions={dict(counts)}")
    print(f"output={args.output}")
    print(f"summary={args.output.with_name('summary.json')}")
    for row in alerts[-10:]:
        print(
            f"ALERT window={row.get('window_start', '-')} pod={row.get('pod', '-')} "
            f"class={row['predicted_class']} anomaly_probability={float(row['anomaly_probability']):.3f}"
        )


if __name__ == "__main__":
    main()
