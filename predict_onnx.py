#!/usr/bin/env python3
"""Predict anomaly classes with an exported ONNX Runtime model."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort


def class_name(class_names, class_id):
    return class_names.get(str(class_id), class_names.get(class_id, str(class_id)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models" / "anomaly_mlp.onnx")
    parser.add_argument("--output", type=Path, default=Path("onnx_predictions.csv"))
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    metadata = json.loads(args.model.with_suffix(".json").read_text(encoding="utf-8"))
    scaler = joblib.load(args.model.with_suffix(".scaler.joblib"))
    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("input dataset has no rows")
    features = metadata["features"]
    missing = set(features) - set(rows[0])
    if missing:
        raise SystemExit(f"input dataset is missing model features: {sorted(missing)}")
    inputs = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float32)
    inputs = scaler.transform(inputs).astype(np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.model), sess_options=options, providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], {"features": inputs})[0]
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    classes = [int(value) for value in metadata["classes"]]
    class_names = metadata.get("class_names", {
        0: "normal", 1: "ddos", 2: "portscan",
    })
    normal_index = classes.index(0) if 0 in classes else None
    probability_fields = [f"probability_{class_name(class_names, item)}" for item in classes]
    fields = list(rows[0]) + [
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
            "confidence": float(row_probabilities[prediction]),
            "anomaly_probability": float(anomaly_probability),
            "has_signal": int(has_signal),
            "is_anomaly": int(
                has_signal and anomaly_probability >= args.threshold
                and (metadata["target"] == "label" or prediction != 0)
            ),
        })
        row.update(dict(zip(probability_fields, row_probabilities)))
        if row["is_anomaly"]:
            alerts.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    alerts_output = args.output.with_name("alerts.csv")
    with alerts_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(alerts)
    counts = Counter(row["predicted_class"] for row in rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "onnx_mlp",
        "model": str(args.model),
        "target": metadata["target"],
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


if __name__ == "__main__":
    main()
