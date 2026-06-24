#!/usr/bin/env python3
"""Serve the ONNX MLP anomaly detector over HTTP."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from flask import Flask, Response, jsonify, request


def class_name(class_names, class_id):
    return class_names.get(str(class_id), class_names.get(class_id, str(class_id)))


def load_model(model_path):
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    scaler = joblib.load(model_path.with_suffix(".scaler.joblib"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return metadata, scaler, session


def create_app(model_path, threshold):
    metadata, scaler, session = load_model(model_path)
    features = metadata["features"]
    classes = [int(value) for value in metadata["classes"]]
    class_names = metadata.get("class_names", {})
    normal_index = classes.index(0) if 0 in classes else None
    metrics = {
        "requests": 0,
        "rows": 0,
        "alerts": 0,
        "latency_ms_total": 0.0,
        "latency_ms_max": 0.0,
        "classes": Counter(),
    }
    app = Flask(__name__)

    @app.get("/healthz")
    def healthz():
        return jsonify({
            "status": "ok",
            "model": str(model_path),
            "target": metadata["target"],
            "feature_count": len(features),
        })

    @app.get("/metadata")
    def model_metadata():
        return jsonify({
            "model": str(model_path),
            "target": metadata["target"],
            "classes": classes,
            "class_names": class_names,
            "features": features,
            "threshold": threshold,
        })

    @app.get("/metrics")
    def prometheus_metrics():
        lines = [
            "# HELP anomaly_inference_requests_total Total prediction requests.",
            "# TYPE anomaly_inference_requests_total counter",
            f"anomaly_inference_requests_total {metrics['requests']}",
            "# HELP anomaly_inference_rows_total Total predicted rows.",
            "# TYPE anomaly_inference_rows_total counter",
            f"anomaly_inference_rows_total {metrics['rows']}",
            "# HELP anomaly_inference_alerts_total Total anomaly alerts.",
            "# TYPE anomaly_inference_alerts_total counter",
            f"anomaly_inference_alerts_total {metrics['alerts']}",
            "# HELP anomaly_inference_latency_ms_total Total inference latency in milliseconds.",
            "# TYPE anomaly_inference_latency_ms_total counter",
            f"anomaly_inference_latency_ms_total {metrics['latency_ms_total']}",
            "# HELP anomaly_inference_latency_ms_max Maximum observed inference latency in milliseconds.",
            "# TYPE anomaly_inference_latency_ms_max gauge",
            f"anomaly_inference_latency_ms_max {metrics['latency_ms_max']}",
        ]
        for label, count in sorted(metrics["classes"].items()):
            lines.append(
                f'anomaly_inference_predicted_class_total{{class="{label}"}} {count}'
            )
        return Response("\n".join(lines) + "\n", mimetype="text/plain")

    @app.post("/predict")
    def predict():
        payload = request.get_json(force=True)
        rows = payload.get("rows", payload if isinstance(payload, list) else [payload])
        if not isinstance(rows, list) or not rows:
            return jsonify({"error": "request must contain a row object or rows list"}), 400
        missing = sorted(set(features) - set(rows[0]))
        if missing:
            return jsonify({"error": "missing required features", "features": missing}), 400
        matrix = np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float32)
        matrix = scaler.transform(matrix).astype(np.float32)
        started = time.perf_counter()
        logits = session.run(["logits"], {"features": matrix})[0]
        latency_ms = (time.perf_counter() - started) * 1000
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        predictions = probabilities.argmax(axis=1)
        result_rows = []
        for prediction, row_probabilities in zip(predictions, probabilities):
            prediction = int(prediction)
            anomaly_probability = (
                1.0 - row_probabilities[normal_index] if normal_index is not None
                else float(prediction != 0)
            )
            predicted_class = class_name(class_names, prediction)
            result = {
                "predicted_class_id": prediction,
                "predicted_class": predicted_class,
                "confidence": float(row_probabilities[prediction]),
                "anomaly_probability": float(anomaly_probability),
                "is_anomaly": int(anomaly_probability >= threshold and (
                    metadata["target"] == "label" or prediction != 0
                )),
                "probabilities": {
                    class_name(class_names, class_id): float(row_probabilities[index])
                    for index, class_id in enumerate(classes)
                },
            }
            result_rows.append(result)
            metrics["classes"][predicted_class] += 1
            metrics["alerts"] += int(result["is_anomaly"])
        metrics["requests"] += 1
        metrics["rows"] += len(rows)
        metrics["latency_ms_total"] += latency_ms
        metrics["latency_ms_max"] = max(metrics["latency_ms_max"], latency_ms)
        return jsonify({
            "model": str(model_path),
            "target": metadata["target"],
            "threshold": threshold,
            "rows": result_rows,
            "batch_size": len(rows),
            "latency_ms": latency_ms,
        })

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models" / "anomaly_mlp.onnx")
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app = create_app(args.model, args.threshold)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
