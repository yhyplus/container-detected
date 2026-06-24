#!/usr/bin/env python3
"""Compare sklearn Random Forest and ONNX Runtime MLP inference performance."""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
import psutil
from sklearn.metrics import accuracy_score, f1_score


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("benchmark dataset has no rows")
    return rows


def matrix(rows, features):
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float32)


def measure(name, predict, inputs, expected, repeats, model_path):
    process = psutil.Process()
    predict(inputs)
    latencies = []
    cpu_start = process.cpu_times()
    wall_start = time.perf_counter()
    for _ in range(repeats):
        start = time.perf_counter()
        predictions = predict(inputs)
        latencies.append((time.perf_counter() - start) * 1000)
    wall_seconds = time.perf_counter() - wall_start
    cpu_end = process.cpu_times()
    cpu_seconds = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
    sorted_latency = sorted(latencies)
    return {
        "model": name,
        "accuracy": accuracy_score(expected, predictions),
        "macro_f1": f1_score(expected, predictions, average="macro", zero_division=0),
        "batch_rows": len(inputs),
        "repeats": repeats,
        "latency_ms_mean": statistics.mean(latencies),
        "latency_ms_p95": sorted_latency[max(0, int(len(sorted_latency) * 0.95) - 1)],
        "throughput_windows_sec": len(inputs) * repeats / wall_seconds,
        "cpu_percent_process_aggregate": cpu_seconds / wall_seconds * 100,
        "rss_mib": process.memory_info().rss / 1024 ** 2,
        "model_size_kib": model_path.stat().st_size / 1024,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--rf-model", type=Path, default=Path(__file__).parent / "models" / "anomaly_detector.joblib")
    parser.add_argument("--onnx-model", type=Path, default=Path(__file__).parent / "models" / "anomaly_mlp.onnx")
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    args = parser.parse_args()

    rows = load_rows(args.input)
    expected = np.asarray([int(row["class_id"]) for row in rows])
    rf_bundle = joblib.load(args.rf_model)
    rf_inputs = matrix(rows, rf_bundle["feature_names"])
    results = [measure(
        "sklearn_random_forest", rf_bundle["model"].predict, rf_inputs, expected,
        args.repeats, args.rf_model,
    )]

    metadata = json.loads(args.onnx_model.with_suffix(".json").read_text(encoding="utf-8"))
    scaler = joblib.load(args.onnx_model.with_suffix(".scaler.joblib"))
    onnx_inputs = scaler.transform(matrix(rows, metadata["features"])).astype(np.float32)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.onnx_model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    results.append(measure(
        "onnxruntime_mlp",
        lambda values: session.run(["logits"], {"features": values})[0].argmax(axis=1),
        onnx_inputs, expected, args.repeats, args.onnx_model,
    ))
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
