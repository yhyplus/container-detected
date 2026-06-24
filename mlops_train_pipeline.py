#!/usr/bin/env python3
"""Run the anomaly detector training pipeline with MLflow tracking."""

import argparse
import csv
import json
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_TRAIN_RUNS = [
    ROOT / "runs" / "train-v3-expanded-20260603",
    ROOT / "runs" / "train-v3-large-20260603",
]
DEFAULT_EVAL_RUNS = [
    ROOT / "runs" / "eval-v3-expanded-20260603",
    ROOT / "runs" / "eval-v3-large-20260603",
]
MODEL_FILES = [
    ROOT / "models" / "anomaly_detector.joblib",
    ROOT / "models" / "anomaly_detector.json",
    ROOT / "models" / "anomaly_binary_detector.joblib",
    ROOT / "models" / "anomaly_binary_detector.json",
    ROOT / "models" / "anomaly_mlp.onnx",
    ROOT / "models" / "anomaly_mlp.onnx.data",
    ROOT / "models" / "anomaly_mlp.pt",
    ROOT / "models" / "anomaly_mlp.scaler.joblib",
    ROOT / "models" / "anomaly_mlp.json",
    ROOT / "models" / "anomaly_mlp_binary.onnx",
    ROOT / "models" / "anomaly_mlp_binary.onnx.data",
    ROOT / "models" / "anomaly_mlp_binary.pt",
    ROOT / "models" / "anomaly_mlp_binary.scaler.joblib",
    ROOT / "models" / "anomaly_mlp_binary.json",
    ROOT / "models" / "anomaly_mlp_weights.json",
]
DEPLOY_FILES = [
    ROOT / "models" / "anomaly_mlp_weights.json",
    ROOT / "Dockerfile.inference-pure",
    ROOT / "k8s" / "anomaly-inference.yaml",
]


def load_mlflow():
    try:
        import mlflow  # type: ignore
    except Exception:
        return None
    return mlflow


def run(command):
    print("+", " ".join(str(part) for part in command))
    subprocess.run(command, cwd=ROOT, check=True)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def combine_runs(run_dirs, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "training_multiclass_windows.csv"
    rows = []
    fieldnames = None
    for run_dir in run_dirs:
        path = run_dir / "training_multiclass_windows.csv"
        current = read_rows(path)
        if not current:
            raise SystemExit(f"dataset has no rows: {path}")
        if fieldnames is None:
            fieldnames = list(current[0])
        rows.extend(current)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output, rows


def dataset_summary(name, run_dirs, combined_path, rows):
    scenarios = Counter(row.get("scenario", "") for row in rows)
    classes = Counter(row.get("class_id", "") for row in rows)
    labels = Counter(row.get("label", "") for row in rows)
    features = [
        field for field in rows[0]
        if field not in {
            "scenario", "attack_type", "attack_subtype", "anomaly_type",
            "class_id", "label", "is_target", "window_coverage_ratio",
            "window_start", "pod",
        }
    ]
    return {
        "name": name,
        "combined_path": str(combined_path.relative_to(ROOT)),
        "source_runs": [str(path.relative_to(ROOT)) for path in run_dirs],
        "rows": len(rows),
        "columns": len(rows[0]),
        "feature_count": len(features),
        "features": features,
        "scenario_distribution": dict(sorted(scenarios.items())),
        "class_id_distribution": dict(sorted(classes.items())),
        "label_distribution": dict(sorted(labels.items())),
    }


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def copy_artifacts(version_dir):
    version_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in MODEL_FILES + DEPLOY_FILES:
        if path.exists():
            destination = version_dir / path.name
            shutil.copy2(path, destination)
            copied.append(str(destination.relative_to(ROOT)))
    return copied


def model_record(name, metadata_path, artifact_paths, benchmark=None):
    metadata = load_json(metadata_path)
    report = metadata.get("evaluation_report") or {}
    record = {
        "name": name,
        "target": metadata.get("target"),
        "training_runs": metadata.get("training_runs", []),
        "evaluation_runs": metadata.get("evaluation_runs", []),
        "training_rows": metadata.get("training_rows"),
        "training_classes": metadata.get("training_classes"),
        "accuracy": report.get("accuracy"),
        "macro_f1": (report.get("macro avg") or {}).get("f1-score"),
        "weighted_f1": (report.get("weighted avg") or {}).get("f1-score"),
        "metadata_path": str(metadata_path.relative_to(ROOT)),
        "artifacts": artifact_paths,
        "created_at": metadata.get("created_at"),
    }
    if "class_weight" in metadata:
        record["class_weight"] = metadata["class_weight"]
    if benchmark:
        record["benchmark"] = benchmark
    return record


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(path, data_registry, model_registry):
    models = model_registry["models"]
    metric_rows = []
    for model in models:
        metric_rows.append([
            model["name"],
            model["target"],
            f"{model['accuracy'] * 100:.2f}%" if model.get("accuracy") is not None else "-",
            f"{model['macro_f1']:.3f}" if model.get("macro_f1") is not None else "-",
        ])
    dataset_rows = [
        [
            item["name"],
            item["rows"],
            item["feature_count"],
            item["combined_path"],
        ]
        for item in data_registry["datasets"]
    ]
    text = f"""# MLOps Model Evaluation Report

Generated at: `{model_registry['generated_at']}`

## Dataset Versions

{markdown_table(['Dataset', 'Rows', 'Features', 'Combined CSV'], dataset_rows)}

## Model Metrics

{markdown_table(['Model', 'Target', 'Accuracy', 'Macro F1'], metric_rows)}

## Benchmark Summary

Benchmark input: `{model_registry['benchmark_input']}`

"""
    benchmark = model_registry.get("benchmark", [])
    if benchmark:
        rows = [
            [
                item["model"],
                f"{item['accuracy'] * 100:.2f}%",
                f"{item['macro_f1']:.3f}",
                f"{item['latency_ms_mean']:.4f} ms",
                f"{item['latency_ms_p95']:.4f} ms",
                f"{item['model_size_kib']:.1f} KiB",
            ]
            for item in benchmark
        ]
        text += markdown_table(
            ["Model", "Accuracy", "Macro F1", "Mean Latency", "P95 Latency", "Size"],
            rows,
        )
    text += """

## Notes

- Random Forest six-class training uses balanced class weights.
- Random Forest binary training uses no class weight after evaluation showed fewer normal false positives at the default 0.5 threshold.
- ONNX MLP is exported with the scaler artifact and is the preferred lightweight deployment candidate.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def log_mlflow(mlflow, model_registry, data_registry, report_path):
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("k8s-anomaly-mlops")
    with mlflow.start_run(run_name=model_registry["version"]):
        for item in data_registry["datasets"]:
            mlflow.log_param(f"{item['name']}_rows", item["rows"])
            mlflow.log_param(f"{item['name']}_features", item["feature_count"])
        for model in model_registry["models"]:
            prefix = model["name"]
            if model.get("accuracy") is not None:
                mlflow.log_metric(f"{prefix}_accuracy", model["accuracy"])
            if model.get("macro_f1") is not None:
                mlflow.log_metric(f"{prefix}_macro_f1", model["macro_f1"])
        for item in model_registry.get("benchmark", []):
            prefix = item["model"]
            mlflow.log_metric(f"{prefix}_latency_ms_mean", item["latency_ms_mean"])
            mlflow.log_metric(f"{prefix}_model_size_kib", item["model_size_kib"])
        for path in MODEL_FILES:
            if path.exists():
                mlflow.log_artifact(str(path), artifact_path="models")
        mlflow.log_artifact(str(ROOT / "registry" / "data_registry.json"), artifact_path="registry")
        mlflow.log_artifact(str(ROOT / "registry" / "model_registry.json"), artifact_path="registry")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        validation_report = ROOT / "reports" / "data_validation_report.json"
        if validation_report.exists():
            mlflow.log_artifact(str(validation_report), artifact_path="reports")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", action="append", type=Path, default=[])
    parser.add_argument("--eval-run", action="append", type=Path, default=[])
    parser.add_argument("--version", default=datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    train_runs = [path if path.is_absolute() else ROOT / path for path in (args.train_run or DEFAULT_TRAIN_RUNS)]
    eval_runs = [path if path.is_absolute() else ROOT / path for path in (args.eval_run or DEFAULT_EVAL_RUNS)]
    version = args.version
    combined_train, train_rows = combine_runs(train_runs, ROOT / "runs" / f"train-v3-combined-{version}")
    combined_eval, eval_rows = combine_runs(eval_runs, ROOT / "runs" / f"eval-v3-combined-{version}")
    run([
        "python3", "mlops_data_validate.py",
        "--train", str(combined_train.relative_to(ROOT)),
        "--eval", str(combined_eval.relative_to(ROOT)),
        "--output", "reports/data_validation_report.json",
    ])
    data_registry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "datasets": [
            dataset_summary("train", train_runs, combined_train, train_rows),
            dataset_summary("eval", eval_runs, combined_eval, eval_rows),
        ],
    }
    write_json(ROOT / "registry" / "data_registry.json", data_registry)

    if not args.skip_training:
        common_train = sum([["--run-dir", str(path.relative_to(ROOT))] for path in train_runs], [])
        common_eval = sum([["--eval-run-dir", str(path.relative_to(ROOT))] for path in eval_runs], [])
        run(["python3", "train_model.py", *common_train, *common_eval, "--output", "models/anomaly_detector.joblib"])
        run([
            "python3", "train_model.py", *common_train, *common_eval,
            "--target", "label", "--class-weight", "none",
            "--output", "models/anomaly_binary_detector.joblib",
        ])
        run(["python3", "train_mlp_onnx.py", *common_train, *common_eval, "--output", "models/anomaly_mlp.onnx"])
        run([
            "python3", "train_mlp_onnx.py", *common_train, *common_eval,
            "--target", "label", "--output", "models/anomaly_mlp_binary.onnx",
        ])
        run(["python3", "export_mlp_weights.py"])

    benchmark_path = ROOT / "runs" / f"eval-v3-combined-{version}" / "benchmark.json"
    run([
        "python3", "benchmark_models.py",
        "--input", str(combined_eval.relative_to(ROOT)),
        "--output", str(benchmark_path.relative_to(ROOT)),
    ])
    benchmark = load_json(benchmark_path)
    version_dir = ROOT / "registry" / "models" / version
    copied = copy_artifacts(version_dir)
    existing_registry = load_json(ROOT / "registry" / "model_registry.json") if (ROOT / "registry" / "model_registry.json").exists() else {}
    existing_production_version = (
        existing_registry.get("production_version")
        or existing_registry.get("previous_production_version")
    )
    model_registry = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "stage": "Candidate",
        "stage_updated_at": datetime.now(timezone.utc).isoformat(),
        "stage_note": "created by mlops_train_pipeline.py; promote after deployment validation",
        "previous_production_version": existing_production_version,
        "benchmark_input": str(combined_eval.relative_to(ROOT)),
        "data_validation_report": "reports/data_validation_report.json",
        "benchmark": benchmark,
        "model_artifact_snapshot": copied,
        "models": [
            model_record("random_forest_multiclass", ROOT / "models" / "anomaly_detector.json", [
                "models/anomaly_detector.joblib", "models/anomaly_detector.json",
            ], benchmark[0]),
            model_record("random_forest_binary", ROOT / "models" / "anomaly_binary_detector.json", [
                "models/anomaly_binary_detector.joblib", "models/anomaly_binary_detector.json",
            ]),
            model_record("onnx_mlp_multiclass", ROOT / "models" / "anomaly_mlp.json", [
                "models/anomaly_mlp.onnx", "models/anomaly_mlp.pt",
                "models/anomaly_mlp.scaler.joblib", "models/anomaly_mlp.json",
            ], benchmark[1]),
            model_record("onnx_mlp_binary", ROOT / "models" / "anomaly_mlp_binary.json", [
                "models/anomaly_mlp_binary.onnx", "models/anomaly_mlp_binary.pt",
                "models/anomaly_mlp_binary.scaler.joblib", "models/anomaly_mlp_binary.json",
            ]),
        ],
    }
    write_json(ROOT / "registry" / "model_registry.json", model_registry)
    shutil.copy2(ROOT / "registry" / "model_registry.json", version_dir / "model_registry.json")
    shutil.copy2(ROOT / "registry" / "data_registry.json", version_dir / "data_registry.json")
    validation_path = ROOT / "reports" / "data_validation_report.json"
    if validation_path.exists():
        shutil.copy2(validation_path, version_dir / "data_validation_report.json")
    report_path = ROOT / "reports" / "model_eval_report.md"
    write_report(report_path, data_registry, model_registry)
    write_json(ROOT / "reports" / "model_eval_report.json", {
        "data_registry": data_registry,
        "model_registry": model_registry,
    })

    if not args.no_mlflow:
        mlflow = load_mlflow()
        if mlflow is None:
            print("MLflow is not installed; local registry and reports were still generated.")
        else:
            log_mlflow(mlflow, model_registry, data_registry, report_path)
            print("MLflow run logged under ./mlruns")

    print(f"data_registry={ROOT / 'registry' / 'data_registry.json'}")
    print(f"model_registry={ROOT / 'registry' / 'model_registry.json'}")
    print(f"report={report_path}")


if __name__ == "__main__":
    main()
