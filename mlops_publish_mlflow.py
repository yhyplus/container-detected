#!/usr/bin/env python3
"""Publish the current production model snapshot into MLflow Model Registry."""

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "random_forest_multiclass": ["anomaly_detector.joblib", "anomaly_detector.json"],
    "random_forest_binary": ["anomaly_binary_detector.joblib", "anomaly_binary_detector.json"],
    "onnx_mlp_multiclass": [
        "anomaly_mlp.onnx", "anomaly_mlp.onnx.data", "anomaly_mlp.pt",
        "anomaly_mlp.scaler.joblib", "anomaly_mlp.json", "anomaly_mlp_weights.json",
    ],
    "onnx_mlp_binary": [
        "anomaly_mlp_binary.onnx", "anomaly_mlp_binary.onnx.data",
        "anomaly_mlp_binary.pt", "anomaly_mlp_binary.scaler.joblib",
        "anomaly_mlp_binary.json",
    ],
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_registered_model(client, name):
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name)


def existing_version(client, name, project_version):
    try:
        versions = client.search_model_versions(f"name = '{name}'")
    except Exception:
        return None
    for version in versions:
        tags = dict(version.tags or {})
        if tags.get("project_version") == project_version:
            return version
    return None


def publish_model(client, registered_name, source_dir, model, registry):
    ensure_registered_model(client, registered_name)
    project_version = registry["production_version"] or registry["version"]
    found = existing_version(client, registered_name, project_version)
    if found:
        version = found
        created = False
    else:
        version = client.create_model_version(
            name=registered_name,
            source=source_dir.resolve().as_uri(),
            description=(
                f"{model['name']} published from local registry version "
                f"{project_version}"
            ),
        )
        created = True
    client.set_registered_model_alias(registered_name, "production", version.version)
    tags = {
        "project": "k8s-anomaly-detection",
        "project_version": project_version,
        "local_stage": registry.get("stage", "Production"),
        "target": str(model.get("target")),
        "accuracy": str(model.get("accuracy")),
        "macro_f1": str(model.get("macro_f1")),
    }
    for key, value in tags.items():
        client.set_model_version_tag(registered_name, version.version, key, value)
    return {
        "name": registered_name,
        "version": version.version,
        "created": created,
        "source": source_dir.resolve().as_uri(),
        "alias": "production",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=Path("registry/model_registry.json"))
    parser.add_argument("--production-dir", type=Path, default=Path("registry/production"))
    parser.add_argument("--name-prefix", default="k8s_anomaly")
    parser.add_argument("--output", type=Path, default=Path("reports/mlflow_publish_report.json"))
    args = parser.parse_args()

    import mlflow

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("k8s-anomaly-mlops")
    client = mlflow.tracking.MlflowClient()
    registry = load_json(args.registry)
    production_dir = args.production_dir
    if not production_dir.exists():
        raise SystemExit(f"missing production dir: {production_dir}")

    published = []
    project_version = registry["production_version"] or registry["version"]
    with mlflow.start_run(run_name=f"publish-{project_version}") as run:
        mlflow.set_tag("mlops_action", "publish_models")
        mlflow.set_tag("project_version", project_version)
        mlflow.log_param("production_version", project_version)
        for model in registry.get("models", []):
            model_name = model["name"]
            files = DEFAULT_MODELS.get(model_name, [])
            source_dir = production_dir / model_name
            source_dir.mkdir(parents=True, exist_ok=True)
            for file_name in files:
                source = production_dir / file_name
                if source.exists():
                    target = source_dir / file_name
                    if not target.exists() or source.read_bytes() != target.read_bytes():
                        target.write_bytes(source.read_bytes())
            registered_name = f"{args.name_prefix}_{model_name}"
            item = publish_model(client, registered_name, source_dir, model, registry)
            published.append(item)
            mlflow.log_param(f"{model_name}_registered_name", registered_name)
            if model.get("accuracy") is not None:
                mlflow.log_metric(f"{model_name}_accuracy", model["accuracy"])
            if model.get("macro_f1") is not None:
                mlflow.log_metric(f"{model_name}_macro_f1", model["macro_f1"])
            mlflow.log_artifacts(str(source_dir), artifact_path=f"registered_models/{registered_name}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "tracking_uri": mlflow.get_tracking_uri(),
            "published": published,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(ROOT / "registry" / "model_registry.json"), artifact_path="registry")
        mlflow.log_artifact(str(args.output), artifact_path="reports")

    print(f"mlflow_publish_report={args.output}")
    for item in published:
        print(
            f"{item['name']} version={item['version']} "
            f"created={item['created']} alias={item['alias']}"
        )


if __name__ == "__main__":
    main()
