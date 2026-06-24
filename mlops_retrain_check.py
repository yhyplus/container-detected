#!/usr/bin/env python3
"""Check simple MLOps retraining trigger rules."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def load_json(path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-registry", type=Path, default=Path("registry/data_registry.json"))
    parser.add_argument("--model-registry", type=Path, default=Path("registry/model_registry.json"))
    parser.add_argument("--drift-report", type=Path, default=Path("reports/drift_report.json"))
    parser.add_argument("--new-data-rows", type=int, default=0)
    parser.add_argument("--min-new-rows", type=int, default=300)
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--max-drift-score", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=Path("reports/retrain_check.json"))
    args = parser.parse_args()

    data_registry = load_json(args.data_registry) or {}
    model_registry = load_json(args.model_registry) or {}
    drift_report = load_json(args.drift_report) or {}
    reasons = []
    if args.new_data_rows >= args.min_new_rows:
        reasons.append(f"new_data_rows {args.new_data_rows} >= {args.min_new_rows}")
    for model in model_registry.get("models", []):
        accuracy = model.get("accuracy")
        if accuracy is not None and accuracy < args.min_accuracy:
            reasons.append(f"{model['name']} accuracy {accuracy:.4f} < {args.min_accuracy:.4f}")
    drift_score = drift_report.get("overall_drift_score")
    if drift_score is not None and drift_score > args.max_drift_score:
        reasons.append(f"drift_score {drift_score:.4f} > {args.max_drift_score:.4f}")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "should_retrain": bool(reasons),
        "reasons": reasons,
        "thresholds": {
            "min_new_rows": args.min_new_rows,
            "min_accuracy": args.min_accuracy,
            "max_drift_score": args.max_drift_score,
        },
        "current": {
            "new_data_rows": args.new_data_rows,
            "drift_score": drift_score,
            "data_version": data_registry.get("version"),
            "model_version": model_registry.get("version"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"retrain_check={args.output}")
    print(f"should_retrain={report['should_retrain']} reasons={reasons}")


if __name__ == "__main__":
    main()
