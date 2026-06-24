#!/usr/bin/env python3
"""Train a one-class Isolation Forest using normal traffic windows only."""

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

from train_model import feature_names, read_rows


def matrix(rows, features):
    return np.asarray(
        [[float(row[name]) for name in features] for row in rows],
        dtype=np.float64,
    )


def anomaly_probabilities(model, values, score_scale):
    anomaly_margin = -model.decision_function(values)
    scaled = np.clip(anomaly_margin / score_scale, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--eval-run-dir", type=Path, action="append", default=[])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent / "models" / "anomaly_isolation_forest.joblib",
    )
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--contamination", type=float, default=0.02)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0 < args.contamination <= 0.5:
        raise SystemExit("contamination must be in (0, 0.5]")
    if not 0 <= args.threshold <= 1:
        raise SystemExit("threshold must be in [0, 1]")

    all_train_rows = read_rows(args.run_dir)
    normal_rows = [row for row in all_train_rows if int(row["label"]) == 0]
    if not normal_rows:
        raise SystemExit("training data contains no normal rows")

    features = feature_names(normal_rows)
    scaler = StandardScaler()
    normal_x = scaler.fit_transform(matrix(normal_rows, features))
    model = IsolationForest(
        n_estimators=args.trees,
        contamination=args.contamination,
        random_state=args.seed,
        n_jobs=args.jobs,
    )
    model.fit(normal_x)

    normal_margins = np.abs(model.decision_function(normal_x))
    score_scale = max(float(np.median(normal_margins)), 1e-6)
    report = None
    evaluation_rows = 0
    evaluation_classes = {}
    if args.eval_run_dir:
        eval_rows = read_rows(args.eval_run_dir)
        eval_x = scaler.transform(matrix(eval_rows, features))
        expected = np.asarray([int(row["label"]) for row in eval_rows], dtype=np.int64)
        probabilities = anomaly_probabilities(model, eval_x, score_scale)
        predictions = (probabilities >= args.threshold).astype(np.int64)
        report = classification_report(
            expected, predictions, output_dict=True, zero_division=0
        )
        evaluation_rows = len(eval_rows)
        evaluation_classes = dict(sorted(Counter(expected.tolist()).items()))
        print(f"held_out_accuracy={accuracy_score(expected, predictions):.4f}")
        print(classification_report(expected, predictions, zero_division=0))

    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_names": features,
        "target": "label",
        "class_names": {0: "normal", 1: "anomaly"},
        "algorithm": "isolation_forest",
        "score_scale": score_scale,
        "default_threshold": args.threshold,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "isolation_forest",
        "training_mode": "normal_only",
        "target": "label",
        "features": features,
        "training_runs": [str(path) for path in args.run_dir],
        "evaluation_runs": [str(path) for path in args.eval_run_dir],
        "source_training_rows": len(all_train_rows),
        "training_rows": len(normal_rows),
        "training_classes": {0: len(normal_rows)},
        "evaluation_rows": evaluation_rows,
        "evaluation_classes": evaluation_classes,
        "trees": args.trees,
        "contamination": args.contamination,
        "default_threshold": args.threshold,
        "score_scale": score_scale,
        "evaluation_report": report,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"model={args.output}")
    print(
        f"normal_training_rows={len(normal_rows)} "
        f"source_training_rows={len(all_train_rows)}"
    )


if __name__ == "__main__":
    main()
