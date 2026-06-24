#!/usr/bin/env python3
"""Train a supervised anomaly classifier from collected Pixie windows."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


EXCLUDED_COLUMNS = {
    "scenario", "attack_type", "attack_subtype", "anomaly_type",
    "class_id", "label", "is_target",
    "window_coverage_ratio", "window_start", "pod",
}
MULTICLASS_NAMES = {
    0: "normal",
    1: "ddos",
    2: "portscan",
    3: "lateral_movement",
    4: "resource_anomaly",
    5: "syscall_anomaly",
}


def read_rows(run_dirs):
    rows = []
    for run_dir in run_dirs:
        path = run_dir / "training_multiclass_windows.csv"
        if not path.exists():
            raise SystemExit(f"dataset does not exist: {path}")
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise SystemExit("dataset has no rows")
    return rows


def feature_names(rows):
    return [name for name in rows[0] if name not in EXCLUDED_COLUMNS]


def matrix(rows, features):
    return [[float(row[name]) for name in features] for row in rows]


def labels(rows, target):
    return [int(row[target]) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True,
                        help="training run directory; repeat to combine runs")
    parser.add_argument("--eval-run-dir", type=Path, action="append", default=[],
                        help="optional held-out run directory; repeat as needed")
    parser.add_argument("--target", choices=["class_id", "label"], default="class_id")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "models" / "anomaly_detector.joblib")
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    args = parser.parse_args()

    train_rows = read_rows(args.run_dir)
    features = feature_names(train_rows)
    train_labels = labels(train_rows, args.target)
    model = RandomForestClassifier(
        n_estimators=args.trees,
        class_weight=None if args.class_weight == "none" else args.class_weight,
        random_state=args.seed,
        n_jobs=args.jobs,
    )
    model.fit(matrix(train_rows, features), train_labels)

    report = None
    if args.eval_run_dir:
        eval_rows = read_rows(args.eval_run_dir)
        predictions = model.predict(matrix(eval_rows, features))
        expected = labels(eval_rows, args.target)
        report = classification_report(expected, predictions, output_dict=True, zero_division=0)
        print(f"held_out_accuracy={accuracy_score(expected, predictions):.4f}")
        print(classification_report(expected, predictions, zero_division=0))

    artifact = {
        "model": model,
        "feature_names": features,
        "target": args.target,
        "class_names": (
            MULTICLASS_NAMES
            if args.target == "class_id" else {0: "normal", 1: "anomaly"}
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "features": features,
        "training_runs": [str(path) for path in args.run_dir],
        "evaluation_runs": [str(path) for path in args.eval_run_dir],
        "class_weight": args.class_weight,
        "training_rows": len(train_rows),
        "training_classes": dict(sorted(Counter(train_labels).items())),
        "evaluation_report": report,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"model={args.output}")
    print(f"training_rows={len(train_rows)} classes={dict(sorted(Counter(train_labels).items()))}")


if __name__ == "__main__":
    main()
