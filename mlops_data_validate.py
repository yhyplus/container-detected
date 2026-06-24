#!/usr/bin/env python3
"""Validate training/evaluation datasets before model training."""

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXCLUDED_COLUMNS = {
    "scenario", "attack_type", "attack_subtype", "anomaly_type",
    "class_id", "label", "is_target", "window_coverage_ratio",
    "window_start", "pod",
}
REQUIRED_COLUMNS = {
    "scenario", "class_id", "label", "window_coverage_ratio", "window_start", "pod",
    "cpu_pct", "rss_mib", "rx_kib_sec", "tx_kib_sec",
    "connection_open_count", "http_request_count", "dns_request_count",
    "sensor_event_count", "process_spawn_count", "sensitive_file_access_count",
}
REQUIRED_SCENARIOS = {
    "normal", "http_flood", "network_connections", "dns_flood",
    "portscan", "lateral_movement", "cpu", "syscall_anomaly",
}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numeric_features(rows):
    return [name for name in rows[0] if name not in EXCLUDED_COLUMNS]


def validate_dataset(name, path, min_rows, min_class_rows):
    rows = read_rows(path)
    issues = []
    warnings = []
    if not rows:
        return {
            "name": name,
            "path": str(path.relative_to(ROOT) if path.is_absolute() else path),
            "rows": 0,
            "passed": False,
            "issues": ["dataset has no rows"],
            "warnings": [],
        }
    columns = set(rows[0])
    missing_columns = sorted(REQUIRED_COLUMNS - columns)
    if missing_columns:
        issues.append(f"missing required columns: {missing_columns}")
    if len(rows) < min_rows:
        issues.append(f"rows {len(rows)} < minimum {min_rows}")

    scenarios = Counter(row.get("scenario", "") for row in rows)
    missing_scenarios = sorted(REQUIRED_SCENARIOS - set(scenarios))
    if missing_scenarios:
        warnings.append(f"missing expected scenarios: {missing_scenarios}")

    classes = Counter(row.get("class_id", "") for row in rows)
    for class_id in ["0", "1", "2", "3", "4", "5"]:
        if classes.get(class_id, 0) < min_class_rows:
            issues.append(f"class_id={class_id} rows {classes.get(class_id, 0)} < {min_class_rows}")

    labels = Counter(row.get("label", "") for row in rows)
    for label in ["0", "1"]:
        if labels.get(label, 0) < min_class_rows:
            issues.append(f"label={label} rows {labels.get(label, 0)} < {min_class_rows}")

    features = numeric_features(rows)
    feature_reports = []
    for feature in features:
        missing = 0
        non_numeric = 0
        values = []
        for row in rows:
            raw = row.get(feature, "")
            if raw == "":
                missing += 1
                continue
            try:
                value = float(raw)
            except ValueError:
                non_numeric += 1
                continue
            if math.isnan(value) or math.isinf(value):
                non_numeric += 1
                continue
            values.append(value)
        missing_rate = missing / len(rows)
        non_numeric_rate = non_numeric / len(rows)
        zero_rate = values.count(0.0) / len(values) if values else 1.0
        if missing_rate > 0:
            issues.append(f"{feature} missing_rate={missing_rate:.3f}")
        if non_numeric_rate > 0:
            issues.append(f"{feature} non_numeric_rate={non_numeric_rate:.3f}")
        if zero_rate > 0.98:
            warnings.append(f"{feature} zero_rate={zero_rate:.3f}")
        feature_reports.append({
            "feature": feature,
            "missing_rate": missing_rate,
            "non_numeric_rate": non_numeric_rate,
            "zero_rate": zero_rate,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        })

    return {
        "name": name,
        "path": str(path.relative_to(ROOT) if path.is_absolute() else path),
        "rows": len(rows),
        "columns": len(rows[0]),
        "feature_count": len(features),
        "scenario_distribution": dict(sorted(scenarios.items())),
        "class_id_distribution": dict(sorted(classes.items())),
        "label_distribution": dict(sorted(labels.items())),
        "features": feature_reports,
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
    }


def write_markdown(path, report):
    lines = [
        "# Data Validation Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Overall status: `{'PASS' if report['passed'] else 'FAIL'}`",
        "",
    ]
    for dataset in report["datasets"]:
        lines.extend([
            f"## {dataset['name']}",
            "",
            f"Path: `{dataset['path']}`",
            "",
            f"Rows: `{dataset['rows']}`",
            "",
            f"Status: `{'PASS' if dataset['passed'] else 'FAIL'}`",
            "",
        ])
        if dataset["issues"]:
            lines.append("Issues:")
            lines.extend(f"- {item}" for item in dataset["issues"])
            lines.append("")
        if dataset["warnings"]:
            lines.append("Warnings:")
            lines.extend(f"- {item}" for item in dataset["warnings"][:20])
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--min-train-rows", type=int, default=300)
    parser.add_argument("--min-eval-rows", type=int, default=120)
    parser.add_argument("--min-class-rows", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("reports/data_validation_report.json"))
    args = parser.parse_args()

    train_report = validate_dataset("train", args.train, args.min_train_rows, args.min_class_rows)
    eval_report = validate_dataset("eval", args.eval, args.min_eval_rows, args.min_class_rows)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": train_report["passed"] and eval_report["passed"],
        "datasets": [train_report, eval_report],
        "thresholds": {
            "min_train_rows": args.min_train_rows,
            "min_eval_rows": args.min_eval_rows,
            "min_class_rows": args.min_class_rows,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), report)
    print(f"data_validation_report={args.output}")
    print(f"passed={report['passed']}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
