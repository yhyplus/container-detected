#!/usr/bin/env python3
"""Decide whether a candidate model version is safe to deploy."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_json(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def model_map(registry):
    return {item["name"]: item for item in registry.get("models", [])}


def write_markdown(path, report):
    lines = [
        "# Release Gate Report",
        "",
        f"Generated at: `{report['generated_at']}`",
        "",
        f"Decision: `{'PASS' if report['passed'] else 'FAIL'}`",
        "",
        f"Candidate version: `{report['candidate_version']}`",
        "",
        f"Production baseline: `{report.get('production_version') or 'none'}`",
        "",
    ]
    if report["issues"]:
        lines.append("## Blocking Issues")
        lines.extend(f"- {item}" for item in report["issues"])
        lines.append("")
    if report["warnings"]:
        lines.append("## Warnings")
        lines.extend(f"- {item}" for item in report["warnings"])
        lines.append("")
    lines.append("## Model Checks")
    lines.append("")
    lines.append("| Model | Accuracy | Macro F1 | Baseline Accuracy | Baseline Macro F1 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in report["models"]:
        lines.append(
            f"| {item['name']} | {item.get('accuracy')} | {item.get('macro_f1')} | "
            f"{item.get('baseline_accuracy')} | {item.get('baseline_macro_f1')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=Path("registry/model_registry.json"))
    parser.add_argument("--production", type=Path, default=Path("registry/production/model_registry.json"))
    parser.add_argument("--data-validation", type=Path, default=Path("reports/data_validation_report.json"))
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--min-macro-f1", type=float, default=0.95)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    parser.add_argument("--max-macro-f1-drop", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("reports/release_gate_report.json"))
    args = parser.parse_args()

    candidate = load_json(args.candidate, {})
    production = load_json(args.production, {})
    validation = load_json(args.data_validation, {"passed": True})
    issues = []
    warnings = []

    if not candidate:
        issues.append(f"missing candidate registry: {args.candidate}")
    if validation and not validation.get("passed", False):
        issues.append("data validation did not pass")
    if not production:
        warnings.append("no production baseline registry found; absolute thresholds only")

    baseline = model_map(production or {})
    model_reports = []
    for model in candidate.get("models", []):
        name = model["name"]
        accuracy = model.get("accuracy")
        macro_f1 = model.get("macro_f1")
        base = baseline.get(name, {})
        baseline_accuracy = base.get("accuracy")
        baseline_macro_f1 = base.get("macro_f1")
        if accuracy is not None and accuracy < args.min_accuracy:
            issues.append(f"{name} accuracy {accuracy:.4f} < {args.min_accuracy:.4f}")
        if macro_f1 is not None and macro_f1 < args.min_macro_f1:
            issues.append(f"{name} macro_f1 {macro_f1:.4f} < {args.min_macro_f1:.4f}")
        if baseline_accuracy is not None and accuracy is not None:
            drop = baseline_accuracy - accuracy
            if drop > args.max_accuracy_drop:
                issues.append(f"{name} accuracy drop {drop:.4f} > {args.max_accuracy_drop:.4f}")
        if baseline_macro_f1 is not None and macro_f1 is not None:
            drop = baseline_macro_f1 - macro_f1
            if drop > args.max_macro_f1_drop:
                issues.append(f"{name} macro_f1 drop {drop:.4f} > {args.max_macro_f1_drop:.4f}")
        model_reports.append({
            "name": name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "baseline_accuracy": baseline_accuracy,
            "baseline_macro_f1": baseline_macro_f1,
        })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "passed": not issues,
        "candidate_version": candidate.get("version"),
        "production_version": production.get("production_version") or production.get("version"),
        "thresholds": {
            "min_accuracy": args.min_accuracy,
            "min_macro_f1": args.min_macro_f1,
            "max_accuracy_drop": args.max_accuracy_drop,
            "max_macro_f1_drop": args.max_macro_f1_drop,
        },
        "issues": issues,
        "warnings": warnings,
        "models": model_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), report)
    print(f"release_gate_report={args.output}")
    print(f"passed={report['passed']}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
