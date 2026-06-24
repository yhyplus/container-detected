#!/usr/bin/env python3
"""Automatically run the training pipeline when retraining rules are triggered."""

import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command, dry_run=False):
    print("+", " ".join(str(part) for part in command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def run_capture(command, dry_run=False):
    print("+", " ".join(str(part) for part in command))
    if dry_run:
        return ""
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    return result.stdout


def deploy_to_k3s(dry_run=False):
    run(["docker", "build", "-f", "Dockerfile.inference-pure", "-t", "anomaly-inference:latest", "."], dry_run=dry_run)
    run(["docker", "save", "anomaly-inference:latest", "-o", "/tmp/anomaly-inference-latest.tar"], dry_run=dry_run)
    run([
        "kubectl", "--kubeconfig", "/home/idolsingerydd/.kube/config",
        "-n", "anomaly-test", "delete", "job", "anomaly-image-import", "--ignore-not-found=true",
    ], dry_run=dry_run)
    run([
        "kubectl", "--kubeconfig", "/home/idolsingerydd/.kube/config",
        "apply", "-f", "k8s/anomaly-image-import-job.yaml",
    ], dry_run=dry_run)
    run([
        "kubectl", "--kubeconfig", "/home/idolsingerydd/.kube/config",
        "-n", "anomaly-test", "wait", "--for=condition=complete",
        "job/anomaly-image-import", "--timeout=90s",
    ], dry_run=dry_run)
    run([
        "kubectl", "--kubeconfig", "/home/idolsingerydd/.kube/config",
        "apply", "-f", "k8s/anomaly-inference.yaml",
    ], dry_run=dry_run)
    run([
        "kubectl", "--kubeconfig", "/home/idolsingerydd/.kube/config",
        "-n", "anomaly-test", "rollout", "restart", "deployment/anomaly-inference",
    ], dry_run=dry_run)
    run([
        "kubectl", "--kubeconfig", "/home/idolsingerydd/.kube/config",
        "-n", "anomaly-test", "rollout", "status",
        "deployment/anomaly-inference", "--timeout=120s",
    ], dry_run=dry_run)
    run_capture([
        "kubectl", "--kubeconfig", "/home/idolsingerydd/.kube/config",
        "-n", "anomaly-test", "exec", "deployment/anomaly-generator", "--",
        "python", "-c",
        "import urllib.request; print(urllib.request.urlopen('http://anomaly-inference:8080/healthz', timeout=5).read().decode())",
    ], dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-data-rows", type=int, default=0)
    parser.add_argument("--min-new-rows", type=int, default=300)
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument("--max-drift-score", type=float, default=0.25)
    parser.add_argument("--version", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--build-image", action="store_true")
    parser.add_argument("--deploy", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--publish-mlflow", action="store_true")
    parser.add_argument("--skip-release-gate", action="store_true")
    parser.add_argument("--rollback-on-failure", action="store_true")
    args = parser.parse_args()

    report_path = ROOT / "reports" / "retrain_check.json"
    check_command = [
        sys.executable,
        "mlops_retrain_check.py",
        "--new-data-rows", str(args.new_data_rows),
        "--min-new-rows", str(args.min_new_rows),
        "--min-accuracy", str(args.min_accuracy),
        "--max-drift-score", str(args.max_drift_score),
        "--output", str(report_path.relative_to(ROOT)),
    ]
    run(check_command, dry_run=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    should_retrain = args.force or report["should_retrain"]
    version = args.version or datetime.now(timezone.utc).strftime("auto-%Y%m%d-%H%M%S")

    decision = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "forced": args.force,
        "dry_run": args.dry_run,
        "should_retrain": should_retrain,
        "reasons": report.get("reasons", []),
        "actions": [],
    }

    if not should_retrain:
        print("No retraining trigger matched; leaving the current model in place.")
        decision["actions"].append("skip_retraining")
    else:
        try:
            pipeline_command = [sys.executable, "mlops_train_pipeline.py", "--version", version]
            if args.no_mlflow:
                pipeline_command.append("--no-mlflow")
            run(pipeline_command, dry_run=args.dry_run)
            decision["actions"].append("train_evaluate_export")

            run([sys.executable, "export_mlp_weights.py"], dry_run=args.dry_run)
            decision["actions"].append("export_pure_python_weights")

            if not args.skip_release_gate:
                run([sys.executable, "mlops_release_gate.py"], dry_run=args.dry_run)
                decision["actions"].append("release_gate_passed")

            if args.build_image and not args.deploy:
                run(["docker", "build", "-f", "Dockerfile.inference-pure", "-t", "anomaly-inference:latest", "."], dry_run=args.dry_run)
                run(["docker", "save", "anomaly-inference:latest", "-o", "/tmp/anomaly-inference-latest.tar"], dry_run=args.dry_run)
                decision["actions"].append("build_inference_image")
            if args.deploy:
                deploy_to_k3s(dry_run=args.dry_run)
                decision["actions"].append("build_and_deploy_to_k3s")
            if args.promote:
                promote_command = [
                    sys.executable, "mlops_model_registry.py", "promote",
                    "--note", "validated by mlops_auto_retrain.py",
                ]
                if args.publish_mlflow:
                    promote_command.append("--publish-mlflow")
                run(promote_command, dry_run=args.dry_run)
                decision["actions"].append("promote_model_registry")
                if args.publish_mlflow:
                    decision["actions"].append("publish_mlflow_model_registry")
            decision["status"] = "success"
        except subprocess.CalledProcessError as exc:
            decision["status"] = "failed"
            decision["error"] = str(exc)
            decision["traceback"] = traceback.format_exc()
            if "release_gate_passed" not in decision["actions"]:
                run([sys.executable, "mlops_model_registry.py", "set-stage", "Rejected", "--note", "release gate or training failed"], dry_run=args.dry_run)
                decision["actions"].append("mark_candidate_rejected")
            if args.rollback_on_failure:
                try:
                    run([sys.executable, "mlops_model_registry.py", "rollback", "--note", "rollback after auto retrain failure"], dry_run=args.dry_run)
                    decision["actions"].append("rollback_model_registry")
                except subprocess.CalledProcessError as rollback_exc:
                    decision["rollback_error"] = str(rollback_exc)
            output = ROOT / "reports" / "auto_retrain_report.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"auto_retrain_report={output}")
            raise

    output = ROOT / "reports" / "auto_retrain_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"auto_retrain_report={output}")
    print(f"should_retrain={should_retrain} actions={decision['actions']}")


if __name__ == "__main__":
    main()
