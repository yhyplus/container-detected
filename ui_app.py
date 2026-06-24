#!/usr/bin/env python3
"""Local web console for Pixie collection, training, and anomaly prediction."""

import csv
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for


ROOT = Path(__file__).resolve().parent
LOCAL_TZ = ZoneInfo("Asia/Shanghai")
RUNS_DIR = ROOT / "runs"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "ui-logs"
MODEL_FILES = {
    "random_forest": {
        "binary": MODELS_DIR / "anomaly_binary_detector.joblib",
        "multiclass": MODELS_DIR / "anomaly_detector.joblib",
    },
    "onnx_mlp": {
        "binary": MODELS_DIR / "anomaly_mlp_binary.onnx",
        "multiclass": MODELS_DIR / "anomaly_mlp.onnx",
    },
}
COLLECTION_PROFILES = {
    "low": {"workers": "2", "memory_mib": "96"},
    "medium": {"workers": "4", "memory_mib": "128"},
    "high": {"workers": "6", "memory_mib": "160"},
    "eval": {"workers": "3", "memory_mib": "112"},
}
EXPERIMENTS = {
    "http_flood": {"label": "DDoS：HTTP 洪泛", "description": "并发请求 Nginx 服务，观察 HTTP RPS 激增。"},
    "network_connections": {"label": "DDoS：TCP 短连接", "description": "快速建立短连接，观察网络包速率异常。"},
    "dns_flood": {"label": "DDoS：DNS 洪泛", "description": "高频解析随机域名，观察 DNS 请求异常。"},
    "portscan": {"label": "端口扫描", "description": "在集群内扫描 Nginx Pod 的多个 TCP 端口。"},
    "lateral_movement": {"label": "横向移动：扫描内网", "description": "仅扫描 anomaly-test 中受控 Pod 和 Service 的常见端口。"},
    "cpu": {"label": "资源异常：CPU 飙升", "description": "持续计算，观察容器 CPU 使用率异常升高。"},
    "syscall_anomaly": {"label": "系统调用异常", "description": "在测试容器中启动非预期进程并访问敏感文件。"},
}
CLASS_LABELS = {
    "normal": "正常",
    "anomaly": "异常",
    "ddos": "DDoS 洪泛",
    "portscan": "端口扫描",
    "lateral_movement": "横向移动",
    "resource_anomaly": "资源异常",
    "syscall_anomaly": "系统调用异常",
}
jobs = {}
jobs_lock = threading.Lock()
app = Flask(__name__)
metrics_cache = {"expires_at": 0.0, "payload": None}
metrics_lock = threading.Lock()


def timestamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def safe_run_dir(name):
    path = RUNS_DIR / name
    if not path.is_dir() or path.parent != RUNS_DIR:
        abort(404)
    return path


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_mtime_text(path):
    if not path.exists():
        return "-"
    return datetime.fromtimestamp(path.stat().st_mtime, LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def local_time_text(value):
    if not value or value == "-":
        return "-"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def percent(value):
    if value is None:
        return "-"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "-"


def mlops_pipeline_summary():
    registry = read_json(ROOT / "registry" / "model_registry.json")
    data_registry = read_json(ROOT / "registry" / "data_registry.json")
    validation = read_json(ROOT / "reports" / "data_validation_report.json")
    release_gate = read_json(ROOT / "reports" / "release_gate_report.json")
    retrain = read_json(ROOT / "reports" / "auto_retrain_report.json")
    publish = read_json(ROOT / "reports" / "mlflow_publish_report.json")

    models = registry.get("models", [])
    datasets = data_registry.get("datasets", [])
    actions = set(retrain.get("actions", []))
    published = publish.get("published", [])
    release_passed = release_gate.get("passed")
    validation_passed = validation.get("passed")
    if validation_passed is None and validation:
        validation_passed = not validation.get("errors")

    steps = [
        {
            "key": "validate",
            "title": "数据校验",
            "status": "ok" if validation_passed else "missing",
            "detail": "检查训练/评估数据字段、标签、样本量和类别分布",
            "evidence": f"reports/data_validation_report.json · {file_mtime_text(ROOT / 'reports' / 'data_validation_report.json')}",
        },
        {
            "key": "train",
            "title": "训练与评估",
            "status": "ok" if models else "missing",
            "detail": f"已管理 {len(models)} 个模型，训练集/评估集版本随模型记录",
            "evidence": f"registry/model_registry.json · {file_mtime_text(ROOT / 'registry' / 'model_registry.json')}",
        },
        {
            "key": "mlflow",
            "title": "MLflow 记录",
            "status": "ok" if published else "missing",
            "detail": f"已发布 {len(published)} 个注册模型，记录指标、参数、模型文件和版本",
            "evidence": f"reports/mlflow_publish_report.json · {file_mtime_text(ROOT / 'reports' / 'mlflow_publish_report.json')}",
        },
        {
            "key": "gate",
            "title": "发布门禁",
            "status": "ok" if release_passed else "missing",
            "detail": "根据准确率、Macro F1、模型文件完整性判断是否允许发布",
            "evidence": f"reports/release_gate_report.json · {file_mtime_text(ROOT / 'reports' / 'release_gate_report.json')}",
        },
        {
            "key": "deploy",
            "title": "部署准备",
            "status": "ok" if "build_and_deploy_to_k3s" in actions or registry.get("production_path") else "missing",
            "detail": "生产模型快照可用于构建推理镜像并部署到 anomaly-test 命名空间",
            "evidence": f"registry/production · {file_mtime_text(ROOT / 'registry' / 'production')}",
        },
        {
            "key": "retrain",
            "title": "重训练触发",
            "status": "ok" if retrain else "missing",
            "detail": "根据新数据量、指标阈值和漂移结果决定是否触发下一轮训练",
            "evidence": f"reports/auto_retrain_report.json · {file_mtime_text(ROOT / 'reports' / 'auto_retrain_report.json')}",
        },
    ]

    model_rows = []
    for model in models:
        model_rows.append({
            "name": model.get("name", "-"),
            "target": "六分类" if model.get("target") == "class_id" else "二分类",
            "accuracy": percent(model.get("accuracy")),
            "macro_f1": f"{float(model.get('macro_f1')):.3f}" if model.get("macro_f1") is not None else "-",
            "rows": model.get("training_rows", "-"),
        })

    dataset_rows = []
    for dataset in datasets:
        dataset_rows.append({
            "name": dataset.get("name", "-"),
            "rows": dataset.get("rows", "-"),
            "features": dataset.get("feature_count", "-"),
            "path": dataset.get("combined_path", "-"),
        })

    retrain_display = dict(retrain)
    if retrain_display:
        retrain_display["generated_at"] = local_time_text(retrain_display.get("generated_at"))

    return {
        "version": registry.get("production_version") or registry.get("version") or "-",
        "stage": registry.get("stage", "-"),
        "generated_at": local_time_text(registry.get("generated_at", "-")),
        "latest_retrain": retrain_display,
        "steps": steps,
        "models": model_rows,
        "datasets": dataset_rows,
        "mlflow_models": [
            {
                "name": item.get("name", "-"),
                "version": item.get("version", "-"),
                "alias": item.get("alias", "-"),
                "created": item.get("created", False),
            }
            for item in published
        ],
    }


def parse_prometheus_metrics(text):
    values = {}
    classes = {}
    buckets = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        metric, _, value_text = line.rpartition(" ")
        if not metric or not value_text:
            continue
        try:
            value = float(value_text)
        except ValueError:
            continue
        name, labels = (metric.split("{", 1) + [""])[:2] if "{" in metric else (metric, "")
        labels = labels.rstrip("}")
        if name == "anomaly_inference_predicted_class_total":
            label = "unknown"
            for part in labels.split(","):
                if part.startswith("class="):
                    label = part.split("=", 1)[1].strip('"')
            classes[label] = value
        elif name == "anomaly_inference_anomaly_probability_bucket":
            label = "+Inf"
            for part in labels.split(","):
                if part.startswith("le="):
                    label = part.split("=", 1)[1].strip('"')
            buckets[label] = value
        else:
            values[name] = value
    return {"values": values, "classes": classes, "buckets": buckets}


def load_inference_metrics():
    urls = [
        os.environ.get("INFERENCE_METRICS_URL", ""),
        "http://127.0.0.1:18080/metrics",
        "http://127.0.0.1:8080/metrics",
    ]
    errors = []
    for url in [item for item in urls if item]:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                text = response.read().decode("utf-8", errors="replace")
            parsed = parse_prometheus_metrics(text)
            parsed.update({
                "ok": True,
                "source": url,
                "raw": text,
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
            })
            return parsed
        except (OSError, urllib.error.URLError) as exc:
            errors.append(f"{url}: {exc}")
    try:
        command = [
            "kubectl", "--kubeconfig", "/home/idolsingerydd/.kube/config",
            "-n", "anomaly-test", "exec", "deployment/anomaly-generator", "--",
            "python", "-c",
            "import urllib.request; print(urllib.request.urlopen('http://anomaly-inference:8080/metrics', timeout=3).read().decode())",
        ]
        result = subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, timeout=6, check=False
        )
        if result.returncode == 0:
            parsed = parse_prometheus_metrics(result.stdout)
            parsed.update({
                "ok": True,
                "source": "k8s://anomaly-test/anomaly-inference:8080/metrics",
                "raw": result.stdout,
                "fetched_at": datetime.now().isoformat(timespec="seconds"),
            })
            return parsed
        errors.append(result.stderr.strip() or result.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"kubectl metrics fallback: {exc}")
    return {
        "ok": False,
        "error": "; ".join(errors) if errors else "no metrics endpoint configured",
        "values": {},
        "classes": {},
        "buckets": {},
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
    }


def as_float(row, field):
    try:
        return float(row.get(field, 0))
    except (TypeError, ValueError):
        return 0.0


def summarize_prediction_run(run_dir):
    rows = read_csv(run_dir / "predictions.csv")
    if not rows:
        return None
    alerts = [row for row in rows if row.get("is_anomaly") == "1"]
    observed = [row for row in rows if row.get("has_signal") == "1"]
    latest = rows[-1]
    metadata = read_json(run_dir / "summary.json")
    model_type = "multiclass" if any(
        field in latest for field in ["probability_ddos", "probability_portscan", "probability_lateral_movement"]
    ) else "binary"
    return {
        "name": run_dir.name,
        "model_type": model_type,
        "algorithm": metadata.get("algorithm", "random_forest"),
        "windows": len(rows),
        "observed": len(observed),
        "alerts": len(alerts),
        "latest_time": latest.get("window_start", "-"),
        "latest_alert_probability": max(
            [as_float(row, "anomaly_probability") for row in alerts], default=0.0
        ),
    }


def prediction_runs():
    summaries = []
    run_dirs = sorted(
        RUNS_DIR.glob("live-*"),
        key=lambda path: (path / "predictions.csv").stat().st_mtime
        if (path / "predictions.csv").exists() else path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        summary = summarize_prediction_run(run_dir)
        if summary:
            summaries.append(summary)
    return summaries


def training_runs():
    return [
        path.name for path in sorted(RUNS_DIR.iterdir())
        if path.is_dir() and is_current_dataset(path / "training_multiclass_windows.csv")
        and not path.name.startswith("live-")
    ]


def is_current_dataset(path):
    rows = read_csv(path)
    scenarios = {row.get("scenario") for row in rows}
    return bool(
        rows and {"attack_type", "attack_subtype", "sensor_event_count"} <= set(rows[0])
        and {"lateral_movement", "cpu", "syscall_anomaly"} <= scenarios
    )


def model_file(algorithm, model_type):
    if algorithm not in MODEL_FILES or model_type not in MODEL_FILES[algorithm]:
        abort(400)
    return MODEL_FILES[algorithm][model_type]


def evaluation_runs():
    return [name for name in training_runs() if name.startswith("eval")]


def model_training_runs():
    return [name for name in training_runs() if not name.startswith("eval")]


def start_job(kind, command, env=None, run_name=None):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex[:10]
    log_path = LOGS_DIR / f"{job_id}.log"
    job = {
        "id": job_id,
        "kind": kind,
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": "",
        "log_path": log_path,
        "run_name": run_name,
    }
    with jobs_lock:
        jobs[job_id] = job

    def execute():
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                text=True, check=False,
            )
        with jobs_lock:
            job["status"] = "completed" if result.returncode == 0 else "failed"
            job["finished_at"] = datetime.now().isoformat(timespec="seconds")
            job["returncode"] = result.returncode

    threading.Thread(target=execute, daemon=True).start()
    return job_id


def public_job(job):
    return {
        key: value for key, value in job.items()
        if key != "log_path"
    }


@app.get("/")
def dashboard():
    recent_runs = prediction_runs()
    return render_template(
        "dashboard.html",
        prediction_runs=recent_runs[:12],
        latest_run=recent_runs[0] if recent_runs else None,
        training_runs=model_training_runs(),
        evaluation_runs=evaluation_runs(),
        jobs=sorted(jobs.values(), key=lambda item: item["started_at"], reverse=True)[:12],
        profiles=COLLECTION_PROFILES,
        experiments=EXPERIMENTS,
        mlops_pipeline=mlops_pipeline_summary(),
    )


@app.get("/api/status")
def status():
    with jobs_lock:
        current_jobs = [
            public_job(job) for job in sorted(
                jobs.values(), key=lambda item: item["started_at"], reverse=True
            )[:12]
        ]
    runs = prediction_runs()[:12]
    return jsonify({
        "jobs": current_jobs,
        "prediction_runs": runs,
        "latest_run": runs[0] if runs else None,
    })


@app.get("/api/inference_metrics")
def inference_metrics():
    now = time.monotonic()
    with metrics_lock:
        if metrics_cache["payload"] and metrics_cache["expires_at"] > now:
            return jsonify(metrics_cache["payload"])
        payload = load_inference_metrics()
        metrics_cache["payload"] = payload
        metrics_cache["expires_at"] = now + 0.8
        return jsonify(payload)


@app.get("/api/mlops_pipeline")
def mlops_pipeline():
    return jsonify(mlops_pipeline_summary())


@app.post("/actions/predict")
def predict():
    model_type = request.form.get("model_type", "binary")
    algorithm = request.form.get("algorithm", "random_forest")
    selected_model = model_file(algorithm, model_type)
    try:
        threshold = min(1.0, max(0.0, float(request.form.get("threshold", "0.7"))))
        lookback = min(3600, max(30, int(request.form.get("lookback", "120"))))
    except ValueError:
        abort(400)
    run_name = f"live-{timestamp()}"
    env = os.environ.copy()
    env.update({
        "OUT": str(RUNS_DIR / run_name),
        "MODEL": str(selected_model),
        "ALGORITHM": algorithm,
        "THRESHOLD": str(threshold),
        "LOOKBACK_SECONDS": str(lookback),
    })
    start_job("实时检测", [str(ROOT / "predict_live.sh")], env=env, run_name=run_name)
    return redirect(url_for("dashboard"))


@app.post("/actions/collect")
def collect():
    profile_name = request.form.get("profile", "")
    profile = COLLECTION_PROFILES.get(profile_name)
    if not profile:
        abort(400)
    try:
        duration = min(3600, max(30, int(request.form.get("duration", "600"))))
    except ValueError:
        abort(400)
    run_name = f"{'eval' if profile_name == 'eval' else 'train-' + profile_name}-{timestamp()}"
    env = os.environ.copy()
    env.update({
        "OUT": str(RUNS_DIR / run_name),
        "DURATION": str(duration),
        "WORKERS": profile["workers"],
        "MEMORY_MIB": profile["memory_mib"],
    })
    start_job(f"采集 {profile_name}", [str(ROOT / "run_scenarios.sh")], env=env, run_name=run_name)
    return redirect(url_for("dashboard"))


@app.post("/actions/experiment")
def experiment():
    scenario = request.form.get("scenario", "")
    available_experiments = EXPERIMENTS
    if scenario not in available_experiments:
        abort(400)
    model_type = request.form.get("model_type", "multiclass")
    algorithm = request.form.get("algorithm", "random_forest")
    selected_model = model_file(algorithm, model_type)
    try:
        duration = min(300, max(30, int(request.form.get("duration", "60"))))
        threshold = min(1.0, max(0.0, float(request.form.get("threshold", "0.7"))))
    except ValueError:
        abort(400)
    run_name = f"live-experiment-{scenario}-{timestamp()}"
    env = os.environ.copy()
    env.update({
        "OUT": str(RUNS_DIR / run_name),
        "MODEL": str(selected_model),
        "ALGORITHM": algorithm,
        "SCENARIO": scenario,
        "DURATION": str(duration),
        "THRESHOLD": str(threshold),
    })
    start_job(f"模拟并检测 {available_experiments[scenario]['label']}",
              [str(ROOT / "run_experiment.sh")], env=env, run_name=run_name)
    return redirect(url_for("dashboard"))


@app.post("/actions/train")
def train():
    model_type = request.form.get("model_type", "binary")
    algorithm = request.form.get("algorithm", "random_forest")
    selected_model = model_file(algorithm, model_type)
    selected_runs = request.form.getlist("run")
    valid_runs = set(model_training_runs())
    if not selected_runs or any(name not in valid_runs for name in selected_runs):
        abort(400)
    command = [
        str(ROOT / ("train_mlp_onnx.py" if algorithm == "onnx_mlp" else "train_model.py"))
    ]
    for name in selected_runs:
        command.extend(["--run-dir", str(RUNS_DIR / name)])
    eval_run = request.form.get("eval_run", "")
    if eval_run:
        if eval_run not in set(evaluation_runs()):
            abort(400)
        command.extend(["--eval-run-dir", str(RUNS_DIR / eval_run)])
    if model_type == "binary":
        command.extend(["--target", "label"])
    output = (
        MODELS_DIR / ("anomaly_mlp_binary.onnx" if model_type == "binary" else "anomaly_mlp.onnx")
        if algorithm == "onnx_mlp" else selected_model
    )
    command.extend(["--output", str(output)])
    start_job(f"训练 {algorithm} {model_type}", command)
    return redirect(url_for("dashboard"))


@app.post("/actions/mlops_dry_run")
def mlops_dry_run():
    command = [
        sys.executable,
        str(ROOT / "mlops_auto_retrain.py"),
        "--force",
        "--dry-run",
        "--deploy",
        "--promote",
        "--publish-mlflow",
        "--rollback-on-failure",
    ]
    start_job("MLOps Pipeline 演示 dry-run", command)
    return redirect(url_for("dashboard") + "#mlops-panel")


@app.get("/runs/<name>")
def run_report(name):
    run_dir = safe_run_dir(name)
    rows = read_csv(run_dir / "predictions.csv")
    if not rows:
        abort(404)
    alerts = [row for row in rows if row.get("is_anomaly") == "1"]
    display_rows = list(reversed(alerts if alerts else rows))[:100]
    return render_template(
        "run_report.html",
        run_name=name,
        summary=summarize_prediction_run(run_dir),
        rows=display_rows,
        showing_alerts=bool(alerts),
        labels=CLASS_LABELS,
    )


@app.get("/jobs/<job_id>")
def job_report(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        abort(404)
    log = job["log_path"].read_text(encoding="utf-8") if job["log_path"].exists() else ""
    return render_template("job_report.html", job=job, log=log)


@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        abort(404)
    log = job["log_path"].read_text(encoding="utf-8") if job["log_path"].exists() else ""
    return jsonify({"job": public_job(job), "log": log})


if __name__ == "__main__":
    RUNS_DIR.mkdir(exist_ok=True)
    app.run(
        host=os.environ.get("UI_HOST", "0.0.0.0"),
        port=int(os.environ.get("UI_PORT", "5000")),
        debug=False,
    )
