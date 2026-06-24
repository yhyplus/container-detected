#!/usr/bin/env python3
"""Serve the trained MLP with only the Python standard library.

This service is intended for the local K3s demo path where using the existing
python:3.12-alpine image is more reliable than pulling a large ONNX Runtime
image. The predictions still use the trained MLP weights exported by
export_mlp_weights.py.
"""

import argparse
import json
import math
import os
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def class_name(class_names, class_id):
    return class_names.get(str(class_id), class_names.get(class_id, str(class_id)))


def dot_row(row, weights, bias):
    outputs = []
    for neuron_weights, neuron_bias in zip(weights, bias):
        total = neuron_bias
        for value, weight in zip(row, neuron_weights):
            total += value * weight
        outputs.append(total)
    return outputs


def softmax(logits):
    max_value = max(logits)
    exps = [math.exp(value - max_value) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def dashboard_html():
    return """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Anomaly Inference Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9e0e8;
      --accent: #0f766e;
      --accent-2: #2563eb;
      --warn: #b45309;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 18px 24px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { margin: 0; font-size: 22px; font-weight: 700; }
    .status { display: flex; align-items: center; gap: 10px; color: var(--muted); font-size: 14px; }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--warn); }
    .dot.ok { background: var(--accent); }
    main { max-width: 1220px; margin: 0 auto; padding: 22px; }
    .grid { display: grid; gap: 14px; }
    .kpis { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .charts { grid-template-columns: minmax(0, 1.1fr) minmax(0, 0.9fr); margin-top: 14px; }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }
    .label { color: var(--muted); font-size: 13px; margin-bottom: 8px; }
    .value { font-size: 30px; font-weight: 750; line-height: 1.1; overflow-wrap: anywhere; }
    .sub { color: var(--muted); font-size: 13px; margin-top: 8px; }
    h2 { margin: 0 0 14px; font-size: 16px; }
    .bar-row {
      display: grid;
      grid-template-columns: minmax(105px, 155px) minmax(0, 1fr) 58px;
      gap: 10px;
      align-items: center;
      margin: 10px 0;
      font-size: 13px;
    }
    .track { height: 12px; background: #eef2f6; border-radius: 999px; overflow: hidden; }
    .fill { height: 100%; width: 0%; background: var(--accent-2); transition: width .25s ease; }
    .fill.alert { background: var(--danger); }
    .table { width: 100%; border-collapse: collapse; font-size: 13px; }
    .table th, .table td { padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; }
    .table th { color: var(--muted); font-weight: 600; }
    .footer { margin-top: 14px; color: var(--muted); font-size: 13px; }
    @media (max-width: 860px) {
      header { align-items: flex-start; flex-direction: column; }
      main { padding: 14px; }
      .kpis, .charts { grid-template-columns: 1fr; }
      .bar-row { grid-template-columns: minmax(90px, 125px) minmax(0, 1fr) 48px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>异常检测在线监控</h1>
    <div class="status"><span id="status-dot" class="dot"></span><span id="status-text">连接中</span></div>
  </header>
  <main>
    <section class="grid kpis">
      <div class="card"><div class="label">预测请求</div><div id="requests" class="value">0</div><div class="sub">累计请求数</div></div>
      <div class="card"><div class="label">异常告警</div><div id="alerts" class="value">0</div><div id="alert-rate" class="sub">告警率 0%</div></div>
      <div class="card"><div class="label">平均延迟</div><div id="latency" class="value">0 ms</div><div id="latency-max" class="sub">最大 0 ms</div></div>
      <div class="card"><div class="label">内存占用</div><div id="memory" class="value">0 MiB</div><div id="cpu" class="sub">CPU 0s</div></div>
    </section>
    <section class="grid charts">
      <div class="card">
        <h2>预测类别分布</h2>
        <div id="classes"></div>
      </div>
      <div class="card">
        <h2>异常概率分布</h2>
        <div id="probability"></div>
      </div>
    </section>
    <section class="grid charts">
      <div class="card">
        <h2>运行状态</h2>
        <table class="table">
          <tbody>
            <tr><th>预测行数</th><td id="rows">0</td></tr>
            <tr><th>异常概率均值</th><td id="prob-avg">0</td></tr>
            <tr><th>异常概率范围</th><td id="prob-range">0 - 0</td></tr>
            <tr><th>服务运行时间</th><td id="uptime">0s</td></tr>
          </tbody>
        </table>
      </div>
      <div class="card">
        <h2>接口</h2>
        <table class="table">
          <tbody>
            <tr><th>健康检查</th><td><a href="/healthz">/healthz</a></td></tr>
            <tr><th>模型元数据</th><td><a href="/metadata">/metadata</a></td></tr>
            <tr><th>原始指标</th><td><a href="/metrics">/metrics</a></td></tr>
          </tbody>
        </table>
      </div>
    </section>
    <div id="updated" class="footer">等待刷新</div>
  </main>
  <script>
    function parseMetrics(text) {
      const values = {};
      const classes = {};
      const buckets = {};
      for (const raw of text.split("\\n")) {
        const line = raw.trim();
        if (!line || line.startsWith("#")) continue;
        const match = line.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*)(\\{([^}]*)\\})?\\s+(-?\\d+(?:\\.\\d+)?(?:e[-+]?\\d+)?)$/);
        if (!match) continue;
        const name = match[1];
        const labels = match[3] || "";
        const value = Number(match[4]);
        if (name === "anomaly_inference_predicted_class_total") {
          const label = (labels.match(/class="([^"]+)"/) || [])[1] || "unknown";
          classes[label] = value;
        } else if (name === "anomaly_inference_anomaly_probability_bucket") {
          const label = (labels.match(/le="([^"]+)"/) || [])[1] || "+Inf";
          buckets[label] = value;
        } else {
          values[name] = value;
        }
      }
      return { values, classes, buckets };
    }
    function fmt(value, digits = 2) {
      return Number.isFinite(value) ? value.toFixed(digits) : "0.00";
    }
    function barRows(container, data, colorClass = "") {
      const entries = Object.entries(data).filter(([, value]) => value > 0);
      const max = Math.max(1, ...entries.map(([, value]) => value));
      container.innerHTML = entries.length ? "" : '<div class="sub">暂无数据</div>';
      for (const [label, value] of entries) {
        const row = document.createElement("div");
        row.className = "bar-row";
        row.innerHTML = `<div>${label}</div><div class="track"><div class="fill ${colorClass}" style="width:${Math.max(3, value / max * 100)}%"></div></div><div>${value}</div>`;
        container.appendChild(row);
      }
    }
    function seconds(value) {
      if (value >= 3600) return `${fmt(value / 3600, 1)}h`;
      if (value >= 60) return `${fmt(value / 60, 1)}m`;
      return `${fmt(value, 0)}s`;
    }
    async function refresh() {
      const statusDot = document.getElementById("status-dot");
      const statusText = document.getElementById("status-text");
      try {
        const response = await fetch("/metrics", { cache: "no-store" });
        const parsed = parseMetrics(await response.text());
        const m = parsed.values;
        const requests = m.anomaly_inference_requests_total || 0;
        const rows = m.anomaly_inference_rows_total || 0;
        const alerts = m.anomaly_inference_alerts_total || 0;
        const latencyTotal = m.anomaly_inference_latency_ms_total || 0;
        const probSum = m.anomaly_inference_anomaly_probability_sum || 0;
        document.getElementById("requests").textContent = requests;
        document.getElementById("alerts").textContent = alerts;
        document.getElementById("alert-rate").textContent = `告警率 ${fmt(rows ? alerts / rows * 100 : 0, 1)}%`;
        document.getElementById("latency").textContent = `${fmt(requests ? latencyTotal / requests : 0, 3)} ms`;
        document.getElementById("latency-max").textContent = `最大 ${fmt(m.anomaly_inference_latency_ms_max || 0, 3)} ms`;
        document.getElementById("memory").textContent = `${fmt((m.anomaly_inference_process_resident_memory_bytes || 0) / 1048576, 1)} MiB`;
        document.getElementById("cpu").textContent = `CPU ${fmt(m.anomaly_inference_process_cpu_seconds_total || 0, 3)}s`;
        document.getElementById("rows").textContent = rows;
        document.getElementById("prob-avg").textContent = fmt(rows ? probSum / rows : 0, 4);
        document.getElementById("prob-range").textContent = `${fmt(m.anomaly_inference_anomaly_probability_min || 0, 4)} - ${fmt(m.anomaly_inference_anomaly_probability_max || 0, 4)}`;
        document.getElementById("uptime").textContent = seconds(m.anomaly_inference_process_uptime_seconds || 0);
        const bucketLabels = ["0.1", "0.3", "0.5", "0.7", "0.9", "1.0", "+Inf"];
        const bucketData = {};
        let previous = 0;
        for (const label of bucketLabels) {
          const cumulative = parsed.buckets[label] || 0;
          bucketData[`<= ${label}`] = Math.max(0, cumulative - previous);
          previous = cumulative;
        }
        barRows(document.getElementById("classes"), parsed.classes);
        barRows(document.getElementById("probability"), bucketData, "alert");
        statusDot.classList.add("ok");
        statusText.textContent = "在线";
        document.getElementById("updated").textContent = `最后刷新 ${new Date().toLocaleString()}`;
      } catch (error) {
        statusDot.classList.remove("ok");
        statusText.textContent = "连接失败";
        document.getElementById("updated").textContent = String(error);
      }
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class InferenceModel:
    def __init__(self, weights_path):
        self.weights_path = weights_path
        self.metadata = json.loads(weights_path.read_text(encoding="utf-8"))
        self.features = self.metadata["features"]
        self.classes = [int(value) for value in self.metadata["classes"]]
        self.class_names = self.metadata.get("class_names", {})
        self.normal_index = self.classes.index(0) if 0 in self.classes else None
        self.mean = self.metadata["scaler"]["mean"]
        self.scale = self.metadata["scaler"]["scale"]
        self.layers = self.metadata["layers"]

    def predict_one(self, row, threshold):
        values = [
            (float(row[name]) - mean) / scale
            for name, mean, scale in zip(self.features, self.mean, self.scale)
        ]
        activations = values
        for layer in self.layers:
            activations = dot_row(activations, layer["weight"], layer["bias"])
            if layer["activation"] == "relu":
                activations = [max(0.0, value) for value in activations]
        probabilities = softmax(activations)
        prediction_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        class_id = self.classes[prediction_index]
        anomaly_probability = (
            1.0 - probabilities[self.normal_index]
            if self.normal_index is not None
            else float(class_id != 0)
        )
        predicted_class = class_name(self.class_names, class_id)
        return {
            "predicted_class_id": class_id,
            "predicted_class": predicted_class,
            "confidence": probabilities[prediction_index],
            "anomaly_probability": anomaly_probability,
            "is_anomaly": int(
                anomaly_probability >= threshold
                and (self.metadata["target"] == "label" or class_id != 0)
            ),
            "probabilities": {
                class_name(self.class_names, class_id): probabilities[index]
                for index, class_id in enumerate(self.classes)
            },
        }


def make_handler(model, threshold):
    started_at = time.time()
    anomaly_probability_buckets = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    metrics = {
        "requests": 0,
        "rows": 0,
        "alerts": 0,
        "latency_ms_total": 0.0,
        "latency_ms_max": 0.0,
        "anomaly_probability_sum": 0.0,
        "anomaly_probability_min": None,
        "anomaly_probability_max": None,
        "anomaly_probability_buckets": Counter(),
        "classes": Counter(),
    }

    def resident_memory_bytes():
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            statm = Path("/proc/self/statm").read_text(encoding="utf-8").split()
            return int(statm[1]) * page_size
        except Exception:
            return 0

    class Handler(BaseHTTPRequestHandler):
        server_version = "AnomalyMLP/1.0"

        def send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_html(self, status, html):
            body = html.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in {"/", "/dashboard"}:
                self.send_html(200, dashboard_html())
                return
            if path == "/healthz":
                self.send_json(200, {
                    "status": "ok",
                    "model": str(model.weights_path),
                    "target": model.metadata["target"],
                    "feature_count": len(model.features),
                })
                return
            if path == "/metadata":
                self.send_json(200, {
                    "model": str(model.weights_path),
                    "target": model.metadata["target"],
                    "classes": model.classes,
                    "class_names": model.class_names,
                    "features": model.features,
                    "threshold": threshold,
                })
                return
            if path == "/metrics":
                lines = [
                    "# HELP anomaly_inference_requests_total Total prediction requests.",
                    "# TYPE anomaly_inference_requests_total counter",
                    f"anomaly_inference_requests_total {metrics['requests']}",
                    "# HELP anomaly_inference_rows_total Total predicted rows.",
                    "# TYPE anomaly_inference_rows_total counter",
                    f"anomaly_inference_rows_total {metrics['rows']}",
                    "# HELP anomaly_inference_alerts_total Total anomaly alerts.",
                    "# TYPE anomaly_inference_alerts_total counter",
                    f"anomaly_inference_alerts_total {metrics['alerts']}",
                    "# HELP anomaly_inference_latency_ms_total Total inference latency in milliseconds.",
                    "# TYPE anomaly_inference_latency_ms_total counter",
                    f"anomaly_inference_latency_ms_total {metrics['latency_ms_total']}",
                    "# HELP anomaly_inference_latency_ms_max Maximum observed inference latency in milliseconds.",
                    "# TYPE anomaly_inference_latency_ms_max gauge",
                    f"anomaly_inference_latency_ms_max {metrics['latency_ms_max']}",
                    "# HELP anomaly_inference_anomaly_probability_sum Sum of anomaly probabilities.",
                    "# TYPE anomaly_inference_anomaly_probability_sum counter",
                    f"anomaly_inference_anomaly_probability_sum {metrics['anomaly_probability_sum']}",
                    "# HELP anomaly_inference_anomaly_probability_min Minimum observed anomaly probability.",
                    "# TYPE anomaly_inference_anomaly_probability_min gauge",
                    "anomaly_inference_anomaly_probability_min "
                    f"{metrics['anomaly_probability_min'] if metrics['anomaly_probability_min'] is not None else 0.0}",
                    "# HELP anomaly_inference_anomaly_probability_max Maximum observed anomaly probability.",
                    "# TYPE anomaly_inference_anomaly_probability_max gauge",
                    "anomaly_inference_anomaly_probability_max "
                    f"{metrics['anomaly_probability_max'] if metrics['anomaly_probability_max'] is not None else 0.0}",
                    "# HELP anomaly_inference_process_cpu_seconds_total CPU seconds consumed by this process.",
                    "# TYPE anomaly_inference_process_cpu_seconds_total counter",
                    f"anomaly_inference_process_cpu_seconds_total {time.process_time()}",
                    "# HELP anomaly_inference_process_resident_memory_bytes Resident memory used by this process.",
                    "# TYPE anomaly_inference_process_resident_memory_bytes gauge",
                    f"anomaly_inference_process_resident_memory_bytes {resident_memory_bytes()}",
                    "# HELP anomaly_inference_process_uptime_seconds Process uptime in seconds.",
                    "# TYPE anomaly_inference_process_uptime_seconds gauge",
                    f"anomaly_inference_process_uptime_seconds {time.time() - started_at}",
                ]
                cumulative = 0
                for bucket in anomaly_probability_buckets:
                    cumulative += metrics["anomaly_probability_buckets"][bucket]
                    lines.append(
                        f'anomaly_inference_anomaly_probability_bucket{{le="{bucket}"}} {cumulative}'
                    )
                lines.append(
                    f'anomaly_inference_anomaly_probability_bucket{{le="+Inf"}} {metrics["rows"]}'
                )
                lines.append(f"anomaly_inference_anomaly_probability_count {metrics['rows']}")
                for label, count in sorted(metrics["classes"].items()):
                    lines.append(
                        f'anomaly_inference_predicted_class_total{{class="{label}"}} {count}'
                    )
                body = ("\n".join(lines) + "\n").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                self.send_json(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", "0"))
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
            except json.JSONDecodeError:
                self.send_json(400, {"error": "request body must be JSON"})
                return
            rows = payload.get("rows", payload if isinstance(payload, list) else [payload])
            if not isinstance(rows, list) or not rows:
                self.send_json(400, {"error": "request must contain a row object or rows list"})
                return
            missing = sorted(set(model.features) - set(rows[0]))
            if missing:
                self.send_json(400, {"error": "missing required features", "features": missing})
                return

            started = time.perf_counter()
            result_rows = [model.predict_one(row, threshold) for row in rows]
            latency_ms = (time.perf_counter() - started) * 1000
            for result in result_rows:
                metrics["classes"][result["predicted_class"]] += 1
                metrics["alerts"] += int(result["is_anomaly"])
                probability = result["anomaly_probability"]
                metrics["anomaly_probability_sum"] += probability
                metrics["anomaly_probability_min"] = (
                    probability if metrics["anomaly_probability_min"] is None
                    else min(metrics["anomaly_probability_min"], probability)
                )
                metrics["anomaly_probability_max"] = (
                    probability if metrics["anomaly_probability_max"] is None
                    else max(metrics["anomaly_probability_max"], probability)
                )
                for bucket in anomaly_probability_buckets:
                    if probability <= bucket:
                        metrics["anomaly_probability_buckets"][bucket] += 1
                        break
            metrics["requests"] += 1
            metrics["rows"] += len(rows)
            metrics["latency_ms_total"] += latency_ms
            metrics["latency_ms_max"] = max(metrics["latency_ms_max"], latency_ms)
            self.send_json(200, {
                "model": str(model.weights_path),
                "target": model.metadata["target"],
                "threshold": threshold,
                "rows": result_rows,
                "batch_size": len(rows),
                "latency_ms": latency_ms,
            })

        def log_message(self, fmt, *args):
            print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=Path("models/anomaly_mlp_weights.json"))
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    model = InferenceModel(args.weights)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, args.threshold))
    print(f"serving {args.weights} on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
