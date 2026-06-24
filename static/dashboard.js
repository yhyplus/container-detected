const escapeHtml = (value) => {
  const node = document.createElement("span");
  node.textContent = value ?? "";
  return node.innerHTML;
};

const badge = (text, kind = "") => `<span class="badge ${kind}">${escapeHtml(text)}</span>`;

const fmt = (value, digits = 2) => Number.isFinite(value) ? value.toFixed(digits) : "0.00";

function durationText(seconds) {
  if (!Number.isFinite(seconds)) return "0s";
  if (seconds >= 3600) return `${fmt(seconds / 3600, 1)}h`;
  if (seconds >= 60) return `${fmt(seconds / 60, 1)}m`;
  return `${fmt(seconds, 0)}s`;
}

function renderJobs(jobs) {
  const body = document.querySelector("#jobs-body");
  if (!body) return;
  if (!jobs.length) {
    body.innerHTML = '<tr><td colspan="5">暂无后台任务</td></tr>';
    return;
  }
  body.innerHTML = jobs.map((job) => {
    const kind = job.status === "failed" ? "badge-bad" : job.status === "completed" ? "badge-ok" : "badge-warn";
    return `<tr><td>${escapeHtml(job.kind)}</td><td>${badge(job.status, kind)}</td>
      <td>${escapeHtml(job.started_at)}</td><td>${escapeHtml(job.run_name || "-")}</td>
      <td><a href="/jobs/${encodeURIComponent(job.id)}">查看日志</a></td></tr>`;
  }).join("");
}

function renderRuns(runs) {
  const body = document.querySelector("#runs-body");
  if (!body) return;
  if (!runs.length) {
    body.innerHTML = '<tr><td colspan="9">暂无检测记录</td></tr>';
    return;
  }
  body.innerHTML = runs.map((run) => {
    const state = run.alerts ? badge("发现异常", "badge-bad") : badge("正常", "badge-ok");
    const mode = badge(run.model_type === "multiclass" ? "六分类" : "二分类");
    const algorithm = run.algorithm === "isolation_forest"
      ? "方案 C：孤立森林"
      : run.algorithm === "onnx_mlp" ? "方案 B：ONNX" : "方案 A：RF";
    return `<tr><td><a href="/runs/${encodeURIComponent(run.name)}">${escapeHtml(run.name)}</a></td>
      <td>${algorithm}</td><td>${mode}</td><td>${state}</td><td>${run.windows}</td><td>${run.observed}</td>
      <td>${run.alerts}</td><td>${(run.latest_alert_probability * 100).toFixed(1)}%</td>
      <td>${escapeHtml(run.latest_time)}</td></tr>`;
  }).join("");
}

async function refreshDashboard() {
  try {
    const response = await fetch("/api/status", {cache: "no-store"});
    if (!response.ok) return;
    const payload = await response.json();
    renderJobs(payload.jobs);
    renderRuns(payload.prediction_runs);
    const latestName = payload.latest_run?.name || "";
    const currentName = document.body.dataset.latestRun || "";
    if (latestName && latestName !== currentName) window.location.reload();
  } catch (_) {
    // The next interval retries if the local server is temporarily busy.
  }
}

function renderBars(selector, data, warn = false) {
  const container = document.querySelector(selector);
  if (!container) return;
  const entries = Object.entries(data).filter(([, value]) => value > 0);
  if (!entries.length) {
    container.innerHTML = '<div class="muted small">暂无数据</div>';
    return;
  }
  const max = Math.max(1, ...entries.map(([, value]) => value));
  container.innerHTML = entries.map(([label, value]) => {
    const width = Math.max(3, value / max * 100);
    return `<div class="bar-row"><div>${escapeHtml(label)}</div><div class="bar-track"><div class="bar-fill ${warn ? "bar-warn" : ""}" style="width:${width}%"></div></div><div>${value}</div></div>`;
  }).join("");
}

function renderInferenceMetrics(payload) {
  const state = document.querySelector("#inference-state");
  if (!state) return;
  if (!payload.ok) {
    state.className = "badge badge-bad";
    state.textContent = "监控未连接";
    document.querySelector("#metric-updated").textContent = payload.error || "-";
    return;
  }
  state.className = "badge badge-ok";
  state.textContent = "实时在线";
  const m = payload.values || {};
  const requests = m.anomaly_inference_requests_total || 0;
  const rows = m.anomaly_inference_rows_total || 0;
  const alerts = m.anomaly_inference_alerts_total || 0;
  const latencyTotal = m.anomaly_inference_latency_ms_total || 0;
  const probSum = m.anomaly_inference_anomaly_probability_sum || 0;
  document.querySelector("#metric-requests").textContent = requests;
  document.querySelector("#metric-alerts").textContent = alerts;
  document.querySelector("#metric-alert-rate").textContent = `告警率 ${fmt(rows ? alerts / rows * 100 : 0, 1)}%`;
  document.querySelector("#metric-latency").textContent = `${fmt(requests ? latencyTotal / requests : 0, 3)} ms`;
  document.querySelector("#metric-latency-max").textContent = `最大 ${fmt(m.anomaly_inference_latency_ms_max || 0, 3)} ms`;
  document.querySelector("#metric-memory").textContent = `${fmt((m.anomaly_inference_process_resident_memory_bytes || 0) / 1048576, 1)} MiB`;
  document.querySelector("#metric-cpu").textContent = `CPU ${fmt(m.anomaly_inference_process_cpu_seconds_total || 0, 3)}s`;
  document.querySelector("#metric-rows").textContent = rows;
  document.querySelector("#metric-prob-avg").textContent = fmt(rows ? probSum / rows : 0, 4);
  document.querySelector("#metric-prob-range").textContent = `${fmt(m.anomaly_inference_anomaly_probability_min || 0, 4)} - ${fmt(m.anomaly_inference_anomaly_probability_max || 0, 4)}`;
  document.querySelector("#metric-uptime").textContent = durationText(m.anomaly_inference_process_uptime_seconds || 0);
  document.querySelector("#metric-updated").textContent = payload.fetched_at || "-";

  const bucketLabels = ["0.1", "0.3", "0.5", "0.7", "0.9", "1.0", "+Inf"];
  const bucketData = {};
  let previous = 0;
  for (const label of bucketLabels) {
    const cumulative = (payload.buckets || {})[label] || 0;
    bucketData[`<= ${label}`] = Math.max(0, cumulative - previous);
    previous = cumulative;
  }
  renderBars("#metric-classes", payload.classes || {});
  renderBars("#metric-probability", bucketData, true);
}

function renderCompactList(selector, rows, emptyText, renderRow) {
  const container = document.querySelector(selector);
  if (!container) return;
  if (!rows.length) {
    container.innerHTML = `<div class="muted small">${escapeHtml(emptyText)}</div>`;
    return;
  }
  container.innerHTML = rows.map(renderRow).join("");
}

function renderMlopsPipeline(payload) {
  const stage = document.querySelector("#mlops-stage");
  if (!stage) return;
  stage.textContent = payload.stage || "-";
  stage.className = payload.stage === "Production" ? "badge badge-ok" : "badge badge-warn";
  document.querySelector("#mlops-version").textContent = payload.version || "-";
  document.querySelector("#mlops-updated").textContent = payload.generated_at || "-";
  const latestRetrain = payload.latest_retrain || {};
  const runStatus = document.querySelector("#mlops-run-status");
  const runTime = document.querySelector("#mlops-run-time");
  if (runStatus) {
    const actions = latestRetrain.actions?.length ? ` · ${latestRetrain.actions.length} 步` : "";
    runStatus.textContent = latestRetrain.status ? `${latestRetrain.status}${actions}` : "-";
    runStatus.className = latestRetrain.status === "success" ? "status-line status-ok" : "status-line status-warn";
  }
  if (runTime) runTime.textContent = latestRetrain.generated_at || "-";

  const steps = document.querySelector("#mlops-steps");
  if (steps) {
    steps.innerHTML = (payload.steps || []).map((step) => {
      const ok = step.status === "ok";
      return `<div class="pipeline-step ${ok ? "step-ok" : "step-missing"}">
        <div class="step-dot"></div>
        <h3>${escapeHtml(step.title)}</h3>
        <p class="muted small">${escapeHtml(step.detail)}</p>
        <div class="evidence">${escapeHtml(step.evidence)}</div>
      </div>`;
    }).join("");
  }

  renderCompactList("#mlops-models", payload.models || [], "暂无模型版本", (model) =>
    `<div class="compact-row"><b>${escapeHtml(model.name)}</b><span>${escapeHtml(model.target)} · Acc ${escapeHtml(model.accuracy)} · F1 ${escapeHtml(model.macro_f1)}</span></div>`
  );
  renderCompactList("#mlops-datasets", payload.datasets || [], "暂无数据版本", (dataset) =>
    `<div class="compact-row"><b>${escapeHtml(dataset.name)}</b><span>${escapeHtml(dataset.rows)} 行 · ${escapeHtml(dataset.features)} 特征</span></div>`
  );
  renderCompactList("#mlops-mlflow", payload.mlflow_models || [], "暂无发布记录", (model) =>
    `<div class="compact-row"><b>${escapeHtml(model.name)}</b><span>v${escapeHtml(model.version)} · ${escapeHtml(model.alias)}</span></div>`
  );
}

async function refreshInferenceMetrics() {
  try {
    const response = await fetch("/api/inference_metrics", {cache: "no-store"});
    if (!response.ok) return;
    renderInferenceMetrics(await response.json());
  } catch (error) {
    renderInferenceMetrics({ok: false, error: String(error)});
  }
}

async function refreshMlopsPipeline() {
  try {
    const response = await fetch("/api/mlops_pipeline", {cache: "no-store"});
    if (!response.ok) return;
    renderMlopsPipeline(await response.json());
  } catch (_) {
    // The next refresh will retry.
  }
}

refreshDashboard();
refreshInferenceMetrics();
refreshMlopsPipeline();
setInterval(refreshDashboard, 2000);
setInterval(refreshInferenceMetrics, 1000);
setInterval(refreshMlopsPipeline, 3000);

document.querySelectorAll("[data-algorithm-select]").forEach((algorithmSelect) => {
  const form = algorithmSelect.closest("form");
  const modelTypeSelect = form?.querySelector("[data-model-type-select]");
  const thresholdInput = form?.querySelector("[data-threshold-input]");
  if (!modelTypeSelect) return;
  const syncModelType = () => {
    const isolationForest = algorithmSelect.value === "isolation_forest";
    const multiclass = [...modelTypeSelect.options].find((option) => option.value === "multiclass");
    if (multiclass) multiclass.disabled = isolationForest;
    if (isolationForest) {
      modelTypeSelect.value = "binary";
      if (thresholdInput) thresholdInput.value = "0.5";
    }
  };
  algorithmSelect.addEventListener("change", syncModelType);
  syncModelType();
});

document.querySelectorAll("[data-experiment-algorithm]").forEach((algorithmSelect) => {
  const modelType = algorithmSelect.closest("form")?.querySelector("[data-experiment-model-type]");
  if (!modelType) return;
  const syncExperimentMode = () => {
    modelType.value = algorithmSelect.value === "isolation_forest" ? "binary" : "multiclass";
  };
  algorithmSelect.addEventListener("change", syncExperimentMode);
  syncExperimentMode();
});
