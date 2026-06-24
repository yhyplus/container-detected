#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PF_LOG="${ROOT}/ui-logs/port-forward-anomaly-inference.log"
mkdir -p "${ROOT}/ui-logs"

if ! (echo > /dev/tcp/127.0.0.1/18080) >/dev/null 2>&1; then
  kubectl --kubeconfig /home/idolsingerydd/.kube/config \
    -n anomaly-test port-forward svc/anomaly-inference 18080:8080 \
    > "$PF_LOG" 2>&1 &
  PF_PID=$!
  trap 'kill "$PF_PID" >/dev/null 2>&1 || true' EXIT
  export INFERENCE_METRICS_URL="http://127.0.0.1:18080/metrics"
  export INFERENCE_PREDICT_URL="http://127.0.0.1:18080/predict"
else
  export INFERENCE_METRICS_URL="${INFERENCE_METRICS_URL:-http://127.0.0.1:18080/metrics}"
  export INFERENCE_PREDICT_URL="${INFERENCE_PREDICT_URL:-http://127.0.0.1:18080/predict}"
fi

exec python3 "$ROOT/ui_app.py"
