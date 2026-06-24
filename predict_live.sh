#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUBECTL="${KUBECTL:-kubectl --kubeconfig /home/idolsingerydd/.kube/config}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="${OUT:-$ROOT/runs/live-$RUN_ID}"
LOOKBACK_SECONDS="${LOOKBACK_SECONDS:-120}"
THRESHOLD="${THRESHOLD:-0.7}"
MODEL="${MODEL:-$ROOT/models/anomaly_binary_detector.joblib}"
ALGORITHM="${ALGORITHM:-random_forest}"
PUBLISH_STATE_FILE="${PUBLISH_STATE_FILE:-$OUT/.published_online_windows}"
NAMESPACE="anomaly-test"

mkdir -p "$OUT"
"$ROOT/collect_pixie.sh" "$OUT/raw"
GENERATOR_IP="$($KUBECTL -n "$NAMESPACE" get pod -l app=anomaly-generator -o jsonpath='{.items[0].status.podIP}')"
WINDOW_ARGS=()
if [[ -n "${PREDICTION_START_TIME:-}" ]]; then
  WINDOW_ARGS+=(--start-time "$PREDICTION_START_TIME")
fi
if [[ -n "${PREDICTION_END_TIME:-}" ]]; then
  WINDOW_ARGS+=(--end-time "$PREDICTION_END_TIME")
fi
python3 "$ROOT/build_live_windows.py" \
  --raw-dir "$OUT/raw" \
  --output "$OUT/windows.csv" \
  --lookback-seconds "$LOOKBACK_SECONDS" \
  --generator-ip "$GENERATOR_IP" \
  "${WINDOW_ARGS[@]}"
case "$ALGORITHM" in
  random_forest) PREDICTOR="$ROOT/predict_model.py" ;;
  onnx_mlp) PREDICTOR="$ROOT/predict_onnx.py" ;;
  isolation_forest) PREDICTOR="$ROOT/predict_isolation_forest.py" ;;
  *) echo "Unsupported prediction algorithm: $ALGORITHM" >&2; exit 1 ;;
esac
python3 "$PREDICTOR" \
    --model "$MODEL" \
    --input "$OUT/windows.csv" \
    --output "$OUT/predictions.csv" \
    --threshold "$THRESHOLD"

python3 "$ROOT/publish_inference_metrics.py" \
    --input "$OUT/windows.csv" \
    --state-file "$PUBLISH_STATE_FILE" || true
