#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUBECTL="${KUBECTL:-kubectl --kubeconfig /home/idolsingerydd/.kube/config}"
SCENARIO="${SCENARIO:?SCENARIO is required}"
DURATION="${DURATION:-60}"
WORKERS="${WORKERS:-8}"
MEMORY_MIB="${MEMORY_MIB:-128}"
CONNECTION_DELAY="${CONNECTION_DELAY:-0.02}"
PORT_SCAN_MAX="${PORT_SCAN_MAX:-1000}"
PORT_SCAN_DELAY="${PORT_SCAN_DELAY:-0.003}"
SETTLE_SECONDS="${SETTLE_SECONDS:-8}"
PIXIE_EXPORT_BUFFER_SECONDS="${PIXIE_EXPORT_BUFFER_SECONDS:-120}"
ONLINE_INFERENCE_INTERVAL="${ONLINE_INFERENCE_INTERVAL:-15}"
NAMESPACE="anomaly-test"

case "$SCENARIO" in
  cpu|http_flood|network_connections|dns_flood|portscan|lateral_movement|syscall_anomaly) ;;
  *) echo "Unsupported experiment scenario: $SCENARIO" >&2; exit 1 ;;
esac

$KUBECTL -n "$NAMESPACE" create configmap anomaly-simulator \
  --from-file=simulate.py="$ROOT/simulate.py" \
  --dry-run=client -o yaml | $KUBECTL apply -f -
$KUBECTL apply -f "$ROOT/k8s/anomaly-generator.yaml"
$KUBECTL -n "$NAMESPACE" rollout restart deployment/anomaly-generator
$KUBECTL -n "$NAMESPACE" rollout status deployment/anomaly-generator --timeout=180s

TARGET_IP="$($KUBECTL -n "$NAMESPACE" get service nginx-test -o jsonpath='{.spec.clusterIP}')"
TARGET_POD_IP="$($KUBECTL -n "$NAMESPACE" get pod -l app=nginx-test -o jsonpath='{.items[0].status.podIP}')"
TARGET_HOST="$TARGET_IP"
if [[ "$SCENARIO" == portscan ]]; then
  TARGET_HOST="$TARGET_POD_IP"
fi
$KUBECTL -n "$NAMESPACE" exec deployment/anomaly-generator -- \
  rm -f /tmp/anomaly-sensor-events.csv
echo "Simulating scenario: $SCENARIO for ${DURATION}s"
PREDICTION_START_TIME="$(date --iso-8601=seconds)"
mkdir -p "$OUT/raw" "$OUT/realtime"
PUBLISH_STATE_FILE="$OUT/.published_online_windows"
$KUBECTL -n "$NAMESPACE" exec deployment/anomaly-generator -- \
  python /app/simulate.py --mode "$SCENARIO" --duration "$DURATION" \
    --workers "$WORKERS" --memory-mib "$MEMORY_MIB" \
    --connection-delay "$CONNECTION_DELAY" \
    --port-scan-max "$PORT_SCAN_MAX" --port-scan-delay "$PORT_SCAN_DELAY" \
    --scan-hosts "$TARGET_POD_IP,$TARGET_IP" \
    --url "http://$TARGET_IP/" --host "$TARGET_HOST" &
SIM_PID=$!

TICK=0
while kill -0 "$SIM_PID" >/dev/null 2>&1; do
  sleep "$ONLINE_INFERENCE_INTERVAL"
  if ! kill -0 "$SIM_PID" >/dev/null 2>&1; then
    break
  fi
  TICK=$((TICK + 1))
  NOW_TIME="$(date --iso-8601=seconds)"
  echo "Realtime inference tick $TICK at $NOW_TIME"
  if "$ROOT/collect_pixie.sh" "$OUT/raw"; then
    "$ROOT/collect_sensor.sh" "$OUT/raw/sensor_events.csv" || true
    GENERATOR_IP="$($KUBECTL -n "$NAMESPACE" get pod -l app=anomaly-generator -o jsonpath='{.items[0].status.podIP}')"
    if python3 "$ROOT/build_live_windows.py" \
      --raw-dir "$OUT/raw" \
      --output "$OUT/realtime/windows-$TICK.csv" \
      --lookback-seconds "$((DURATION + PIXIE_EXPORT_BUFFER_SECONDS))" \
      --generator-ip "$GENERATOR_IP" \
      --start-time "$PREDICTION_START_TIME" \
      --end-time "$NOW_TIME"; then
      python3 "$ROOT/publish_inference_metrics.py" \
        --input "$OUT/realtime/windows-$TICK.csv" \
        --state-file "$PUBLISH_STATE_FILE" || true
    else
      echo "Realtime inference tick $TICK produced no windows yet"
    fi
  else
    echo "Realtime Pixie collection failed at tick $TICK; final report will retry"
  fi
done
wait "$SIM_PID"
PREDICTION_END_TIME="$(date --iso-8601=seconds)"

echo "Waiting ${SETTLE_SECONDS}s for Pixie telemetry"
sleep "$SETTLE_SECONDS"
"$ROOT/collect_sensor.sh" "$OUT/raw/sensor_events.csv"
export PREDICTION_START_TIME PREDICTION_END_TIME PUBLISH_STATE_FILE
LOOKBACK_SECONDS="$((DURATION + SETTLE_SECONDS + PIXIE_EXPORT_BUFFER_SECONDS))" "$ROOT/predict_live.sh"
