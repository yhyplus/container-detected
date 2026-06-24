#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KUBECTL="${KUBECTL:-kubectl --kubeconfig /home/idolsingerydd/.kube/config}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUT="${OUT:-$ROOT/runs/$RUN_ID}"
RESUME="${RESUME:-0}"
CONFIG_FILE="$OUT/run.env"
if [[ "$RESUME" == 1 && -e "$CONFIG_FILE" ]]; then
  source "$CONFIG_FILE"
fi
DURATION="${DURATION:-90}"
WORKERS="${WORKERS:-8}"
SETTLE_SECONDS="${SETTLE_SECONDS:-15}"
MEMORY_MIB="${MEMORY_MIB:-128}"
CONNECTION_DELAY="${CONNECTION_DELAY:-0.02}"
PORT_SCAN_MAX="${PORT_SCAN_MAX:-1000}"
PORT_SCAN_DELAY="${PORT_SCAN_DELAY:-0.003}"
NAMESPACE="anomaly-test"
TABLES=(process_stats network_stats conn_stats http_events dns_events sensor_events)

mkdir -p "$OUT"
if [[ -e "$OUT/manifest.csv" ]]; then
  if [[ "$RESUME" != 1 ]]; then
    echo "Run directory already exists: $OUT" >&2
    echo "Set RESUME=1 with the same OUT to continue it." >&2
    exit 1
  fi
else
  printf 'scenario,start_time,end_time,label,generator_ip\n' > "$OUT/manifest.csv"
fi
if [[ ! -e "$CONFIG_FILE" ]]; then
  printf 'DURATION=%q\nWORKERS=%q\nSETTLE_SECONDS=%q\nMEMORY_MIB=%q\nCONNECTION_DELAY=%q\nPORT_SCAN_MAX=%q\nPORT_SCAN_DELAY=%q\n' \
    "$DURATION" "$WORKERS" "$SETTLE_SECONDS" "$MEMORY_MIB" "$CONNECTION_DELAY" "$PORT_SCAN_MAX" "$PORT_SCAN_DELAY" > "$CONFIG_FILE"
fi

$KUBECTL -n "$NAMESPACE" create configmap anomaly-simulator \
  --from-file=simulate.py="$ROOT/simulate.py" \
  --dry-run=client -o yaml | $KUBECTL apply -f -
$KUBECTL apply -f "$ROOT/k8s/anomaly-generator.yaml"
$KUBECTL -n "$NAMESPACE" rollout restart deployment/anomaly-generator
$KUBECTL -n "$NAMESPACE" rollout status deployment/anomaly-generator --timeout=180s
GENERATOR_IP="$($KUBECTL -n "$NAMESPACE" get pod -l app=anomaly-generator -o jsonpath='{.items[0].status.podIP}')"
TARGET_IP="$($KUBECTL -n "$NAMESPACE" get service nginx-test -o jsonpath='{.spec.clusterIP}')"
TARGET_POD_IP="$($KUBECTL -n "$NAMESPACE" get pod -l app=nginx-test -o jsonpath='{.items[0].status.podIP}')"

scenario_complete() {
  local scenario="$1"
  local table
  awk -F, -v scenario="$scenario" 'NR > 1 && $1 == scenario { found = 1 } END { exit !found }' \
    "$OUT/manifest.csv" || return 1
  for table in "${TABLES[@]}"; do
    [[ -e "$OUT/raw/$scenario/$table.csv" ]] || return 1
  done
}

run_one() {
  local scenario="$1"
  local label="$2"
  local start end host
  if scenario_complete "$scenario"; then
    echo "Skipping completed scenario: $scenario"
    return
  fi
  echo "Running scenario: $scenario"
  host="$TARGET_IP"
  if [[ "$scenario" == portscan ]]; then
    host="$TARGET_POD_IP"
  fi
  $KUBECTL -n "$NAMESPACE" exec deployment/anomaly-generator -- \
    rm -f /tmp/anomaly-sensor-events.csv
  start="$(date --iso-8601=seconds)"
  $KUBECTL -n "$NAMESPACE" exec deployment/anomaly-generator -- \
    python /app/simulate.py --mode "$scenario" --duration "$DURATION" \
      --workers "$WORKERS" --memory-mib "$MEMORY_MIB" \
      --connection-delay "$CONNECTION_DELAY" \
      --port-scan-max "$PORT_SCAN_MAX" --port-scan-delay "$PORT_SCAN_DELAY" \
      --scan-hosts "$TARGET_POD_IP,$TARGET_IP" \
      --url "http://$TARGET_IP/" --host "$host"
  end="$(date --iso-8601=seconds)"
  sleep "$SETTLE_SECONDS"
  "$ROOT/collect_pixie.sh" "$OUT/raw/$scenario"
  "$ROOT/collect_sensor.sh" "$OUT/raw/$scenario/sensor_events.csv"
  printf '%s,%s,%s,%s,%s\n' "$scenario" "$start" "$end" "$label" "$GENERATOR_IP" >> "$OUT/manifest.csv"
}

run_one normal 0
run_one http_flood 1
run_one network_connections 1
run_one dns_flood 1
run_one portscan 1
run_one lateral_movement 1
run_one cpu 1
run_one syscall_anomaly 1

python3 "$ROOT/build_dataset.py" --run-dir "$OUT"
python3 "$ROOT/validate_dataset.py" "$OUT/training_windows.csv"
python3 "$ROOT/validate_dataset.py" "$OUT/training_multiclass_windows.csv"
echo "Dataset written to $OUT/training_windows.csv"
echo "Multiclass dataset written to $OUT/training_multiclass_windows.csv"
