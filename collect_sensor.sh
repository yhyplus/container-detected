#!/usr/bin/env bash
set -euo pipefail

KUBECTL="${KUBECTL:-kubectl --kubeconfig /home/idolsingerydd/.kube/config}"
OUT="${1:?usage: collect_sensor.sh OUTPUT_FILE}"
NAMESPACE="anomaly-test"

mkdir -p "$(dirname "$OUT")"
if ! $KUBECTL -n "$NAMESPACE" exec deployment/anomaly-generator -- \
  cat /tmp/anomaly-sensor-events.csv > "$OUT"; then
  printf 'time_,pod,event_type,dst_ip,dst_port,detail\n' > "$OUT"
fi
