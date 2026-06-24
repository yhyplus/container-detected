#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

NEW_DATA_ROWS="${NEW_DATA_ROWS:-0}"
MIN_NEW_ROWS="${MIN_NEW_ROWS:-300}"
MIN_ACCURACY="${MIN_ACCURACY:-0.95}"
MAX_DRIFT_SCORE="${MAX_DRIFT_SCORE:-0.25}"

python3 monitor_drift.py
python3 mlops_monitor_snapshot.py || true
python3 mlops_auto_retrain.py \
  --new-data-rows "$NEW_DATA_ROWS" \
  --min-new-rows "$MIN_NEW_ROWS" \
  --min-accuracy "$MIN_ACCURACY" \
  --max-drift-score "$MAX_DRIFT_SCORE" \
  --build-image \
  --deploy \
  --promote \
  --rollback-on-failure
