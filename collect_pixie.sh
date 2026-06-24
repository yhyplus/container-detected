#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PX="${PX:-/home/idolsingerydd/bin/px}"
export PX_CLOUD_ADDR="${PX_CLOUD_ADDR:-getcosmic.ai}"
OUT="${1:?usage: collect_pixie.sh OUTPUT_DIR}"
mkdir -p "$OUT"

if ! "$PX" script list -o table >/dev/null 2>&1 && [[ -z "${PX_DIRECT_VIZIER_ADDR:-}" ]]; then
  echo "Pixie CLI is not authenticated. Run: $PX auth login" >&2
  exit 1
fi

for script in "$ROOT"/pxl/*.pxl; do
  name="$(basename "$script" .pxl)"
  tmp="$OUT/$name.csv.tmp"
  for attempt in 1 2 3; do
    if "$PX" run -f "$script" -o csv > "$tmp"; then
      mv "$tmp" "$OUT/$name.csv"
      break
    fi
    if [[ "$attempt" == 3 ]]; then
      rm -f "$tmp"
      echo "Pixie export failed after $attempt attempts: $name" >&2
      exit 1
    fi
    echo "Retrying Pixie export ($attempt/3): $name" >&2
    sleep 8
  done
done
