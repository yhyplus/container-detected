#!/usr/bin/env python3
"""Persist a point-in-time snapshot of online inference metrics."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import ui_app


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("reports/monitoring_history.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("reports/monitoring_latest.json"))
    args = parser.parse_args()

    payload = ui_app.load_inference_metrics()
    snapshot = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "ok": payload.get("ok", False),
        "source": payload.get("source"),
        "values": payload.get("values", {}),
        "classes": payload.get("classes", {}),
        "buckets": payload.get("buckets", {}),
        "error": payload.get("error"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(snapshot, sort_keys=True) + "\n")
    args.summary.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"monitoring_snapshot={args.summary}")
    print(f"ok={snapshot['ok']} source={snapshot.get('source')}")


if __name__ == "__main__":
    main()
