#!/usr/bin/env python3
"""Manage lightweight model registry stages for this project."""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REGISTRY_PATH = ROOT / "registry" / "model_registry.json"
HISTORY_PATH = ROOT / "registry" / "model_registry_history.json"
STAGES = {"Candidate", "Production", "Archived", "Rejected"}


def load_json(path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_history(event):
    history = load_json(HISTORY_PATH, {"events": []})
    history["events"].append(event)
    write_json(HISTORY_PATH, history)


def registry_summary(registry):
    return {
        "version": registry.get("version"),
        "stage": registry.get("stage", "Candidate"),
        "production_version": registry.get("production_version"),
        "previous_production_version": registry.get("previous_production_version"),
        "models": [
            {
                "name": item.get("name"),
                "accuracy": item.get("accuracy"),
                "macro_f1": item.get("macro_f1"),
            }
            for item in registry.get("models", [])
        ],
    }


def set_stage(stage, note):
    if stage not in STAGES:
        raise SystemExit(f"unsupported stage: {stage}")
    registry = load_json(REGISTRY_PATH, {})
    if not registry:
        raise SystemExit(f"missing registry: {REGISTRY_PATH}")
    old_stage = registry.get("stage", "Candidate")
    event = {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": "set_stage",
        "version": registry.get("version"),
        "from_stage": old_stage,
        "to_stage": stage,
        "note": note,
    }
    registry["stage"] = stage
    registry["stage_updated_at"] = event["time"]
    if note:
        registry["stage_note"] = note
    write_json(REGISTRY_PATH, registry)
    append_history(event)
    print(json.dumps(registry_summary(registry), indent=2, sort_keys=True))


def promote(note, publish_mlflow=False):
    registry = load_json(REGISTRY_PATH, {})
    if not registry:
        raise SystemExit(f"missing registry: {REGISTRY_PATH}")
    version = registry.get("version")
    current_production = registry.get("production_version") or registry.get("previous_production_version")
    production_dir = ROOT / "registry" / "production"
    version_dir = ROOT / "registry" / "models" / version
    if not version_dir.exists():
        raise SystemExit(f"missing model snapshot: {version_dir}")
    if production_dir.exists():
        archive_root = ROOT / "registry" / "archived"
        archive_root.mkdir(parents=True, exist_ok=True)
        archived_dir = archive_root / (current_production or datetime.now(timezone.utc).strftime("unknown-%Y%m%d-%H%M%S"))
        if archived_dir.exists():
            shutil.rmtree(archived_dir)
        shutil.copytree(production_dir, archived_dir)
    if production_dir.exists():
        shutil.rmtree(production_dir)
    shutil.copytree(version_dir, production_dir)
    event_time = datetime.now(timezone.utc).isoformat()
    registry.update({
        "stage": "Production",
        "stage_updated_at": event_time,
        "production_version": version,
        "previous_production_version": current_production,
        "production_path": str(production_dir.relative_to(ROOT)),
    })
    if note:
        registry["stage_note"] = note
    write_json(REGISTRY_PATH, registry)
    write_json(production_dir / "model_registry.json", registry)
    data_registry = load_json(ROOT / "registry" / "data_registry.json", {})
    if data_registry:
        write_json(production_dir / "data_registry.json", data_registry)
    append_history({
        "time": event_time,
        "event": "promote",
        "version": version,
        "previous_production_version": current_production,
        "note": note,
    })
    if publish_mlflow:
        subprocess.run([sys.executable, "mlops_publish_mlflow.py"], cwd=ROOT, check=True)
    print(json.dumps(registry_summary(registry), indent=2, sort_keys=True))


def rollback(note):
    registry = load_json(REGISTRY_PATH, {})
    previous = registry.get("previous_production_version")
    if not previous:
        raise SystemExit("no previous production version recorded")
    previous_dir = ROOT / "registry" / "archived" / previous
    production_dir = ROOT / "registry" / "production"
    if not previous_dir.exists():
        raise SystemExit(f"missing archived production snapshot: {previous_dir}")
    if production_dir.exists():
        shutil.rmtree(production_dir)
    shutil.copytree(previous_dir, production_dir)
    event_time = datetime.now(timezone.utc).isoformat()
    registry.update({
        "stage": "Production",
        "stage_updated_at": event_time,
        "production_version": previous,
        "previous_production_version": registry.get("production_version"),
        "production_path": str(production_dir.relative_to(ROOT)),
    })
    if note:
        registry["stage_note"] = note
    write_json(REGISTRY_PATH, registry)
    append_history({
        "time": event_time,
        "event": "rollback",
        "version": previous,
        "note": note,
    })
    print(json.dumps(registry_summary(registry), indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("status")
    set_parser = sub.add_parser("set-stage")
    set_parser.add_argument("stage", choices=sorted(STAGES))
    set_parser.add_argument("--note", default="")
    promote_parser = sub.add_parser("promote")
    promote_parser.add_argument("--note", default="")
    promote_parser.add_argument("--publish-mlflow", action="store_true")
    rollback_parser = sub.add_parser("rollback")
    rollback_parser.add_argument("--note", default="")
    args = parser.parse_args()

    if args.command == "status":
        print(json.dumps(registry_summary(load_json(REGISTRY_PATH, {})), indent=2, sort_keys=True))
    elif args.command == "set-stage":
        set_stage(args.stage, args.note)
    elif args.command == "promote":
        promote(args.note, publish_mlflow=args.publish_mlflow)
    elif args.command == "rollback":
        rollback(args.note)


if __name__ == "__main__":
    main()
