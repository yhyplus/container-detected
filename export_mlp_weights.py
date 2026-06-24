#!/usr/bin/env python3
"""Export the trained PyTorch MLP and scaler to a pure-Python JSON artifact."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import torch


def tensor_to_list(state, name):
    return state[name].detach().cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/anomaly_mlp.pt"))
    parser.add_argument("--metadata", type=Path, default=Path("models/anomaly_mlp.json"))
    parser.add_argument("--scaler", type=Path, default=Path("models/anomaly_mlp.scaler.joblib"))
    parser.add_argument("--output", type=Path, default=Path("models/anomaly_mlp_weights.json"))
    args = parser.parse_args()

    state = torch.load(args.model, map_location="cpu")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    scaler = joblib.load(args.scaler)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_model": str(args.model),
        "source_metadata": str(args.metadata),
        "target": metadata["target"],
        "features": metadata["features"],
        "classes": metadata["classes"],
        "class_names": metadata.get("class_names", {}),
        "training_rows": metadata.get("training_rows"),
        "training_runs": metadata.get("training_runs", []),
        "evaluation_runs": metadata.get("evaluation_runs", []),
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "layers": [
            {
                "weight": tensor_to_list(state, "layers.0.weight"),
                "bias": tensor_to_list(state, "layers.0.bias"),
                "activation": "relu",
            },
            {
                "weight": tensor_to_list(state, "layers.2.weight"),
                "bias": tensor_to_list(state, "layers.2.bias"),
                "activation": "relu",
            },
            {
                "weight": tensor_to_list(state, "layers.4.weight"),
                "bias": tensor_to_list(state, "layers.4.bias"),
                "activation": "linear",
            },
        ],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"weights={args.output}")


if __name__ == "__main__":
    main()
