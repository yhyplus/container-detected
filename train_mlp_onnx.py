#!/usr/bin/env python3
"""Train a PyTorch MLP and export it to ONNX for runtime comparison."""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from train_model import MULTICLASS_NAMES, feature_names, labels, read_rows


class MLP(nn.Module):
    def __init__(self, input_size, class_count):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, class_count),
        )

    def forward(self, inputs):
        return self.layers(inputs)


def matrix(rows, features):
    return np.asarray([[float(row[name]) for name in features] for row in rows], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--eval-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--target", choices=["class_id", "label"], default="class_id")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "models" / "anomaly_mlp.onnx")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train_rows = read_rows(args.run_dir)
    features = feature_names(train_rows)
    scaler = StandardScaler()
    train_x = scaler.fit_transform(matrix(train_rows, features)).astype(np.float32)
    train_y = np.asarray(labels(train_rows, args.target), dtype=np.int64)
    classes = sorted(set(train_y.tolist()))
    if classes != list(range(len(classes))):
        raise SystemExit(f"target classes must be contiguous from zero: {classes}")

    model = MLP(len(features), len(classes))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=args.batch_size, shuffle=True,
    )
    model.train()
    for _ in range(args.epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    report = None
    if args.eval_run_dir:
        eval_rows = read_rows(args.eval_run_dir)
        eval_x = scaler.transform(matrix(eval_rows, features)).astype(np.float32)
        expected = labels(eval_rows, args.target)
        with torch.no_grad():
            predictions = model(torch.from_numpy(eval_x)).argmax(dim=1).numpy()
        report = classification_report(expected, predictions, output_dict=True, zero_division=0)
        print(f"held_out_accuracy={accuracy_score(expected, predictions):.4f}")
        print(classification_report(expected, predictions, zero_division=0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output.with_suffix(".pt"))
    joblib.dump(scaler, args.output.with_suffix(".scaler.joblib"))
    dummy = torch.zeros((1, len(features)), dtype=torch.float32)
    try:
        torch.onnx.export(
            model, dummy, args.output,
            input_names=["features"], output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit("ONNX export requires the onnx and onnxscript packages") from exc
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": args.target,
        "features": features,
        "classes": classes,
        "class_names": (
            MULTICLASS_NAMES if args.target == "class_id" else {0: "normal", 1: "anomaly"}
        ),
        "training_runs": [str(path) for path in args.run_dir],
        "evaluation_runs": [str(path) for path in args.eval_run_dir],
        "training_rows": len(train_rows),
        "training_classes": dict(sorted(Counter(train_y.tolist()).items())),
        "evaluation_report": report,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"onnx_model={args.output}")


if __name__ == "__main__":
    main()
