#!/usr/bin/env python3
"""Send live feature windows to the deployed inference service.

The local experiment pipeline still writes predictions.csv for reports. This
script additionally calls the K3s inference service so the online monitoring
dashboard reflects the traffic that was just simulated.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path


DEFAULT_URLS = [
    "http://127.0.0.1:18080/predict",
    "http://127.0.0.1:8080/predict",
]


def post_json(url, payload, timeout):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--url", default=os.environ.get("INFERENCE_PREDICT_URL", ""))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--state-file", type=Path, default=None)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("online_inference_publish=skipped reason=no_rows")
        return
    published = set()
    if args.state_file and args.state_file.exists():
        published = {
            line.strip() for line in args.state_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    if args.state_file:
        filtered = []
        for row in rows:
            key = f"{row.get('window_start', '')}|{row.get('pod', '')}"
            if key not in published:
                filtered.append(row)
        rows = filtered
        if not rows:
            print("online_inference_publish=skipped reason=no_new_windows")
            return

    urls = [args.url] if args.url else DEFAULT_URLS
    errors = []
    for url in urls:
        try:
            sent = 0
            responses = []
            for index in range(0, len(rows), args.batch_size):
                batch = rows[index:index + args.batch_size]
                status, payload = post_json(url, {"rows": batch}, args.timeout)
                responses.append({"status": status, "batch_size": len(batch)})
                sent += len(batch)
            if args.state_file:
                args.state_file.parent.mkdir(parents=True, exist_ok=True)
                with args.state_file.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(f"{row.get('window_start', '')}|{row.get('pod', '')}\n")
            print(f"online_inference_publish=ok url={url} rows={sent} responses={responses}")
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            errors.append(f"{url}: {exc}")
    try:
        payload = json.dumps({"rows": rows})
        command = [
            "kubectl", "--kubeconfig", "/home/idolsingerydd/.kube/config",
            "-n", "anomaly-test", "exec", "-i", "deployment/anomaly-generator", "--",
            "python", "-c",
            (
                "import json,sys,urllib.request;"
                "data=sys.stdin.read().encode();"
                "req=urllib.request.Request('http://anomaly-inference:8080/predict',"
                "data=data,headers={'Content-Type':'application/json'});"
                "print(urllib.request.urlopen(req, timeout=10).read().decode())"
            ),
        ]
        result = subprocess.run(
            command, input=payload, text=True, capture_output=True, timeout=20, check=False
        )
        if result.returncode == 0:
            if args.state_file:
                args.state_file.parent.mkdir(parents=True, exist_ok=True)
                with args.state_file.open("a", encoding="utf-8") as handle:
                    for row in rows:
                        handle.write(f"{row.get('window_start', '')}|{row.get('pod', '')}\n")
            print(f"online_inference_publish=ok url=k8s://anomaly-inference rows={len(rows)}")
            print(result.stdout.strip())
            return
        errors.append(result.stderr.strip() or result.stdout.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"kubectl publish fallback: {exc}")
    print("online_inference_publish=failed " + " | ".join(errors))
    sys.exit(0)


if __name__ == "__main__":
    main()
