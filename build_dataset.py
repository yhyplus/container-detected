#!/usr/bin/env python3
"""Aggregate Pixie CSV exports into labeled pod-level training windows."""

import argparse
import csv
import math
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path


WINDOW_SECONDS = 10
CLASS_IDS = {
    "normal": 0,
    "http_flood": 1,
    "network_connections": 1,
    "dns_flood": 1,
    "portscan": 2,
    "lateral_movement": 3,
    "cpu": 4,
    "syscall_anomaly": 5,
}
ATTACK_TYPES = {
    "normal": "normal",
    "http_flood": "ddos",
    "network_connections": "ddos",
    "dns_flood": "ddos",
    "portscan": "portscan",
    "lateral_movement": "lateral_movement",
    "cpu": "resource_anomaly",
    "syscall_anomaly": "syscall_anomaly",
}
TARGET_PODS = {
    "normal": {"anomaly-test/anomaly-generator", "anomaly-test/nginx-test"},
    "http_flood": {"anomaly-test/nginx-test"},
    "network_connections": {"anomaly-test/anomaly-generator"},
    "dns_flood": {"anomaly-test/anomaly-generator"},
    "portscan": {"anomaly-test/anomaly-generator"},
    "lateral_movement": {"anomaly-test/anomaly-generator"},
    "cpu": {"anomaly-test/anomaly-generator"},
    "syscall_anomaly": {"anomaly-test/anomaly-generator"},
}
FIELDS = [
    "scenario", "attack_type", "attack_subtype", "class_id", "label", "is_target",
    "window_coverage_ratio", "window_start", "pod", "cpu_pct", "rss_mib",
    "disk_read_kib_sec", "disk_write_kib_sec", "rx_kib_sec", "tx_kib_sec",
    "rx_packets_sec", "tx_packets_sec",
    "connection_open_count", "connection_active_max", "remote_endpoint_count",
    "unique_dst_ip_count", "unique_dst_port_count",
    "connection_bytes_sent", "connection_bytes_recv", "http_request_count",
    "http_rps", "http_error_rate", "http_resp_kib", "http_latency_avg_ms",
    "http_latency_p95_ms", "dns_request_count", "dns_failure_rate",
    "sensor_event_count", "process_spawn_count", "sensitive_file_access_count",
]
FEATURE_FIELDS = FIELDS[9:]


def parse_time(value):
    match = re.match(
        r"(\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d)(?:\.(\d+))?"
        r"(?:\s*(Z|[+-]\d\d:?\d\d))?",
        value,
    )
    if not match:
        raise ValueError(f"unsupported timestamp: {value!r}")
    fraction = (match.group(3) or "")[:6].ljust(6, "0")
    offset = (match.group(4) or "").replace("Z", "+00:00")
    if offset and ":" not in offset:
        offset = f"{offset[:3]}:{offset[3:]}"
    stamp = datetime.fromisoformat(f"{match.group(1)}T{match.group(2)}.{fraction}{offset}")
    return stamp.astimezone().replace(tzinfo=None) if stamp.tzinfo else stamp


def number(value, kind=None):
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except ValueError:
        pass
    match = re.match(r"\s*(-?[\d.]+)\s*([A-Za-z%µ/]+)", value)
    if not match:
        return 0.0
    amount, unit = float(match.group(1)), match.group(2)
    if kind == "bytes":
        return amount * {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}.get(unit, 1)
    if kind == "latency":
        return amount * {"ns": 1e-6, "µs": 1e-3, "us": 1e-3, "ms": 1, "s": 1000}.get(unit, 1)
    if kind == "duration_ns":
        return amount * {"ns": 1, "µs": 1e3, "us": 1e3, "ms": 1e6, "s": 1e9}.get(unit, 1)
    return amount


def window_start(value):
    stamp = parse_time(value)
    return stamp.replace(second=stamp.second // WINDOW_SECONDS * WINDOW_SECONDS, microsecond=0)


def normalize_pod(pod):
    if pod.startswith("anomaly-test/anomaly-generator"):
        return "anomaly-test/anomaly-generator"
    if pod.startswith("anomaly-test/nginx-test"):
        return "anomaly-test/nginx-test"
    return pod


def read_rows(path, start, end):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        if "time_" not in row:
            continue
        stamp = parse_time(row["time_"])
        if start <= stamp <= end:
            row["_window"] = window_start(row["time_"])
            if "pod" in row:
                row["pod"] = normalize_pod(row["pod"])
            result.append(row)
    return result


def delta(rows, field):
    kind = "bytes" if "bytes" in field else "duration_ns" if field.endswith("_ns") else None
    values = [number(row.get(field), kind) for row in rows]
    return max(values) - min(values) if values else 0.0


def percentile(values, fraction):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def grouped(rows, keys):
    result = defaultdict(list)
    for row in rows:
        result[tuple(row.get(key, "") for key in keys)].append(row)
    return result


def base_features(scenario, label, stamp, pod):
    row = {field: 0 for field in FIELDS}
    row.update({
        "scenario": scenario,
        "attack_type": ATTACK_TYPES[scenario],
        "attack_subtype": scenario,
        "class_id": CLASS_IDS[scenario],
        "label": int(label),
        "is_target": int(pod in TARGET_PODS[scenario]),
        "window_start": stamp.isoformat(),
        "pod": pod,
    })
    return row


def aggregate_scenario(raw_dir, scenario, label, start, end, generator_ip=""):
    tables = {
        name: read_rows(raw_dir / f"{name}.csv", start, end)
        for name in [
            "process_stats", "network_stats", "conn_stats", "http_events", "dns_events",
            "sensor_events",
        ]
    }
    pods = {
        row.get("pod", "") for rows in tables.values() for row in rows if row.get("pod")
    }
    rows = {}
    stamp = start.replace(second=start.second // WINDOW_SECONDS * WINDOW_SECONDS, microsecond=0)
    while stamp <= end:
        for pod in pods:
            rows[(stamp, pod)] = base_features(scenario, label, stamp, pod)
        stamp += timedelta(seconds=WINDOW_SECONDS)

    process_groups = grouped(tables["process_stats"], ["_window", "pod", "upid"])
    for (stamp, pod, _), samples in process_groups.items():
        out = rows.setdefault((stamp, pod), base_features(scenario, label, stamp, pod))
        out["cpu_pct"] += 100 * (delta(samples, "cpu_utime_ns") + delta(samples, "cpu_ktime_ns")) / 1e9 / WINDOW_SECONDS
        out["rss_mib"] += sum(number(row.get("rss_bytes"), "bytes") for row in samples) / len(samples) / 1024 ** 2
        out["disk_read_kib_sec"] += delta(samples, "read_bytes") / 1024 / WINDOW_SECONDS
        out["disk_write_kib_sec"] += delta(samples, "write_bytes") / 1024 / WINDOW_SECONDS

    for (stamp, pod), samples in grouped(tables["network_stats"], ["_window", "pod"]).items():
        out = rows.setdefault((stamp, pod), base_features(scenario, label, stamp, pod))
        out["rx_kib_sec"] = delta(samples, "rx_bytes") / 1024 / WINDOW_SECONDS
        out["tx_kib_sec"] = delta(samples, "tx_bytes") / 1024 / WINDOW_SECONDS
        out["rx_packets_sec"] = delta(samples, "rx_packets") / WINDOW_SECONDS
        out["tx_packets_sec"] = delta(samples, "tx_packets") / WINDOW_SECONDS

    conn_groups = grouped(tables["conn_stats"], ["_window", "pod"])
    for (stamp, pod), samples in conn_groups.items():
        out = rows.setdefault((stamp, pod), base_features(scenario, label, stamp, pod))
        out["connection_open_count"] = sum(delta(group, "conn_open") for group in grouped(samples, ["upid", "remote_addr", "remote_port"]).values())
        out["connection_active_max"] = max(number(row.get("conn_active")) for row in samples)
        out["remote_endpoint_count"] = len({(row.get("remote_addr"), row.get("remote_port")) for row in samples})
        out["unique_dst_ip_count"] = len({row.get("remote_addr") for row in samples if row.get("remote_addr")})
        out["unique_dst_port_count"] = len({row.get("remote_port") for row in samples if row.get("remote_port")})
        out["connection_bytes_sent"] = sum(delta(group, "bytes_sent") for group in grouped(samples, ["upid", "remote_addr", "remote_port"]).values())
        out["connection_bytes_recv"] = sum(delta(group, "bytes_recv") for group in grouped(samples, ["upid", "remote_addr", "remote_port"]).values())

    for (stamp, pod), samples in grouped(tables["sensor_events"], ["_window", "pod"]).items():
        out = rows.setdefault((stamp, pod), base_features(scenario, label, stamp, pod))
        destinations = {
            (row.get("dst_ip"), row.get("dst_port"))
            for row in samples if row.get("dst_ip") or row.get("dst_port")
        }
        out["sensor_event_count"] = len(samples)
        out["remote_endpoint_count"] = max(out["remote_endpoint_count"], len(destinations))
        out["unique_dst_ip_count"] = max(
            out["unique_dst_ip_count"],
            len({row.get("dst_ip") for row in samples if row.get("dst_ip")}),
        )
        out["unique_dst_port_count"] = max(
            out["unique_dst_port_count"],
            len({row.get("dst_port") for row in samples if row.get("dst_port")}),
        )
        out["process_spawn_count"] = sum(
            row.get("event_type") == "process_start" for row in samples
        )
        out["sensitive_file_access_count"] = sum(
            row.get("event_type") == "sensitive_file_access" for row in samples
        )

    for (stamp, pod), samples in grouped(tables["http_events"], ["_window", "pod"]).items():
        out = rows.setdefault((stamp, pod), base_features(scenario, label, stamp, pod))
        latencies = [number(row.get("latency"), "latency") for row in samples]
        out["http_request_count"] = len(samples)
        out["http_rps"] = len(samples) / WINDOW_SECONDS
        out["http_error_rate"] = sum(number(row.get("resp_status")) >= 400 for row in samples) / len(samples)
        out["http_resp_kib"] = sum(number(row.get("resp_body_size"), "bytes") for row in samples) / 1024
        out["http_latency_avg_ms"] = sum(latencies) / len(latencies)
        out["http_latency_p95_ms"] = percentile(latencies, 0.95)

    dns_rows = tables["dns_events"]
    has_dns_client_ip = any(row.get("local_addr") not in {"", "-"} for row in dns_rows)
    if generator_ip and has_dns_client_ip:
        dns_rows = [row for row in dns_rows if row.get("local_addr") == generator_ip]
    for (stamp, pod), samples in grouped(dns_rows, ["_window", "pod"]).items():
        out = rows.setdefault((stamp, pod), base_features(scenario, label, stamp, pod))
        out["dns_request_count"] = len(samples)
        out["dns_failure_rate"] = sum(number(row.get("rcode")) != 0 for row in samples) / len(samples)

    for (stamp, _), row in rows.items():
        overlap_start = max(stamp, start)
        overlap_end = min(stamp + timedelta(seconds=WINDOW_SECONDS), end)
        overlap = max(0.0, (overlap_end - overlap_start).total_seconds())
        row["window_coverage_ratio"] = overlap / WINDOW_SECONDS
    return rows.values()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.run_dir / "training_windows.csv"
    multiclass_output = args.run_dir / "training_multiclass_windows.csv"
    all_rows = []
    with (args.run_dir / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        for item in csv.DictReader(handle):
            all_rows.extend(aggregate_scenario(
                args.run_dir / "raw" / item["scenario"],
                item["scenario"], item["label"],
                parse_time(item["start_time"]), parse_time(item["end_time"]),
                item.get("generator_ip", ""),
            ))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(all_rows, key=lambda row: (row["window_start"], row["pod"])))
    multiclass_rows = [
        row for row in all_rows
        if row["is_target"] == 1 and row["window_coverage_ratio"] >= 0.5
        and any(float(row[field]) != 0 for field in FEATURE_FIELDS)
    ]
    with multiclass_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(multiclass_rows, key=lambda row: (row["window_start"], row["pod"])))


if __name__ == "__main__":
    main()
