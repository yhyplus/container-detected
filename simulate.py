#!/usr/bin/env python3
"""Generate isolated K3s workload patterns for Pixie anomaly collection."""

import argparse
import csv
from datetime import datetime, timezone
import os
import random
import socket
import string
import subprocess
import threading
import time


SENSOR_FIELDS = ["time_", "pod", "event_type", "dst_ip", "dst_port", "detail"]


class ExperimentSensor:
    """Record controlled experiment events that are absent from local Pixie tables."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.seen = set()
        self.seen_second = ""
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=SENSOR_FIELDS).writeheader()

    def emit(self, event_type, dst_ip="", dst_port="", detail="", deduplicate=False):
        stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        second = stamp[:19]
        key = (event_type, str(dst_ip), str(dst_port), detail)
        with self.lock:
            if second != self.seen_second:
                self.seen.clear()
                self.seen_second = second
            if deduplicate and key in self.seen:
                return
            self.seen.add(key)
            with open(self.path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=SENSOR_FIELDS)
                writer.writerow({
                    "time_": stamp,
                    "pod": "anomaly-test/anomaly-generator",
                    "event_type": event_type,
                    "dst_ip": dst_ip,
                    "dst_port": dst_port,
                    "detail": detail,
                })


def deadline_after(seconds):
    return time.monotonic() + seconds


def http_request(url):
    try:
        subprocess.run(
            ["wget", "-q", "-O", "/dev/null", url],
            check=False,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        pass


def normal_traffic(end, url):
    while time.monotonic() < end:
        http_request(url)
        time.sleep(1)


def cpu_pressure(end):
    value = 1
    while time.monotonic() < end:
        value = (value * 1664525 + 1013904223) & 0xFFFFFFFF


def memory_pressure(end, memory_mib):
    blocks = [bytearray(1024 * 1024) for _ in range(memory_mib)]
    for block in blocks:
        block[0] = 1
    while time.monotonic() < end:
        time.sleep(0.2)


def http_flood(end, url, workers):
    run_workers(lambda: http_request(url), end, workers)


def run_workers(operation, end, workers):
    def work():
        while time.monotonic() < end:
            operation()

    threads = [threading.Thread(target=work) for _ in range(max(1, workers))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def connection_flood(end, host, port, workers, delay):
    def connect_once():
        try:
            with socket.create_connection((host, port), timeout=1):
                pass
        except OSError:
            pass
        time.sleep(delay)

    run_workers(connect_once, end, workers)


def dns_flood(end, workers):
    alphabet = string.ascii_lowercase

    def resolve_once():
        name = "".join(random.choice(alphabet) for _ in range(16)) + ".invalid"
        try:
            socket.getaddrinfo(name, 80)
        except OSError:
            pass

    run_workers(resolve_once, end, workers)


def port_scan(end, host, max_port, workers, delay, sensor):
    next_port = 1
    lock = threading.Lock()

    def scan_once():
        nonlocal next_port
        with lock:
            port = next_port
            next_port = 1 if next_port >= max_port else next_port + 1
        try:
            with socket.create_connection((host, port), timeout=0.05):
                pass
        except OSError:
            pass
        sensor.emit("port_scan_attempt", host, port, deduplicate=True)
        time.sleep(delay)

    run_workers(scan_once, end, workers)


def lateral_movement_scan(end, hosts, workers, delay, sensor):
    ports = [22, 80, 443, 445, 2375, 3306, 5432, 6379, 8080, 10250]
    targets = [(host, port) for host in hosts for port in ports]
    next_target = 0
    lock = threading.Lock()

    def scan_once():
        nonlocal next_target
        with lock:
            host, port = targets[next_target]
            next_target = (next_target + 1) % len(targets)
        try:
            with socket.create_connection((host, port), timeout=0.05):
                pass
        except OSError:
            pass
        sensor.emit("lateral_scan_attempt", host, port, deduplicate=True)
        time.sleep(delay)

    run_workers(scan_once, end, workers)


def syscall_anomaly(end, sensor):
    while time.monotonic() < end:
        subprocess.run(["sh", "-c", "true"], check=False)
        sensor.emit("process_start", detail="sh -c true")
        try:
            with open("/etc/passwd", encoding="utf-8") as handle:
                handle.read(64)
            sensor.emit("sensitive_file_access", detail="/etc/passwd")
        except OSError:
            pass
        time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=[
        "normal", "cpu", "memory", "http_flood", "network_connections", "dns_flood",
        "portscan", "lateral_movement", "syscall_anomaly",
    ])
    parser.add_argument("--duration", type=int, default=90)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--memory-mib", type=int, default=128)
    parser.add_argument("--connection-delay", type=float, default=0.02)
    parser.add_argument("--port-scan-max", type=int, default=1000)
    parser.add_argument("--port-scan-delay", type=float, default=0.003)
    parser.add_argument("--scan-hosts", default="")
    parser.add_argument("--sensor-output", default="/tmp/anomaly-sensor-events.csv")
    parser.add_argument("--url", default=os.getenv(
        "TARGET_URL", "http://nginx-test.anomaly-test.svc.cluster.local/"
    ))
    parser.add_argument("--host", default=os.getenv(
        "TARGET_HOST", "nginx-test.anomaly-test.svc.cluster.local"
    ))
    parser.add_argument("--port", type=int, default=80)
    args = parser.parse_args()
    end = deadline_after(args.duration)
    sensor = ExperimentSensor(args.sensor_output)

    baseline = threading.Thread(target=normal_traffic, args=(end, args.url), daemon=True)
    baseline.start()

    if args.mode == "normal":
        baseline.join()
    elif args.mode == "cpu":
        threads = [
            threading.Thread(target=cpu_pressure, args=(end,))
            for _ in range(max(1, args.workers))
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    elif args.mode == "memory":
        memory_pressure(end, args.memory_mib)
    elif args.mode == "http_flood":
        http_flood(end, args.url, args.workers)
    elif args.mode == "network_connections":
        connection_flood(end, args.host, args.port, args.workers, args.connection_delay)
    elif args.mode == "dns_flood":
        dns_flood(end, args.workers)
    elif args.mode == "portscan":
        port_scan(end, args.host, args.port_scan_max, args.workers, args.port_scan_delay, sensor)
    elif args.mode == "lateral_movement":
        hosts = [host for host in args.scan_hosts.split(",") if host] or [args.host]
        lateral_movement_scan(end, hosts, args.workers, args.port_scan_delay, sensor)
    elif args.mode == "syscall_anomaly":
        syscall_anomaly(end, sensor)


if __name__ == "__main__":
    main()
