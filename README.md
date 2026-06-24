# Container Anomaly Detection

基于 eBPF 的容器异常监测、数据集生成与 MLOps 流水线。

## Pixie anomaly dataset pipeline

This pipeline generates isolated workloads in the `anomaly-test` namespace and
aggregates Pixie observations and controlled experiment sensor events into
pod-level 10-second training windows.

Scenarios:

| Scenario | Generator behavior | Main Pixie features |
| --- | --- | --- |
| `normal` | One HTTP request per second | Baseline resource and traffic values |
| `http_flood` | Concurrent HTTP requests to the test Nginx service | `http_rps`, response bytes |
| `network_connections` | Repeated short TCP connections to the test Nginx service | network bytes and packet rate |
| `dns_flood` | Queries randomized `.invalid` names through cluster DNS | DNS request count and failure rate |
| `portscan` | Scans TCP ports on the isolated Nginx Pod | network packet rate |
| `lateral_movement` | Scans common ports on controlled Pod and Service addresses | unique destination IP and port counts |
| `cpu` | Runs sustained computation | CPU usage |
| `syscall_anomaly` | Starts an unexpected process and reads `/etc/passwd` inside the test container | process starts and sensitive file accesses |

The generator only contacts the isolated Nginx service, Nginx Pod, and cluster
DNS. The lateral-movement simulation is deliberately constrained to addresses
inside `anomaly-test`. It does not scan your real LAN or send load to public
endpoints.

## Prerequisites

Authenticate the Pixie CLI once:

```bash
/home/idolsingerydd/bin/px auth login
```

Run the full collection:

```bash
cd /home/idolsingerydd/pixie-data/pipeline
./run_scenarios.sh
```

The raw diagnostic file is written under `runs/<timestamp>/training_windows.csv`.
For model training, use `runs/<timestamp>/training_multiclass_windows.csv`. The
multiclass file keeps only the target pod for each controlled experiment and
discards windows with less than 50% scenario coverage. Keep different runs
separate and split training and evaluation data by run ID rather than randomly
splitting adjacent windows.

Useful overrides:

```bash
DURATION=180 WORKERS=4 ./run_scenarios.sh
```

The TCP connection scenario uses a small per-worker delay to avoid exhausting
the generator Pod's ephemeral ports. Adjust it only when needed:

```bash
CONNECTION_DELAY=0.01 ./run_scenarios.sh
```

To resume an interrupted run, reuse its output directory explicitly. Completed
scenarios with a manifest entry and all raw exports are skipped. Collection
settings are saved in `run.env` and automatically reused:

```bash
OUT="$PWD/runs/train-v3-low" RESUME=1 ./run_scenarios.sh
```

`label` remains the binary normal/anomalous target. For the course-project
six-class diagnosis, use `class_id`:

| `class_id` | `attack_type` |
| --- | --- |
| `0` | `normal` |
| `1` | `ddos` |
| `2` | `portscan` |
| `3` | `lateral_movement` |
| `4` | `resource_anomaly` |
| `5` | `syscall_anomaly` |

`attack_subtype` keeps the DDoS subtype: `http_flood`,
`network_connections`, or `dns_flood`.

Drop `scenario`, `attack_type`, `attack_subtype`, `class_id`, `label`, `is_target`,
`window_coverage_ratio`, `window_start`, and `pod` from model input features.
Use either `label` or `class_id` as the prediction target.

The current local Vizier deployment leaves `conn_stats` empty. To keep
port-scan experiments measurable, the pipeline combines Pixie data with
`sensor_events.csv`, a controlled experiment sensor written by `simulate.py`.
It supplies per-window `unique_dst_ip_count`, `unique_dst_port_count`,
`process_spawn_count`, and `sensitive_file_access_count`. This is suitable for
repeatable course experiments, but it is not a production host sensor. For
production detection, replace it with working connection-level eBPF telemetry
and a runtime sensor such as Falco or Tetragon.

## Terminology

**Valid telemetry** means that a 10-second window contains at least one
non-zero observed feature, such as CPU usage, packet rate, HTTP requests,
destination ports, or sensor events. Empty windows remain visible in prediction
CSV files but do not trigger alerts.

**Smoke test** means a short end-to-end check that confirms the scripts can run,
export data, build windows, and invoke the model. It catches integration errors;
it does not prove model accuracy. Accuracy must be measured on a separately
collected evaluation dataset.

Some Pixie deployments export DNS `local_addr` as `-`. In that case DNS counts
are scoped by the controlled scenario time window rather than an exact client
Pod address. Collect DNS experiments in an otherwise quiet test namespace and
keep `dns_request_count` as the primary DNS anomaly signal.

## Training

Train a six-class random forest after collection:

```bash
python3 train_model.py --run-dir runs/train-v3-low
```

Repeat `--run-dir` to combine training runs. Use a separate run for evaluation
so adjacent windows from one experiment do not leak across the split:

```bash
python3 train_model.py \
  --run-dir runs/train-v3-low \
  --eval-run-dir runs/eval-v3
```

The model bundle is written to `models/anomaly_detector.joblib`, with training
metadata in the adjacent JSON file. Pass `--target label` to train a binary
normal-versus-anomaly classifier instead.

For production-style alerting, train the binary detector separately:

```bash
python3 train_model.py \
  --run-dir runs/train-v3-low \
  --target label \
  --output models/anomaly_binary_detector.joblib
```

Train a one-class Isolation Forest using only normal (`label=0`) windows:

```bash
python3 train_isolation_forest.py \
  --run-dir runs/train-v4-large-20260623 \
  --eval-run-dir runs/eval-v4-large-20260623 \
  --contamination 0.15 \
  --output models/anomaly_isolation_forest.joblib
```

Anomalous windows are used only for held-out evaluation, never for fitting the
Isolation Forest. The default threshold `0.5` corresponds to the learned
normal/anomalous decision boundary. This model is intended as a sensitive
unknown-anomaly detector and may produce more false positives than the
supervised Random Forest and MLP models.

## Prediction

Predict anomaly classes for an existing aggregated dataset:

```bash
python3 predict_model.py \
  --input runs/train-v3-low/training_windows.csv \
  --output runs/train-v3-low/predictions.csv
```

Run a one-shot live check against the most recent Pixie data:

```bash
./predict_live.sh
```

The live command exports recent Pixie tables, aggregates 10-second windows, and
writes `runs/live-<timestamp>/predictions.csv`. It uses the binary detector by
default. A row becomes an alert when its anomaly probability is at least `0.7`.
Adjust the lookback and alert threshold when needed:

```bash
LOOKBACK_SECONDS=300 THRESHOLD=0.8 ./predict_live.sh
```

Use the normal-only Isolation Forest for binary alerting:

```bash
ALGORITHM=isolation_forest \
MODEL="$PWD/models/anomaly_isolation_forest.joblib" \
THRESHOLD=0.5 \
./predict_live.sh
```

Use the six-class detector when investigating the likely anomaly type:

```bash
MODEL="$PWD/models/anomaly_detector.joblib" ./predict_live.sh
```

Use Scheme B ONNX Runtime instead:

```bash
ALGORITHM=onnx_mlp MODEL="$PWD/models/anomaly_mlp.onnx" ./predict_live.sh
```

Rows with no observed feature signal are retained in the prediction CSV with
`has_signal=0`, but they do not emit alerts. The training dataset also excludes
these empty telemetry windows.

Each prediction also writes `summary.json` and `alerts.csv`. Use `alerts.csv`
when only the actionable rows are needed.

## Web Console

Start the local web console:

```bash
./start_ui.sh
```

Then open `http://127.0.0.1:5000`. The page can select Scheme A Random Forest or
Scheme B ONNX MLP for detection, start binary or six-class live checks, collect
low/medium/high/evaluation datasets, retrain models from selected runs, inspect
readable prediction reports, and show background task logs. The recommended
starting point is **One-click anomaly experiment**: choose an isolated DDoS,
port-scan, lateral-movement, CPU, or system-call scenario and let the UI
simulate activity before automatically running detection. The dashboard and task log page poll
for updates every two seconds, so manual browser refreshes are not required.

## Dataset Labels

Both models train from `training_multiclass_windows.csv`. The file contains
10-second Pod-level feature windows and two label columns:

| Scenario | `attack_type` | `class_id` for six-class | `label` for binary |
| --- | --- | --- | --- |
| `normal` | `normal` | `0` | `0` |
| `http_flood` | `ddos` | `1` | `1` |
| `network_connections` | `ddos` | `1` | `1` |
| `dns_flood` | `ddos` | `1` | `1` |
| `portscan` | `portscan` | `2` | `1` |
| `lateral_movement` | `lateral_movement` | `3` | `1` |
| `cpu` | `resource_anomaly` | `4` | `1` |
| `syscall_anomaly` | `syscall_anomaly` | `5` | `1` |

`train_model.py` uses `class_id` by default. Passing `--target label` trains the
binary normal-versus-anomaly model from the same rows.

## ONNX MLP Comparison

Scheme B trains a PyTorch MLP, exports it to ONNX, and runs inference with ONNX
Runtime:

```bash
python3 train_mlp_onnx.py \
  --run-dir runs/train-v3-low \
  --run-dir runs/train-v3-medium \
  --run-dir runs/train-v3-high \
  --eval-run-dir runs/eval-v3
python3 predict_onnx.py \
  --input runs/eval-v3/training_multiclass_windows.csv
```

Compare Random Forest and ONNX Runtime MLP with the same held-out dataset:

```bash
python3 benchmark_models.py \
  --input runs/eval-v3/training_multiclass_windows.csv \
  --output runs/eval-v3/benchmark.json
```

The benchmark is an in-process microbenchmark. Very small batches can make ONNX
throughput look unusually high and can distort short CPU measurements. For a
deployment report, add a longer cgroup-constrained container benchmark.

## Bootstrap Baseline

The initial v3 bootstrap model was trained from `runs/train-v3-bootstrap` and
evaluated on the separately collected `runs/eval-v3-bootstrap` dataset:

| Model | Target | Held-out accuracy |
| --- | --- | --- |
| Scheme A Random Forest | six-class | `95%` |
| Scheme A Random Forest | binary | `100%` |
| Scheme B ONNX MLP | six-class | `90%` |
| Scheme B ONNX MLP | binary | `90%` |

These are integration baselines, not final course-report claims: the evaluation
set contains only 20 filtered windows. Collect longer low, medium, high, and
evaluation runs before drawing conclusions.

## Improving Accuracy

Do not tune against adjacent windows from the same run. Collect separate
training and evaluation runs, then train on several workload intensities:

```bash
OUT="$PWD/runs/train-v3-low" DURATION=600 WORKERS=2 PORT_SCAN_DELAY=0.006 ./run_scenarios.sh
OUT="$PWD/runs/train-v3-medium" DURATION=600 WORKERS=4 PORT_SCAN_DELAY=0.004 ./run_scenarios.sh
OUT="$PWD/runs/train-v3-high" DURATION=600 WORKERS=6 PORT_SCAN_DELAY=0.003 ./run_scenarios.sh
python3 train_model.py \
  --run-dir runs/train-v3-low \
  --run-dir runs/train-v3-medium \
  --run-dir runs/train-v3-high \
  --eval-run-dir runs/eval-v3
```

Use distinct `OUT` values for named runs. Include multiple normal runs during
quiet and ordinary service activity. A 90% target should be accepted only on
held-out runs with enough windows for every class, not on the training set.
