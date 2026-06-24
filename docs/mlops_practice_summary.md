# MLOps 实践总结

项目二在项目一的 Kubernetes 异常检测实验基础上，引入 MLflow Tracking、本地 registry、自动化训练流水线、评估报告和漂移检测脚本，目标是把离线模型实验推进为可复现、可追踪、可评估、可迭代的 MLOps 闭环。

## 1. 当前 MLOps 闭环

```text
数据 run 版本
  -> 合并训练/评估数据集
  -> 自动训练 Random Forest 与 ONNX MLP
  -> 生成模型文件与评估指标
  -> MLflow 记录参数、指标和 artifact
  -> 本地 JSON registry 记录数据/模型版本
  -> Markdown 报告用于答辩和复现
  -> 漂移检测脚本监控新数据分布变化
```

## 2. 主要交付物

```text
mlops_train_pipeline.py
monitor_drift.py
registry/data_registry.json
registry/model_registry.json
reports/model_eval_report.md
reports/model_eval_report.json
reports/drift_report.md
reports/drift_report.json
mlruns/
serve_onnx.py
Dockerfile.inference
k8s/anomaly-inference.yaml
monitor_predictions.py
mlops_retrain_check.py
```

## 3. MLflow Tracking

训练流水线会将以下内容记录到 MLflow：

- 数据集行数与特征数。
- Random Forest 和 ONNX MLP 的 Accuracy、Macro F1。
- Benchmark 延迟和模型大小。
- 模型 artifact，包括 `.joblib`、`.onnx`、`.pt`、`.scaler.joblib` 和元数据 JSON。
- 本地 registry 与评估报告。

启动 MLflow UI：

```bash
cd /home/idolsingerydd/pixie-data/pipeline
mlflow ui --backend-store-uri ./mlruns
```

然后在浏览器中查看 MLflow 页面。

## 4. 一键训练与记录

默认使用当前最终训练集和评估集：

```bash
cd /home/idolsingerydd/pixie-data/pipeline
python3 mlops_train_pipeline.py --version 20260603-final
```

也可以显式指定 run：

```bash
python3 mlops_train_pipeline.py \
  --train-run runs/train-v3-expanded-20260603 \
  --train-run runs/train-v3-large-20260603 \
  --eval-run runs/eval-v3-expanded-20260603 \
  --eval-run runs/eval-v3-large-20260603 \
  --version 20260603-final
```

## 5. 漂移检测

默认对比训练集与评估集：

```bash
python3 monitor_drift.py
```

也可以对比训练集与某次 live run：

```bash
python3 monitor_drift.py \
  --reference runs/train-v3-combined-20260603/training_multiclass_windows.csv \
  --current runs/live-xxxx/windows.csv \
  --output reports/live_drift_report.json
```

## 6. 后续扩展

- 构建并部署 ONNX MLP K3s 内部推理服务。
- 将实时预测结果、告警数量、类别分布和漂移分数接入监控面板。
- 使用 MLflow Model Registry 管理 `Staging`、`Production` 和 `Archived` 阶段。
- 当新数据累积或漂移分数升高时，触发重训练。
