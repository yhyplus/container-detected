# 项目二 MLOps 工作状态对照

本文档对照项目一答辩 PPT 中的待开展工作，说明当前项目二已经实现和仍需后续完善的部分。

## 1. 模型版本管理

状态：已完善为轻量 Model Registry。

已完成：

- 使用 MLflow Tracking 记录训练 run。
- 生成 `registry/model_registry.json`。
- 记录模型文件、训练数据版本、评估数据版本、评估指标、模型参数。
- 在 `registry/models/20260603-final/` 保存模型 artifact 快照，支持回滚。
- 新增 `mlops_model_registry.py`，支持 `Candidate / Production / Archived / Rejected` 阶段管理。
- 支持 `promote` 将候选版本晋级为 Production，并把旧 Production 归档。
- 支持 `rollback` 从 archived 快照恢复上一版 Production。
- 记录 `registry/model_registry_history.json`，保留晋级、回滚、阶段变更历史。
- 新增 `mlops_publish_mlflow.py`，将 Production 快照同步发布到 MLflow Registered Models。
- 当前 MLflow 中已注册 4 个模型：RF 六分类、RF 二分类、ONNX MLP 六分类、ONNX MLP 二分类，并设置 `production` alias。

相关文件：

```text
mlops_train_pipeline.py
mlops_model_registry.py
mlops_publish_mlflow.py
registry/model_registry.json
registry/model_registry_history.json
registry/models/20260603-final/
registry/production/
mlruns/
reports/model_eval_report.md
```

仍可完善：

- 后续可接入 MLflow 原生 Model Registry 或镜像仓库审批系统。

## 2. 自动化训练流程

状态：已实现带数据质量校验的训练、评估、导出、记录自动化。

已完成：

- `mlops_train_pipeline.py` 自动合并训练/评估 run。
- 自动训练 Random Forest 六分类、Random Forest 二分类、MLP 六分类、MLP 二分类。
- 自动导出 ONNX。
- 自动运行 benchmark。
- 自动生成 registry 和评估报告。
- 自动写入 MLflow。
- 新增 `mlops_data_validate.py`，训练前校验训练集/评估集行数、字段完整性、类别覆盖、缺失率、非数值、零值比例。
- 数据校验失败时训练流水线会停止，避免坏数据进入模型训练。

相关文件：

```text
mlops_train_pipeline.py
mlops_data_validate.py
reports/data_validation_report.md
reports/model_eval_report.md
registry/data_registry.json
registry/model_registry.json
```

仍可完善：

- 生产环境可继续接入 CI/CD 或 Argo Workflows，形成固定周期的数据采集训练调度。

## 3. 模型部署

状态：已通过正式镜像部署到本地 K3s 的 `anomaly-test` namespace。

已完成：

- 新增 ONNX 推理 HTTP 服务 `serve_onnx.py`。
- 新增容器构建文件 `Dockerfile.inference`。
- 新增 K3s 部署清单 `k8s/anomaly-inference.yaml`。
- 新增本地 K3s 演示部署清单 `k8s/anomaly-inference-hostpath.yaml`。
- 为避免 WSL 环境下镜像拉取和宿主机 Conda 动态库问题，新增纯 Python 推理服务 `serve_mlp_pure.py`。
- 将训练好的 MLP 权重导出为 `models/anomaly_mlp_weights.json`，K3s Pod 通过 hostPath 只读挂载项目目录并直接加载该权重。
- 新增 `Dockerfile.inference-pure`，基于 `python:3.12-alpine` 构建正式推理镜像 `anomaly-inference:latest`，避免安装 ONNX Runtime 等大依赖。
- 新增 `k8s/anomaly-image-import-job.yaml`，在无宿主机 sudo 交互权限时，通过 K3s Job 将本地镜像 tar 导入 K3s containerd。
- 已将 `anomaly-inference:latest` 导入 K3s containerd。
- 已将 `Deployment/anomaly-inference` 从 hostPath 演示版切换为正式镜像版。
- 当前运行镜像为 `anomaly-inference:latest`，模型权重位于镜像内 `/app/models/anomaly_mlp_weights.json`。
- 在 K3s 中创建并验证 `Deployment/anomaly-inference` 和 `Service/anomaly-inference`。
- 服务提供：
  - `GET /healthz`
  - `GET /metadata`
  - `POST /predict`
  - `GET /metrics`

相关文件：

```text
serve_onnx.py
serve_mlp_pure.py
export_mlp_weights.py
Dockerfile.inference
Dockerfile.inference-pure
k8s/anomaly-inference.yaml
k8s/anomaly-inference-hostpath.yaml
k8s/anomaly-image-import-job.yaml
docs/mlops_deployment_plan.md
```

仍可完善：

- 后续生产环境可将 `anomaly-inference:latest` 推送到正式镜像仓库。
- 将 `predict_live.sh` 的结果直接发送到推理服务，而不是本地脚本推理。

## 4. 模型监控

状态：已实现离线预测监控、漂移检测，以及 K3s 中的在线指标采集。

已完成：

- `monitor_predictions.py` 汇总预测类别分布、告警数量、异常概率分布。
- `monitor_drift.py` 对比训练集与当前数据的特征分布漂移。
- 生成 JSON 与 Markdown 报告。
- 推理服务暴露 Prometheus 风格 `/metrics`，包括请求数、预测行数、告警数、类别分布、推理延迟。
- `/metrics` 已扩展异常概率分布，包括异常概率 sum/min/max 和 bucket 直方图。
- `/metrics` 已暴露推理进程资源指标，包括 CPU seconds、RSS memory bytes、uptime seconds。
- 推理服务内置 `/dashboard` 可视化监控页面，可展示请求数、告警数、延迟、内存占用、类别分布、异常概率分布和运行状态。
- 新增 `k8s/anomaly-inference-monitor-cronjob.yaml`，在 K3s 中周期性抓取推理服务指标并写入 Job 日志。
- 新增 `mlops_monitor_snapshot.py`，可将在线指标快照持久化为 `reports/monitoring_history.jsonl` 和 `reports/monitoring_latest.json`。
- K3s 侧新增 `anomaly-monitor-history` CronJob，用于周期性输出 JSON 格式监控快照日志。

相关文件：

```text
monitor_predictions.py
monitor_drift.py
mlops_monitor_snapshot.py
serve_mlp_pure.py
k8s/anomaly-inference-monitor-cronjob.yaml
reports/monitoring_history.jsonl
reports/drift_report.md
reports/prediction_monitor_report.md
```

仍可完善：

- 当前已经具备内置 Dashboard、CronJob 日志快照和本地 JSONL 历史快照。
- 后续生产环境仍可接入 Prometheus / Grafana，支持历史曲线、长期存储和告警规则。

## 5. 持续评估与重训练

状态：已实现带发布门禁、自动部署、健康验证、失败处理和模型晋级的重训练闭环。

已完成：

- `mlops_retrain_check.py` 根据新数据量、模型准确率和漂移分数判断是否建议重训练。
- `mlops_auto_retrain.py` 在触发条件满足时自动执行 `mlops_train_pipeline.py`，并重新导出在线推理权重。
- `mlops_auto_retrain.py --build-image` 可在重训练后重新构建 `anomaly-inference:latest` 镜像和镜像 tar。
- `mlops_auto_retrain.py --deploy` 可自动导入镜像到 K3s、滚动重启推理 Deployment 并验证 `/healthz`。
- `mlops_auto_retrain.py --promote` 可在部署验证后调用 `mlops_model_registry.py promote` 晋级 Production。
- 新增 `mlops_release_gate.py`，新模型上线前会检查数据校验结果、绝对指标阈值，以及相对当前 Production 的指标退化幅度。
- `mlops_auto_retrain.py --rollback-on-failure` 可在训练、门禁或部署失败后写入失败报告，并尝试执行 registry 回滚。
- 新增 `mlops_periodic_retrain.sh`，可由宿主机 cron/CI 调用，串联漂移检测、监控快照、重训练、部署和晋级。
- `docs/mlops_retraining_policy.md` 描述触发条件、重训练流程、模型替换规则和回滚策略。

相关文件：

```text
mlops_retrain_check.py
mlops_auto_retrain.py
mlops_release_gate.py
mlops_periodic_retrain.sh
docs/mlops_retraining_policy.md
reports/retrain_check.json
reports/auto_retrain_report.json
reports/release_gate_report.json
```

仍可完善：

- 接入宿主机 cron、CI/CD 或 Argo Workflows 做固定周期无人值守调度。
- 将人工审批、消息通知和更严格的部署失败自动回滚进一步平台化。

## 总结

当前项目二已经完成了轻量 MLOps 闭环：

```text
MLflow 实验追踪
  -> 数据/模型 registry
  -> 自动化训练与评估
  -> 模型 artifact 快照
  -> ONNX 推理服务代码
  -> K3s 部署清单
  -> 预测监控与漂移检测
  -> 重训练触发检查
```

尚未完全完成的是生产级持续化能力：

```text
周期性自动重训练
CI/CD 自动发布与回滚
正式镜像仓库和生产级 Prometheus / Grafana 长期监控
```
