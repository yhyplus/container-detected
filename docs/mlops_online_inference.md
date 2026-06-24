# 在线推理与告警流程

当前在线推理流程由 `predict_live.sh` 触发。

## 1. 推理流程

```text
导出最近 Pixie 表
  -> build_live_windows.py 聚合 10 秒窗口
  -> 加载 Random Forest 或 ONNX MLP
  -> 生成 predictions.csv
  -> 根据阈值生成 alerts.csv
  -> 写 summary.json
```

## 2. 阈值说明

离线二分类评估通常使用 0.5 阈值：

```text
P(anomaly) >= 0.5 -> anomaly
P(anomaly) < 0.5 -> normal
```

在线告警默认使用 0.7 阈值：

```text
P(anomaly) >= 0.7 -> alert
```

这样可以降低误报率。

## 3. 输出文件

```text
predictions.csv  # 所有窗口预测
alerts.csv       # 触发告警的窗口
summary.json     # 本次预测摘要
```

## 4. MLOps 监控指标

后续可持续监控：

- 每分钟窗口数量。
- 告警数量。
- 预测类别分布。
- 异常概率分布。
- 推理延迟。
- 特征漂移分数。

