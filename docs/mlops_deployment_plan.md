# 模型部署计划

当前项目采用本地训练、本地推理、K3s 负责生成实验流量和 Pixie/eBPF 数据采集的方式。项目二的部署目标是优先将 ONNX MLP 作为轻量化推理模型，逐步封装为可部署服务。

## 1. 当前模型文件

```text
models/anomaly_detector.joblib
models/anomaly_binary_detector.joblib
models/anomaly_mlp.onnx
models/anomaly_mlp.scaler.joblib
models/anomaly_mlp_binary.onnx
models/anomaly_mlp_binary.scaler.joblib
```

## 2. 推荐部署路径

```text
Pixie/eBPF 导出最近窗口
  -> build_live_windows.py 聚合特征
  -> ONNX Runtime 推理服务
  -> 输出预测类别和异常概率
  -> 告警写入 alerts.csv 或推送到监控系统
```

## 3. 为什么优先部署 ONNX MLP

- 模型文件小，约 8.3 KiB。
- 推理延迟低。
- 不依赖 sklearn Python 对象序列化。
- 更适合封装为 HTTP/gRPC 推理服务。

## 4. K3s 部署规划

后续可以将 ONNX 推理服务打包为容器镜像，并部署到 K3s：

```text
Deployment: anomaly-inference
Service: anomaly-inference
ConfigMap/Volume: ONNX 模型与 scaler
Input: 10 秒窗口特征
Output: predicted_class、confidence、anomaly_probability
```

当前项目已经提供部署准备文件：

```text
serve_onnx.py
Dockerfile.inference
k8s/anomaly-inference.yaml
```

本地启动推理服务：

```bash
cd /home/idolsingerydd/pixie-data/pipeline
python3 serve_onnx.py --model models/anomaly_mlp.onnx --port 8080
```

构建镜像：

```bash
docker build -f Dockerfile.inference -t anomaly-inference:latest .
```

部署到 K3s 前，需要将镜像导入 K3s 使用的 container runtime。之后执行：

```bash
kubectl --kubeconfig /home/idolsingerydd/.kube/config apply -f k8s/anomaly-inference.yaml
```

## 5. 当前限制

- 已提供推理服务代码和部署 YAML，但还没有在当前会话中实际构建镜像并部署到 K3s。
- 当前在线检测仍主要通过脚本触发。
