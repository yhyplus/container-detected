# 《机器学习系统》答辩 PPT 大纲

项目名称：基于 Pixie/eBPF 的 Kubernetes 异常检测数据集生成与分类实验

> 这版大纲更适合答辩展示：重点讲“实验做了什么、数据集怎么生成、模型怎么训练、结果如何”，避免把 PPT 做成代码说明书。

## 1. 封面

标题：机器学习系统答辩

副标题：基于 Pixie/eBPF 的 Kubernetes 异常检测数据集生成与分类实验

一句话介绍：

本实验在 Kubernetes 隔离环境中主动制造多类异常行为，利用 Pixie/eBPF 采集运行时观测数据，构建 10 秒窗口级异常检测数据集，并训练模型识别异常类型。

## 2. 实验目标：我们到底做了什么？

本实验主要做了三件事：

1. 搭建一个可控的 Kubernetes 异常实验环境。
2. 自动生成正常流量和多类异常流量，采集 Pixie/eBPF 观测数据，构建机器学习数据集。
3. 训练并比较两类模型：
   - Random Forest：作为传统机器学习基线。
   - PyTorch MLP：作为神经网络模型，并导出为 ONNX 做推理对比。

展示重点：

- 不是直接使用公开数据集，而是自己生成数据集。
- 数据来自真实运行的 Pod、网络流量、HTTP/DNS 请求、资源使用和受控安全事件。
- 最终任务是二分类告警和六分类异常诊断。

## 3. 实验环境与数据来源

实验运行在 Kubernetes 集群中的 `anomaly-test` 命名空间。

核心组件：

- `anomaly-generator`：异常生成器 Pod，负责制造不同异常行为。
- `nginx-test`：被访问和被扫描的测试服务。
- Pixie/eBPF：采集运行时观测数据。
- 实验传感器 `sensor_events.csv`：补充 Pixie 本地部署中缺失的端口扫描、横向移动、进程启动和敏感文件访问事件。

采集的数据表包括：

- `process_stats.csv`：CPU、内存、磁盘读写等进程资源数据。
- `network_stats.csv`：网络收发字节数、包速率等。
- `http_events.csv`：HTTP 请求数量、响应大小、延迟、错误率等。
- `dns_events.csv`：DNS 请求数量与失败率。
- `sensor_events.csv`：端口扫描、横向移动、进程启动、敏感文件访问等受控事件。

## 4. 数据集生成总流程

建议这一页画流程图：

```text
启动隔离实验环境
        ↓
按场景制造正常/异常行为
        ↓
Pixie/eBPF 导出原始观测表
        ↓
实验传感器补充安全事件
        ↓
按 Pod 和时间聚合为 10 秒窗口
        ↓
生成带标签的 training_multiclass_windows.csv
        ↓
训练模型并在独立评估集上测试
```

核心思想：

- 每个异常场景运行一段固定时间。
- 期间持续采集运行时指标。
- 把连续时间序列切成 10 秒窗口。
- 每个窗口对应一行样本。
- 每行样本带有异常类别标签。

## 5. 实验场景：如何制造不同类型的数据？

本实验设计了 8 个原始场景，最终映射为 6 类诊断标签。

| 原始场景 | 实验行为 | 六分类标签 |
| --- | --- | --- |
| `normal` | 每秒 1 次 HTTP 请求，模拟正常访问 | normal |
| `http_flood` | 多线程并发请求 Nginx 服务 | ddos |
| `network_connections` | 高频建立 TCP 短连接 | ddos |
| `dns_flood` | 高频查询随机 `.invalid` 域名 | ddos |
| `portscan` | 扫描测试 Pod 的多个 TCP 端口 | portscan |
| `lateral_movement` | 扫描受控 Pod/Service 的常见端口 | lateral_movement |
| `cpu` | 持续计算制造 CPU 压力 | resource_anomaly |
| `syscall_anomaly` | 启动非预期进程并读取 `/etc/passwd` | syscall_anomaly |

展示时可以强调：

- DDoS 被设计成 3 个子场景，但在六分类中合并为 `ddos`。
- 横向移动和端口扫描只在隔离命名空间内发生，不扫描真实网络。
- 系统调用异常通过受控脚本产生，不是攻击真实系统。

## 6. 一次数据采集实验是怎样运行的？

本项目最终合并了两轮训练采集：

- `train-v3-expanded-20260603`：每个场景 180 秒。
- `train-v3-large-20260603`：每个场景 360 秒。

以较大的 `train-v3-large-20260603` 为例：

采集参数：

| 参数 | 值 | 含义 |
| --- | --- | --- |
| `DURATION` | 360 秒 | 每个场景的主要运行时间 |
| `WORKERS` | 6 | 并发工作线程数 |
| `SETTLE_SECONDS` | 10 秒 | 场景启动后的稳定等待时间 |
| `MEMORY_MIB` | 128 | 内存压力参数 |
| `CONNECTION_DELAY` | 0.02 秒 | TCP 短连接间隔 |
| `PORT_SCAN_MAX` | 1000 | 端口扫描最大端口号 |
| `PORT_SCAN_DELAY` | 0.004 秒 | 端口扫描间隔 |

独立评估集也合并了两轮采集：

- `eval-v3-expanded-20260603`：每个场景 90 秒。
- `eval-v3-large-20260603`：每个场景 180 秒。

以较大的 `eval-v3-large-20260603` 为例：

| 参数 | 值 |
| --- | --- |
| `DURATION` | 180 秒 |
| `WORKERS` | 4 |
| `PORT_SCAN_MAX` | 700 |
| `PORT_SCAN_DELAY` | 0.005 秒 |

为什么训练集和评估集参数不同：

- 让模型在略有差异的实验条件下评估。
- 避免只记住某一次采集的固定模式。
- 更接近真实场景中的变化。

## 7. 数据样本如何形成？

原始数据不是直接拿来训练，而是先做窗口聚合。

聚合方式：

- 时间粒度：10 秒一个窗口。
- 空间粒度：Pod 级别。
- 一行样本表示：某个 Pod 在某 10 秒内的行为特征。

样本字段包括：

- 标签字段：`class_id`、`label`、`attack_type`、`attack_subtype`。
- 辅助字段：`scenario`、`window_start`、`pod`、`is_target`、`window_coverage_ratio`。
- 模型输入特征：资源、网络、HTTP、DNS、安全事件等数值特征。

训练时会去掉：

- 场景名、标签、时间、Pod 名称等字段。
- 这样做是为了避免模型“偷看答案”。

## 8. 数据集规模与类别分布

最终用于主要结果展示的数据集：

训练集：`runs/train-v3-combined-20260603/training_multiclass_windows.csv`

- 总样本数：496 行。
- 类别分布：
  - normal：114
  - ddos：163
  - portscan：55
  - lateral_movement：56
  - resource_anomaly：54
  - syscall_anomaly：54

评估集：`runs/eval-v3-combined-20260603/training_multiclass_windows.csv`

- 总样本数：249 行。
- 类别分布：
  - normal：54
  - ddos：82
  - portscan：27
  - lateral_movement：29
  - resource_anomaly：28
  - syscall_anomaly：29

展示时要主动说明：

- 这是一个课程实验级自建数据集，规模已经从 bootstrap 的 33 条训练样本扩展到 496 条训练样本。
- 重点是完成“可控数据生成 -> 特征聚合 -> 模型训练 -> 实时检测”的完整闭环。
- 后续提升方向是继续扩大采集轮次、服务种类和异常强度。

## 9. 特征设计：模型看到的是什么？

模型输入共 26 个数值特征。

可以分成 5 类：

| 特征组 | 代表特征 | 能捕捉的异常 |
| --- | --- | --- |
| 资源特征 | `cpu_pct`、`rss_mib`、磁盘读写速率 | CPU/内存/磁盘异常 |
| 网络特征 | `rx_kib_sec`、`tx_kib_sec`、包速率 | 网络洪泛、连接异常 |
| 连接特征 | 目标 IP 数、目标端口数、连接数 | 端口扫描、横向移动 |
| HTTP/DNS 特征 | `http_rps`、HTTP 错误率、DNS 请求数 | HTTP 洪泛、DNS 洪泛 |
| 安全事件特征 | 进程启动次数、敏感文件访问次数 | 系统调用异常 |

展示重点：

- 特征不是随机选的，而是对应异常行为设计的。
- 每类异常都会在不同特征组上留下“行为指纹”。

## 10. 标签设计：二分类与六分类

本实验同时支持两个任务。

任务 A：二分类告警

| `label` | 含义 |
| --- | --- |
| 0 | 正常 |
| 1 | 异常 |

任务 B：六分类诊断

| `class_id` | 类别 |
| --- | --- |
| 0 | normal |
| 1 | ddos |
| 2 | portscan |
| 3 | lateral_movement |
| 4 | resource_anomaly |
| 5 | syscall_anomaly |

答辩时建议这样讲：

- 二分类解决“要不要告警”。
- 六分类解决“异常大概是什么类型”。
- 本实验更关注六分类，因为它更能体现诊断价值。

## 11. 模型一：Random Forest 基线

Random Forest 不是神经网络，而是传统机器学习模型。

为什么仍然使用它：

- 小数据集上通常比较稳。
- 对特征缩放不敏感。
- 能作为神经网络模型的对照基线。

训练参数：

| 参数 | 值 |
| --- | --- |
| 模型 | `RandomForestClassifier` |
| 树数量 | 300 |
| 类别权重 | 六分类使用 `balanced`；二分类优化后使用 `none` |
| 随机种子 | 42 |
| 并行数 | 1 |
| 默认目标 | `class_id` 六分类 |

输出文件：

- 六分类模型：`models/anomaly_detector.joblib`
- 二分类模型：`models/anomaly_binary_detector.joblib`

## 12. 模型二：PyTorch MLP 神经网络

本实验真正使用的神经网络是 MLP，也就是多层感知机。

网络结构：

```text
输入层：26 维特征
   ↓
全连接层：26 -> 64
   ↓
ReLU
   ↓
全连接层：64 -> 32
   ↓
ReLU
   ↓
输出层：32 -> 类别数
```

类别数：

- 六分类时：6 个输出神经元。
- 二分类时：2 个输出神经元。

为什么选择 MLP：

- 输入是表格型数值特征，不是图片或文本。
- MLP 适合处理固定维度的结构化特征。
- 网络较小，便于导出 ONNX 做部署实验。

## 13. MLP 训练参数

MLP 训练配置：

| 参数 | 值 |
| --- | --- |
| 框架 | PyTorch |
| 输入维度 | 26 |
| 隐藏层 | 64、32 |
| 激活函数 | ReLU |
| 损失函数 | CrossEntropyLoss |
| 优化器 | Adam |
| 学习率 | 0.001 |
| Epochs | 120 |
| Batch size | 64 |
| 随机种子 | 42 |
| 特征标准化 | StandardScaler |
| 导出格式 | ONNX |
| ONNX opset | 18 |

训练流程：

1. 读取 `training_multiclass_windows.csv`。
2. 去掉标签、时间、Pod、场景等非输入字段。
3. 使用 `StandardScaler` 对 26 个特征标准化。
4. 使用交叉熵损失训练 MLP。
5. 在独立评估集上计算准确率和 F1。
6. 导出 `.pt`、`.onnx` 和标准化器文件。

输出文件：

- 六分类 ONNX：`models/anomaly_mlp.onnx`
- 二分类 ONNX：`models/anomaly_mlp_binary.onnx`
- 标准化器：`models/anomaly_mlp.scaler.joblib`

## 14. 实验结果：分类效果

主要评估结果来自合并后的独立评估集 `eval-v3-combined-20260603`。

| 模型 | 任务 | Accuracy | Macro F1 |
| --- | --- | --- | --- |
| Random Forest | 六分类 | 98.39% | 0.985 |
| PyTorch MLP / ONNX | 六分类 | 99.20% | 0.993 |
| Random Forest | 二分类 | 99.60% | - |
| PyTorch MLP / ONNX | 二分类 | 99.60% | - |

结果解读：

- 六分类任务上，Random Forest 和 MLP 都取得了较高准确率。
- MLP 在本次合并数据集上准确率更高，同时模型更小，推理更快。
- Random Forest 二分类去掉 `balanced` 类别权重后，在 0.5 默认分类阈值下准确率从 89.16% 提升到 99.60%。
- 对课程规模数据集而言，传统模型和轻量神经网络都能完成有效分类。

## 15. 实验结果：推理性能

在相同评估数据上进行微基准测试：

| 模型 | 平均延迟 | p95 延迟 | 模型大小 |
| --- | --- | --- | --- |
| Random Forest | 约 10.63 ms | 约 12.02 ms | 约 944 KiB |
| ONNX MLP | 约 0.025 ms | 约 0.036 ms | 约 8.3 KiB |

展示时可以这样总结：

- Random Forest：传统机器学习基线，六分类表现稳定。
- ONNX MLP：模型更小，推理更快，本次六分类准确率也更高。
- 两者体现了“准确率”和“部署效率”的取舍。

## 16. 实时检测实验

训练完成后，系统可以进行实时预测。

实时检测流程：

```text
导出最近 Pixie 数据
        ↓
构建最近时间窗口
        ↓
加载模型
        ↓
输出每个窗口的异常概率和预测类别
        ↓
超过阈值则写入 alerts.csv
```

默认告警阈值：

- 异常概率 >= 0.7 时触发告警。

输出结果：

- `predictions.csv`：所有窗口的预测结果。
- `alerts.csv`：只包含告警窗口。
- `summary.json`：本次预测摘要。

## 17. 可视化与演示入口

本项目还实现了本地 Web Console。

演示时可以展示：

- 选择模型：Random Forest 或 ONNX MLP。
- 选择任务：二分类或六分类。
- 一键生成异常实验。
- 查看预测报告。
- 查看后台采集、训练、预测日志。

建议演示顺序：

1. 打开 Web Console。
2. 选择一个异常场景，比如 HTTP 洪泛或端口扫描。
3. 点击 One-click anomaly experiment。
4. 展示生成的告警结果和预测类别。

## 18. 实验结论

本实验完成了一个完整的机器学习系统闭环：

```text
异常场景设计
  -> 数据采集
  -> 数据集生成
  -> 特征工程
  -> 模型训练
  -> 离线评估
  -> 实时检测
```

主要结论：

- 自建数据集能够覆盖多种 Kubernetes 运行时异常。
- 10 秒 Pod 窗口可以把多源观测数据转换为可训练样本。
- Random Forest 六分类表现稳定，可作为传统机器学习基线。
- MLP/ONNX 在本次合并数据集上六分类准确率更高，同时模型更小、推理更快。
- 六分类诊断比单纯二分类告警更有实用价值。

## 19. 不足与改进方向

当前不足：

- 当前数据集已经扩展到 496 条训练样本和 249 条独立评估样本，但相对真实生产系统仍然偏小。
- 数据来自单一测试命名空间和固定测试服务，场景复杂度有限。
- 本地 Pixie/Vizier 的 `conn_stats` 为空，因此连接级行为需要实验传感器补充。
- 目前实验在单一测试服务上进行，真实业务复杂度更高。

改进方向：

- 继续增加采集轮次，扩大训练集和评估集。
- 引入更多服务类型和更多异常强度。
- 使用生产级运行时安全传感器，如 Falco 或 Tetragon。
- 对 ONNX 模型做更长时间、容器资源限制下的性能测试。
- 尝试更复杂的神经网络或时序模型，但前提是先扩大数据集。

## 20. 结束页

总结一句话：

本项目的重点不是简单套模型，而是从零构造一个可复现的 Kubernetes 异常检测数据集，并在此基础上完成传统机器学习模型和神经网络模型的训练、评估与实时检测对比。

结束语：

谢谢老师，欢迎提问。

## 可直接给 ChatGPT / WPS AI 的生成提示词

请根据以下要求生成一份中文答辩 PPT：

- 主题：《机器学习系统》课程项目答辩。
- 项目名称：基于 Pixie/eBPF 的 Kubernetes 异常检测数据集生成与分类实验。
- PPT 风格：学术答辩风格，重点突出实验过程和数据集生成，不要做成纯技术文档。
- 页数：18 到 20 页。
- 重点内容：
  - 实验做了什么。
  - Kubernetes 隔离实验环境。
  - 正常流量和异常流量如何生成。
  - Pixie/eBPF 和实验传感器如何采集数据。
  - 如何把原始数据聚合成 10 秒 Pod 级窗口。
  - 训练集和评估集规模与类别分布：训练集 496 条，评估集 249 条。
  - 使用了哪些模型：Random Forest 基线、PyTorch MLP 神经网络、ONNX Runtime 推理。
  - MLP 网络结构：26 -> 64 -> 32 -> 6，ReLU 激活。
  - 训练参数：Adam，学习率 0.001，CrossEntropyLoss，epochs 120，batch size 64，StandardScaler，seed 42。
  - Random Forest 参数：300 棵树，六分类 class_weight balanced，二分类 class_weight none，seed 42。
  - 实验结果：Random Forest 六分类 98.39%，ONNX MLP 六分类 99.20%；Random Forest 二分类 99.60%，ONNX MLP 二分类 99.60%。
  - 结论：自建数据集完成闭环，Random Forest 是稳定基线，ONNX MLP 在本次合并数据集上准确率更高、模型更小、推理更快。
- 请多使用流程图、表格和对比图，减少大段代码说明。
