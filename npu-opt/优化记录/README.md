# Ascend 910B2 优化记录

本目录集中保存本项目在 Ascend 910B2 上形成的自研基准、诊断探针、结构化结果和测评报告。`npu-opt/框架特性/`、`npu-opt/npu亲和算子与操作/` 与 `npu-opt/torch-compile/` 继续保存上游参考资料，不与实测记录混放。

## 记录索引

| 主题 | 时间 | 状态 | 报告 | 结构化结果 |
| --- | --- | --- | --- | --- |
| YOLO11-L / RT-DETR-L 训练优化 | 2026-07-31 至 2026-08-02 | 6 项训练优化和 1 项 MSDA 正确性修复已进入代码 | [训练优化报告](./检测模型训练/报告/yolo11l_rtdetr_910b2_training_optimization_report.md) | [JSON](./检测模型训练/结果/yolo11l_rtdetr_910b2_results.json) |
| PE-Spatial 全系列接入与 L/14 NPU 优化 | 2026-08-07 | 五档结构已接入；L/14 官方权重和 RoPE 快路径已验证 | [PE-Spatial 报告](./pe_spatial/pe_spatial_l14_910b2_report.md) | [JSON](./pe_spatial/pe_spatial_l14_910b2_results.json) |
| Conv-BN-ReLU 图化，CANN 9.1.0-beta.3 | 2026-07-30 | 历史环境；手动 graphed callables 有收益 | [历史报告](./torch_compile/报告/conv_bn_relu_910b2_benchmark_report.md) | 报告内记录 |
| Conv-BN-ReLU 编译，CANN 9.1.0 / Triton-Ascend 3.2.2 | 2026-08-02 | 正式版原始栈下 Inductor 不可用 | [正式版复测](./torch_compile/报告/conv_bn_relu_cann_9.1.0_triton_ascend_3.2.2_report.md) | 报告内记录 |
| reduce-overhead 临时兼容实验 | 2026-08-03 | 20M Conv 成功；YOLO11-L 失败；RT-DETR-L 中止；实现已撤回 | [终止实验报告](./torch_compile/报告/reduce_overhead_conv_yolo_rtdetr_910b2_20260803.md) | [JSON](./torch_compile/结果/reduce_overhead_conv_yolo_rtdetr_910b2_20260803.json) |

## 目录说明

- `检测模型训练/脚本/`：YOLO11-L、RT-DETR-L 单卡和 DDP 完整训练 step 基准，以及 NPU 利用率采样器。
- `检测模型训练/结果/`：检测模型逐项优化的结构化数据。
- `torch_compile/基准/`：Conv-BN-ReLU、Eager DVM 和 YOLO NPUGraph 基准。
- `torch_compile/探针/`：用于隔离诊断 DVM、Triton 和 worker 行为的最小程序。
- `pe_spatial/`：PE-Spatial-L/14 的 Ascend RoPE 消融、真实权重结果和两卡 DDP smoke。
- 各主题的报告文件保存人工审计结论；报告中的提交 SHA 均表示测试时的历史快照，不使用当前分支头回填。

## 当前建议

- YOLO11-L 和 RT-DETR-L 生产训练继续使用 Eager、`TASK_QUEUE_ENABLE=2`、internal format 和 FP16，并采用训练优化报告中已经验证并进入代码的优化。
- 固定 shape 的孤立 Conv-BN-ReLU 块可以参考历史报告评估 NPUGraph，但不能把微基准收益直接外推到完整检测模型。
- 当前代码未保留 2026-08-03 的临时 Inductor 兼容层，不应在 YOLO11-L 或 RT-DETR-L 生产训练中启用 `reduce-overhead`。

## 使用约定

- 从仓库根目录运行脚本，显式传入 `--output`，不要把生成数据写回本目录后覆盖历史结果。
- TaskQueue、可见设备和编译 cache 必须在导入 TorchNPU 前由独立进程设置。
- 对图编译结果同时检查 loss、梯度、参数、BN running stats 和 optimizer state；只有时延下降不能视为训练可用。
