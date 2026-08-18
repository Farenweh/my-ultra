# Ascend reduce-overhead 训练实验终止报告

## 结论

2026-08-03 在单张 Ascend 910B2、CANN 9.1.0、PyTorch/TorchNPU 2.12.0 和 Triton-Ascend 3.2.2 上进行项目内临时兼容实验。规则的 20M 参数 Conv-BN-ReLU 堆叠能够正确完成 `torch.compile(mode="reduce-overhead")` 训练，并在稳态快于 Queue2 Eager；同一方案未能跑通 YOLO11-L，RT-DETR-L 在测试完成前被终止。

实验兼容层、改写代码、原始日志和专用编译 cache 均已删除。本文只保存历史测量结果，**不表示当前代码已经支持该路径**。生产训练仍推荐 Eager 与 `TASK_QUEUE_ENABLE=2`。

## 环境与临时兼容范围

| 项目 | 配置 |
| --- | --- |
| 硬件 | 单张 Ascend 910B2 |
| CANN | 9.1.0 正式版，910B 算子包 |
| PyTorch | 2.12.0 |
| torch_npu | 2.12.0 |
| Triton-Ascend | 3.2.2 |
| 编译模式 | `torch.compile(mode="reduce-overhead")`，`TASK_QUEUE_ENABLE=1` |
| 生产对照 | Eager，`TASK_QUEUE_ENABLE=2` |

临时兼容层处理了 PyTorch/TorchNPU `CantSplit` 接口差异、训练态 BatchNorm 回退、Triton `where` 布尔条件和 NPU 上错误的 CUDA channels-last layout 传播，并在编译 microstep 前标记 NPUGraph step。所有改动均仅用于实验，现已撤回。

## 20M Conv-BN-ReLU

负载为 35 个训练态 `Conv2d(bias=False) → BatchNorm2d → ReLU` 块，输入 `[2,64,32,32]`，共 `20,219,392` 个参数，使用 BF16 autocast、SGD，并计入 forward、loss、backward 和 optimizer step。

10 步正确性对照通过：loss、梯度、参数更新、BN running stats 和 optimizer state 的最大绝对差均为 0。

| 测量 | Eager | reduce-overhead | 加速 |
| --- | ---: | ---: | ---: |
| Queue1 配对 AB | 24.8635 ms | 11.2907 ms | 2.202× |
| Queue1 配对 BA | 25.9425 ms | 10.2920 ms | 2.521× |
| 独立稳态生产对照 | Queue2：17.3768 ms | Queue1：10.3607 ms | **1.677×** |

独立长测中，candidate 平均为 `10.3651 ms/step`，Queue2 Eager 平均为 `17.2138 ms/step`。以中位数计算，candidate 时延降低约 40.4%，等价吞吐提升约 67.7%。

代价和资源情况：

- 清空专用 cache 后的真正冷编译首步为 `363,177.53 ms`，约 6 分 3 秒。
- 缓存命中后的 candidate 首步仍为 `3,320.85 ms`；Queue2 Eager 首步为 `223.34 ms`。
- candidate 峰值 allocated 为 `203.343 MiB`，Queue2 Eager 为 `203.423 MiB`，该负载没有观察到显著显存增加。
- candidate 的 20 个利用率样本中，AICore 平均 `70.3%`、NPU 平均 `99.4%`；Queue2 Eager 的 32 个样本分别为 `42.97%` 和 `58.47%`。

## YOLO11-L 失败原因

首个确定阻塞点是 SPPF 连续 MaxPool 的 FakeTensor 布局推导与 NPU 真实 internal format 不一致。编译器期望中间张量：

```text
shape  = (1, 256, 4, 4)
stride = (4096, 16, 4, 1)
```

实际 NPU 存储 stride 为：

```text
stride = (4096, 1600, 64, 1)
```

已经筛查但未形成有效修复的路径：

- 仅把 MaxPool 回退到原生 NPU 算子，仍会在 FakeTensor/真实 Tensor 边界触发布局校验错误。
- 在 SPPF 或 MaxPool 周围 graph break 会产生约 28 个子图，冷编译达到数十分钟，并继续暴露 `Expected standard deleter`、ACLop Conv2D capture、静态地址和原地 autograd 不兼容，以及 allocator 地址空间不足。
- 将 SPPF 改写为数值等价形式后，Eager 前向一致，但相应 backward Triton kernel 编译超过 7 分钟仍未完成，未通过完整 optimizer step 验证。

因此失败并非模型参数量本身导致，而是 YOLO 的池化复用、concat、分支和多尺度特征图触发了简单 Conv 堆叠没有覆盖的 layout、图切分及 NPUGraph 内存管理路径。

## RT-DETR-L 与材料状态

RT-DETR-L 编译 smoke 已启动，但在得到 forward/backward 结果前按要求终止，不能给出正确性、可用性或性能结论。

本次实验的临时兼容实现、测试文件、专用 `/dev/shm` cache 和新生成的 debug trace 均已清理；源码迁移使用的 PyTorch、TorchNPU、Triton 和 LLVM 实验工作区也已删除。由于原始运行日志不再保留，本文与[结构化结果](../结果/reduce_overhead_conv_yolo_rtdetr_910b2_20260803.json)是该次实验的最终审计记录。
