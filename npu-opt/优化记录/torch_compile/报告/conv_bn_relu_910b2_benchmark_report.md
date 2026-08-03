# Ascend 910B2 Conv-BN-ReLU 训练编译/图化测评报告

> 历史报告：本文数据采自 CANN 9.1.0-beta.3 软件栈。CANN 9.1.0 正式版与 Triton-Ascend 3.2.2 的复测结论见
> [conv_bn_relu_cann_9.1.0_triton_ascend_3.2.2_report.md](./conv_bn_relu_cann_9.1.0_triton_ascend_3.2.2_report.md)。

- 测试日期：2026-07-30
- 代码基线：`0a209b1c3895ca497dc122063845e487e1059a72`
- 可复现脚本：[benchmark_conv_bn_relu.py](../基准/benchmark_conv_bn_relu.py)
- Eager DVM/FFN 复现脚本：[benchmark_eager_dvm.py](../基准/benchmark_eager_dvm.py)

## 结论

当前 910B2 软件栈上，取得明确且可复现稳态加速的首选方案是：

> 用 `torch_npu.npu.make_graphed_callables()` 捕获 Conv-BN-ReLU 模块的前向和反向，优化器留在 Eager；固定 shape，并在捕获和重放时使用相同的精度上下文。主测试使用 BF16 autocast，FP32 也能加速。

主形状 `[32, 64, 56, 56]` 的最终结果如下：

- 3 张物理 NPU、6 个独立进程的 BF16 AB/BA 配对测试全部加速，范围为 `1.083×–1.304×`，简单平均为 `1.183×`，中位数为 `1.178×`。
- 为排除“Eager 总是先跑”的顺序偏差，Candidate-first（BA）三次也全部加速，范围为 `1.083×–1.149×`，平均为 `1.126×`。
- 6 次正确性检查的 Eager/Candidate 前向输出都确认为 `torch.bfloat16`；10 步 loss 逐项相同，BN 计数均为 `10/10`，状态最大绝对差为 `3.73e-9`，梯度最大绝对差不超过 `9.54e-7`。
- `SGD(momentum=0.9)` 的独立测试为 `1.214×`，状态、梯度和 momentum buffer 均通过自动判据。
- FP32 的一次 Candidate-first 敏感性测试也训练正确并取得 `1.135×`。BF16 在这个孤立块上并不比 FP32 绝对更快，不应只为该微基准切换训练 dtype。

该 API 不是 `torch.compile` backend，但它是 [NPUGraph 高级图化接口](../../../框架特性/pytorch_npugraph_desc.md)。[NPUGraphs 文档](../../../torch-compile/pytorch_compile_npugraph_desc.md) 也将细粒度控制指向该路径。自动 `torch.compile(..., backend="npugraphs")` 和 `npugraph_ex` 虽然训练正确，却分别只有约 `0.64×–0.66×` 和 `0.60×`，不适合这个粒度的块。

规模放大到 YOLO11-L、640×640、batch 8 后，自动 `npugraphs` 相对图捕获兼容的 TaskQueue=1 Eager 转为 `1.261×` 加速，但仍比当前 Ultralytics 默认 TaskQueue=2 Eager 慢 `15.6%`；手动整网 `make_graphed_callables` 则比默认 Eager 快 `1.082×`。这是固定形状合成训练筛选，尚未完成真实数据长训练、DDP 和收敛验证。

当天最新 TorchNPU `v2.12.0` 源码还可通过 `TORCH_NPU_LAZY_FUSION=True` 启用 Eager DVM。它能正确训练，但四张卡 BF16 配对只有约 `1.012×` 平均加速，范围为 `0.998×–1.023×`，其中一张卡轻微变慢；FP32 单次配对为 `0.913×`。该收益远小于 `make_graphed_callables`，暂不作为默认推荐。

这个结论不能外推到 Transformer：补测的 SwiGLU-FFN+LayerNorm 在同步双卡交叉测试中取得约 `1.110×`，而 ViT GELU-MLP+LayerNorm 只有约 `1.009×`。DVM 是否值得开启高度取决于 MatMul 后能否接上可融合的 pointwise 链。

本结论覆盖单卡微基准、固定形状 YOLO11-L 合成训练筛选和短程状态检查，不等价于完整 YOLO 真实数据长训练收敛验证。

## 测试平台

| 项目             | 值                                                                   |
| ---------------- | -------------------------------------------------------------------- |
| NPU              | 8 × Ascend 910B2，每卡 `65536 MiB` 物理 HBM                          |
| 驱动 / `npu-smi` | `25.5.2`                                                             |
| CANN             | `9.1.0-beta.3`                                                       |
| OS               | Linux aarch64，kernel `4.19.90-2107.6.0.0251.71.oe1.bclinux.aarch64` |
| Python           | `3.12.13`                                                            |
| PyTorch          | `2.12.0+cpu`，git `7661cd9c6b841b62b7f411aa52ec51f05457263b`         |
| TorchNPU         | `2.12.0`，git `fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d`             |
| Triton           | distribution `3.5.0`，runtime `3.2.0`                                |
| Triton-Ascend    | `3.2.1`                                                              |

测试开始时 8 张卡均为 `Health=OK`。NPU3 有历史可纠正 HBM 单比特计数 `24`，当前计数、双比特错误和隔离页均为 0；性能测试使用 NPU0/1/2/4/7，避开 NPU3。

环境有两项值得注意的版本风险：

1. 重新下载的 `torch_npu==2.12.0` wheel 与已安装 wheel 的 SHA256 完全相同，但它与 `torch==2.12.0` 的部分 Inductor 私有 API 不匹配。
2. `triton` distribution metadata 为 `3.5.0`，而实际导入 runtime 为 `3.2.0`；这是 Triton-Ascend 覆盖同名 package 目录后的状态。
3. 已发布的 `torch_npu==2.12.0` wheel（git `fa0f83fe`）没有编入 Eager DVM，也没有真正的 Inductor-DVM loader；当天最新 `v2.12.0` 源码（git `656c7917`）才包含这两条实现。

Triton-Ascend 官方 package index 在测试日的最新稳定版就是 `3.2.1`，主环境已经安装该版本。为排除新版本已修复问题的可能，还从官方 `release/3.2.2` 分支的 `c22984ba` 构建了
`triton_ascend-3.2.2+gitc22984ba` aarch64 wheel，并只安装到临时 venv。该分支尚无 `3.2.2` release wheel；源码在 GCC 13 下构建失败，改用 Clang 18 后成功，未覆盖主环境的 `3.2.1`。

## 基准设计

主测模块为：

```python
nn.Sequential(
    nn.Conv2d(64, 64, 3, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(inplace=False),
)
```

测试条件：

- 主 Conv 微基准固定 `TASK_QUEUE_ENABLE=1`，脚本在导入 TorchNPU 前设置并写入结果 JSON。
- 模型参数和输入为 FP32；主测试在捕获与重放两侧都使用 `torch.autocast("npu", dtype=torch.bfloat16, cache_enabled=False)`。
- loss 为 `output.float().square().mean()`；优化器为 `SGD(lr=1e-3, momentum=0, foreach=False)`，另测 `momentum=0.9`。
- 每步计时包含 `zero_grad -> forward -> loss -> backward -> optimizer.step`，optimizer 不入图。
- 捕获样例是运行时第一个 batch 的 clone；计时时轮换两个不同 `data_ptr()` 的 NPU batch，包含 wrapper 内部的输入 `copy_()` 成本。
- 正确性阶段从相同 CPU `state_dict` 出发，比较 10 步 loss、输出 dtype、梯度、参数、BN running stats、`num_batches_tracked` 和 optimizer state。
- BF16 loss 默认判据为 `atol=1e-4, rtol=1e-3`，FP32 loss 为 `atol=1e-5, rtol=1e-4`。BF16 路径的 FP32 参数、梯度和 optimizer state 另用 `atol=1e-4, rtol=1e-2`，避免用宽松的输出容差掩盖训练状态错误。
- 性能阶段预热 20 步，然后测 10 个 block，每个 block 为 20 个完整训练 step；只在 block 边界同步。
- 每项使用独立进程。主候选在同一张卡分别跑 Eager-first（AB）和 Candidate-first（BA），避免固定顺序形成单向偏差。

终审时还用小形状 `[4, 16, 32, 32]`、输出通道 16、`input_offset=5` 的两个高对比 batch 做了 6 步 correctness-only 验证。loss、状态和梯度最大绝对差分别为 `4.17e-7`、`2.98e-7` 和 `3.05e-5`，严格判据通过，证明重放没有滞留在捕获样例。

## 正式结果

### BF16 `make_graphed_callables` AB/BA

| 物理 NPU | 顺序 | Eager median | Candidate median |    加速比 |
| -------: | ---- | -----------: | ---------------: | --------: |
|        0 | AB   |  `1.5725 ms` |      `1.3029 ms` | `1.2069×` |
|        0 | BA   |  `1.5735 ms` |      `1.3717 ms` | `1.1471×` |
|        1 | AB   |  `1.5028 ms` |      `1.1523 ms` | `1.3042×` |
|        1 | BA   |  `1.6236 ms` |      `1.4133 ms` | `1.1488×` |
|        4 | AB   |  `1.6116 ms` |      `1.3311 ms` | `1.2108×` |
|        4 | BA   |  `1.4331 ms` |      `1.3235 ms` | `1.0829×` |

汇总：

| 汇总方式 | Eager median 均值 | Candidate median 均值 | run-level 加速比均值 |
| -------- | ----------------: | --------------------: | -------------------: |
| AB 三次  |       `1.5623 ms` |           `1.2621 ms` |            `1.2406×` |
| BA 三次  |       `1.5434 ms` |           `1.3695 ms` |            `1.1262×` |
| 全部六次 |       `1.5528 ms` |           `1.3158 ms` |            `1.1834×` |

顺序对绝对时延和加速比有明显影响，因此不使用只有三个样本的 Student-t 置信区间，也不只报告较高的 AB 数字。六次每个方案各有 10 个计时 block，block CV 范围为 `0.38%–4.42%`；所有独立进程的中位数配对仍一致指向加速。

首次图捕获 setup 为 `30.1–41.2 ms`，同进程第二次捕获为 `10.3–12.0 ms`。用首次捕获成本除以各 run 的稳态单步节省，约需 `86–376` 个 step 回本，中位数约 `135` step；因此该方案适合固定 shape 的长训练，而不适合只运行几十步的短任务。

脚本在每一侧测量后显式执行循环引用回收、NPU cache 清理和同步。最终六次的 Eager/Candidate 峰值分别稳定为 `232.97/220.79 MiB`，不再受前一张图的存活时间影响；这些清理动作位于计时区间之外，因为真实训练会持续复用同一张图。

所有 `make_graphed_callables` 运行都有一次 `AccumulateGrad stream mismatch` warning。本次单卡前反向、BN 状态和性能均正常，但 DDP、梯度 hook、梯度累积和保留 autograd graph 必须另行验证。

### 数值和优化器敏感性

| 变体                       | 顺序 | 正确性 | Eager median | Candidate median |    加速比 |
| -------------------------- | ---- | ------ | -----------: | ---------------: | --------: |
| B32 BF16，SGD momentum 0.9 | BA   | 通过   |  `1.8989 ms` |      `1.5646 ms` | `1.2137×` |
| B32 FP32，SGD momentum 0   | BA   | 通过   |  `1.3516 ms` |      `1.1912 ms` | `1.1346×` |

Momentum 测试的状态、梯度和 optimizer state 最大绝对差分别为 `7.45e-9`、`4.77e-7` 和 `8.58e-7`。FP32 的 loss 逐项相同，状态/梯度最大绝对差为 `3.73e-9/2.79e-9`。

同卡 BA 测量中，BF16/FP32 Candidate 的峰值已分配显存为 `220.79/221.79 MiB`，差异很小；BF16 Candidate 绝对时延比 FP32 高约 `18.6%`，但相对加速略高。应沿用完整训练原本需要的 dtype，而不是依据这个孤立块切换精度。

### 自动图 backend

| 方案                                        | 形状 / dtype / 顺序 | 正确性     | Eager median | Candidate median |    加速比 | 结论                 |
| ------------------------------------------- | ------------------- | ---------- | -----------: | ---------------: | --------: | -------------------- |
| `torch.compile(..., backend="npugraphs")`   | 主形状 BF16 / AB    | 通过       |  `1.6800 ms` |      `2.5465 ms` | `0.6597×` | 变慢                 |
| `torch.compile(..., backend="npugraphs")`   | 主形状 BF16 / BA    | 通过       |  `1.4646 ms` |      `2.2937 ms` | `0.6385×` | 变慢                 |
| `torch.compile(..., backend="npugraph_ex")` | 主形状 BF16 / BA    | 通过       |  `1.5237 ms` |      `2.5587 ms` | `0.5955×` | 变慢                 |
| `aot_eager`                                 | 小形状 FP32 / BA    | 通过       |  `1.3625 ms` |      `2.3156 ms` | `0.5884×` | 调试基线，无性能收益 |
| TorchAir-GE                                 | 小形状 FP32 / BA    | **不通过** |  `1.2395 ms` |      `1.8691 ms` | `0.6631×` | 数值超差且变慢       |

`npugraphs` counters 为 `calls_captured=4`、`unique_graphs=1`、AOTAutograd `total=1/ok=1`，并记录 `npugraph_recorded_non_static_inputs=8`。它确实完成图捕获，而不是静默回退；对单个小卷积块而言，自动 AOT/图管理和非静态输入处理成本超过重放收益。

两条路径的捕获范围基本相同：都图化模块前向和反向，loss 与 optimizer 都留在 Eager。差异在重放热路径：`make_graphed_callables` 把模块包装成一个定制 autograd node，前向主要执行输入指针判断、必要的 `copy_()` 和 forward graph replay，反向主要复制传入梯度并 replay backward graph。自动 `npugraphs` 还要经过 Dynamo/AOTAutograd boxed callable 和 NPUGraph Tree；每步调用 `npugraph_mark_step_begin()`，并处理 generation/path、输入 data pointer、输出存活期、alias/mutation、静态输入不变量、非静态输入批量复制和输出 tensor 重建。它用额外运行时工作换取多调用路径和张量生命周期安全，在约 `1.5 ms` 的小块上成本大于省下的 kernel launch 开销。

TorchAir-GE 的 4 步 FP32 检查中，loss、状态和梯度最大绝对差分别为 `7.71e-5`、`1.08e-4` 和 `2.50e-2`，超出脚本判据；首次 candidate step 约 `6.24 s`。

`npugraphs` 使用 `zero_grad(set_to_none=False)` 时，在本脚本已验证的 mark 位置上都会在第二次 backward 触发 stale-output 错误。正式数据使用 `set_to_none=True`。

#### YOLO11-L 规模敏感性

为验证自动图管理成本能否被大网络摊薄，补测了 `25,372,160` 参数的 YOLO11-L。使用固定 `640×640`、batch 8、BF16 autocast、真实 detection loss 和 SGD momentum `0.9`，计时包含模型前向、loss、backward 和 optimizer step；输入在两个不同 `data_ptr()` 的合成 batch 间轮换，标签为每张图一个固定目标，不包含 dataloader。下表四次均在同一张物理 NPU0 上以独立进程顺序执行：

| 路径                                | `TASK_QUEUE_ENABLE` |   Median step | 相对 Queue=1 Eager | 相对 Queue=2 Eager |
| ----------------------------------- | ------------------: | ------------: | -----------------: | -----------------: |
| Eager                               |                   1 | `187.9525 ms` |          `1.0000×` |          `0.6857×` |
| 自动 `npugraphs`，`fullgraph=False` |                   1 | `148.9998 ms` |          `1.2614×` |          `0.8649×` |
| 当前 Ultralytics 默认 Eager         |                   2 | `128.8688 ms` |          `1.4585×` |          `1.0000×` |
| `make_graphed_callables`            |                   1 | `119.0750 ms` |          `1.5784×` |          `1.0822×` |

大网络确实改变了自动路径的结论：相对图捕获兼容的 Queue=1 Eager，自动 `npugraphs` 从小卷积块的变慢转为 `1.261×` 加速。测试时的 Ultralytics 会把 `TASK_QUEUE_ENABLE` 无条件设为 2，而 NPUGraph 捕获明确只支持 0/1；若不在 NPU 初始化前改回 1，自动路径会直接报错。本轮已将默认设置改为 `setdefault("TASK_QUEUE_ENABLE", "2")`，基准脚本也在导入 TorchNPU/Ultralytics 前显式设置队列，因此 Queue=1/2 可控且会写入 JSON。Queue=2 本身在同卡上将 Eager 加速到 `128.87 ms`，因此自动图相对实际默认基线仍慢 `15.6%`。手动整网捕获用 Queue=1 仍达到 `119.07 ms`，相对 Queue=2 Eager 加速 `1.082×`。

同一 YOLO11-L 负载还用 PyTorch allocator、CANN `aclrtGetMemUsageInfo` 和 `npu-smi` 三层口径复测了 Queue=1/2 的峰值内存：

| 内存口径                    |       Queue=1 |       Queue=2 | Queue=2 - Queue=1 |
| --------------------------- | ------------: | ------------: | ----------------: |
| PyTorch peak allocated      | `4961.89 MiB` | `4941.38 MiB` |      `-20.51 MiB` |
| PyTorch peak reserved       |    `5404 MiB` |    `5004 MiB` |        `-400 MiB` |
| CANN `APP memPeakSize`      |    `5404 MiB` |    `5562 MiB` |        `+158 MiB` |
| `npu-smi` 进程 HBM 采样峰值 |     `5738 MB` |     `5896 MB` |         `+158 MB` |

Queue=2 将动态 tiling、workspace 申请和 kernel 下发放入多级流水，CANN 记录到约 `558 MiB` 不在 PyTorch allocator 统计中的额外 APP/workspace（`5562-5004`）。与此同时，该执行时序让 PyTorch caching allocator 本轮少保留 `400 MiB`，所以进程总 HBM 的净峰值只增加 `158 MiB`，约为 Queue=1 APP 峰值的 `2.9%`。仅查看 `torch.npu.max_memory_allocated()` 会错误地得出“没有增加”的结论。额外 workspace 取决于算子 workspace、队列中并存任务和模型形态，`158 MiB` 只代表本轮 YOLO11-L、batch 8、640×640 BF16，不应视为所有模型的固定常数。

自动路径首次训练 step 约 `47.9 s`，并不是一个完整整网图：`npu.get_npu_format`/`npu.npu_format_cast` 缺少 FakeTensor/Meta 实现，`fullgraph=True` 会直接失败；允许 graph break 后共形成 30 个 AOT graph、27 次 format 相关 graph break、1040 个记录的非静态输入。图被切碎后，每段都承担 Graph Tree 管理成本，限制了大网络摊薄效果。手动路径不经过 FakeTensor tracing，整网 capture setup 为约 `1.77 s`，但仍出现 `AccumulateGrad stream mismatch` warning。

三条路径均完成了有限 loss、backward 和参数更新；4 步 BF16 状态对照中，候选与 Eager 存在训练轨迹差异，但两次跨卡 Eager 重复也出现相同量级差异，原因是 BF16 数值扰动会被 Task-Aligned Assigner 的 top-k 离散选择放大，现有数据不能将其归因于图语义错误。该补测仍是固定形状合成训练筛选，没有覆盖真实数据长训练收敛、DDP、梯度累积和多尺度，不能直接作为生产启用结论。

### Eager DVM

发布 wheel 中设置 `TORCH_NPU_LAZY_FUSION=True` 会被静默忽略：二进制不含 lazy-fusion marker，`torch_npu._C.dvm` 不存在，开启 dump 后没有输出 DVM graph/kernel。因此没有把该 wheel 的 on/off 结果误报为 DVM 性能。

为验证更新说明中的真实实现，从当天最新 `v2.12.0` 源码 `656c791722ca376248c5e927c1e36012f107a2a6` 构建了隔离 wheel `torch_npu-2.12.0+git656c791`。该 wheel 在 CANN `9.1.0-beta.3` 上成功构建并满足：

- `torch_npu._C.dvm=True`，二进制包含 `TORCH_NPU_LAZY_FUSION`；
- `TASK_QUEUE_ENABLE=1` 和 `TORCH_NPU_LAZY_FUSION="True dump_as_text ..."` 能生成 DVM graph/kernel；
- 官方三步训练和 BN backward output-mask 用例通过；
- 标准 Conv-BN-ReLU 的 4 步 FP32 on/off loss 逐项相同，BN 计数 `4/4`；参数/BN 状态、梯度和 momentum buffer 最大绝对差分别为 `1.49e-8`、`4.70e-8` 和 `1.02e-7`，严格判据通过。

主形状 `[32, 64, 56, 56]`、BF16 autocast、SGD momentum `0.9` 的完整训练 step 结果为：

| 物理 NPU | 顺序 | DVM off median | DVM on median | off/on 加速比 |
| -------: | ---- | -------------: | ------------: | ------------: |
|        0 | AB   |    `1.9406 ms` |   `1.9449 ms` |     `0.9978×` |
|        1 | BA   |    `2.0316 ms` |   `2.0020 ms` |     `1.0148×` |
|        4 | AB   |    `1.8813 ms` |   `1.8384 ms` |     `1.0233×` |
|        7 | BA   |    `1.9535 ms` |   `1.9300 ms` |     `1.0122×` |

四次 run-level 加速比简单平均为 `1.0120×`，中位数为 `1.0135×`；平均 off/on median 时延比为 `1.0119×`，3/4 次加速。两侧最终 loss 每次都完全相同，输出均为 BF16，峰值显存均为 `233.115 MiB`。收益只有约 1%，接近跨进程噪声尺度，不能视为强加速结论。

FP32 的一次 AB 敏感性测试中，off/on median 为 `1.6826/1.8432 ms`，DVM 只有 `0.9128×`，明显变慢。标准块的实际 ATen 路径中，Conv/Conv backward 不支持 Eager DVM，训练态 BN forward 会回退，ReLU backward 也构成边界；只有 BN backward 的部分内部计算及相邻点算可能获益，符合实测收益有限的结果。

### FFN 与 LayerNorm 敏感性

为检验更符合 Transformer/ViT 的负载，使用同一源码 wheel 测了两个 BF16 完整训练块：

- ViT GELU-MLP：`LayerNorm(768) -> Linear(768,3072,bias=True) -> GELU(tanh) -> Linear(3072,768,bias=True) -> residual add`，输入 `[32,196,768]`。
- Transformer SwiGLU：`LayerNorm(1024) -> gate/up Linear(1024,2816,bias=False) -> SiLU(gate)*up -> down Linear(2816,1024,bias=False) -> residual add`，输入 `[4,512,1024]`。

两者都使用 SGD momentum `0.9`，计时包含 forward、loss、backward 和 optimizer step。初始逐进程 AB/BA 结果存在明显设备热状态和顺序漂移，因此正式数据采用同步双卡交叉：第一轮在两张卡同时跑 off/on，第二轮交换卡上的模式；每个进程预热 500 step，再测 `20 step × 20` 组。

| 负载               | 物理 NPU | DVM off median | DVM on median |    off/on |
| ------------------ | -------: | -------------: | ------------: | --------: |
| ViT GELU-MLP       |        0 |    `2.8405 ms` |   `2.8217 ms` | `1.0067×` |
| ViT GELU-MLP       |        1 |    `3.0046 ms` |   `2.9708 ms` | `1.0114×` |
| Transformer SwiGLU |        4 |    `3.2329 ms` |   `2.6861 ms` | `1.2036×` |
| Transformer SwiGLU |        7 |    `3.0556 ms` |   `2.9862 ms` | `1.0232×` |

双卡加速比几何平均：

- ViT GELU-MLP：median `1.0090×`，mean `1.0109×`，基本持平。
- Transformer SwiGLU：median `1.1097×`，mean `1.1066×`，有明确收益；但单卡范围为 `1.023×–1.204×`，幅度对卡和平台状态敏感。

独立 111 步 on/off 状态比较中，ViT 的 loss 完全相同，模型状态、梯度和 momentum 最大绝对差分别为 `4.47e-8/4.77e-7/1.25e-6`；SwiGLU 对应为 `3.91e-8/2.38e-7/3.87e-7`，两者均通过 BF16 训练判据。

DVM dump 解释了差异：

- 原生 LayerNorm forward/backward 没有 DVM 注册，两种网络都会在 LayerNorm 处 flush 和回退。
- ViT 主要生成彼此分开的 `addmm`、GELU/GELU backward 和 residual/loss 点算图，跨算子融合有限。
- SwiGLU 生成 `matmul+silu`、`matmul+mul`、`matmul+residual add+loss pow` 等融合段，并融合部分 `silu_backward`/乘法反向，能实质减少 launch 和中间张量读写。
- MatMul backward 本身仍不支持 DVM；每个新的 MatMul 也会先 flush，因此 DVM 不会把多个独立 GEMM 合成一个 kernel。

所以，对于 FFN+LN，不能把“有 FFN”视为充分条件：显式 SwiGLU pointwise 链是本轮有效候选，普通 GELU-MLP 基本没有稳态收益，LayerNorm 自身也不会被加速。

## Inductor 兼容性筛选

小形状 `[4, 16, 32, 32]`、输出通道 16 的 FP32 筛选得到：

| 方案                                        | 结果         | 原因 / 加速比                                                                           |
| ------------------------------------------- | ------------ | --------------------------------------------------------------------------------------- |
| Inductor-Triton（发布 wheel + `3.2.1`）     | 编译失败     | TorchNPU 调用无参 `CantSplit()`，PyTorch 2.12 要求 `CantSplit(expr, remaining)`         |
| Inductor-Triton（最新源码 wheel + `3.2.1`） | 编译失败     | 真正进入 Triton 调度，但仍触发同一 `CantSplit` 私有 API 不匹配                          |
| Inductor `reduce-overhead`                  | 编译失败     | 同一 `CantSplit` 私有 API 不匹配                                                        |
| Inductor + 进程内 `CantSplit` 兼容补丁      | 编译失败     | BN+ReLU backward Triton Python 代码缩进无效，触发 `IndentationError`                    |
| Inductor + all-fallback + `reduce-overhead` | 可训练但变慢 | `0.7004×`；全部算子回退后绕开代码生成，失去编译收益                                     |
| Inductor-DVM（发布 wheel）                  | 未真正启用   | wheel 未注册 DVM loader，`npu_backend="dvm"` 实际回落到 Triton；此前断言不是 DVM kernel |
| Inductor-DVM（最新源码 wheel）              | 无法启动     | 已进入真正 DVM loader，但当前 Python 3.12 无 `torch_mlir`，触发 `NameError: ir`         |
| Inductor-MLIR                               | 无法启动     | 当前无 `torch_mlir`；文档 aarch64 wheel 目录只有 Python 3.10/3.11，无 Python 3.12 wheel |

这里的“不可用”只针对当前软件栈上的 `torch.compile` 标准 Conv-BN-ReLU 训练，不表示 Triton-Ascend 或 DVM 对所有算子、所有调用方式都不可用。尤其要区分两条 DVM 路径：Inductor-DVM 是 `torch.compile` 的代码生成后端，本环境无法启动；Eager DVM 不经过 `torch.compile`，最新源码 wheel 已验证可以训练和生成 DVM kernel，只是该卷积块收益很小。

还在隔离 venv 中安装 `torch==2.10.0` 与 `torch_npu==2.10.0.post4`。该组合不再触发无参 `CantSplit`，但仍为 BN 训练融合核生成错误缩进的 Python 代码并以 `IndentationError` 失败，因此不建议为此降级主环境。

### Triton-Ascend 版本复核

隔离构建的 `3.2.2+gitc22984ba` 在同一小形状上得到：

1. 不加兼容补丁时仍在 Triton 编译前触发无参 `CantSplit()` 错误，说明第一处阻塞属于 `torch_npu` 与 PyTorch Inductor 私有 API 的不匹配。
2. 加 `CantSplit` 兼容补丁后，仍生成与 `3.2.1` 相同的错误缩进：BN backward 融合核的 `r2 = ...` 等语句无对应代码块却额外缩进，Python 在调用 Triton 编译器前即报 `IndentationError`。
3. 绕开 Inductor 直接运行 Triton-Ascend 官方 vector-add 时，Triton 内核已编译，但 NPU launcher 头文件编译失败。`torch_npu 2.12.0` 的 `AclInterface.h` 要求 `aclmdlRICondHandle`、`aclmdlRICondTaskParams`，当前 CANN `9.1.0-beta.3` 的 ACL 头文件不含这些类型。

因此，把 Triton-Ascend 从稳定版 `3.2.1` 单独更新到未发布的 `3.2.2` 源码不能修复方案 1。公开可下载的最新 CANN 社区版就是当前 `9.1.0-beta.3`，并不存在可直接升级的 9.1 正式社区包。当天最新 TorchNPU 源码 wheel 配合稳定版 Triton-Ascend `3.2.1` 的复测也仍在 `CantSplit` 处失败。该源码 wheel 已能在 beta3 上构建并运行 Eager DVM，但 Inductor-Triton 仍需修复 `CantSplit` 和 BN reduction 代码生成；只替换 Triton-Ascend 不足以进入可测性能的状态。

## 推荐用法

与最终基准一致的关键模式如下：

```python
import torch
import torch_npu

model = ConvBNReLU().npu().train()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, foreach=False)

# 捕获会执行 warmup/前反向，先保存初始状态；捕获样例不再作为运行 batch 使用。
initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
capture_input = torch.randn(32, 64, 56, 56, device="npu")
with torch.autocast("npu", dtype=torch.bfloat16, cache_enabled=False):
    graphed_model = torch_npu.npu.make_graphed_callables(model, (capture_input,))
model.load_state_dict(initial_state)
optimizer.zero_grad(set_to_none=True)

for batch in loader:
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("npu", dtype=torch.bfloat16, cache_enabled=False):
        output = graphed_model(batch)
        loss = loss_fn(output)
    loss.backward()
    optimizer.step()
```

必须满足：

1. 捕获和重放使用同一 autocast dtype，且 `cache_enabled=False`。只在重放时开启 autocast 不会改变已经捕获的图。
2. 捕获样例与运行 batch 分离；新 batch 的 shape、dtype 和 device 必须匹配。wrapper 会在指针不同时把数据 `copy_()` 到静态输入。
3. 捕获后恢复模型状态并清理梯度，否则 BN running stats 和 `.grad` 会包含捕获副作用。
4. 固定 shape 和图结构。多尺度训练需要按 shape 管理不同的 graphed callable。
5. 优化器留在 Eager，这也符合 [Inductor 约束](../../../torch-compile/pytorch_inductor_desc.md#约束说明) 和 [NPUGraph_EX 约束](../../../torch-compile/pytorch_npugraph_ex_desc.md)。
6. 在完整任务上验证收敛、显存和吞吐后再扩大使用。

## 复现命令

主测试 AB：

```bash
CONV_BENCH_CACHE=$(mktemp -d /tmp/convbench-XXXXXX)
export ASCEND_RT_VISIBLE_DEVICES=0
export TASK_QUEUE_ENABLE=1
export TORCHINDUCTOR_CACHE_DIR="$CONV_BENCH_CACHE/inductor"
export TORCHINDUCTOR_COMPILE_DIR="$CONV_BENCH_CACHE/compile"
export TRITON_CACHE_DIR="$CONV_BENCH_CACHE/triton"

python npu-opt/优化记录/torch_compile/基准/benchmark_conv_bn_relu.py \
    --scheme make_graphed_callables \
    --device 0 \
    --task-queue 1 \
    --shape 32,64,56,56 \
    --out-channels 64 \
    --input-variants 2 \
    --dtype amp_bf16 \
    --correctness-steps 10 \
    --warmup-steps 20 \
    --measure-steps 20 \
    --repeats 10 \
    --performance-order eager_first \
    --phase both \
    --output /tmp/convbench-ab.json
```

将 `--performance-order` 改为 `candidate_first` 并在独立进程运行，即得到 BA 配对。主形状 `npugraphs`/`npugraph_ex` 只需替换 `--scheme`。

高对比输入的 correctness-only 检查命令为：

```bash
python npu-opt/优化记录/torch_compile/基准/benchmark_conv_bn_relu.py \
    --scheme make_graphed_callables --device 0 \
    --shape 4,16,32,32 --out-channels 16 --dtype amp_bf16 \
    --input-variants 2 --input-offset 5 --correctness-steps 6 \
    --phase correctness --output /tmp/convbench-high-contrast.json
```

AOT_Eager 和 TorchAir-GE 表格行使用以下参数：

```bash
SMALL_ARGS=(
    --device 0 --shape 4,16,32,32 --out-channels 16 --dtype fp32
    --correctness-steps 4 --warmup-steps 10 --measure-steps 10 --repeats 5
    --performance-order candidate_first --phase both
)
python npu-opt/优化记录/torch_compile/基准/benchmark_conv_bn_relu.py \
    --scheme aot_eager "${SMALL_ARGS[@]}" --output /tmp/convbench-aot.json
python npu-opt/优化记录/torch_compile/基准/benchmark_conv_bn_relu.py \
    --scheme torchair_ge "${SMALL_ARGS[@]}" --output /tmp/convbench-torchair.json
```

Inductor 筛选使用更短的两步参数；每条命令都应像主命令一样在独立进程和新 cache 中运行：

```bash
INDUCTOR_ARGS=(
    --device 0 --shape 4,16,32,32 --out-channels 16 --dtype fp32
    --correctness-steps 2 --warmup-steps 2 --measure-steps 2 --repeats 2
    --performance-order candidate_first --phase both
)

# 复现 CantSplit API 错误。
python npu-opt/优化记录/torch_compile/基准/benchmark_conv_bn_relu.py \
    --scheme inductor "${INDUCTOR_ARGS[@]}" --output /tmp/convbench-inductor.json

# 仅验证 CantSplit API 不匹配之后的下一个失败点。
python npu-opt/优化记录/torch_compile/基准/benchmark_conv_bn_relu.py \
    --scheme inductor --cantsplit-compat "${INDUCTOR_ARGS[@]}" \
    --output /tmp/convbench-inductor-compat.json

# 仅验证全部算子回退，不代表 Inductor 生成了 NPU kernel。
NPU_INDUCTOR_FALLBACK_LIST=allfallback \
    python npu-opt/优化记录/torch_compile/基准/benchmark_conv_bn_relu.py \
    --scheme inductor_reduce_overhead --cantsplit-compat "${INDUCTOR_ARGS[@]}" \
    --output /tmp/convbench-inductor-fallback.json
```

将第一条命令的 `--scheme` 替换为 `inductor_reduce_overhead` 或 `inductor_dvm`，可分别复现另外两个失败路径。

Eager DVM 必须在安装包含 DVM binding 的源码 wheel 后，以独立进程分别运行 on/off。Conv、ViT GELU-MLP 和
Transformer SwiGLU 的命令只需替换 `--workload`：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python npu-opt/优化记录/torch_compile/基准/benchmark_eager_dvm.py \
    --workload conv_bn_relu --lazy-fusion off --task-queue 1 \
    --dtype amp_bf16 --warmup-steps 500 --measure-steps 20 --repeats 20 \
    --output /tmp/dvm-conv-off.json
ASCEND_RT_VISIBLE_DEVICES=0 python npu-opt/优化记录/torch_compile/基准/benchmark_eager_dvm.py \
    --workload conv_bn_relu --lazy-fusion on --task-queue 1 \
    --dtype amp_bf16 --warmup-steps 500 --measure-steps 20 --repeats 20 \
    --output /tmp/dvm-conv-on.json
```

将 `conv_bn_relu` 替换为 `vit_gelu` 或 `transformer_swiglu` 可复现 FFN+LayerNorm 测试；交换 on/off
顺序并交换物理 NPU 可组成报告中的 AB/BA 与双卡交叉设计。使用 `--state-output` 可在独立正确性 run
保存模型、梯度和 optimizer 状态。

## 限制

- 这是单个 Conv-BN-ReLU 块的微基准，不代表完整 YOLO 模型的端到端加速比。
- 只测试单卡、固定 shape；未测试 DDP、SyncBatchNorm、梯度累积、动态 shape、多尺度训练或完整收敛曲线。
- CANN 为 `9.1.0-beta.3`，不是最终商用 `9.1.0`；升级后应重跑。
- FP32 仅做一次敏感性测试；未用 FP16/GradScaler 得出长训练数值结论。
- 统计来自当前机器当日状态。AB/BA、多卡独立进程和 block 中位数降低了偏差，但不能替代目标任务实测。

终审还专门检查了捕获期 autocast：早期探索中“捕获在 FP32、只在重放开启 BF16”的数据已全部废弃，未进入本报告。最终脚本记录 Eager/Candidate 输出 dtype，并将 dtype、有限性、结构和 dtype-specific 容差纳入 `correctness_pass`，防止再次混淆不同精度路径。
