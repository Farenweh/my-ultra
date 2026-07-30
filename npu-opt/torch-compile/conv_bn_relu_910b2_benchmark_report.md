# Ascend 910B2 Conv-BN-ReLU 训练编译/图化测评报告

- 测试日期：2026-07-30
- 代码基线：`0567fa91236a1f328a2fcfe2b592d1341b854041`
- 可复现脚本：[benchmark_conv_bn_relu.py](./benchmark_conv_bn_relu.py)

## 结论

当前 910B2 软件栈上，唯一同时满足本次“训练正确且稳态加速”判据的方案是：

> 用 `torch_npu.npu.make_graphed_callables()` 捕获 Conv-BN-ReLU 模块的前向和反向，优化器留在 Eager；固定 shape，并在捕获和重放时使用相同的精度上下文。主测试使用 BF16 autocast，FP32 也能加速。

主形状 `[32, 64, 56, 56]` 的最终结果如下：

- 3 张物理 NPU、6 个独立进程的 BF16 AB/BA 配对测试全部加速，范围为 `1.083×–1.304×`，简单平均为 `1.183×`，中位数为 `1.178×`。
- 为排除“Eager 总是先跑”的顺序偏差，Candidate-first（BA）三次也全部加速，范围为 `1.083×–1.149×`，平均为 `1.126×`。
- 6 次正确性检查的 Eager/Candidate 前向输出都确认为 `torch.bfloat16`；10 步 loss 逐项相同，BN 计数均为 `10/10`，状态最大绝对差为 `3.73e-9`，梯度最大绝对差不超过 `9.54e-7`。
- `SGD(momentum=0.9)` 的独立测试为 `1.214×`，状态、梯度和 momentum buffer 均通过自动判据。
- FP32 的一次 Candidate-first 敏感性测试也训练正确并取得 `1.135×`。BF16 在这个孤立块上并不比 FP32 绝对更快，不应只为该微基准切换训练 dtype。

该 API 不是 `torch.compile` backend，但它是 [NPUGraph 高级图化接口](../框架特性/pytorch_npugraph_desc.md)。[NPUGraphs 文档](./pytorch_compile_npugraph_desc.md) 也将细粒度控制指向该路径。自动 `torch.compile(..., backend="npugraphs")` 和 `npugraph_ex` 虽然训练正确，却分别只有约 `0.64×–0.66×` 和 `0.60×`，不适合这个粒度的块。

本结论只覆盖单卡微基准和短程状态等价性，不等价于完整 YOLO 长训练收敛验证。

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

测试开始时 8 张卡均为 `Health=OK`。NPU3 有历史可纠正 HBM 单比特计数 `24`，当前计数、双比特错误和隔离页均为 0；性能测试使用 NPU0/1/2/4，避开 NPU3。

环境有两项值得注意的版本风险：

1. 重新下载的 `torch_npu==2.12.0` wheel 与已安装 wheel 的 SHA256 完全相同，但它与 `torch==2.12.0` 的部分 Inductor 私有 API 不匹配。
2. `triton` distribution metadata 为 `3.5.0`，而实际导入 runtime 为 `3.2.0`；这是 Triton-Ascend 覆盖同名 package 目录后的状态。

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

TorchAir-GE 的 4 步 FP32 检查中，loss、状态和梯度最大绝对差分别为 `7.71e-5`、`1.08e-4` 和 `2.50e-2`，超出脚本判据；首次 candidate step 约 `6.24 s`。

`npugraphs` 使用 `zero_grad(set_to_none=False)` 时，在本脚本已验证的 mark 位置上都会在第二次 backward 触发 stale-output 错误。正式数据使用 `set_to_none=True`。

## Inductor 兼容性筛选

小形状 `[4, 16, 32, 32]`、输出通道 16 的 FP32 筛选得到：

| 方案                                        | 结果         | 原因 / 加速比                                                                           |
| ------------------------------------------- | ------------ | --------------------------------------------------------------------------------------- |
| Inductor-Triton                             | 编译失败     | TorchNPU 调用无参 `CantSplit()`，PyTorch 2.12 要求 `CantSplit(expr, remaining)`         |
| Inductor `reduce-overhead`                  | 编译失败     | 同一 `CantSplit` 私有 API 不匹配                                                        |
| Inductor + 进程内 `CantSplit` 兼容补丁      | 编译失败     | BN+ReLU backward Triton Python 代码缩进无效，触发 `IndentationError`                    |
| Inductor + all-fallback + `reduce-overhead` | 可训练但变慢 | `0.7004×`；全部算子回退后绕开代码生成，失去编译收益                                     |
| Inductor-DVM                                | 编译失败     | scheduler tiling/fusion 路径触发 `AssertionError`                                       |
| Inductor-MLIR                               | 无法启动     | 当前无 `torch_mlir`；文档 aarch64 wheel 目录只有 Python 3.10/3.11，无 Python 3.12 wheel |

还在隔离 venv 中安装 `torch==2.10.0` 与 `torch_npu==2.10.0.post4`。该组合不再触发无参 `CantSplit`，但仍为 BN 训练融合核生成错误缩进的 Python 代码并以 `IndentationError` 失败，因此不建议为此降级主环境。

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
5. 优化器留在 Eager，这也符合 [Inductor 约束](./pytorch_inductor_desc.md#约束说明) 和 [NPUGraph_EX 约束](./pytorch_npugraph_ex_desc.md)。
6. 在完整任务上验证收敛、显存和吞吐后再扩大使用。

## 复现命令

主测试 AB：

```bash
CONV_BENCH_CACHE=$(mktemp -d /tmp/convbench-XXXXXX)
export ASCEND_RT_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR="$CONV_BENCH_CACHE/inductor"
export TORCHINDUCTOR_COMPILE_DIR="$CONV_BENCH_CACHE/compile"
export TRITON_CACHE_DIR="$CONV_BENCH_CACHE/triton"

python npu-opt/torch-compile/benchmark_conv_bn_relu.py \
    --scheme make_graphed_callables \
    --device 0 \
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
python npu-opt/torch-compile/benchmark_conv_bn_relu.py \
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
python npu-opt/torch-compile/benchmark_conv_bn_relu.py \
    --scheme aot_eager "${SMALL_ARGS[@]}" --output /tmp/convbench-aot.json
python npu-opt/torch-compile/benchmark_conv_bn_relu.py \
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
python npu-opt/torch-compile/benchmark_conv_bn_relu.py \
    --scheme inductor "${INDUCTOR_ARGS[@]}" --output /tmp/convbench-inductor.json

# 仅验证 CantSplit API 不匹配之后的下一个失败点。
python npu-opt/torch-compile/benchmark_conv_bn_relu.py \
    --scheme inductor --cantsplit-compat "${INDUCTOR_ARGS[@]}" \
    --output /tmp/convbench-inductor-compat.json

# 仅验证全部算子回退，不代表 Inductor 生成了 NPU kernel。
NPU_INDUCTOR_FALLBACK_LIST=allfallback \
    python npu-opt/torch-compile/benchmark_conv_bn_relu.py \
    --scheme inductor_reduce_overhead --cantsplit-compat "${INDUCTOR_ARGS[@]}" \
    --output /tmp/convbench-inductor-fallback.json
```

将第一条命令的 `--scheme` 替换为 `inductor_reduce_overhead` 或 `inductor_dvm`，可分别复现另外两个失败路径。

## 限制

- 这是单个 Conv-BN-ReLU 块的微基准，不代表完整 YOLO 模型的端到端加速比。
- 只测试单卡、固定 shape；未测试 DDP、SyncBatchNorm、梯度累积、动态 shape、多尺度训练或完整收敛曲线。
- CANN 为 `9.1.0-beta.3`，不是最终商用 `9.1.0`；升级后应重跑。
- FP32 仅做一次敏感性测试；未用 FP16/GradScaler 得出长训练数值结论。
- 统计来自当前机器当日状态。AB/BA、多卡独立进程和 block 中位数降低了偏差，但不能替代目标任务实测。

终审还专门检查了捕获期 autocast：早期探索中“捕获在 FP32、只在重放开启 BF16”的数据已全部废弃，未进入本报告。最终脚本记录 Eager/Candidate 输出 dtype，并将 dtype、有限性、结构和 dtype-specific 容差纳入 `correctness_pass`，防止再次混淆不同精度路径。
