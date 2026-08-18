# CANN 9.1.0 正式版下大型 Conv-BN-ReLU 训练编译测评

- 测试日期：2026-08-02
- 硬件：Ascend 910B2
- 复现脚本：[benchmark_conv_bn_relu.py](../基准/benchmark_conv_bn_relu.py)
- 历史 beta3 报告：[conv_bn_relu_910b2_benchmark_report.md](./conv_bn_relu_910b2_benchmark_report.md)
- 后续临时兼容实验：[reduce_overhead_conv_yolo_rtdetr_910b2_20260803.md](./reduce_overhead_conv_yolo_rtdetr_910b2_20260803.md)

> 本文记录未应用项目内临时兼容层时的结论。2026-08-03 的后续实验曾让规则 Conv-BN-ReLU 堆叠通过 Inductor，但该实现未保留，不能据此认为当前代码已支持该路径。

## 结论

当前软件栈已经修复旧 CANN 头文件导致的 Triton launcher 编译问题，直接 Triton-Ascend 内核可以正常编译和执行；但 `torch.compile(..., backend="inductor")` 训练大型 Conv-BN-ReLU 堆叠仍不可用。阻塞点已经从 CANN/launcher 前移到 TorchNPU 与 PyTorch Inductor 私有接口、BN 双 reduction 代码生成及 Triton frontend 配置生成。

生产建议仍是 **Eager + `TASK_QUEUE_ENABLE=2`**。固定形状且能够接受 Queue1 时，`npugraphs` 相对 Queue1 Eager 有约 `1.10–1.11×` 加速，但仍比本机最佳 Queue2 Eager 慢约 `3–4%`。AOT Eager 可以正确训练但没有加速；TorchAir 虽显示约 `1.39×`，却破坏训练态 BN 状态和 backward 正确性，不能使用。

## 软件栈与负载

| 组件 | 版本 |
| --- | --- |
| CANN | `9.1.0`，内部版本 `V100R001C25B114` |
| 910B 算子包 | `Ascend-cann-910b-ops 9.1.0`，内部版本 `V100R001C11SPC001B243` |
| PyTorch | `2.12.0+cpu` |
| torch_npu | `2.12.0`，git `fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d` |
| Triton distribution | `3.5.0` |
| Triton-Ascend distribution | `3.2.2` |
| `triton.__version__` | `3.2.0` |

主负载由 9 个训练态 `Conv2d(bias=False) → BatchNorm2d → ReLU` 串联组成：输入 `[8,64,28,28]`，隐藏/输出通道 512，共 `19,178,496` 个参数；使用 AMP FP16、SGD、固定 shape，计时覆盖 forward、loss、backward 和 optimizer step。候选均与独立 Eager 初始化进行 loss、输出、参数、BN buffer、梯度和 optimizer 状态对照。

## 结果

| 路径 | 正确性 | 稳态结果 | 判断 |
| --- | --- | --- | --- |
| Eager，Queue2，internal format 关 | 通过 | `7.613 ms/step`，`1050.9 img/s` | **当前推荐基线** |
| AOT Eager，Queue2 | 通过 | AB `7.953 ms`，BA `7.597 ms`；相对配对 Eager 为 `0.939×/0.979×` | 可用但变慢 |
| `npugraphs`，Queue1，internal format 关 | 通过；4 步输出完全一致，状态/梯度通过 | AB/BA 为 `1.100×/1.112×`，candidate `7.976/7.758 ms` | Queue1 下有效，仍略慢于 Queue2 Eager |
| `npugraph_ex`，Queue1，internal format 关 | AMP 训练状态和梯度在容差内；输出最大绝对差 `0.0134` | AB/BA 为 `1.226×/1.081×`，candidate `8.199/8.158 ms` | 比最佳 Eager 慢约 `7.4%`，不推荐 |
| TorchAir / `backend="npu"` | **失败** | 表面约 `1.39×` | BN running stats 和 backward 错误，拒绝 |
| Inductor 默认 / `reduce-overhead` | 编译失败 | 无有效时延 | 当前不可用 |
| Inductor MLIR | 启动失败 | 缺少 `torch_mlir` | 当前不可用 |
| Inductor DVM | 未真正注册 | 当前 wheel 静默落回 Triton | 当前不可用 |
| 直接 Triton-Ascend vector-add | 精确通过 | 可编译、可执行 | 证明 CANN launcher 链路已修复 |

AOT Eager 的三步正确性检查中，loss 最大绝对差 `8.05e-7`、模型状态最大绝对差 `8.15e-6`、梯度最大绝对差 `9.36e-5`，均通过 AMP FP16 判据。其 candidate 峰值 allocated 为 `286.28 MiB`，与 Eager 的 `286.23 MiB` 基本一致。

材料迁移后又在正式版环境做了轻量复核：扩展后的脚本构造出 `19,178,496` 参数，Eager 对照的 loss、状态和梯度差均为 0，`correctness_pass=true`；直连 Triton vector-add 的 `max_abs_error=0`、`allclose=true`。

`npugraph_ex` 的 candidate 峰值 allocated/reserved 为 `544/836 MiB`，相对同组 Queue1 Eager 的 `368/456 MiB` 明显增加。`npugraphs` candidate 为约 `360/530 MiB`，内存代价小得多。

## Inductor 不可用原因

1. 未加 workaround 时，TorchNPU 仍按旧接口调用无参 `CantSplit()`，而 PyTorch 2.12 要求 `CantSplit(expr, remaining)`。
2. 进程内兼容 `CantSplit` 后，BN backward 的双 reduction 会生成缩进错误并引用未正确构造的中间变量，Python 代码在 Triton 编译前即失败。
3. 继续绕过该处后，Triton frontend 还会在配置生成阶段触发 `IndexError`/`NoTritonConfigs`。
4. `reduce-overhead` 与默认 Inductor 共用上述代码生成链，因此不能绕开问题。
5. `NPU_INDUCTOR_FALLBACK_LIST=allfallback` 虽能让训练运行，却把算子全部回退并失去代码生成收益，不属于可用的 Inductor 加速方案。

CANN 9.1.0 正式版和正确的 910B 算子包已经让官方 Triton vector-add 正常工作，说明不再是 910/910B 算子包或 ACL launcher 头文件问题。要修复 `torch.compile`，需要 TorchNPU/Inductor 适配 PyTorch 2.12 的 `CantSplit` API，并修正 BN reduction 与 Triton config 代码生成；只继续更新 CANN 或 Triton-Ascend 小版本不足以解决。

## 其他后端说明

- 当前 torch_npu wheel 没有注册可观察的 Eager DVM binding；设置 lazy-fusion 环境变量不会生成 DVM graph/kernel。`inductor_dvm` 也没有加载真实 DVM backend，因此不能把其失败或性能归因于 DVM。
- TorchAir 会减少 step 时延，但 1/2/4 步检查都出现 BN running mean/variance 被部分错误更新，不能依靠放宽数值容差接受。
- `npugraphs` 只适合固定 shape，并要求 Queue0/1。多尺度训练、动态 batch、DDP 和真实长训练仍需单独验证。

## 复现

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0
export TASK_QUEUE_ENABLE=2

python docs/my-docs/npu-opt/优化记录/torch_compile/基准/benchmark_conv_bn_relu.py \
    --scheme aot_eager \
    --device 0 \
    --task-queue 2 \
    --shape 8,64,28,28 \
    --out-channels 512 \
    --blocks 9 \
    --dtype amp_fp16 \
    --correctness-steps 3 \
    --warmup-steps 10 \
    --measure-steps 5 \
    --repeats 6 \
    --phase both
```

将 `--scheme` 替换为 `inductor`、`inductor_reduce_overhead`、`npugraphs` 或 `npugraph_ex` 可复现相应路径。图捕获测试应在独立进程中设置 Queue1，并使用新的 Inductor/Triton cache；不要在一个已初始化为 Queue2 的进程中切换。

本报告仅针对固定 shape 的标准卷积堆叠训练。它足以判断当前编译后端可用性，但不能代替 YOLO11-L/RT-DETR-L 真实数据、多尺度、DDP 和长训练收敛验证。
