# 融合优化器

优化器需要对大量参数、梯度和动量状态执行相似操作。逐个处理小张量会产生较多 Host 到 Device 的任务下发和算子调度开销。TorchNPU 的 `NpuFusedAdamW` 将同 dtype 的参数和梯度合并到连续的 NPU 缓冲区，再按参数组执行大张量更新，从而减少调度次数。AdamW 的计算仍在 NPU 上执行，这里的下发不能理解为每步将模型参数搬回 CPU 计算。

此实现保留参数组各自的学习率、weight decay 等配置，合并参数、梯度和一阶/二阶动量存储，并重绑相关张量的数据指针。它需要较大的连续缓冲区，可能增加峰值显存占用。融合后应保持参数对象稳定，DDP 也需要避免让梯度成为 bucket view；本仓库在融合优化器模式下使用 `gradient_as_bucket_view=False`。

当前昇腾AI处理器支持NPU亲和的融合优化器功能，针对表1中的优化器做了昇腾AI处理器亲和性优化，使性能提升。

**表 1**  融合优化器使用方法

|**优化器名称**|**样例原代码**|**修改后代码**|
|--|--|--|
|sgd|optimizer = **torch.optim.SGD**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedSGD**(model.parameters(), lr=args.lr)|
|adadelta|optimizer = **torch.optim.Adadelta**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedAdadelta**(model.parameters(), lr=args.lr)|
|lamb|optimizer = **torch.optim.Lamb**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedLamb**(model.parameters(), lr=args.lr)|
|adamw|optimizer = **torch.optim.AdamW**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedAdamW**(model.parameters(), lr=args.lr)|
|adamp|optimizer = **torch.optim.AdamP**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedAdamP**(model.parameters(), lr=args.lr)|
|bert_adam|optimizer = **BertAdam**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedBertAdam**(model.parameters(), lr=args.lr)|
|rmsprop|optimizer = **torch.optim.RMSprop**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedRMSprop**(model.parameters(), lr=args.lr)|
|rmsprop_tf|optimizer = **torch.optim.RMSpropTF**(model.parameters(), lr=args.lr)|optimizer = **torch_npu.optim.NpuFusedRMSpropTF**(model.parameters(), lr=args.lr)|

> **说明：** 上表沿用迁移示意。`Lamb`、`AdamP`、`RMSpropTF`、`BertAdam` 并非当前 `torch.optim` 的原生接口，需要使用对应第三方实现；`BertAdam` 常见于 `pytorch_pretrained_bert.optimization`。

## 本地实测：小通道 BN 与融合存储布局

以下是 2026-10-18 在 CANN 9.2.0、PyTorch 2.14.0、TorchNPU 2.14.0（产品版 26.2.0）、Ascend 910B2 上的排查与仓库修复记录。最终方案使用未修改的原版 TorchNPU，在创建融合优化器前准备正确的 BN 参数物理布局。

### 第一层故障：参数组长度与组合缓冲区不一致

YOLO26n 默认 FP16 AMP、NpuFusedAdamW、内部格式开启时，首次反缩放梯度就报：

```text
RuntimeError: (index + size) (2572296) > torch_npu.get_storage_size(combined_tensor) (2572280)
```

真实调用链为 `GradScaler.unscale_()` → `_maybe_init_combined_params_and_grads()` → `get_part_combined_tensor()`。这是组合缓冲区的长度不一致；本例没有耗尽显存。

| 计量方式 | 读取内容 | 原实现的用途 |
| --- | --- | --- |
| `tensor.storage().size()` | 底层 storage 字节数除以元素大小 | 计算参数组切片长度 |
| `torch_npu.get_storage_size(tensor)` | NPU 描述符 `storage_sizes_` 各维乘积 | 分配组合缓冲区、推进指针和检查切片边界 |

逐张量记录发现，YOLO26n 只有两个 BN 梯度存在差值；对应参数自身的两种长度都为 8。

| 梯度 | shape | `storage().size()` | `get_storage_size()` |
| --- | --- | ---: | ---: |
| `model.2.m.0.cv1.bn.weight.grad` | `[8]` | 16 | 8 |
| `model.2.m.0.cv1.bn.bias.grad` | `[8]` | 16 | 8 |

二者都是连续的 FP32 梯度、format=2、storage offset=0，且不共享 storage。全模型梯度按旧口径累计 2,572,296，按描述符累计 2,572,280，正好多算 `8+8=16`。`.contiguous()` 对这些已经连续的张量不起作用。

Pose 模型还存在 12 个 BN85：各自 weight/bias 梯度的底层 storage 为 96、描述符为 85，另多算 `12×2×11=264`；连同骨干 BN8 的 16，精确解释了该模型的 280 元素越界。不能仅处理某个特定 BN8 层。

### 第二层故障：只改长度会掩盖相邻参数覆盖

曾尝试把 fused base 的 `param_size` / `grad_size` 两行统一为 `get_storage_size()`。它消除了运行异常，完整短训、保存重载和推理也能完成，但严格数值对照未通过：

- 第一步融合与普通 AdamW 完全一致。
- 第二次 `BatchNorm2d.forward` 之后、优化器更新之前，BN.bias 的 8 个值以及紧随其后参数的前 8 个值已被清零。
- 梯度值、一阶/二阶动量仍基本一致，参数差可达 0.1 以上。首次 clone 异常梯度也无法阻止覆盖。
- 同一现象在 SGD、Adam、RMSprop 上独立复现，因此不能只按“loss 有限、参数有变化、没有异常”验收。

这些阶段快照均使用同步 CPU 副本，并用逐层 hook 锁定 BN 前向。两行候选已经撤销，原始依赖已恢复；候选脚本已归档停用，不能作为修复重放。

官方实现解释了布局冲突的来源：输入为 `NC1HWC0` 而 BN 参数为 ND 时，BN 会临时对 weight/bias 等执行 `unsafe_format_cast`；该 helper 修改 NPU 格式和存储描述符，不重新分配底层 storage。从 C0 格式返回 ND 时，也可能留下“底层 16、描述符 8”的梯度。原版模块迁移逻辑在 `jit_compile=False` 时提前返回，跳过了 BN 参数原本的格式 3 准备。

结合实测，紧密融合的 ND 小参数槽没有为这条 BN 路径保留 C0 对齐区域，后续 BN 执行会覆盖相邻槽。官方源码证明存在描述符转换，逐层快照证明覆盖发生在 BN.forward；底层 CANN 内核具体哪条指令清零 padding 尚未直接定位，不能将其视为已完成的二进制级归因。

### 最终修复：融合前为 BN 参数保留 C0 布局

仓库的 `ultralytics/utils/torch_utils.py::prepare_npu_fused_optimizer()` 在首次融合和加载优化器状态之前：

1. 找出 `BatchNorm1d` / `BatchNorm2d` 中通道数不是 16 的整数倍的层。
2. 仅对位于 NPU、需要训练的 weight/bias，将 `parameter.data` 转为 `ACL_FORMAT_NC1HWC0`（格式 3）。保留 Parameter 对象、逻辑形状和数值。
3. 原版融合优化器随后将梯度转为参数格式，参数、梯度和动量都拥有一致的 C0 存储。例如 BN8 每个槽真实保留 16 个元素，BN85 保留 96 个元素。

`BaseTrainer.build_optimizer()` 只在实际选择 NPU 融合优化器且内部格式未关闭时调用这个 helper。CPU、普通优化器和已经 C0 对齐的 BN 不做转换；冻结参数不进入融合存储。DDP 的 `gradient_as_bucket_view=False`、FP16 AMP、默认 GradScaler、内部格式和融合裁剪均保留。

这里不能用 `numel()` 替代存储长度，也不能在参数已经融合后随意替换参数对象。布局准备必须在第一次组合和优化器状态恢复之前完成。对于 DDP，还必须在 Reducer 构造之前完成：TorchNPU Reducer 会缓存参数的物理长度和偏移，之后才将 BN8 从 8 元素改为 16 元素，会使缓存与真实布局不一致。因此本仓库先构建 dataloader/优化器、准备格式，再包装 DDP，最后创建 EMA 并恢复 checkpoint。优化器仍按原来的 `auto`、MuSGD 和缺类回退规则选择，没有复制一套提前选择逻辑。

双卡回归还触发了一处 EMA 集成缺陷：格式 3 不支持 `_foreach_lerp_` 时会回退到逐张量更新；原回退路径遍历 `named_buffers()`，却从 `state_dict()` 取值，因而找不到 DDP 的非持久化对齐 buffer。现将 buffer 来源统一为模型的 `named_buffers()`，与 foreach 路径保持一致，并加入强制回退、非持久化 float/int buffer 连续同步的回归测试。

### 数值验收

原版 TorchNPU 加 BN 格式 3 的多步对照结果如下，普通对应优化器接收相同梯度值：

| 优化器 | 多步参数最大绝对差 | 状态对照 |
| --- | ---: | --- |
| NpuFusedAdamW，YOLO/C3k2 | 1.49e-8 | 一阶不超过 9.31e-10，二阶为 0；恢复后与连续训练参数和状态差均为 0 |
| NpuFusedSGD | 0 | momentum 为 0 |
| NpuFusedAdam | 2.98e-8 | 一阶不超过 4.66e-10，二阶为 0 |
| NpuFusedRMSprop | 0 | square_avg、momentum 为 0；centered grad_avg 不超过 2.91e-11 |

各组连续运行至少三步，并检查 forward/backward 不会改写参数。仓库回归测试还覆盖 BN8、BN85、参数身份、数值与 AdamW 状态恢复。

默认融合优化器、FP16 和内部格式开启时，YOLO11n/YOLO26n 检测、YOLO26n 实例分割/姿态、分类、OBB、语义分割、深度这 8 项公开 API 均完成至少 16 次迭代、真实参数更新、保存重载、独立验证和推理。两个 YOLO 检测模型从真实原始 optimizer checkpoint 恢复，global_step 从 16 推进到 32。用户当前完整 DINOv3-L C3k2-QMS 在 640 输入下也通过 8 次迭代及完整训推链路。RT-DETR 默认融合路径的8iter训练加8iter真实续训同样通过，共14次有效更新，并完成真实/空目标损失、去噪分支、保存重载、验证和推理。相关存储/数值/EMA/路由/运行模式回归46项通过，DDP生命周期与配置56项通过；纯CPU关闭后端自动加载另有14项通过、2项真实NPU测试预期跳过，重复项不累加。短训不用于评价正式收敛精度。

双卡 HCCL 最终回归同样保留默认融合优化器、FP16 和内部格式。YOLO26n 初始16iter每rank7次真实更新，独立进程从366项真实optimizer状态及GradScaler恢复后，再16iter全部更新，global_step从16推进到32；每rank累计23次真实更新。两rank梯度范数、参数变化和scale逐值一致，前后向期间全模型参数意外改写为0。独立双卡验证完成4/4图，训练、恢复和独立验证三个进程阶段均退出0。

原生Conv-BN8双卡三步与普通AdamW参考的最大梯度差为1.49e-8、参数差7.45e-9。后置转换的本次数值对照也通过，所以将DDP前移表述为确保物理缓存契约；已复现的双卡业务异常是上述EMA回退问题。

原始依赖 SHA256、已否决候选和无需数据集的最小复现保存在：

```text
/root/work/filestorage/zhoufei/hermite/deps/patches/torch_npu-fused-storage-2.14/
```

最终修复随仓库代码生效，当前环境没有保留两行依赖补丁。启动新 Python / torchrun worker 后加载新代码。`USE_ASCEND_FUSED_OPTIMIZER=0` 仍是临时绕过方式，不作为本次最终修复。

### 官方源码依据

- [融合优化器基类](https://github.com/Ascend/pytorch/blob/v2.14.0-26.2.0/torch_npu/optim/npu_fused_optim_base.py#L51-L52) 与 [组合缓冲区实现](https://github.com/Ascend/pytorch/blob/v2.14.0-26.2.0/torch_npu/utils/combine_tensors.py#L24)：原始计量差异。
- [NPU 存储描述符长度](https://github.com/Ascend/pytorch/blob/v2.14.0-26.2.0/torch_npu/csrc/aten/common/StorageSizeNpu.cpp#L8-L15) 与 [TypedStorage 长度](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/storage.py#L1236-L1244)：两种长度的含义。
- [BN 前向的临时格式转换](https://github.com/Ascend/op-plugin/blob/5ac12689b6a61ec9db3fc260eab17bbc5447567e/op_plugin/ops/aclops/BatchNormKernelNpu.cpp#L158-L185)、[BN 反向返回梯度的转换](https://github.com/Ascend/op-plugin/blob/5ac12689b6a61ec9db3fc260eab17bbc5447567e/op_plugin/ops/aclops/BatchNormBackwardKernelNpu.cpp#L284-L290) 与 [unsafe cast helper](https://github.com/Ascend/pytorch/blob/2c8682df9271e545e7f50f8c45e2395563a05483/torch_npu/csrc/framework/FormatHelper.cpp#L521-L531)：说明布局风险的官方源码路径，固定修订不等同于当前 wheel 的逐指令反汇编证明。
- [原版模块的格式准备](https://github.com/Ascend/pytorch/blob/v2.14.0-26.2.0/torch_npu/utils/_module.py#L110-L119)：`jit_compile=False` 时跳过 BN 参数的格式 3 准备。

- [TorchNPU Reducer 的物理长度缓存](https://github.com/Ascend/pytorch/blob/v2.14.0-26.2.0/torch_npu/csrc/distributed/reducer.cpp#L969-L1018) 与 [按目标物理长度复制](https://github.com/Ascend/pytorch/blob/v2.14.0-26.2.0/torch_npu/csrc/aten/common/CopyMemoryKernel.cpp#L25-L53)：支持在 DDP 构造前完成布局准备。后置转换的缓存不一致是源码审查发现的契约风险，与已复现的 EMA `KeyError` 分开归因。
