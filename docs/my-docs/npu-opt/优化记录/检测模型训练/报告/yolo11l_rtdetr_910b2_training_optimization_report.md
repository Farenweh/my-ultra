# YOLO11-L 与 RT-DETR-L 在 Ascend 910B2 上的训练优化报告

## 1. 最终结论

本轮以重写后的 `npu` 基线 `0a209b1c3` 为起点，继续把 YOLO11-L、RT-DETR-L 及通用 Trainer 候选逐项拆开验证。只有正确性通过且完整 step 实测有效的改动才保留；无稳定收益或端到端退化的候选均未进入代码。2026-08-02 测试完成时，本地 `npu` 与 `npu-opt` 均指向代码提交 `0bfc29006`，报告和复现脚本当时作为非跟踪审计材料保留；2026-08-03 这些材料统一归档并纳入版本控制。

最终保留 6 项训练路径优化和 1 项 MSDA 基线正确性修复，并统一 5 项昇腾专用配置和 1 项全局配置的三态语义；5 项 Ascend 运行时环境默认值不再覆盖用户显式配置：

| 模型/路径                           | 改动                             |      基线 |    优化后 |             结果 |
| ----------------------------------- | -------------------------------- | --------: | --------: | ---------------: |
| YOLO11-L，FP16，640，batch 32       | NPU fused grad clip              | 248.75 ms | 231.17 ms |      **1.0760x** |
| YOLO11-L，2×NPU，每卡 b16，累积 2   | DDP 仅在更新边界同步             | 236.38 ms | 227.39 ms |      **1.0395x** |
| RT-DETR-L，FP16，640，batch 8       | 跨层批量 Hungarian cost          | 220.98 ms | 210.00 ms |      **1.0523x** |
| RT-DETR-L，FP16，640，batch 8       | decoder top-k 高级索引 → gather  | 208.48 ms | 202.46 ms |      **1.0298x** |
| RT-DETR-L，2×NPU，每卡 b4，累积 2   | DDP 仅在更新边界同步             | 257.42 ms | 239.98 ms |      **1.0727x** |
| RT-DETR Hungarian padding，batch 8  | 逐图 NPU `setitem` → 一次 gather |  1.045 ms |  0.217 ms | 节省 0.828 ms/次 |
| RT-DETR Hungarian padding，batch 48 | 逐图 NPU `setitem` → 一次 gather |  5.519 ms |  0.225 ms | 节省 5.294 ms/次 |
| RT-DETR denoising index，batch 8    | NPU 构造并 D2H → CPU 向量化      |  0.345 ms |  0.223 ms | 节省 0.122 ms/次 |
| RT-DETR denoising index，batch 48   | NPU 构造并 D2H → CPU 向量化      |  0.720 ms |  0.623 ms | 节省 0.097 ms/次 |

推荐生产路径仍是 PyTorch Eager、TaskQueue 2、internal format 和 FP16。短训练或显式选择 AdamW/SGD 时使用 NPU fused optimizer 和自动 fused grad clip；长训练的 `optimizer=auto` 当前会选择 MuSGD，保持其优化算法不变。RT-DETR 额外默认启用跨层批量 Hungarian，多卡且 `accumulate>1` 时自动只在 optimizer 更新边界同步 DDP 梯度。

## 2. 环境与测量口径

- 硬件：8 × Ascend 910B2，每卡 65,536 MiB HBM
- Python：3.12.13
- PyTorch：2.12.0+cpu（Ascend 适配形态）
- torch_npu：2.12.0，git `fa0f83fe49d309dcbc31e264e9e6ed6e5dc49d2d`
- Ultralytics：8.4.114
- 基线：`npu` 分支 `0a209b1c3895ca497dc122063845e487e1059a72`
- 原性能选型环境：CANN 9.1.0-beta.3；`npu-smi` 25.5.2
- 迁移回归环境：CANN 9.1.0 正式版、torch 2.12.0、torch_npu 2.12.0
- 默认模式：TaskQueue 2、internal format 开、JIT compile 关、FP16 AMP、NPU fused optimizer、EMA 开

完整训练 step 包含检测 loss、backward、AMP unscale、梯度裁剪、optimizer step 和 EMA；首步编译不计入稳态。正式对照采用同一物理卡 A→B→B→A，每个进程独立初始化并使用相同 seed、输入形状和目标数。

- fused grad clip：预热 20 step，5 × 20 个测量 step；每个版本两次。
- 跨层 Hungarian：预热 20 step，10 × 20 个测量 step；每个版本两次，串行运行且无其他 NPU 基准并发。
- decoder top-k gather：独占物理 NPU 7，预热 20 step，5 × 20 个测量 step，按 A→B→B→A 运行。
- DDP 梯度累积：物理 NPU 6/7，MuSGD，FP16，`accumulate=2`，GradScaler `init_scale=16`；预热 20 microstep，5 × 20 个测量 microstep，按 A→B→B→A 运行。每次均执行 60 次 optimizer update，且没有 overflow。
- DDP 筛查：2 卡、每 rank batch 32，最终口径预热 100 step，10 × 20 个测量 step，并用 HCCL 取较慢 rank 的时延。
- 微基准在同一进程按 A/B/B/A 交替，NPU 同步后计时，并校验有效元素或索引逐元素一致。

## 3. 已提交的有效改动

### 3.1 自动启用融合梯度裁剪

基线提交：`d9931e58f`（直接重写原 `99008628a`，吸收原独立提交 `59d9d49bc` 和 `cb6fb348e`）

`USE_ASCEND_FUSED_GRAD_CLIP` 直接使用 `None/False/True` 三态：未设置且融合优化器未被显式禁用时，Trainer 会根据 optimizer 是否提供 `clip_grad_norm_fused_()` 自动选择融合路径；不支持该方法的 optimizer 回退到标准实现。显式设为 `1` 仍是严格模式，显式设为 `0` 可禁用。该实现从融合裁剪首次进入当前历史时即存在，不再依赖后置修复提交。

同卡 ABBA：

- 普通裁剪：248.75 ms，128.64 img/s
- 融合裁剪：231.17 ms，138.43 img/s
- 加速比：`1.0760x`
- 两对加速比：`1.0746x / 1.0775x`
- 峰值已分配显存：增加约 96.8 MiB

数值对照中，FP32 norm 和裁剪后梯度完全一致；FP16 norm 相对误差 `2.14e-4`，最大梯度绝对误差 `4.88e-4`。

同一迁移提交还统一让 DDP 和最终 Validator 复用已经解析出的设备类型。`[0,1]`、`(0,1)`、`"0, 1"` 和 `"0,1"` 在 Ascend 平台会自动选择 NPU/HCCL，无需添加 `npu:` 前缀；双卡 RT-DETR-L 已完成训练、保存、重载和最终验证。

### 3.2 向量化 Hungarian GT padding

提交：`cd886700f`

旧路径按图创建 padding 并发出多次 NPU `setitem`。新路径根据 `gt_groups` 一次生成 padding 索引，再 gather box/class。batch 内含空图的 `[2, 0, 1]` 用例与逐图匹配结果严格一致。

该改动随 batch 增大收益明显：batch 8 节省 0.828 ms/次，batch 48 节省 5.294 ms/次。

### 3.3 合并 RT-DETR 跨层 Hungarian cost 传输

提交：`038b39b2f`

标准 RT-DETR-L 会对 encoder 输出和 6 个 decoder 层分别构造 cost 并 D2H。新路径将 `[layer, batch]` 展平，一次构造和传输 cost，随后仍逐层、逐图独立调用 SciPy `linear_sum_assignment`；GT 索引减去层偏移后与旧路径一致。

同卡串行 ABBA：

- 逐层 cost D2H：220.98 ms，36.20 img/s
- 合并 cost D2H：210.00 ms，38.10 img/s
- 加速比：`1.0523x`
- 两对加速比：`1.0536x / 1.0509x`
- 峰值已分配/保留显存无变化

该功能只在 Ascend NPU、`use_uni_match=False` 且没有外部 match indices 时启用，可用全局配置 `USE_BATCHED_HUNGARIAN=0` 回退。

### 3.4 在 CPU 向量化构造 denoising 正样本索引

提交：`3877fa5c2`

旧实现先在 NPU 构造 `pos_idx`，随后立即 `.cpu()`。第一版 CPU 实现使用按图 Python 循环，batch 48 出现退化，因此没有直接提交；最终版本在 CPU 上整批构造并 split，batch 8 和 batch 48 均取得正收益。

### 3.5 使用 gather 选择 RT-DETR decoder 查询

提交：`e96dc7cba`

原实现把 `[batch, query]` 的 top-k 索引展平，并在 CPU 构造 `batch_ind`，随后对 NPU 上的 feature、anchor 和 score 做三次高级索引。新实现保留二维 top-k 索引，沿 query 维直接 `gather`，删除 CPU 索引的构造、复制和混合设备高级索引。

独占卡同卡 ABBA：

- 高级索引：208.48 ms，38.37 img/s
- gather：202.46 ms，39.51 img/s
- 加速比：`1.0298x`
- 两对加速比：`1.0370x / 1.0228x`
- 峰值已分配/保留显存无变化

batch 48 的单次同卡 A/B 为 592.92/590.89 ms（`1.0034x`），收益随卷积和 MSDA 计算量增加而被摊薄，但没有大 batch 退化。CPU 和 910B2 FP16 上 forward 逐元素完全一致，backward 在浮点容差内一致。

### 3.6 梯度累积时只在更新边界同步 DDP

提交：`0bfc29006`

原训练循环对每个 microstep 都执行 DDP gradient all-reduce，即使该步只是在本地累积梯度、不会更新 optimizer。新路径复用原有的 `ni - last_opt_step >= self.accumulate` 判据：非更新 microstep 使用 `DistributedDataParallel.no_sync()`，并按 PyTorch DDP 语义让 context 同时覆盖 forward 和 backward；更新边界仍执行一次完整同步。单卡和 `accumulate=1` 不进入该路径，动态 warmup、跨 epoch 的未满累积窗口、optimizer 步频和停止时的现有行为均不变。

物理 NPU 6/7、生产长训练默认 MuSGD、无 overflow 的 ABBA：

| 模型与口径                    | 每个 backward 同步 | 仅更新边界同步 |  加速比 |        全局吞吐 |
| ----------------------------- | -----------------: | -------------: | ------: | --------------: |
| YOLO11-L，每 rank b16，累积 2 |          236.38 ms |      227.39 ms | 1.0395x | 135.38→140.73/s |
| RT-DETR-L，每 rank b4，累积 2 |          257.42 ms |      239.98 ms | 1.0727x |   31.08→33.34/s |

YOLO 两对加速比为 `1.0124x/1.0664x`，RT-DETR 为 `1.1039x/1.0426x`，四对均同方向；两种模型的峰值已分配显存均没有变化。`accumulate=2` 时每次 optimizer update 的梯度同步次数由 2 次降至 1 次，但通信可与 backward 重叠，因此墙钟收益小于 50%。累积倍数或 world size 增大时通信消除比例会更高，实际收益仍应按目标集群复测。

两卡 HCCL 正确性对照中，累积两步后的梯度最大差为 `2.98e-8`，SGD step 后参数最大差为 `7.45e-9`，momentum 最大差为 `2.98e-8`。差异仅来自浮点归约顺序。

### 3.7 将 MSDA fast-path 缓存收归模块实例

基线提交：`ce6ece687`（直接重写原 `34aa1b29c`）

原实现把 `spatial_shapes` 和 `level_start_index` 放在 `utils.py` 的模块级全局字典中。TensorBoard 即使 trace 深拷贝模型，副本仍会写入这个共享字典，后续训练命中 inference tensor 时 MMCV autograd 会报：

```text
Inference tensors cannot be saved for backward
```

现在每个 `MSDeformAttn` 实例持有单条最近使用缓存；深拷贝模型与训练模型不再共享。缓存 Tensor 始终在关闭 inference mode 的上下文创建，模块发生设备或精度迁移时由 `_apply()` 清空设备绑定缓存。该缓存不是 buffer，不进入 EMA foreach/state_dict tensor 列表。独立修复提交 `a5d5c31c2` 已从历史删除。

同卡两次运行对照中，全局缓存与实例缓存的完整训练 step 分别为 `230.570 ms` 和 `231.813 ms`，实例缓存平均慢 `0.539%`；两对比值为 `0.9642x/1.0265x`，方向相反，没有形成稳定退化。峰值 reserved 不变，allocated 仅增加 `0.0049 MiB`。

真实 RT-DETR-L Trainer 已通过 TensorBoard graph trace、2 个训练 step、epoch 验证、`best/last/raw` 权重保存、重新加载 `best.pt` 和最终验证；没有 EMA foreach 回退、inference tensor 或跨设备缓存错误。

### 3.8 Ascend 环境默认值与三态配置

环境默认值、融合优化器、JIT compile 和 internal format 配置归属提交：`3571ba27d`（直接重写原 `83919eb13`）。

以下环境变量仅在进程启动前没有设置时写入推荐默认值；显式值（包括空字符串）均原样保留：

| 环境变量                 | 默认值                     |
| ------------------------ | -------------------------- |
| `TASK_QUEUE_ENABLE`      | `2`                        |
| `ACLNN_CACHE_LIMIT`      | `500000`                   |
| `CPU_AFFINITY_CONF`      | `1`                        |
| `PYTORCH_NPU_ALLOC_CONF` | `expandable_segments:True` |
| `HOST_CACHE_CAPACITY`    | `50`                       |

因此默认训练性能配置不变，同时 NPU Graph、缓存容量、绑核、内存分配器和 Host Cache 专项测试可在导入 Ultralytics 前显式覆盖。该行为从亲和配置提交起即存在，原独立 TaskQueue 修复提交 `7f620ca7b` 已从历史删除。

五项 `USE_ASCEND_*` 仅在 `IS_ASCEND=True` 时创建，并直接解析为 `None/False/True`；非 Ascend 环境不导出这些名字。Hungarian 开关改名为全局 `USE_BATCHED_HUNGARIAN`，但 fast path 仍只在已验证的 Ascend NPU 条件下启用。所有配置均为未设置→`None`、显式 `0/1`→`False/True`。

| 配置                          | `None` 策略                                          | 历史归属    |
| ----------------------------- | ---------------------------------------------------- | ----------- |
| `USE_ASCEND_FUSED_GRAD_CLIP`  | 优化器允许融合且提供融合裁剪方法时启用，否则标准裁剪 | `d9931e58f` |
| `USE_ASCEND_FUSED_OPTIMIZER`  | 融合类存在时使用，不存在时回退标准优化器             | `3571ba27d` |
| `USE_ASCEND_JIT_COMPILE`      | 关闭 JIT compile                                     | `3571ba27d` |
| `USE_ASCEND_INTERNAL_FORMAT`  | 开启 internal format                                 | `3571ba27d` |
| `USE_ASCEND_DDP_BUFFER_ALIGN` | 多卡 Ascend 训练时启用 512 字节通信对齐              | `12777fcdf` |
| `USE_BATCHED_HUNGARIAN`       | 满足 Ascend NPU、形状和匹配模式条件时启用            | `038b39b2f` |

显式 `1` 的融合优化器和融合裁剪保持严格模式，能力缺失会报错；显式 `0` 强制回退。融合优化器为 `None` 时，DDP 保守关闭 `gradient_as_bucket_view`，避免自动选中融合类后发生 `p.grad` data pointer 冲突。

## 4. 被实测否决的候选

| 候选                                       |         A |         B | 结论                                                             |
| ------------------------------------------ | --------: | --------: | ---------------------------------------------------------------- |
| RT-DETR 在 dataloader 侧预计算 `gt_groups` | 221.16 ms | 231.91 ms | `0.9536x`，端到端退化，撤销                                      |
| YOLO 前景 host sync + 固定常量缓存         | 234.01 ms | 233.02 ms | 仅 `1.0042x`；两对为 `1.0083x/1.0002x`，不可靠，撤销             |
| YOLO DDP 关闭 unused 扫描                  | 264.71 ms | 262.73 ms | 平均 `1.0075x`，但两对为 `0.9928x/1.0225x`，方向相反，提交已移除 |
| YOLO target max 由 CPU 提供                | 232.65 ms | 233.53 ms | `0.9963x`，没有收益                                              |
| YOLO TAL 无条件执行冲突消解                | 232.65 ms | 231.96 ms | 仅 `1.0030x`，低于噪声门槛                                       |
| torch_npu fused clip 改为 branchless       | 233.53 ms | 237.16 ms | `0.9847x`，且依赖 torch_npu 私有 API                             |
| 独立 NPU stream 异步 EMA                   | 233.53 ms | 234.20 ms | `0.9971x`，无收益且增加 callback 一致性风险                      |
| 静态 GradScaler                            | 233.53 ms | 233.23 ms | 仅快 0.13%，但最终 loss 为 NaN                                   |
| YOLO FP16 → BF16                           | 233.65 ms | 303.53 ms | `0.7698x`，反向显著变慢                                          |
| RT-DETR CDN attention mask 向量化          | 236.02 ms | 240.76 ms | `0.9803x`，撤销                                                  |
| YOLO C3k2 `chunk` → `split`                | 237.22 ms | 236.23 ms | 仅 `1.0042x`；两对方向相反，撤销                                 |
| RT-DETR decoder `npu_fusion_attention`     | 196.90 ms | 203.72 ms | `0.9665x`；虽省 389 MiB，但吞吐退化                              |
| RT-DETR loss 三组索引合并 H2D              | 196.90 ms | 216.84 ms | `0.9081x`，stack/view 代价高于传输节省                           |
| MuSGD `set_to_none=False` 复用梯度         | 287.19 ms | 290.83 ms | `0.9875x`，且改变无梯度参数语义                                  |
| MuSGD 标准裁剪强制 foreach                 | 287.19 ms | 287.50 ms | `0.9989x`，无收益                                                |
| MuSGD 合并不同 scale 的正交化 bucket       |  53.13 ms |  64.10 ms | optimizer 微基准慢 20.7%，撤销                                   |
| MuSGD 按宽度 2 次幂细分 bucket             | 287.71 ms | 296.14 ms | `0.9716x`；padding 减少但 kernel launch 增加                     |
| MuSGD 仅在宽度间隔 >2 时细分 bucket        | 287.71 ms | 294.92 ms | `0.9756x`，仍然退化                                              |
| YOLO NPU HF32 全开                         | 234.93 ms | 236.92 ms | `0.9916x`，保持 Conv 开、Matmul 关的默认值                       |
| RT-DETR Hungarian 4 线程 CPU solver        | 203.36 ms | 214.01 ms | `0.9502x`；两对方向相反，线程调度噪声过大                        |
| DDP gradient bucket view + MuSGD           | 229.86 ms | 231.53 ms | `0.9928x`；仅省约 40 MiB reserved                                |

YOLO loss 两个微路径本身分别节省约 0.089 ms 和 0.298 ms，但完整 step 的第二对几乎无差异。为避免为亚毫秒且不可稳定观测的收益增加缓存状态和语义风险，最终保持基线实现。

本轮单卡 YOLO11-L 没有新增代码提交。当前 fused 路径的 batch 32 阶段中位数为 forward 94.78 ms、backward 128.97 ms、unscale+clip 1.57 ms、EMA 5.24 ms；稳态采样的 AICore/NPU 利用率均值为 74.7%/89.5%。新 profile 中设备热点仍是 TransData 47.94 ms、卷积 forward/backward 39.41/88.60 ms、SiLU backward 32.00 ms 和 BN backward 48.50 ms。CSP 分支的 `split()` 完整 step 只有 0.42% 表面收益，未形成稳定方向。

进一步追踪标量栈后，单个完整 step 只有 7 次 Python/仓库层可见的 `_local_scalar_dense`，均来自已测的 target/TAL/前景条件、GradScaler 和 fused clip；其余约 411 次属于 torch_npu 内部算子或 format 元数据路径，无法从仓库 Python owner 删除。NPU HF32 全关与默认值相差仅 0.006%，全开慢 0.85%，因此保留 Conv 开、Matmul 关的 torch_npu 默认值。`channels_last` 在模型迁移阶段即被 torch_npu 以 `ERR01007` 拒绝，当前 Trainer 只对 CUDA 启用它是必要限制。

RT-DETR-L 的两步设备 profile 中，MSDA forward/backward 合计 22.70 ms/step，TransData、Transpose 和 Cast 分别为 9.49/7.56/6.33 ms/step，BatchMatMul 和 Softmax forward+backward 分别为 5.13/2.17 ms/step。直接切换 `npu_fusion_attention` 虽减少约 389 MiB 显存，却使完整 step 退化 3.47%；`npu_add_layer_norm` 又不支持当前 FP16 activation 与 FP32 affine 参数的混合 dtype backward，因此均未保留。

后续深挖 MuSGD 发现，当前按行数合并不同宽度矩阵会让 YOLO/RT-DETR 的 Newton–Schulz 理论计算量因 padding 分别放大到约 `2.84x/1.77x`。按宽度 2 次幂细分可把 padding 降到约 `1.01x`，但 bucket 数分别由 9/18 增至 23/35。YOLO 完整 step 实测反而由 287.71 ms 增至 296.14 ms；只切分明显宽度间隔也为 294.92 ms。额外 kernel launch 和任务下发成本超过减少的 BMM 计算，因此保持当前批处理策略。

RT-DETR 跨层 Hungarian 的进一步分解显示，每 step 的 56 次 SciPy solver 只占 1.03 ms，cost enqueue、D2H 和 packing 占 2.83 ms。持久 4 线程池虽然把纯 CPU solver 微基准从 0.991 ms 降到 0.717 ms，但完整 step 两对结果为 `0.8805x/1.0297x`，均值退化 4.98%；不引入线程池生命周期和 TaskQueue 争用。

DDP 的两个 rank 都确认 YOLO11-L 没有 unused parameter，但在生产 internal format 配置下额外图遍历的开销没有形成稳定端到端收益，因此保留原默认行为及环境变量覆盖能力。

## 5. Batch 和利用率建议

既有 batch 扫描仍可用于容量规划，但这些数据来自上一轮组合候选，不应当作本轮逐项提交的精确最终时延：

| 模型      | 建议起始 batch/卡 |         吞吐 | AICore 均值 | NPU 利用率均值 | 峰值已分配显存 |
| --------- | ----------------: | -----------: | ----------: | -------------: | -------------: |
| YOLO11-L  |                64 | 152.61 img/s |      78.88% |         95.46% |     37,203 MiB |
| RT-DETR-L |                48 |  84.53 img/s |      88.85% |         92.34% |     28,130 MiB |

YOLO b96 和 RT-DETR b80 可以运行，但边际收益递减且显存余量下降。生产环境建议从 b64/b48 起步，目标密集数据先下调，确认真实峰值后再增加。

RT-DETR-L 的当前 gather 版本在 batch 48 独立稳态运行中采样 53 次，AICore/NPU 利用率中位数为 95%/93%，均值为 88.85%/92.34%；不同物理卡的绝对时延有明显差异，因此该利用率运行不与旧卡结果计算因果加速比。

YOLO 两卡累积 2 的独立长窗口复测覆盖 200 个 microstep：每步同步为 236.83 ms、135.12 img/s，仅更新边界同步为 222.11 ms、144.07 img/s，吞吐提高 6.63%。物理 NPU 2 的稳态采样如下：

| DDP 同步策略      | 样本 | AICore 均值/中位 | NPU 均值/中位 | HBM 带宽均值/中位 |
| ----------------- | ---: | ---------------: | ------------: | ----------------: |
| 每个 microstep    |   50 |      52.00/53.5% |   61.42/62.0% |       20.60/20.0% |
| 仅 optimizer 边界 |   47 |      46.81/47.0% |   65.02/62.0% |       23.13/22.0% |

AICore 采样下降不是计算退化：被删除的 HCCL 规约本身也占设备工作，而且 1 秒采样会受 microstep 相位影响。此处有效工作吞吐提高、NPU 均值和 HBM 带宽同时提高，说明设备空转间隔减少；优化目标应以无 overflow 的完整训练吞吐为主，而不是单独追求更高的 AICore 百分比。

RT-DETR 的等更新数利用率复测中，两侧都完成 101 次 optimizer update 且无 overflow：每步同步为 249.48 ms/microstep、32.07 img/s，仅更新边界同步为 232.21 ms/microstep、34.45 img/s，吞吐提高 7.44%。物理 NPU 4 的 AICore 均值/中位数由 32.34/30.5% 提高到 34.97/35.0%，NPU 利用率由 45.84/44.0% 变为 45.97/44.0%，HBM 带宽由 4.26/5.0% 提高到 5.60/5.0%。另一次更长的逐步同步运行仍比 no-sync 慢 2.92%，方向一致但体现了卡状态漂移，因此正式加速比采用上一节的 ABBA 汇总。

建议优先把每卡 microbatch 增大到真实数据显存允许的稳定点；当全局 batch 仍小于 `nbs`、Trainer 自动得到 `accumulate>1` 时，新 DDP 路径无需额外配置即可生效。`accumulate=1` 时不会改变执行路径或性能。

真实 YOLO11-L Trainer 在 coco128、batch 32 上的取批等待进一步确认了 worker 配置的重要性：`workers=0` 的稳态中位/p95 为 555.80/617.96 ms，而 4 个实际 worker 为 0.315/0.555 ms。当前 `build_dataloader()` 会按 loader batch 数限制 worker；该数据集每轮 4 个 batch，因此默认 `workers=8` 实际正好使用 4 个 worker，已经把数据增强和读取隐藏在 NPU 计算后面。生产配置不要无故设成 `workers=0`；应从默认值起步，并按真实存储和 CPU 配额观察取批等待。

DDP 继续筛查也确认当前 25 MiB bucket 最合适：在 YOLO11-L、MuSGD、no-sync 累积 2 下，25 MiB 为 229.86 ms；10/50/100 MiB 分别为 244.10/239.32/255.33 ms。`gradient_as_bucket_view=True` 慢 0.72%，只少约 40 MiB reserved；`static_graph=True` 与当前 no-sync 组合触发 PyTorch reducer internal assert，均不启用。

## 6. 编译和图后端结论

- 保持 `USE_ASCEND_JIT_COMPILE=0`：YOLO11-L 开启后首步 14 分 45 秒仍未完成；RT-DETR-L 在 SDPA format conversion 报 CANN 不支持。
- 当前 torch/torch_npu/CANN 组合下，Inductor/Triton 后端不能完整编译这两类训练图。
- 手动 NPU Graph 只在固定 shape 合成 YOLO step 上观察到约 8% 收益，且与 TaskQueue 2 冲突，没有覆盖真实 Trainer、增强、DDP、EMA 和收敛。
- DVM 没有在检测网络整体训练上得到稳定可复现收益。

因此默认路径仍是经过实测的 Eager，而不是把图后端作为生产默认值。

## 7. 正确性验证

- 三态配置、核心优化、DDP、MSDA、deformable encoder 和 TensorBoard 集中回归：75 passed，无跳过项，包含实际 NPU 用例。
- 本轮新增路径回归：Trainer/DDP/融合裁剪/EMA 定向 27 passed；独立 NPU 进程中的 RT-DETR loss、MSDA、deformable encoder、AIFI 和 NPU 算子 50 passed。
- Ruff format/check、Python 语法编译、JSON 校验和 `git diff --check` 通过。
- Hungarian：有目标、全空目标、混合空图与逐层/逐图旧路径一致。
- denoising index：不均匀目标数和空图索引严格一致，结果位于 CPU。
- decoder top-k gather：CPU/NPU forward 与旧高级索引逐元素一致，backward 梯度在 FP16 容差内一致。
- DDP no-sync：两卡 HCCL 累积梯度、optimizer step 后参数和 momentum 均与逐步同步路径在 `1e-6` 内一致；最大差分别为 `2.98e-8/7.45e-9/2.98e-8`。
- fused grad clip：FP32/FP16 数值对照通过。
- MSDA：实例隔离、inference/autograd 安全、动态尺寸替换和设备/精度迁移清理测试通过。
- 未设置五项昇腾专用配置和全局 Hungarian 配置时，YOLO11-L 与 RT-DETR-L 均自动选择 `NpuFusedAdamW`，并解析为 JIT 关闭、internal format 开启。
- 真实 coco8 Trainer：YOLO11-L 与 RT-DETR-L 各完成 1 epoch、2 iter，forward/backward、验证、`last/best` 保存和重新加载验证均通过，退出码均为 0。
- 真实两卡 coco128 Trainer：YOLO11-L（全局 b32、`nbs=64`）和 RT-DETR-L（全局 b8、`nbs=16`）均以 `accumulate=2` 完成 2 个 microstep、同步更新边界、epoch 验证、`last/best` 保存和 best 重载验证，退出码均为 0。
- RT-DETR-L 的真实 TensorBoard graph trace 成功；YOLO11-L 的 NPU graph trace 仍因 `npu::get_npu_format` 返回整数而产生非致命警告，不影响训练，模型深拷贝隔离测试通过。
- 当前运行时不再导入已停止演进的 `torchvision_npu`，也不调用计划废弃的 TorchNPU IoU/NMS 接口。CANN 9.1 的 `aclnnNonMaxSuppression` 不支持 Atlas A2/910B2，因此标准框 NMS 在 NPU 完成候选过滤后，将框和分数合并传至 CPU 执行 torchvision 精确 NMS，再把少量索引传回 NPU；IoU、CIoU 和 ProbIoU 使用可自动求导的原生 PyTorch 向量化实现。独立验证性能仍不纳入本报告加速比。

coco8/coco128 smoke 只证明端到端可训练，不代表最终 mAP 或长周期收敛。投产前仍需在目标数据集做固定 seed 的多 epoch A/B。

## 8. 推荐配置

```bash
export TASK_QUEUE_ENABLE=2
export USE_ASCEND_FUSED_OPTIMIZER=1
unset USE_ASCEND_FUSED_GRAD_CLIP
export USE_ASCEND_INTERNAL_FORMAT=1
export USE_ASCEND_JIT_COMPILE=0
export USE_BATCHED_HUNGARIAN=1
export USE_ASCEND_DDP_BUFFER_ALIGN=1
export CPU_AFFINITY_CONF=1
export ACLNN_CACHE_LIMIT=500000
```

训练命令中显式传入 `amp=fp16`；AMP dtype 由公共 `amp` 参数统一控制。

以上显式值适合固定生产环境并对缺失能力严格报错；五项 `USE_ASCEND_*` 和全局 `USE_BATCHED_HUNGARIAN` 均不设置时，在当前 910B2 软件栈上的有效默认行为相同，但融合优化器缺失时会自动回退。

需要注意，当前上游在 `optimizer=auto` 且计划迭代数超过 10,000 时选择 MuSGD。MuSGD 不是 `torch_npu` fused optimizer，也没有 `clip_grad_norm_fused_()`，因此这时会走 MuSGD + 标准梯度裁剪；`USE_ASCEND_FUSED_OPTIMIZER=1` 不会把 MuSGD 静默替换成 SGD。若显式设置 `optimizer=AdamW` 或 `optimizer=SGD`，可启用对应 NPU fused optimizer，但这是训练算法选择，应重新验证收敛，不能只依据吞吐替换。

- YOLO11-L 640：每卡 batch 从 64 起步。
- RT-DETR-L 640：每卡 batch 从 48 起步。
- RT-DETR 当前使用 FP16；Ascend MSDA fallback 不支持 BF16 grid sample。
- DDP no-sync 不需要新环境变量；只要 Trainer 计算出 `accumulate>1` 就自动生效。
- 多卡扩大全局 batch 后应重新检查学习率、梯度累积和收敛。

## 9. 提交与复现材料

核心优化提交，按顺序：

1. `d9931e58f` 从 Ascend 原生迁移提交起支持自动设备路由、NPU 亲和优化器和融合梯度裁剪
2. `3571ba27d` 添加 NPU 亲和配置、解析自动策略并保留五项显式 Ascend 环境配置（重写 `npu` 基线提交）
3. `ce6ece687` 将 MSDA fast-path 缓存收归 `MSDeformAttn` 实例（重写 `npu` 基线提交）
4. `cd886700f` 向量化 RT-DETR Hungarian 目标填充
5. `038b39b2f` 合并 RT-DETR 跨层 Hungarian 代价传输并提供全局开关
6. `3877fa5c2` 在 CPU 构造 RT-DETR 去噪匹配索引
7. `e96dc7cba` 使用 gather 加速 RT-DETR decoder 查询选择
8. `0bfc29006` 避免梯度累积 microstep 重复同步 DDP

复现文件：

- 单卡完整 step：`docs/my-docs/npu-opt/优化记录/检测模型训练/脚本/benchmark_detector_train_npu.py`
- 两卡 DDP 筛查：`docs/my-docs/npu-opt/优化记录/检测模型训练/脚本/benchmark_detector_ddp_npu.py`
- NPU 利用率采样：`docs/my-docs/npu-opt/优化记录/检测模型训练/脚本/monitor_npu_util.py`
- 结构化数据：`docs/my-docs/npu-opt/优化记录/检测模型训练/结果/yolo11l_rtdetr_910b2_results.json`
- 本报告：`docs/my-docs/npu-opt/优化记录/检测模型训练/报告/yolo11l_rtdetr_910b2_training_optimization_report.md`

以上分支名和提交 SHA 均为测试时的历史快照，不使用当前分支头回填。材料归档时工作位于 `npu` 分支，整理前 HEAD 为 `e2fa0f800a6e9f37a3d8b2b6d0335daed88916b2`。
