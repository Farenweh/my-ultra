# LocateAnything

## 模型概况

LocateAnything 是 NVIDIA 发布的 3B 视觉语言定位模型，以 MoonViT 为视觉编码器、Qwen2.5-3B 为语言模型，
使用 Parallel Box Decoding（PBD）并支持 `fast`、`hybrid` 和 `slow` 三种生成模式。当前接入提供目标检测、
短语定位、OCR、GUI 定位、点定位、LoRA/全参数 SFT 和 COCO 验证，不提供模型原生 confidence。

当前代码固定使用模型 revision `c32291ca5e996f5a7a485845b4f57a233936bba0`，以避免远程代码和权重漂移。
官方权重采用非商业研究许可，使用前应确认适用范围。

## 安装

```bash
pip install -e '.[locateanything]'
```

该 extra 固定使用 `transformers==5.14.1`，并安装 Accelerate 和 PEFT。Ascend 环境还需要与 PyTorch、CANN
版本匹配的 `torch_npu`。

## 模型加载参数

```python
from ultralytics import LocateAnything

model = LocateAnything(
    model="nvidia/LocateAnything-3B",
    revision="c32291ca5e996f5a7a485845b4f57a233936bba0",
    device="npu:0",
    dtype="bfloat16",
    local_files_only=True,
    npu_fast_path="auto",
)
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `model` | `"nvidia/LocateAnything-3B"` | Hugging Face 模型 ID，或由本接入导出的 LoRA/全参数目录。 |
| `revision` | `"c32291ca5e996f5a7a485845b4f57a233936bba0"` | 当前只接受该已验证 revision；其他值会直接报错。 |
| `device` | `None` | 使用 Ultralytics 设备选择逻辑自动选择；也可显式传入 `"npu:0"`、`"cuda:0"` 或 `"cpu"`。 |
| `dtype` | `"auto"` | CPU 有效值为 FP32，加速器有效值为 BF16。也接受 `float32/fp32`、`float16/fp16`、`bfloat16/bf16` 或 `torch.dtype`。 |
| `local_files_only` | `False` | `True` 时禁止 Transformers/Hugging Face 在运行时下载。 |
| `npu_fast_path` | `"auto"` | `auto` 在满足条件时使用 910B NPU 快路并允许回退；`off` 关闭；`strict` 在快路不适用时报错。`True/False` 分别映射为 `auto/off`。 |

NPU 快路只在 910B、FP16/BF16、`eval` 且 no-grad 条件下启用。训练会恢复原始可微 forward，
因此不会因加载时选择 `auto` 而失去可训练性。

## 统一推理接口

`model(...)` 等价于 `model.predict(...)`。`source` 可以是单张图片路径、HTTP(S) URL、PIL Image、
HWC NumPy 图像，或由这些输入组成的 `list/tuple`。

```python
results = model.predict(
    source="image.jpg",
    task="detect",
    prompt=["person", "car"],
    multiple=True,
    output="box",
    generation_mode="hybrid",
    max_new_tokens=2048,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    stream=False,
    verbose=False,
)
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `source` | 必填 | 单图或图片列表。列表按顺序逐张推理；公开 `predict()` 不启用 validator 的 batch runtime。 |
| `task` | `"ground"` | `detect`、`ground`、`ground_text`、`detect_text`、`gui`、`point` 或 `raw`。别名 `detection` 和 `ground_gui` 也可使用。 |
| `prompt` | `None` | 查询类别、短语或原始 prompt。除 `detect_text` 外的任务均需要非空 prompt；`detect` 可传字符串或字符串序列。 |
| `multiple` | `True` | `ground` 是否请求所有匹配实例；`False` 改为单实例 prompt。 |
| `output` | `"box"` | `box` 或 `point`；主要用于 `gui`，`point()` 会固定传入 `point`。 |
| `generation_mode` | `None` | `None` 在 NPU 上有效为 `slow`，其他设备为 `hybrid`；可显式选择 `fast`、`hybrid` 或 `slow`。 |
| `max_new_tokens` | `2048` | 每张图最多生成 token 数，必须大于 0。调大可降低大量实例被截断的风险，但会增加最坏时延。 |
| `temperature` | `0.7` | 采样温度；`0` 会关闭 sampling 并使用贪心解码。 |
| `top_p` | `0.9` | nucleus sampling 的累积概率阈值，常用范围为 `(0, 1]`。 |
| `repetition_penalty` | `1.1` | 已生成 token 的重复惩罚；`1.0` 表示不惩罚。 |
| `stream` | `False` | `True` 返回按图片产生 `LocateAnythingResult` 的 iterator，不是逐 token stream。 |
| `verbose` | `False` | 将 verbose 传给上游 `generate()` 用于调试。 |

### 任务快捷方法

| 方法 | 专用参数及默认值 | 等价任务 |
| --- | --- | --- |
| `detect(source, classes, **kwargs)` | `classes` 为字符串或类别序列 | `task="detect"` |
| `ground(source, phrase, multiple=True, **kwargs)` | `multiple=True` 定位全部实例 | `task="ground"` |
| `ground_text(source, text, **kwargs)` | `text` 为待定位文字 | `task="ground_text"` |
| `detect_text(source, **kwargs)` | 无专用可选参数 | `task="detect_text"` |
| `ground_gui(source, phrase, output="box", **kwargs)` | `output="box"`，可改为 `point` | `task="gui"` |
| `point(source, phrase, **kwargs)` | 固定 `output="point"` | `task="point"` |

`**kwargs` 可传入上表中的共通生成参数。需要完全自定义官方 prompt 时可用 `task="raw"`，
此时 `prompt` 会原样传入。

## 结果方法

`predict()` 返回 `list[LocateAnythingResult]`；`stream=True` 时返回 iterator。单张结果包含 `boxes`、
`points`、`labels`、`raw_output`、`parse_warnings`、`stats` 和 `speed`。

| 方法 | 默认值 | 说明 |
| --- | --- | --- |
| `summary(normalize=False)` | `False` | 返回框/点列表；`True` 时坐标归一化到 `[0,1]`。 |
| `to_json(normalize=False)` | `False` | 序列化路径、尺寸、预测、原始输出、警告、统计和耗时。 |
| `plot(boxes=True, points=True, labels=True, line_width=2)` | 见签名 | 返回绘制后的 BGR NumPy 图像，不修改原图。 |
| `show(**plot_kwargs)` | 无 | 使用 Pillow 显示 `plot()` 结果。 |
| `save(filename=None, **plot_kwargs)` | `None` | 默认保存为 `results_<原文件名>`，并返回保存路径。 |

## 训练接口

```python
train_result = model.train(
    data="dataset.yaml",
    method="lora",
    device="0,1,2,3,4,5,6,7",
    epochs=1,
    batch=1,
    max_seq_length=4096,
)
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `data` | 必填 | YOLO detection YAML、官方 recipe JSON 或单个 ShareGPT JSONL。 |
| `method` | `"lora"` | `lora` 或 `full`。LoRA 单卡直接训练、多卡使用 DDP；`full` 至少两卡并使用 FSDP2。 |
| `device` | `None` | 默认继承模型当前设备。可传 `"0"`、`"0,1"`、`"npu:0,1"` 或对应 CUDA 形式。 |
| `epochs` | `1` | 训练 epoch 数，必须大于 0。 |
| `max_steps` | `-1` | 大于 0 时以 optimizer update 数作为停止上限；`-1` 表示仅按 epochs 控制。 |
| `batch` | `1` | 每个 rank 的 micro-batch。全局有效 batch 约为 `batch × world_size × gradient_accumulation_steps`。 |
| `workers` | `4` | 每个 rank 的 DataLoader worker 数。 |
| `max_seq_length` | `4096` | 输入和 PBD 监督的最大序列长度；当前要求 `128 <= value <= 4096`。 |
| `gradient_accumulation_steps` | `1` | 每次 optimizer step 累积的 micro-batch 数，必须大于 0。 |
| `learning_rate` | `2e-5` | AdamW 初始学习率。 |
| `weight_decay` | `0.01` | AdamW weight decay。 |
| `warmup_steps` | `0` | 线性 warmup 的 optimizer update 数，之后使用 cosine 衰减。 |
| `max_grad_norm` | `1.0` | 梯度裁剪上限；小于等于 0 时关闭裁剪。 |
| `save_steps` | `100` | 每多少个 optimizer update 保存 checkpoint；小于等于 0 时关闭周期保存。 |
| `output_dir` | `None` | 默认使用递增的 `runs/locateanything/train*`；传入路径时直接使用该目录。 |
| `resume` | `False` | `True` 从 `output_dir/last_checkpoint` 恢复；也可直接传入具体 `step-*` checkpoint 目录。 |
| `seed` | `0` | Python、NumPy、Torch、DistributedSampler 和 YOLO 负类采样种子；每个 rank 使用 `seed + rank`。 |
| `lora_rank` | `64` | LLM attention/MLP LoRA rank。 |
| `lora_alpha` | `128` | LLM LoRA alpha。 |
| `lora_dropout` | `0.05` | LLM 及可选视觉 LoRA dropout。 |
| `vision_lora_rank` | `0` | `0` 关闭视觉 LoRA；正整数启用，其 alpha 固定为 `2 × vision_lora_rank`。connector 在 LoRA 模式下始终参与训练。 |
| `negative_ratio` | `1.0` | 仅影响 YOLO YAML 适配。负类目标数按 `ceil(正类数 × ratio)` 计算，并受正类数和下一项共同限制。 |
| `max_negative_classes` | `32` | 每张 YOLO 训练图允许采样的最大负类数。 |

训练固定使用 AdamW、线性 warmup + cosine scheduler、`block_size=6` 的 PBD 标签和非 reentrant
gradient checkpointing。这些目前不是可公开覆盖的 `train()` 参数。

`LocateTrainResult` 提供 `save_dir`、`final_model`、`last_checkpoint`、`method`、`steps`、`epochs` 和
`final_loss`。LoRA 产物包含 adapter、connector 和 base-model manifest；全参数产物为分片 Safetensors，两者均可
再次传给 `LocateAnything(final_model)`。

## COCO 验证接口

```python
metrics = model.val(
    data="coco.yaml",
    device="0,1,2,3,4,5,6,7",
    batch=1024,
    protocol="paper",
)
print(metrics.results_dict)
```

`LocateAnything.val()` 默认直接在模型所在的单张 NPU 上运行。`device` 传入多个互不重复的 NPU 编号时，
它会自动通过 torchrun/HCCL 启动对应数量的 worker，每个 rank 加载一份 BF16 + SDPA 模型；K8S 多节点
父进程使用 `device=None` 时会自动选择各节点全部可见 NPU。

### 基础验证参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `data` | `"coco.yaml"` | 必须可解析到 COCO val2017 的 5000 张图、`instances_val2017.json` 和 80 个类别。 |
| `device` | `None` | 普通环境等价于单卡 `"0"`；也可传任意数量互不重复的 NPU 编号。K8S 多节点必须保持 `None`。 |
| `output_dir` | `None` | 默认使用递增的 `runs/locateanything/val*`。`resume=True` 时必须显式指向已存在的验证目录。 |
| `generation_mode` | `"hybrid"` | `fast`、`slow` 或 `hybrid`。每 rank 的 `local_batch>1` 时必须是 `hybrid`。注意这与公开 `predict()` 的 NPU 默认 `slow` 不同。 |
| `max_new_tokens` | `8192` | 每张图最大输出 token 数，必须大于 0。 |
| `temperature` | `0.7` | 采样温度，必须大于等于 0。 |
| `top_p` | `0.9` | nucleus sampling 阈值，必须在 `(0,1]`。 |
| `batch` | `1` | 全局 batch。多卡时必须不小于总 world size 且能被其整除；`local_batch=batch/world_size`。不会在 OOM 后静默降级。 |
| `scheduler` | `"pipeline"` | hybrid MTP/AR 调度策略：`eager`、`hold_ar`、`ar_first`、`pipeline` 或 `adaptive`。 |
| `protocol` | `"paper"` | `paper` 使用逐图 GT 正类 prompt；`closed_set` 使用全部数据集类别、错误类别计 FP、零分参与宏平均；两者都使用短边 840 和 Bilinear。`legacy` 使用原图和旧指标。 |
| `seed` | `0` | 每张图使用 `seed + image_id`，便于在动态 rank 划分下复现采样。 |
| `max_images` | `0` | `0` 验证全部 5000 张；正整数只验证按 image ID 排序后的前 N 张，适合 smoke。 |
| `resume` | `False` | 继续指定 `output_dir` 中的 `predictions.rank*.jsonl`。协议、global/local batch、world size 或节点布局不匹配都会拒绝。 |
| `allow_download` | `False` | 禁止 worker 自动下载数据或模型文件；`True` 允许按当前数据配置和 HF 缓存规则下载。 |

validator 内的 `repetition_penalty` 当前固定为 `1.1`，dtype 固定为 BF16，attention 实现固定为 SDPA；
它们不是 `val()` 可覆盖参数。validator worker 会继承构造模型时的 `npu_fast_path` 策略。

### Continuous batching 与动态调度

下表中的 `None` 表示根据 `local_batch` 和 world size 派生有效值，不是始终关闭。

| 参数 | 签名默认值 | 最终有效值与作用 |
| --- | --- | --- |
| `continuous_batching` | `None` | `local_batch=1` 时为 `False`，大于 1 时为 `True`。样本完成后立即补充新样本，减少长短输出差异导致的空转。 |
| `dynamic_scheduling` | `None` | 单卡为 `False`，多卡/多节点为 `True`。各 rank 通过 TCPStore 原子领取图片，不需要每 batch HCCL 同步。 |
| `refill_batch` | `None` | `local_batch>1` 时默认为 `min(8, local_batch)`，否则为 `0`。显式传 `0` 时 runtime 使用 `max(1, local_batch // 16)` 自动值。 |
| `continuous_window` | `1` | 仅在 `continuous_batching=False` 的旧式窗口路径生效，每窗最多读取 `local_batch × continuous_window` 张图。不能与 continuous batching 同时启用。 |

`continuous_batching=True` 要求 `local_batch>1`且 `generation_mode="hybrid"`。`local_batch=1` 继续使用
单图生成路径，不会因其他默认参数而切换到批量 runtime。

### KV cache、形状与图执行参数

| 参数 | 签名默认值 | 最终有效值与作用 |
| --- | --- | --- |
| `static_kv_cache` | `False` | 使用静态 KV cache。不能与 paged KV 同时开启，且 continuous batching 不支持它。 |
| `paged_kv_cache` | `None` | `local_batch=1` 时为 `False`，大于 1 时为 `True`。在支持的 NPU 快路上使用 block table、scatter 写入和 paged attention。 |
| `max_duplicate_boxes` | `0` | `0` 关闭连续相同 box 的退化终止保护；正整数允许该 box 连续出现指定次数，再次重复时结束当前样本。 |
| `shape_bucketing` | `False` | 将 decode batch 和 KV 长度 padding 到固定桶，有利于稳定形状，但会增加 padding 计算。 |
| `kv_bucket_size` | `128` | shape bucketing 的 KV 长度桶粒度，必须是正整数。 |
| `npu_graph` | `False` | 尝试捕获 NPU Graph；必须同时设置 `shape_bucketing=True`。当前 910B2 实测不建议默认启用。 |
| `visual_batching` | `None` | `local_batch=1` 时为 `False`，大于 1 时为 `True`。将相同 `image_grid_hws` 的 MoonViT 输入打包成 TND 多段序列。 |
| `direct_paged_decode` | `True` | paged 路径直接执行 Qwen decoder layer，避免构造最终不使用的 4D attention mask。 |
| `device_repetition_cache` | `True` | paged 路径把 repetition-penalty token 历史保留在 NPU 并增量追加。 |
| `qsample_reservoir` | `False` | 实验性地预生成 qSample 随机数；当前 910B2 实测无收益，因此默认关闭。 |
| `overlap_prefill` | `True` | continuous + paged NPU 路径在独立 stream 中处理新样本的 MoonViT/prefill，与已激活样本 decode 重叠。 |
| `candidate_top_p` | `True` | 使用语义等价的 candidate top-p 采样快路；需要对照不同 CANN/TorchNPU 行为时可关闭。 |

`static_kv_cache=True` 时需要显式设置 `paged_kv_cache=False` 和 `continuous_batching=False`。上表快路均限定于
eval/no-grad；结束验证并进入训练前会释放非参数推理缓存。

### NPU kernel 融合与 CPU 运行时

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `fused_qkv` | `False` | 拼接 Q/K/V 投影的实验路径；当前 910B2 实测未提升吞吐且增加显存。 |
| `fused_add_rms_norm` | `True` | 融合 attention residual add 和 post-attention RMSNorm。 |
| `fused_mlp` | `False` | Gate/Up + SwiGLU 实验融合；当前 910B2 实测未提升吞吐且增加显存。 |
| `cpu_affinity` | `True` | 根据 `npu-smi topo` 将每个 worker 绑定到对应 CPU/NUMA 范围，并把 PyTorch inter-op 线程限制为 1。 |

### Scheduler 可选值

| 值 | 调度行为 |
| --- | --- |
| `pipeline` | 默认值。paged cache 下尽可能混合推进 AR/MTP 行，并及时处理模式切换。 |
| `eager` | 每轮先推进 MTP，再推进 AR。别名为 `default` 和 `normal`。 |
| `ar_first` | 每轮先推进 AR，再推进 MTP。支持 `ar-first`、`repair_first` 和 `repair-first` 别名。 |
| `hold_ar` | 有 AR 行时最多连续推进 5 步，再让出给 MTP。别名为 `hold` 和 `hold-ar`。 |
| `adaptive` | 当 MTP 小组不超过 3 行时暂时优先 AR，否则按 AR 后 MTP 推进。 |

### 建议配置

论文口径精度复现使用单卡 batch 1：

```python
metrics = model.val(
    data="coco.yaml",
    device="0",
    protocol="paper",
    batch=1,
    generation_mode="hybrid",
    max_new_tokens=8192,
    temperature=0.7,
    top_p=0.9,
    seed=0,
)
```

910B2 8 卡高吞吐验证使用全局 batch 1024（每 rank local batch 128）：

```python
metrics = model.val(
    data="coco.yaml",
    device="0,1,2,3,4,5,6,7",
    batch=1024,
)
```

128 张 smoke 与断点续跑示例：

```python
metrics = model.val(
    device="0,1,2,3,4,5,6,7",
    max_images=128,
    batch=128,
    output_dir="runs/locateanything/val_smoke",
)

# 中断后使用同一目录和同一组生成/协议参数
metrics = model.val(
    device="0,1,2,3,4,5,6,7",
    max_images=128,
    batch=128,
    output_dir="runs/locateanything/val_smoke",
    resume=True,
)
```

### 验证指标与产物

`metrics` 为 `LocateMetrics`，并同时设置到 `model.metrics`。

| 属性/方法 | 说明 |
| --- | --- |
| `results_dict` | 扁平指标，包含 F1@0.50、F1@0.95、Mean F1、mean GT IoU 和非标准固定分数 AP。 |
| `speed` | 总 token、全局 tokens/s、images/s、boxes/s、显存峰值和可用的 NPU 利用率采样。 |
| `counts` | 图片、预测框、未知标签、解析警告、空预测、推理错误和重复保护触发数。 |
| `per_class` | IoU 0.50、0.95 及 Mean 的逐类结果。 |
| `coco_ap` | 固定 `score=1.0` 的非标准 COCO AP 诊断，不参与论文 F1 或 fitness。 |
| `summary(decimals=5)` | 返回逐类摘要，可调整小数位数。 |
| `save_dir` | 实际验证目录。 |

验证目录保留 `predictions.rank*.jsonl`、`predictions.json`、`metrics.json` 和 `summary.txt`。
LocateAnything 不输出 confidence；`predictions.json` 中的 `score=1.0` 只是为了调用 COCO evaluator，不会写回模型结果。

### CD-FSOD 六数据集 zero-shot

通过`model.val_cd_fsod()`可直接启动六数据集验证：

```python
metrics = model.val_cd_fsod(
    device="0,1,2,3,4,5,6,7",
    batch=512,
    protocol="closed_set",
    output_dir="runs/locateanything/cd_fsod",
    max_images_per_dataset=0,
    resume=False,
)
```

验证覆盖ArTaxOr、DIOR、FISH（DeepFish）、NEU-DET、UODD和clipart1k，每个数据集仅执行一次。默认配置
虽然使用现有`*-1shot.yaml`定位测试目录，但运行时只读取`val`与`annotations.val`，不会读取训练标注或
执行few-shot适配。8卡全局batch 512对应每rank 64；六数据集共享一次HCCL启动和一份动态任务队列，
每个rank只加载一次模型。

类别prompt会把下划线和连字符自然化为空格，并排除完整测试集中没有GT的`DUMMY_CLS`。指标以六个
数据集等权平均。`paper`输出论文式F1；`closed_set`以严格计数F1为主，同时保存同一预测的辅助
paper-style F1。固定score AP只作诊断。`max_images_per_dataset`按每个数据集限制图片数，
适合smoke；resume会校验六份标注哈希、模型revision、生成参数、global/local batch和world size。

## 输入、设备和当前边界

- 训练支持官方 ShareGPT JSONL recipe 的图片、多图和纯文本样本，也可在线适配 YOLO detection YAML；明确拒绝视频字段。
- `protocol="paper"`和`protocol="closed_set"`都会以 PIL Bilinear 保持长宽比将短边缩放到 840，再由模型 processor 完成 patch-grid 对齐。
- Ascend 路径以 BF16 + SDPA 完成真实验证；CUDA 首版只承诺静态/mock 兼容。
- 训练上下文上限为 4096；不支持在线 stream packing、视频/摄像头、量化、MagiAttention、ONNX、TensorRT 或 Ascend OM 导出。
- 论文精度使用 batch 1。大 batch 与 NPU 融合路径保持生成语义，但 BF16、padding 和采样执行顺序可造成数值或随机漂移。
- 首次分配 NPU 显存时可能出现一次 `NPUCachingAllocator` 32-padding 诊断，不代表验证失败。

## NPU 记录

- [LocateAnything-3B 910B2 优化报告](../npu-opt/优化记录/locateanything/locateanything_3b_910b2_report.md)
- [结构化基准结果](../npu-opt/优化记录/locateanything/locateanything_3b_910b2_results.json)
- [模型包详细说明](../../../ultralytics/models/locateanything/README.md)
