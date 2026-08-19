# LocateAnything

本模块把 NVIDIA LocateAnything-3B 接入 Ultralytics Python API，提供图片定位、结构化结果，以及独立的原生
PyTorch LoRA/全参数 SFT 训练循环。模型代码和权重固定到 revision
`c32291ca5e996f5a7a485845b4f57a233936bba0`。

## 安装

先按目标硬件安装 PyTorch（Ascend 环境还需匹配版本的 `torch_npu`），再安装可选依赖：

```bash
pip install -e '.[locateanything]'
```

该 extra 固定使用 `transformers==5.14.1`，并安装匹配的 Accelerate 和 PEFT。基础
`import ultralytics` 不会导入这些可选依赖。

## 推理

```python
from ultralytics import LocateAnything

model = LocateAnything(
    "nvidia/LocateAnything-3B",
    device="npu:0",
    dtype="bfloat16",
    npu_fast_path="auto",
)

results = model.detect("image.jpg", classes=["person", "car"])
results = model.ground("image.jpg", "穿红衣服的人", multiple=True)
results = model.detect_text("document.png")
results = model.ground_text("document.png", "订单号")
results = model.ground_gui("screen.png", "搜索按钮", output="point")
results = model.point("image.jpg", "红绿灯")

result = results[0]
print(result.summary())
print(result.to_json())
result.save("result.jpg")
```

`predict()` 接受单张图片或图片列表，以及 `task`、`prompt`、`generation_mode`、
`max_new_tokens` 和 `stream`。NPU 默认使用稳健的 `slow` 模式；CUDA 默认使用 `hybrid`；可以显式为 NPU
选择 `hybrid`。结果只保留模型实际返回的框、点和标签，不生成虚假的 confidence。

## 验证

`model.val()` 默认直接在模型所在的单张NPU上验证完整MS COCO 2017 val：

```python
from ultralytics import LocateAnything

model = LocateAnything("nvidia/LocateAnything-3B", local_files_only=True)
metrics = model.val(data="coco.yaml")
print(metrics.results_dict)
```

传入多个设备时会自动启动相应数量的HCCL worker；K8S多节点环境中使用`device=None`会选择各节点全部
可见NPU。验证器默认使用BF16、SDPA、`hybrid`、8192个输出token和temperature 0.7。`metrics` 与
YOLO/RT-DETR一样是validator持有的指标对象，提供 `results_dict`、`speed`、
`counts`、`per_class`、`coco_ap`、`save_dir` 和 `summary()`。

`protocol="paper"`为默认值：验证专用预处理器以PIL Bilinear保持宽高比将COCO图片短边
缩放到840，再交给模型原生processor完成patch-grid对齐。每张图的prompt只包含该图GT中
出现的正类别，按COCO category id排序并以`</c>`分隔。输出框在缩放图上裁剪后再映射回
原图。传入`protocol="legacy"`可恢复原图、全80类prompt和旧式指标。两种协议的resume分片
不能混用。

传入`protocol="closed_set"`仍使用短边840和相同坐标回映射，但每张图提示数据集全部有效类别。
非GT类别预测保留并计为FP，逐类计数precision/recall的零分也参与宏平均。严格closed-set F1作为
`official_locate_metrics`和fitness；同一次预测还会写入`auxiliary_paper_metrics`，使用完整paper
过滤与safe-mean口径复算，便于量化协议差异。固定score AP仍只是非标准诊断。

`batch`默认仍为1，并统一解释为全局batch。多卡时要求它不小于总world size且能被world size整除，
每rank实际容量为`local_batch = batch / world_size`。`local_batch=1`保持单图生成路径；大于1时必须使用
`generation_mode="hybrid"`，并自动启用continuous batching、paged KV cache和visual batching，
`refill_batch`默认取`min(8, local_batch)`。跨rank动态调度在任意多卡配置下默认启用。
验证器会以 `scheduler="pipeline"` 执行批量MTP/AR和逐行KV cache。
尾批只使用实际剩余样本；如果请求的batch导致OOM，会报告请求值和NPU显存状态，不会静默降级。
`metrics.speed` 另外包含总输出token、全局 `tokens_per_second`、平均每图token数和显存峰值。

`local_batch>1`的hybrid验证默认使用`max_duplicate_boxes=0`，不会提前终止连续生成的相同box，以保持模型
生成语义不变。用户可显式设为正整数启用退化循环保护；正常的`<ref>...</ref>`会重置计数，命中时会在
`generation_stats.stopped_repetition`和`metrics.counts.repetition_stopped_images`中记录。

LocateAnything不输出confidence，因此论文口径主指标是F1@0.50、F1@0.95和IoU 0.50–0.95上的
Mean F1。匹配保留模型生成顺序，普通GT一对一，crowd命中不计FP，每图每类最多计入
100个预测；precision和recall先逐类计算，再按官方FastEval方式聚合。mean GT IoU仅作为
诊断项，不是论文中的Mean。验证器也会生成固定
`score=1.0` 的COCO bbox JSON供 `faster-coco-eval` 参考，但该AP会明确标记为非标准结果。使用
`max_images=8` 可执行smoke；传入已有 `output_dir` 和 `resume=True` 可继续中断的验证。
论文精度使用每设备batch 1报告；单卡默认`batch=1`与其一致。多卡可使用`batch=world_size`保持每设备
batch 1，`metrics.json`和`summary.txt`会同时记录global/local batch及节点布局。

Ascend 910B2在首次分配模型显存时可能输出一次 `NPUCachingAllocator` 的32-padding提示。这是当前
CANN/SoC组合的底层内存格式诊断，不代表推理或验证失败；验证器不会通过全局日志过滤隐藏其他NPU问题。

### CD-FSOD zero-shot

`val_cd_fsod()`会在一次torchrun生命周期内验证ArTaxOr、DIOR、FISH/DeepFish、NEU-DET、UODD和
clipart1k，每个rank只加载一次模型。它只读取六个配置的`val`和`annotations.val`字段，不读取或使用
1/5/10-shot训练标注：

```python
metrics = model.val_cd_fsod(
    device="0,1,2,3,4,5,6,7",
    batch=512,  # 全局batch，每rank local batch=64
    protocol="closed_set",
    output_dir="runs/locateanything/cd_fsod",
)
print(metrics.results_dict)
```

该接口默认仍沿用论文COCO协议；可显式选择上述严格closed-set。类别名中的`_`和`-`会在prompt中转成空格，模型输出的原名与
自然化别名都能映射回原category id；无任何测试GT的`DUMMY_CLS`会被排除，合法category id 0则保留。
逐图prompt使用该图GT正类别，因此结果会明确标记为oracle positive-category，而不是开放类别发现。
主目录生成六数据集等权汇总、CSV和共享rank JSONL，各数据集子目录保存独立指标与原始id预测。
由于ArTaxOr使用字符串image id，非标准AP另用可追溯的整数映射文件调用COCO evaluator。

`npu_fast_path="auto"` 默认在固定revision、910B、BF16/FP16和eval/no-grad条件下启用Qwen GQA
Prompt/Incre Flash Attention、RotaryMul、RMSNorm、MoonViT Fusion Attention及批量top-p qSample。形状或算子
不满足约束时自动回退到原SDPA实现；可设为`"off"`做基线对照，或设为`"strict"`让任何回退立即报错。
该快路径不用于训练，也不会改变CUDA/CPU路径。

大batch验证还会复用持久化qSample缓冲，并按`npu-smi topo`绑定worker到NPU对应的CPU/NUMA范围，
同时将PyTorch inter-op线程限制为1。AR采样只返回sampled token，不再构造随后丢弃的完整概率矩阵。
可传`cpu_affinity=False`关闭绑定。`static_kv_cache=True`和旧式`continuous_window>1`仍是实验开关。
`paged_kv_cache=True`已改为NPU原生scatter + paged attention，不再为MTP回复连续KV视图；
`visual_batching=True`会将相同`image_grid_hws`的MoonViT输入打包为TND多段序列。两者都只在
eval/no-grad的批量验证路径生效，训练仍执行原始可微计算。

continuous paged路径还会在一次forward中混合推进彼此独立的AR/MTP行，并预计算复用36层共享的block
写入索引、mask、block table格式和RoPE。token历史、decode suffix及cache position先在Host整批padding，
再执行单次H2D，避免逐样本小传输和NPU到CPU同步。图片processor使用有界后台预取，rank JSONL由后台线程
按完成顺序逐条flush；正常或异常退出都会先排空已提交记录。

默认还会启用`direct_paged_decode=True`和`device_repetition_cache=True`：前者直接执行Qwen decoder层，跳过
固定revision中最终不会使用的4D attention mask构造；后者把repetition penalty的token历史留在NPU并增量
追加。`overlap_prefill=True`会在独立NPU stream中为新补槽样本执行MoonViT和prefill，现有decode无需等待；
prefill event完成前对应slot不会进入活跃集合。这些快路径都限定于eval/no-grad，训练继续使用原始可微forward。
`qsample_reservoir=True`是用于不同CANN版本A/B的实验开关；当前910B2环境实测无收益，因此默认关闭。

`fused_add_rms_norm=True`默认融合attention residual add和post-attention RMSNorm。`fused_qkv=True`与
`fused_mlp=True`分别启用QKV拼接投影和Gate/Up+SwiGLU实验路径；910B2实测后两者吞吐没有提升且增加
显存，因此默认关闭，仅保留给不同CANN/TorchNPU版本做显式A/B。所有融合只在eval/no-grad使用，训练前会
释放非参数推理缓存并恢复原始可微forward。

8卡高吞吐配置中，旧版每rank `batch=128`对应现在的全局`batch=1024`：

```python
metrics = model.val(
    data="coco.yaml",
    device="0,1,2,3,4,5,6,7",
    batch=1024,
)
```

生成池默认在空闲槽位达到`local_batch // 16`时成组补充，避免每完成一张图就执行一次小规模MoonViT和
prefill；可用`refill_batch`显式调整水位。动态队列通过独立TCPStore跨rank、跨节点原子领取图片，不增加
逐batch HCCL同步点。每张图片完成后立即提交rank JSONL后台写入，因此中断后仍可用`resume=True`重建
未完成队列。单卡`batch=1`继续使用单图生成路径；多卡`batch=world_size`则每rank使用相同路径。

`shape_bucketing=True`可将decode batch和KV长度补到固定桶，`npu_graph=True`则进一步尝试对纯Tensor
Qwen MLP热段捕获ACL Graph。Graph要求同时打开shape bucket，且worker会局部使用
`TASK_QUEUE_ENABLE=1`。当前910B2实测Graph replay慢于eager，因此两项默认关闭，不建议用于正式验证。

## 数据

训练支持官方 ShareGPT JSONL recipe：

```json
{
  "my_data": {
    "annotation": "annotations.jsonl",
    "root": "/data/images",
    "repeat_time": 1.0,
    "data_augment": false
  }
}
```

每条 JSONL 记录使用 `conversations` 和可选的 `image`/`image_list`：

```json
{"conversations":[{"from":"human","value":"<image-1>Locate a single instance that matches the following description: car."},{"from":"gpt","value":"<ref>car</ref><box><100><200><500><700></box>"}],"image":"a.jpg"}
```

也可以直接传入 YOLO detection YAML。适配器在线把归一化 `xywh` 转成 `[0,1000]` 的 `xyxy` token，
用固定种子加入不多于正类别数的负类别；背景图输出 `<box>none</box>`。首版忽略 segmentation、pose 和
OBB 附加标注，并明确拒绝视频字段。

## 训练

```python
result = model.train(
    data="dataset.yaml",       # YOLO YAML、recipe JSON 或单个 JSONL
    method="lora",            # lora / full
    device="0,1,2,3",
    epochs=1,
    batch=1,
    max_seq_length=4096,
)
print(result.final_model)
```

LoRA 默认只注入 LLM attention/MLP（rank 64、alpha 128、dropout 0.05），冻结视觉主干并训练 connector；
多卡使用 DDP。`method="full"` 至少需要两个设备，CUDA 使用 PyTorch FSDP2，Ascend 使用
`torch_npu.distributed.fsdp.fully_shard`。两种方式都由本模块自己的 PyTorch 循环驱动，不依赖 Hugging
Face Trainer、DeepSpeed 或 ZeRO。

checkpoint 保存模型、优化器、scheduler、训练进度和各 rank RNG；全参数 checkpoint 使用
`torch.distributed.checkpoint`，可在不同 world size 下恢复。最终 LoRA 目录包含 adapter、connector 和
base-model manifest；全参数目录是分片 Safetensors。两者都可直接传给 `LocateAnything(output_dir)`。

## 当前边界

- Ascend 路径使用 BF16 + SDPA，训练上下文上限为 4096；CUDA 首版只承诺静态/mock 兼容。
- 不支持 CLI、视频/摄像头、在线 stream packing、量化、MagiAttention、ONNX/TensorRT/Ascend OM 导出。
- 官方权重采用 NVIDIA 非商业许可，仅允许学术及非营利研究用途；使用和再分发前请阅读模型仓库中的
  `LICENSE` 并保留要求的版权与归属声明。
