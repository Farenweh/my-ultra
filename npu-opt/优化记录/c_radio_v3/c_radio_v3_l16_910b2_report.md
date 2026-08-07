# C-RADIOv3 全系列接入与 L/16 Ascend 910B2 优化报告

## 结论

本次新增统一的 `CRADIOv3(scale, pretrained=True)` 视觉 backbone，支持官方 B/L/H/g 四种规格；只有 L/16 下载了真实权重并完成模型、检测训练和 NPU 性能验证，其余规格仅依据固定 revision 的官方 metadata 建立解析表。

原生安全实现不执行 Hugging Face `trust_remote_code`，也不依赖 Transformers。固定 revision 的官方参考实现仅在隔离环境中执行一次：修正 LayerNorm `eps` 后，512×512、FP16、SDPA 路径的 1,048,576 个空间特征与官方结果逐元素完全一致，最大和平均绝对误差均为 0。

910B2 上的 640×640 长窗口 ABBA 结果：

- 官方式 SDPA 且不缓存 CPE：20.451 ms/step。
- 仅 CPE 缓存：19.395 ms/step，提升 **1.054×**。
- 仅融合 attention：19.615 ms/step，提升 **1.043×**。
- 完整自动快路径：19.196 ms/step，提升 **1.065×**。

完整路径超过 640 主场景 3% 的目标，且没有出现超过 1% 的稳定退化，因此保留默认 `auto`。该负载在长窗口中四条路径的 AICore 均为 100%，收益来自减少 CPE 重算和 attention 开销，而不是继续抬高已经饱和的核心利用率。

## 环境

- NPU：Ascend 910B2，64 GiB HBM
- CANN：9.1.0
- PyTorch：2.12.0+cpu
- TorchNPU：2.12.0
- TorchVision：0.27.0+cpu
- timm：1.0.28
- AMP：FP16
- Ascend JIT：关闭
- 日期：2026-08-07
- 100 epoch checkpoint 最终验证复测：2026-08-08

## 模型接入

统一入口：

```python
CRADIOv3(scale: str, pretrained: bool | str = True)
```

| scale | 官方仓库 | 固定 revision | 通道 | 深度 | heads | 参数量 | LayerScale | 真实验证 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| b | `nvidia/C-RADIOv3-B` | `44653a0482cf460bb4f12595fc3cc3dfecc403d1` | 768 | 12 | 12 | 98.3M | 1e-5 | metadata |
| l | `nvidia/C-RADIOv3-L` | `9d0413465e8a91e67bbf2c1ad342815478d1b906` | 1024 | 24 | 16 | 319.9M | 1e-5 | 完整实测 |
| h | `nvidia/C-RADIOv3-H` | `d7fd0e2b0a1761f1af150582e06c41e9a99b0bf8` | 1280 | 32 | 16 | 651.6M | 无 | metadata |
| g | `nvidia/C-RADIOv3-g` | `28e70735780c22cd7fca20f0c509cb9fc3893aeb` | 1536 | 40 | 24 | 1.16B | 无 | metadata |

四种规格均为 patch/stride 16、推荐 512、最大边 2048。输入必须是三通道浮点 BCHW，宽高为 16 的倍数；按 OpenAI CLIP mean/std 归一化，不在 backbone 内缩放或补齐。输出会移除 8 个 CPE 前缀 token：`forward_sequence()` 返回 BNC，`forward()` 返回 BCHW。B/L checkpoint 中保留但不参与 CPE 前向的 `reg_token`，以保证权重结构严格一致。

L 权重共 319,934,464 个 Parameter，另有 6 个归一化 buffer；首次下载和逐张量装载约 355 秒，主机峰值约 3.84 GiB。safetensors 使用 meta-device 建模和逐张量复制，本地 PyTorch checkpoint 只允许 `weights_only=True` 的 tensor 字典；缺失、多余和形状错误都会明确失败。

检测示例为 `ultralytics/cfg/models/rf-det/c-radio-v3-yolo11.yaml`。默认使用 L/16 和 YOLO11-L neck/head，从单层空间特征构造 P3/8、P4/16、P5/32；推荐以 `freeze=1` 冻结第 0 层 backbone。

## Ascend 快路径

`CRADIO_V3_ATTENTION_BACKEND` 支持：

- `auto`：默认；910B2、FP16/BF16、非 JIT 且 shape 满足约束时调用 `npu_fusion_attention`，否则回退 PyTorch SDPA。
- `fusion`：严格要求融合，条件不满足时明确报错。
- `sdpa`：强制标准路径。

QKV projection 后保持 BSND 布局进入融合算子，避免先转 BNSD 再恢复。FP32、JIT、非 NPU、dtype/shape 不支持和算子不可用时不会误入融合路径。`torch_npu` 延迟导入，普通 CPU/CUDA 模型导入不会锁定 NPU 可见设备。

CPE 缓存均为实例上的单条普通属性：

- 可训练时保留官方随机 viewport 和 10% 位置 dropout，只复用基础采样网格。
- eval/no-grad 或整个 backbone 冻结时使用确定性 CPE，并缓存完整位置编码。
- 普通 Tensor 通过参数版本检测权重变化；最终验证重新加载得到的 inference tensor 使用无版本标记，不访问其不存在的 `_version`。
- 参数身份、分辨率、参数版本、设备或 dtype 变化会替换缓存，不无限增长。
- 缓存不进入 state dict 或 DDP buffer；深拷贝、设备迁移和权重重载会清空缓存。

官方 patch projection 保持 `im2patch + Linear`。预验证中 640 输入的 Linear 约 0.075 ms，而等价 Conv2d 约 0.867 ms，因此没有引入较慢的 Conv2d 分支。

## 正确性结果

- L safetensors 严格完整加载，结构、参数量、权重键和 `eps=1e-6` 与官方实现一致。
- 固定输入的官方参考与原生实现在 512、FP16、SDPA 下逐元素完全一致。
- 512、640、800 和 384×640 的 FP32/FP16/BF16 前向均为有限值，BNC/BCHW 完全一致。
- 真实 910B2 上 FP16/BF16 融合 attention 的输出、loss、输入梯度和 QKV 权重梯度与 SDPA 对齐。
- FP32 与 JIT 开启时正确回退；严格模式在不支持条件下明确失败。
- YOLO11-L 冻结 backbone 的两步 FP16 loss 为 10.932062、11.224338，backbone 无梯度，neck/head 参数更新；384×640 推理、757 项 state dict 保存和严格重载通过。
- 两卡 HCCL DDP smoke 通过，平均 loss 为 11.172514；动态缓存未注册为 DDP buffer。
- 100 epoch 训练产生的真实 `best.pt` 在 `torch.inference_mode()` 下重新加载后，NPU warmup、640 前向和 384×640 动态尺寸前向均通过；位置参数确认为 inference tensor，缓存可复用并随尺寸替换。

### 100 epoch checkpoint 最终验证

原训练在第 100 epoch 的在线验证结果为 P=0.749、R=0.658、mAP50=0.712、mAP50-95=0.519；训练结束后重新加载 `best.pt` 时，旧实现因读取 inference tensor 的 `_version` 而中止。

修复后在单张 910B2、FP16、batch 128 上完整处理 COCO val 5000 张图：

- Ultralytics 内部汇总：P=0.749、R=0.653、mAP50=0.714、mAP50-95=0.520。
- faster-coco-eval：AP50=0.720、AP50-95=0.527、AR100=0.680。
- 40/40 batch、warmup、前向、NMS 和 COCOeval 全部完成，没有 inference tensor 或缓存异常。

单卡矩形 batch 与原 16-rank 训练期验证的采样/汇总方式不同，约 0.001 的内部 mAP50-95 差异属于预期；本次修复不改变模型权重或 CPE 数学。

## 性能消融

表中是三次双向顺序重复的中位数。短窗口用于覆盖多分辨率和训练前后向；640 冻结前向另以每轮 300 step 重测，减少一秒粒度遥测和启动抖动。

### 冻结 backbone 前向，batch=1，FP16 AMP

| 分辨率 | official_sdpa | cpe_cache | fusion | full | full 加速 | full 峰值 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 20.211 | 20.171 | 20.281 | 20.187 | 1.001× | 1260.8 |
| 640 | 20.534 | 20.186 | 20.401 | 20.231 | 1.015× | 1280.2 |
| 800 | 24.184 | 23.826 | 23.673 | 23.259 | 1.040× | 1313.6 |
| 640，300-step 长窗口 | 20.451 | 19.395 | 19.615 | **19.196** | **1.065×** | 1280.2 |

短窗口 640 的 1.5% 和长窗口的 6.5% 说明该模型单步只有约 20 ms，短测容易受调度扰动；生产结论采用三轮 300-step 中位数，同时保留原始短测 JSON，避免隐藏波动。

### 完整可训练前向+反向，batch=1，FP16 AMP

| 分辨率 | official_sdpa | cpe_cache | fusion | full | full 加速 | official/full 峰值 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 63.212 | 63.157 | 61.412 | 61.431 | 1.029× | 2873.2 / 2824.8 |
| 640 | 64.925 | 64.630 | 62.522 | **61.892** | **1.049×** | 3467.6 / 3392.2 |

640 训练路径的融合 attention 使峰值分配减少约 75.4 MiB。640 的 AICore 在四条路径均达到 100%；`npu-smi` 的 NPU 百分比受一秒采样相位影响，不据此宣称额外利用率提升。

## 复现

```bash
# 轻量结构、加载器和attention测试
pytest -q tests/test_c_radio_v3_backbone.py tests/test_c_radio_v3_attention_ascend.py

# 官方L权重真实NPU测试
RUN_C_RADIO_V3_L_TESTS=1 pytest -q --slow tests/test_c_radio_v3_real_npu.py

# 四路径ABBA
python npu-opt/优化记录/c_radio_v3/benchmark_c_radio_v3_npu.py \
  --output /tmp/c_radio_v3_l16_results.json

# 两卡DDP
python -m torch.distributed.run --standalone --nproc_per_node 2 \
  npu-opt/优化记录/c_radio_v3/c_radio_v3_ddp_smoke.py
```

结构化原始数据见 `c_radio_v3_l16_910b2_results.json`。

## 限制

- B/H/g 没有下载、实例化或执行真实 checkpoint；不能把 L 的 NPU 性能结论外推到这些规格。
- 首版只输出最终空间特征，不包含 summary、teacher adaptor、文本编码器、视频、ViTDet 窗口或多层输出。
- 外部权重受 NVIDIA Open Model License 约束；仓库只提供加载器和架构实现，不分发权重。
- `trust_remote_code` 只用于已删除的隔离参考环境，不是运行时或安装依赖。
