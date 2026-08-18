# PE-Spatial 全系列接入与 L/14 Ascend 910B2 优化报告

## 结论

本次接入支持官方 PE-Spatial T/S/B/L/G 五种视觉编码器，并为 910B2 增加保持官方 `interleave` 数学语义的 RoPE 自动快路径。

PE-Spatial-L/14 使用官方 303,964,160 参数 checkpoint 实测：

- 448 推理从 27.033 ms 降至 23.090 ms，提升 **1.171×**。
- 448 训练前向+反向从 82.499 ms 降至 69.702 ms，提升 **1.184×**。
- 644 推理从 28.838 ms 降至 23.589 ms，提升 **1.223×**。
- 644 训练前向+反向从 88.127 ms 降至 77.286 ms，提升 **1.140×**。
- 644 长窗口中 AICore 维持约 98%～100%，NPU 遥测约 94%～100%；收益主要表现为相同工作更早完成，而不是继续提高已经接近饱和的核心利用率。

完整快路径超过预设的 3% 加速门槛，默认保留 `auto`。

## 环境

- NPU：Ascend 910B2，64 GiB HBM
- CANN：9.1.0
- PyTorch：2.12.0+cpu
- TorchNPU：2.12.0
- timm：1.0.28
- AMP：FP16
- Ascend JIT：关闭
- checkpoint：`facebook/PE-Spatial-L14-448`，约 1.22 GB
- 日期：2026-08-07

## 模型接入

统一入口为：

```python
PESpatial(scale: str, pretrained: bool | str = True)
```

| scale | 官方 checkpoint | 通道 | patch/stride | 原生分辨率 | 真实权重验证 |
|---|---|---:|---:|---:|---|
| t | PE-Spatial-T16-512 | 192 | 16 | 512 | 结构与模拟权重 |
| s | PE-Spatial-S16-512 | 384 | 16 | 512 | 结构与模拟权重 |
| b | PE-Spatial-B16-512 | 768 | 16 | 512 | 结构与模拟权重 |
| l | PE-Spatial-L14-448 | 1024 | 14 | 448 | 完整实测 |
| g | PE-Spatial-G14-448 | 1536 | 14 | 448 | 结构与 D=96 算子实测 |

实现仅保留 Apache-2.0 的视觉配置、ViT、attention 和 2D RoPE，不包含 PE-Core 文本塔、PE-Lang、视频及评测代码。输入按官方方式从 `[0,1]` 归一化到 `[-1,1]`，输出最终层 patch token；空间接口为 BCHW，序列接口为 BNC。

YOLO 示例为 `ultralytics/cfg/models/rf-det/pe-spatial-yolo11.yaml`，使用 L/14 和 YOLO11-L neck/head。检测 stride 是 `[7,14,28]`，因此 `imgsz=640` 会由现有逻辑调整到 644。

## Ascend 快路径

### 为什么不能直接复用 DINOv3 的推理算子

DINOv3 使用 GPT-NeoX 风格的 `half` RoPE；PE-Spatial 使用 GPT-J 风格的 `interleave`。910B2 上 `npu_apply_rotary_pos_emb` 的 A2 接口只支持 `half`，直接套用会产生错误结果。

本实现采用：

1. QKV projection 后保持 BSND。
2. Q、K 分别调用 `npu_rotary_mul(..., rotary_mode="interleave")`。
3. RoPE 后再转为 BNSD，交给 PyTorch SDPA。
4. class token 使用 `sin=0、cos=1` 与 patch token 一次处理。

`PE_SPATIAL_ROPE_BACKEND` 支持：

- `auto`：默认；按设备、dtype、shape 和 JIT 状态自动融合或回退。
- `rotary_mul`：严格融合，不支持时明确报错。
- `manual`：强制官方 PyTorch 路径。

JIT 开启时由于 910B2 不支持 `interleave` RotaryMul JIT，`auto` 会安全回退。

### 缓存

- RoPE frequency/sin/cos 按设备、dtype 和网格保存单条实例缓存，尺寸变化时替换。
- 冻结位置参数时缓存最近一次绝对位置编码插值；微调、参数版本或尺寸变化时失效。
- 动态缓存不是 module buffer，不参与 DDP buffer broadcast，也不进入 state dict。
- 深拷贝、保存 checkpoint 时主动丢弃缓存，避免保存 NPU 临时张量和扩大 checkpoint。

## 正确性结果

- 官方 L/14 checkpoint 的 293 个 state dict 项严格完整加载，参数量为 303,964,160。
- 448 输出 `(1,1024,32,32)`，644 输出 `(1,1024,46,46)`。
- FP32、FP16、BF16 的 448/644 前向均为有限值，BNC 与 BCHW 完全一致。
- 真实 910B2 上 D=64 和 D=96 的 FP32、FP16、BF16 `interleave` RotaryMul 前后向均通过。
- 官方 L/14 首层 auto/manual 的输出、loss、输入梯度和 QKV 权重梯度完全一致。
- YOLO11-L 冻结 backbone 的两步 FP16 loss/backward/SGD、保存和重新加载通过。
- 两卡 HCCL DDP smoke 通过，最终复跑平均 loss 为 11.904262；动态缓存未被注册为 DDP buffer。

## 性能消融

以下为 3 次双向顺序重复的中位数。`original_manual` 精确复现官方只缓存 frequency、每层为 Q/K 分别计算 sin/cos 并执行手工 RoPE；`cached_manual` 只加入系数缓存；`rotary_mul` 再加入融合算子；`full` 还启用冻结位置编码缓存。

### 纯推理，batch=1，FP16 AMP

| 分辨率 | 路径 | ms/step | 相对原始加速 | 峰值分配 MiB |
|---:|---|---:|---:|---:|
| 448 | original_manual | 27.033 | 1.000× | 1201.06 |
| 448 | cached_manual | 24.465 | 1.105× | 1201.06 |
| 448 | rotary_mul | 23.717 | 1.140× | 1199.06 |
| 448 | full | **23.090** | **1.171×** | 1199.06 |
| 644 | original_manual | 28.838 | 1.000× | 2410.33 |
| 644 | cached_manual | 27.797 | 1.037× | 2410.33 |
| 644 | rotary_mul | 23.687 | 1.218× | 2402.06 |
| 644 | full | **23.589** | **1.223×** | 2404.82 |

### 训练前向+反向，batch=1，FP16 AMP

| 分辨率 | 路径 | ms/step | 相对原始加速 | 峰值分配 MiB |
|---:|---|---:|---:|---:|
| 448 | original_manual | 82.499 | 1.000× | 2824.23 |
| 448 | cached_manual | 79.061 | 1.044× | 2800.68 |
| 448 | rotary_mul | 69.595 | 1.185× | 2992.89 |
| 448 | full | **69.702** | **1.184×** | 2992.89 |
| 644 | original_manual | 88.127 | 1.000× | 3986.53 |
| 644 | cached_manual | 86.831 | 1.015× | 3937.92 |
| 644 | rotary_mul | 77.345 | 1.139× | 4334.88 |
| 644 | full | **77.286** | **1.140×** | 4334.88 |

融合 RoPE 的 backward 会保存算子所需张量，因此训练峰值相对原始 manual 增加约 169 MiB（448）和 348 MiB（644）；推理峰值基本持平。对于冻结 backbone 的检测训练不会保存 backbone backward，实际显存行为更接近推理列。

644 长窗口采样结果：

| 场景 | 路径 | AICore | NPU |
|---|---|---:|---:|
| 推理 | original_manual | 99.4% | 94.3% |
| 推理 | full | 98.4% | 95.7% |
| 训练前后向 | original_manual | 100.0% | 100.0% |
| 训练前后向 | full | 100.0% | 99.0% |

两条路径都已使 AICore 接近饱和，`npu-smi` 的一秒粒度 NPU 遥测存在短窗口波动，因此不据此宣称利用率提升；可重复收益来自减少三角函数、布局变换和 RoPE kernel 的执行时间。

## 复现

```bash
# 正确性与轻量测试
pytest -q tests/test_pe_spatial_backbone.py tests/test_pe_spatial_rope_ascend.py

# 官方L/14真实权重测试
RUN_PE_SPATIAL_L_TESTS=1 pytest -q --slow tests/test_pe_spatial_real_npu.py

# 性能消融
python docs/my-docs/npu-opt/优化记录/pe_spatial/benchmark_pe_spatial_npu.py \
  --output /tmp/pe_spatial_l14_results.json

# 两卡DDP
python -m torch.distributed.run --standalone --nproc_per_node 2 \
  docs/my-docs/npu-opt/优化记录/pe_spatial/pe_spatial_ddp_smoke.py
```

原始结构化结果见 `pe_spatial_l14_910b2_results.json`。

## 限制

- T/S/B/G 没有下载真实 checkpoint；其结构、权重接口和 parser 已覆盖，G 的 D=96 RoPE 已在真实 NPU 验证。
- 首版只输出最后一层视觉特征，不提供多层特征、文本塔或可变 token 打包。
- G/14 checkpoint 约 7.4 GB，实际训练或推理前仍需单独验证其主机内存、下载和峰值显存。
