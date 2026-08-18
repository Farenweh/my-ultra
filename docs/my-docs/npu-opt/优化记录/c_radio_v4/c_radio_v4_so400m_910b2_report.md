# C-RADIOv4 接入与 SO400M/16 Ascend 910B2 优化报告

## 结论

本次新增统一的 `CRADIOv4(scale, pretrained=True)` 视觉 backbone，支持官方 SO400M 和 H 两种规格。默认检测配置使用 SO400M；其固定 revision 官方权重、动态分辨率、冻结检测训练、checkpoint 重载、两卡 HCCL 和 NPU 性能均已验证。H 只完成官方 metadata、meta-device 结构和模拟 checkpoint 验证，没有下载真实权重。

SO400M 在 910B2、640×640、batch 1、FP16 下的完整自动快路径结果：

- 冻结前向：29.934 ms 降至 **23.366 ms**，提升 **1.281×**。
- 可训练前向+反向：94.503 ms 降至 **78.342 ms**，提升 **1.206×**。
- 可训练峰值分配：4291.9 MiB 降至 **4196.4 MiB**，减少约 95.5 MiB。
- 四种训练路径的 AICore 均达到 100%；收益来自更高效的 attention 和减少 CPE 重算，不是继续提高已经饱和的核心占用率。

完整路径在 512、640 和 800 均超过 3% 的目标且没有稳定退化，因此默认保留 `auto`。

## 环境

- NPU：Ascend 910B2，64 GiB HBM
- CANN：9.1.0
- PyTorch：2.12.0+cpu
- TorchNPU：2.12.0
- TorchVision：0.27.0+cpu
- timm：1.0.28
- Transformers：5.14.1，仅用于隔离参考验证
- AMP：FP16
- Ascend JIT：关闭
- 日期：2026-08-14

## 模型接入

统一入口：

```python
CRADIOv4(scale: str, pretrained: bool | str = True)
```

| scale | 官方仓库 | 固定 revision | 通道 | 深度 | heads | head dim | MLP | 参数量 | 真实验证 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| so400m | `nvidia/C-RADIOv4-SO400M` | `c0457f5dc26ca145f954cd4fc5bb6114e5705ad8` | 1152 | 27 | 16 | 72 | 4304 | 431.2M | 完整实测 |
| h | `nvidia/C-RADIOv4-H` | `0057b339059c0b9e1b4ba996f975410ebbfdfcc8` | 1280 | 32 | 16 | 80 | 5120 | 651.6M | metadata、meta结构和模拟权重 |

两种规格均为 patch/stride 16、10 个 CPE 前缀 token、推荐分辨率 512、最大边 2048。输入必须是三通道浮点 BCHW，宽高为 16 的倍数；按 OpenAI CLIP mean/std 归一化，不在 backbone 内缩放或补齐。`forward_sequence()` 返回去除前缀 token 的 BNC，`forward()` 返回 BCHW。

SO400M safetensors 为 1,724,986,512 bytes，严格加载得到 431,237,232 个 Parameter。H 的 meta-device 模型参数总数为 651,645,440。两者的 MLP 维度、前缀 token 和 checkpoint 键均按官方权重形状确定。

项目复用 C-RADIOv3 的纯 PyTorch ViT、CPE 和安全加载基础设施，并增加精确 MLP 隐藏维度和模型 family 策略；C-RADIOv3 的公开接口、权重键和运行路径保持不变。运行时不执行 `trust_remote_code`，不依赖 Transformers。

检测示例为 `ultralytics/cfg/models/rf-det/c-radio-v4-yolo11.yaml`，默认使用 SO400M/16 和 YOLO11-L neck/head，从单层空间特征构造 P3/8、P4/16、P5/32；推荐以 `freeze=1` 冻结第 0 层 backbone。

## Ascend 快路径

`CRADIO_V4_ATTENTION_BACKEND` 支持：

- `auto`：默认；910B2、FP16/BF16/FP32、非 JIT 且 shape 满足约束时调用 `npu_fusion_attention_v3`，否则回退 PyTorch SDPA。
- `fusion`：严格要求融合，条件不满足时明确报错。
- `sdpa`：强制标准路径。

QKV projection 后保持 BSND 布局进入融合算子。SO400M 的 head dim 72 和 H 的 head dim 80 均在真实 910B2 上通过 FP16/BF16/FP32 前后向对齐。v3 将随机数状态改为 Tensor 输出，可进入计算图；当前 Eager 路径与普通融合版性能和显存一致。JIT、非 NPU、dtype/shape 不支持和算子不可用时不会误入融合路径；ACLGraph 目前只支持 BNSD，因此尚未把当前 BSND 路径放入图模式。`torch_npu` 保持延迟导入。

CPE 继续使用实例单条缓存：

- 可训练时保留官方随机 viewport 和 10% 位置 dropout，只复用基础采样网格。
- eval、no-grad 或冻结 backbone 时使用确定性 CPE，并缓存完整位置编码。
- 参数身份、版本、设备、dtype 或分辨率变化会直接替换缓存，不无限增长。
- inference tensor 不读取不存在的 `_version`；深拷贝、设备迁移和权重重载清空缓存。
- 缓存不是 buffer，不进入 state dict 或 DDP 广播。

patch projection 保持官方 `im2patch + Linear`，没有引入在当前 910B2 上更慢的 Conv2d 替代路径。

## 正确性验证

- SO400M 官方 safetensors 严格完整加载，参数量、MLP、10 个前缀 token 和权重形状一致。
- 512、640、800 和 384×640 的 FP32/FP16/BF16 前向均为有限值，BNC/BCHW 一致。
- 独立构造的 Transformers `RadioModel` 在 CPU FP32、64×80 输入上与本实现逐元素完全一致；NPU FP32、512 输入平均绝对误差为 `5.01e-7`。
- NVIDIA 仓库仍使用旧式 `args` 配置，Transformers 5.14.1 的 `RadioConfig.from_pretrained()` 会把 SO400M 错落到 H 默认结构；参考验证因此显式构造 1152/27/16/4304 配置，并转换合并 QKV 权重。该限制不影响项目加载器。
- Transformers 参考模型在 NPU 上拆分 Q/K/V GEMM 后出现 3 个大于 `1e-4` 的离群点，而 CPU 参考完全一致；项目保留官方合并 QKV 路径，不采用该拆分实现。
- head dim 64、72、80 的 FP16/BF16/FP32 Fusion Attention v3 输出、loss、输入梯度和 QKV 权重梯度与 SDPA 对齐。
- 普通融合版与 v3 在三种 dtype 下输出和梯度逐元素一致；完整 SO400M 的 FP16/BF16/FP32 Eager 性能和峰值显存没有稳定差异。
- 冻结 SO400M 的 YOLO 两步训练 loss 为 10.895000、11.012089；backbone 无梯度，neck/head 参数更新，384×640 推理通过。
- 912,386,044 bytes 的 FP16 检测 checkpoint 在 `torch.inference_mode()` 内由 AutoBackend 重载；位置参数确认为 inference tensor，640 warmup 和 384×640 前向均通过。
- 切换 Fusion Attention v3 后两卡 HCCL DDP smoke 通过，平均 loss 为 10.862114，动态缓存未成为 DDP buffer。

## 性能消融

表中为三次正序/逆序交替重复的中位数。推理每条路径每轮 100 step，训练每轮 5 step。

### 冻结 backbone 前向，batch=1，FP16 AMP

| 分辨率 | official_sdpa | cpe_cache | fusion | full | full 加速 | full 峰值 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512，300-step长窗口 | 23.053 | 22.639 | 19.878 | **19.221** | **1.199×** | 1689.6 |
| 640 | 29.934 | 29.549 | 23.723 | **23.366** | **1.281×** | 1709.3 |
| 800 | 38.917 | 38.608 | 32.434 | **32.105** | **1.212×** | 1745.1 |

### 完整可训练前向+反向，batch=1，FP16 AMP

| 分辨率 | official_sdpa | cpe_cache | fusion | full | full 加速 | official/full 峰值 MiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 74.022 | 73.993 | 63.097 | **63.287** | **1.170×** | 3632.3 / 3570.9 |
| 640 | 94.503 | 94.480 | 78.569 | **78.342** | **1.206×** | 4291.9 / 4196.4 |

训练态位置编码保持随机，因此 `cpe_cache` 只复用基础网格，收益接近零；主要训练收益来自融合 attention。冻结/eval 可以额外缓存完整确定性位置编码。

### 普通融合版、v3 与 FP32 补充验证

完整 SO400M、640×640 的正反交替长测如下；v3/普通版差异均远小于 1%，且峰值显存完全一致。

| dtype | 路径 | 冻结前向 ms | 前向+反向 ms |
| --- | --- | ---: | ---: |
| FP16 | 普通融合版 | 23.562 | 78.116 |
| FP16 | Fusion Attention v3 | **23.561** | 78.120 |
| BF16 | 普通融合版 | 23.466 | 78.237 |
| BF16 | Fusion Attention v3 | **23.464** | **78.222** |
| FP32 | SDPA | 45.445 | 150.126 |
| FP32 | 普通融合版 | 40.102 | 130.589 |
| FP32 | Fusion Attention v3 | **40.100** | **130.580** |

FP32 v3 相对 SDPA 的前向提升为 **1.133×**，前向+反向提升为 **1.150×**，完整模型输出逐元素一致。因此自动快路径不再对 FP32 回退。v3 的价值主要是图兼容接口；在当前 Eager BSND 场景下不宣称相对普通融合版有额外性能收益。

## 复现

```bash
# 轻量结构、加载器和attention测试
pytest -q tests/test_c_radio_v4_backbone.py tests/test_c_radio_v4_attention_ascend.py

# 官方SO400M权重真实NPU测试
RUN_C_RADIO_V4_SO400M_TESTS=1 pytest -q --slow tests/test_c_radio_v4_real_npu.py

# 四路径ABBA
python docs/my-docs/npu-opt/优化记录/c_radio_v4/benchmark_c_radio_v4_npu.py \
  --output /tmp/c_radio_v4_so400m_results.json

# 两卡DDP
python -m torch.distributed.run --standalone --nproc_per_node 2 \
  docs/my-docs/npu-opt/优化记录/c_radio_v4/c_radio_v4_ddp_smoke.py
```

完整原始数据见 `c_radio_v4_so400m_910b2_results.json`。

## 限制

- H 没有下载或执行真实 checkpoint，不能把 SO400M 的性能结论直接外推到 H。
- 首版只输出最终空间特征，不包含 summary adaptor、教师输出、文本、视频、ViTDet 窗口或多层特征。
- 首次运行需要下载约 1.72 GB SO400M 权重；仓库不分发外部权重。
- 外部权重受 NVIDIA Open Model License 约束。
