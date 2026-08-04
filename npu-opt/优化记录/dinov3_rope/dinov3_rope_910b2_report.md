# DINOv3 2D RoPE 自动路由与 910B2 性能报告

## 结论

本次改造同时保留了纯推理、可训练和手工回退三条路径，并按真实 Autograd 状态、设备、shape、dtype 与 Ascend JIT 模式自动选择：

1. 不需要梯度且满足 Atlas A2 约束：`npu_apply_rotary_pos_emb`，Q/K 双路融合。
2. 需要梯度且满足约束：`npu_rotary_mul`，保留 Autograd backward。
3. 非 NPU 或约束不满足：原生 PyTorch RoPE。

完整 DINOv3-L 在 Ascend 910B2 上的主要结果：

| 场景 | 基线 | 优化后 | 延迟降低 | 加速比 |
|---|---:|---:|---:|---:|
| 224×224，batch=1，纯推理 | 39.430 ms | 26.730 ms | 32.2% | 1.475× |
| 224×224，batch=1，训练前向+反向 | 103.786 ms | 82.394 ms | 20.6% | 1.260× |
| 640×640，batch=1，纯推理 | 40.502 ms | 27.693 ms | 31.6% | 1.463× |

## 环境

- NPU：Ascend 910B2
- CANN：9.1.0
- Driver / npu-smi：25.5.2
- PyTorch：2.12.0+cpu（通过 TorchNPU 使用 NPU）
- torch_npu wheel：2.12.0，26.1.0 发布线
- Ascend legacy JIT：`jit_compile=False`
- AMP：FP16
- DINOv3 RoPE dtype：FP32
- 日期：2026-08-04

## 实现内容

### 自动路由

新增 `DINOV3_ROPE_BACKEND`：

- `auto`：默认，按运行时是否需要梯度自动选择。
- `inference`：强制 Q/K 双路融合；需要梯度或 shape 不支持时直接报错。
- `trainable`：强制可训练 RotaryMul；JIT/shape 不支持时直接报错。
- `manual`：强制原生 PyTorch。

原来的 `USE_DINOV3_ASCEND_ROPE=0` 继续兼容，并等价于 `manual`。

### BSND 与完整序列

- Q/K 保持 BSND 直至 RoPE 结束，适配 A2 上的推理融合算子。
- CLS/storage token 使用 `sin=0, cos=1` 的单位旋转。
- 删除每层 Q/K 的 prefix 切片后重新拼接。
- 推理融合算子取得独占连续存储，避免原地更新污染共享 QKV projection。

### sin/cos 复用

- eval 模式每个 `(H,W)` 在一次 forward 内只生成一次 sin/cos。
- 训练时，如果 shift/jitter/rescale 均未启用，同样只生成一次。
- 存在随机坐标增强时，仍按原语义每个 block 重新采样。

当前项目 `DINOv3ViT` wrapper 默认把 `rescale_coords` 设为 `None`，因此下游训练同样能够复用 RoPE。

### 约束回退

- `jit_compile=True` 且 `D=64` 时不再错误调用仅支持 `D=128` 的 RotaryMul JIT 路径，而是自动回退。
- 检查 D、B、N、广播 layout、dtype、是否需要 sin/cos 梯度以及设备产品名。
- eval 但仍需要输入/参数梯度时，自动选择可训练路径，不以 `module.training` 代替 Autograd 判断。

## 正确性验证

覆盖内容：

- CPU/manual 数学等价。
- 2D 系数到 BSND 的转换与 prefix 单位旋转。
- Q/K 推理双路融合 stub。
- 可训练双 RotaryMul stub 与 backward。
- JIT=True、D=64 的拒绝与 D=128 的支持。
- 确定性 RoPE 每 forward 只生成一次。
- 随机训练 RoPE 保持每 block 重新生成。
- 真实 910B2 上 FP32/FP16/BF16 算子探针。
- 真实 910B2 上自动路由：有梯度走 trainable，无梯度走 inference。
- 真实 910B2 上 Q/K 前向和梯度与手工实现对齐。
- 小型完整 DINOv3 在 NPU 上的推理输出、训练 loss 和 QKV 权重梯度对齐。

真实算子探针结果：

- `npu_apply_rotary_pos_emb` 的 batch `expand` 系数可用，返回值与输入 Q/K 共用存储。
- `npu_rotary_mul` 的 BSND backward 在 FP32、FP16、BF16 下均成功。
- BF16 的融合与手工小算子存在符合低精度舍入特征的差异；项目默认 FP32 RoPE 路径对齐到 `1e-5`。

## 性能数据

### 代表性 RoPE 子图

shape：`B=2, S=1605, N=16, D=64, prefix=5`，对应 DINOv3-L 的 40×40 patch 网格及 5 个特殊 token。

| 场景 | 基线中位数 | 优化中位数 | 加速比 | 基线峰值分配 | 优化峰值分配 |
|---|---:|---:|---:|---:|---:|
| 纯推理 | 0.290 ms | 0.291 ms | 0.996× | 104.88 MiB | 74.39 MiB |
| 训练前向+反向 | 1.346 ms | 0.966 ms | 1.393× | 149.59 MiB | 149.63 MiB |

纯推理子图的延迟处于测量噪声范围，但峰值分配降低约 29.1%。单独固定路由复测时：

- 推理专用 Q/K 融合：0.269 ms
- 无梯度但强制可训练 RotaryMul：0.303 ms
- 手工小算子：0.423 ms

因此自动推理路径仍选择 `npu_apply_rotary_pos_emb`。子图正式基线中旧实现的 prefix 拼接和两次 RotaryMul 已经很短，完整模型的主要收益来自避免 24 个 block 重复生成二维 sin/cos。

### 完整 DINOv3-L，224×224

配置：24 blocks、embed_dim=1024、16 heads、5 个特殊 token、batch=1、AMP FP16、RoPE FP32、随机初始化权重。

| 场景 | 基线中位数 | 优化中位数 | 延迟降低 | 加速比 | 峰值分配 |
|---|---:|---:|---:|---:|---:|
| 纯推理 | 39.430 ms | 26.730 ms | 32.2% | 1.475× | 两者约 1170.69 MiB |
| 训练前向+反向 | 103.786 ms | 82.394 ms | 20.6% | 1.260× | 两者约 2321.75 MiB |

### 完整 DINOv3-L，640×640

| 场景 | 基线中位数 | 优化中位数 | 延迟降低 | 加速比 | 基线峰值分配 | 优化峰值分配 |
|---|---:|---:|---:|---:|---:|---:|
| 纯推理，batch=1 | 40.502 ms | 27.693 ms | 31.6% | 1.463× | 1225.04 MiB | 1212.53 MiB |

640 训练未纳入本轮正式基准，避免以单 batch 的短测试替代真实检测训练吞吐；224 训练步已经覆盖完整 24 层 forward/backward。

## 基线定义与方法

基线严格复现改造前行为：

- Q/K 先转为 BNSD。
- 只旋转 patch suffix。
- Q、K 分别调用 `npu_rotary_mul`。
- 分别把 prefix 拼回。
- 每个 Transformer block 重新生成 meshgrid、angles、sin、cos。

每组数据包括 warmup，随后进行 3 次独立重复并取中位数；所有计时边界均调用 `torch.npu.synchronize()`。可复现实验脚本为 `benchmark_dinov3_rope_npu.py`。

## 被实测否决的方案

训练路径曾测试把 Q/K 沿 head 维拼接后只调用一次 `npu_rotary_mul`。代表性 shape 下：

- 双调用完整序列：约 1.015 ms
- Q/K packed 单调用：约 1.205 ms

packed 方案慢约 18.7%，原因是 Q/K materialization 与相应 backward 成本超过一次 kernel launch，因此没有进入自动路径。

