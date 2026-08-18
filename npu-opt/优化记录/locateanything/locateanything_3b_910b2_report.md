# LocateAnything-3B Ascend 910B2 推理优化报告

## 结论

固定 revision `c32291ca5e996f5a7a485845b4f57a233936bba0` 已加入受保护的 NPU 快路径。在 8 张
Ascend 910B2、每卡 batch=8、COCO val 前 128 张图的端到端验证中，全局吞吐从 **846.09 tok/s**
提升到 **1093.47 tok/s（+29.24%）**，images/s 从 1.4647 提升到 1.8850（+28.70%），无 OOM 和
推理错误。只看每卡第二个稳定 batch，估算吞吐从 846.71 提升到 1172.18 tok/s（+38.44%）。

## 热点与实现

基线短序列画像包含 1008 次通用 SDPA、3199 次 `_local_scalar_dense`，以及大量逐行 `cat/copy`。
针对固定模型结构实施了以下优化：

- Qwen 36 层 GQA 不再把 2 个 KV head 复制到 16 个 query head；prefill/MTP 使用
  `npu_fused_infer_attention_score_v2`，单 token AR 使用 `npu_incre_flash_attention`。
- Qwen RoPE 使用 `npu_rotary_mul`，73 个 RMSNorm 使用 `npu_rms_norm`。
- MoonViT 每张图使用 `npu_fusion_attention_v3` 和 `npu_rotary_mul`，保持原逐图语义。
- top-p 采样使用一次 `npu_top_k_top_p_sample` qSample，并为每张图片保留独立 NPU RNG。
- bbox/ref MTP 解码改为整批张量运算，只在最终 token 矩阵执行一次 D2H；AR token 同样一次取回。
- repetition penalty、prefill mask 和 KV cache 打包移除 `unique/nonzero/item` 热点，减少 Host 同步和小算子。

所有替换都要求固定结构、910B、eval/no-grad、FP16/BF16、非 JIT 和支持的输入形状。`auto` 在不满足
约束时回退原实现，`strict` 报错，`off` 完全关闭；训练及 CUDA/CPU 不进入快路径。

## 算子探针

代表性 BF16 输入上的最大绝对误差和单算子时延如下：

| 算子 | 基线 | 快路径 | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| Qwen GQA，q=6/k=600 | 0.335 ms | 0.119 ms | 0.00098 |
| Qwen GQA，q=1/k=600 | 0.348 ms | 0.090 ms | 0.00195 |
| Qwen prefill，q=k=600 | 2.407 ms | 0.359 ms | 0.00195 |
| MoonViT TND attention | 0.800 ms | 0.106 ms | 0.00391 |
| RMSNorm | 0.245 ms | 0.061 ms | 0 |

SwiGLU 融合仅带来约 4–5% 的单层收益，却需要约 3.2GB 重排权重缓存，因此没有默认启用。

## 端到端结果

| 配置 | 全局 tok/s | 稳定第二批 tok/s | images/s | 峰值显存 | 推理错误 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `npu_fast_path=off` | 846.09 | 846.71 | 1.4647 | 14.15% | 0 |
| `npu_fast_path=auto` | 1093.47 | 1172.18 | 1.8850 | 13.91% | 0 |

`npu-smi` 的 5 秒离散采样均值从 41.79% 变为 15.41%。这不表示计算退化：生成总时长和吞吐均明显改善；
融合算子缩短了 NPU 忙碌窗口，而自回归解码仍是小 batch、逐步依赖的延迟型负载，低频采样容易落在
Host 调度或 rank 等待区间。此处以同步后的 token 总数/最慢 rank 生成时间作为主要性能判据。

曾测试把 batch=8 的 MoonViT token 打包为单次 TND attention。它使一个 rank 的首批随机序列从约 4900
token 增长到 6435 token，尾时延达到 59.78 秒，最终仅 843.17 tok/s，因此已撤回，不属于默认实现。

## 复现

```bash
python npu-opt/优化记录/locateanything/benchmark_locateanything_npu.py \
  --npu-fast-path off --output runs/locateanything/npu_ab/off_128

python npu-opt/优化记录/locateanything/benchmark_locateanything_npu.py \
  --npu-fast-path auto --output runs/locateanything/npu_ab/auto_128
```

完整结构化数据见 `locateanything_3b_910b2_results.json`。由于模型没有原生 confidence，报告不把固定
`score=1.0` 的非标准 COCO AP 作为快路径正确性判据；相同快路径、相同 seed 的单卡 batch=8 重复短测
raw output 完全一致。

## batch=128 大批量优化（2026-08-18）

针对完整COCO验证，又对每卡batch=128进行了长序列端到端测量。原始首批的8卡合计
582800 token，最慢rank生成214.923秒，即 **2711.67 tok/s**。安全优化版处理同一组
1024张图、生成完全相同的582800 token，全局生成时间202.026秒，达到 **2884.77 tok/s
（+6.38%）**、4.9743 images/s。逐图比较raw output、boxes、labels、token数和解析警告，1024/1024
全部完全一致。峰值显存为44.16%，平均NPU利用率为30.77%（5秒离散采样）。

进入默认路径的大批量优化只包含：

- 复用top-p qSample NPU缓冲，避免每个MTP step重复创建128个NPU张量并`cat`。
- 根据`npu-smi info -t topo`把每个worker绑定到相应NPU的CPU范围，并设置
  `torch.set_num_interop_threads(1)`；每个worker线程数从约205降到69。

早期静态KV微基准虽然从276.55提升到377.41 tok/s，但它改变了左填充布局并导致采样漂移，
因此已用精确左填充实现替换。修正后的单卡batch=128、64-token速度为148.86 tok/s，慢于动态KV的
249.53 tok/s。全程paged KV也在batch=8中从动态KV的3.95秒变为6.06秒：单token paged attention本身
数值完全一致，但MTP恢复连续视图的gather开销更大。因此`static_kv_cache=False`、
`paged_kv_cache=False`、`continuous_window=1`仍是默认值。

从4096张断点恢复后，安全动态KV路径已完成全5000张：5000个image ID唯一且无重复。恢复段的
rank 5中，image 498919自然生成到8194-token上限，使该rank用时284.286秒，其他rank仅用时
169.156至191.147秒。这说明大批量利用率不均的主因是随机自回归输出的rank间长尾：先完成的
7张NPU会等待最慢rank。静态KV实验的问题是它会让原本非长尾的相同样本改变采样路径，并非
只有静态KV才可能碰到token上限。

## AR采样与退化循环终止（2026-08-18）

TorchNPU单算子探针表明，AR的`npu_top_k_top_p_sample`设置`is_need_logits=False`后，48行、151936词表
从32.2 ms降到2.9 ms，且sampled token完全一致。真实8卡batch=8严格A/B中，128张的raw output、
boxes和labels全部一致，吞吐从1153.00增至1162.43 tok/s（+0.82%）。MTP仍保留完整概率，
因为hybrid状态机的0.7/0.6/0.2等概率阈值对数值变化敏感。

完整COCO旧结果中，只有image 498919出现连续相同box：1223次，其他4999张最大值都是1。
新增`max_duplicate_boxes=16`会在第17个连续相同box前输出`im_end`，任何ref或不同box都会重置计数。
在原rank 5尾批的同一113张、同seed下：

- 批时延从284.286秒降到199.231秒（-29.92%）。
- 退化样本从8194 token/1312个box降到953 token/105个box。
- 其余112/112张raw output完全一致。

该终止是默认开启的正确性保护，不会将重复框去重后伪装成模型置信度；可用`max_duplicate_boxes=0`恢复无上限的旧行为。

## Continuous batching与动态任务调度（2026-08-18）

新增的流式验证路径不再把`images[rank::world_size]`永久绑定到各rank。8个worker在模型加载完成后只做
一次公平启动同步，随后从共享原子游标领取图片；rank内固定保留128个活跃槽位，空闲槽位达到水位后批量
执行MoonViT与prefill。结果在单图完成时立即写入rank JSONL，动态分配后仍支持按全部分片重建resume队列。

前2048张COCO、8卡、batch=128的实测结果如下。为与旧路径一致，生成tok/s不包含图片预处理和JSONL
回调；images/s使用完整worker墙钟时间。动态分组会改变BF16批量数值与采样路径，因此两次运行的输出token
总数略有差异，性能比较同时列出端到端图片吞吐。

| 调度 | 补充水位 | 生成tok/s | 端到端images/s | 峰值显存 | 采样NPU利用率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 固定分片、固定batch | 不适用 | 2787.65 | 约4.8023 | 约45% | 约31% |
| continuous + dynamic | 32 | 2801.42 | 4.8084 | 44.82% | 36.16% |
| continuous + dynamic | 8 | 2815.51 | 4.8127 | 48.21% | 37.06% |

水位8相对固定分片的生成吞吐约提升1.0%，端到端图片吞吐约提升0.2%。收益小于理想负载均衡上限，原因是
全局队列耗尽时仍有最多`world_size * batch`个已领取样本留在各NPU活跃池中，无法迁移正在生成的KV状态；
同时更频繁的视觉prefill会抵消部分长尾收益。默认兼容路径仍关闭两个调度开关，根目录完整验证示例显式启用
二者并使用`refill_batch=8`。

## 语义等价的五项后续优化（2026-08-18）

1. batch A/B：前2048张COCO下，每卡batch=192为2862.47 tok/s、4.9185 images/s、峰值显存
   60.76%；batch=256为2527.33 tok/s、4.3190 images/s、79.18%。因此根目录示例选择192。
2. shape/KV bucket：新增batch桶和128-token KV长度桶，但补齐本身会增加计算，只作为Graph的
   显式前置选项，不进入默认路径。
3. NPU Graph：整段decoder因远程模型的`.item()`控制流和DynamicCache重编译而不适合捕获，
   因此将边界缩到纯Tensor Qwen MLP。单层q=12微基准数值完全一致，但eager为0.403 ms，
   Graph replay为18.724 ms，故保留实验开关但默认关闭。
4. native paged KV：使用`npu_scatter_nd_update_`在设备侧写入`[block, kv_head, token, dim]`池，
   MTP/AR均直接调用paged infer attention，并在continuous batching中回收物理block。
5. MoonViT batching：按完全相同的`image_grid_hws`分组，一次运行TND block-diagonal attention，
   投影后再按token数拆回原输入顺序。

8卡、512张、每卡batch=64、continuous + dynamic、`max_new_tokens=1024`的组合A/B如下：

| 配置 | 全局 tok/s | images/s | 峰值显存 | 平均NPU利用率 | F1@0.50 | mean GT IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 动态KV + 逐图MoonViT | 2392.12 | 4.1397 | 27.83% | 31.04% | 0.5608 | 0.4515 |
| native paged KV + 同网格MoonViT分组 | 3647.79 | 6.2201 | 25.08% | 41.16% | 0.5638 | 0.4578 |

组合路径的tok/s提升52.49%，images/s提升50.25%，显存降低2.74个百分点。BF16批处理导致
随机采样路径有轻微漂移（总输出token相差约0.61%），但F1和IoU没有下降。所有新快路都要求
eval + no-grad；进入LoRA/全参数训练前会释放Graph引用，并回到原始可微forward。

最终参数又用8卡、1536张图确认每个rank实际batch=192。在`max_new_tokens=256`的容量/速度验收中，
无OOM或推理错误，输出394822 token，全局5016.04 tok/s、18.0232 images/s，峰值PyTorch显存
51.58%，平均NPU利用率35.05%。因256-token上限会截断长输出，该轮指标不与完整8192-token验证横向比较。

## 第三轮语义等价优化与batch=256（2026-08-18）

本轮继续保留可训练性：所有attention、采样和paged KV变更仅在现有`eval + no-grad` NPU快路生效，
LoRA和全参数训练仍走原始可微forward。实施与取舍如下：

- MTP的`npu_top_k_top_p_sample`使用top-k=1024候选快路，首次及每64步用全词表归一化检查
  top-p覆盖率，不足时回退全词表。512图对照的1871个框和F1完全一致；单独端到端收益在测量噪声内。
- `npu_advance_step_flashattn`需要vLLM式固定speculative buffer，且LocateAnything仍必须D2H判断
  box/point/ref/AR状态；接入不能删除同步点，因此没有强行使用。
- Paged MTP attention复用2048×2048 right-down causal mask并使用`sparse_mode=3`；采样常量张量
  按设备/形状缓存，MTP、AR和混合解码的4–5次小型CPU→NPU metadata拷贝合并为1次。
  batch=192下吞吐与旧基线相差1.38%，属于测量波动，不单独宣称加速。
- MoonViT TND打包按16000个视觉token分块，避免batch=256超过TorchNPU FlashAttention总token上限；
  同时修复continuous refill后新MTP行`uncached=0`被混合AR/MTP路径误判为错误的边界。
- MoonViT累计序列长度不再使用NPU `data_ptr`作为跨batch缓存键；allocator可能复用已释放张量的
  地址，从而让新TND batch误命中旧`cu_seqlens`。现在Host累计长度绑定到当前张量对象的生命周期，
  并在调用FlashAttention前校验末值与token数。8卡batch=128、1280图跨窗口smoke已完整通过。

最终8卡、COCO前2048图、每卡batch=256、`max_new_tokens=256`容量/速度测试完成：

| 配置 | 全局tok/s | images/s | 峰值显存 | 平均NPU利用率 | 推理错误 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 上一轮batch=192基线（1536图） | 8229.58 | 28.18 | 51.60% | 49.31% | 0 |
| 本轮batch=192（1536图） | 8116.12 | 28.08 | 51.31% | 48.53% | 0 |
| 本轮batch=256（2048图） | **9459.50** | **31.88** | 64.51% | 48.92% | 0 |

batch=256相对上一轮batch=192基线的tok/s提高约14.95%，峰值显存仍低于80%，因此根目录
`val.py`的完整验证示例改为batch=256。两个batch测试的图片范围不同，这一数字反映已验证的容量/
吞吐配置，不作为严格的同样本A/B因果加速比。
