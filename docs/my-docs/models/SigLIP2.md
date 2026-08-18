# SigLIP2 So400M NaFlex

## 模型概况

`SigLIP2So400M` 使用 timm 的 SigLIP 2 So400M Patch16 NaFlex 视觉编码器，并将中间 patch token 转换为
NCHW 检测特征。输出通道为 1152，feature stride 为 16，适合接入 YOLO 检测 neck/head。

## 直接使用 backbone

```python
import torch
from ultralytics.nn.modules import SigLIP2So400M

backbone = SigLIP2So400M(pretrained=True).eval().to("npu:0")
images = torch.randn(1, 3, 640, 640, device="npu:0")
features = backbone(images)  # [B, 1152, 40, 40]
```

`pretrained=True` 通过 timm 下载官方权重，`pretrained=False` 随机初始化，也可传入本地 checkpoint 路径。
`forward_sequence()` 返回 NLC token。

## YOLO 接口

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/rf-det/siglip2-yolo11.yaml")
model.train(data="coco.yaml", imgsz=640, freeze=1)
metrics = model.val(data="coco.yaml", imgsz=640)
results = model.predict("image.jpg", imgsz=640)
```

## 约束与 NPU 状态

- 输入必须是三通道浮点 BCHW，并且高宽均为 16 的倍数。
- 默认 `max_num_patches=2500`，因此正方形输入最大为 800×800；可显式提高或设为 `None`。
- wrapper 内部完成 SigLIP 需要的 `[-1, 1]` 归一化。
- 已覆盖 NPU 前向、梯度和 YOLO 构建接口；暂无独立的 SigLIP2 910B2 性能报告。
- [NPU 优化资料索引](../npu-opt/优化记录/README.md)
