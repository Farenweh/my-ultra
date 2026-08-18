# DINOv3

## 模型概况

DINOv3ViT 是 DINOv3 Vision Transformer 的检测 backbone 包装，保留 2D RoPE，并将最后一层 patch token
输出为空间特征图。结构支持 `s`、`sp`、`b`、`l`、`lp`、`hp` 和 `7b`，patch stride 为 16；当前本地
预训练权重目录映射主要面向 `l` 规格。

## 直接使用 backbone

```python
import torch
from ultralytics.nn.modules import DINOv3ViT

backbone = DINOv3ViT("l", pretrained="./weights").eval().to("npu:0")
images = torch.randn(1, 3, 640, 640, device="npu:0")
features = backbone(images)  # [B, 1024, 40, 40]
```

`l` 规格的 `pretrained` 可以是包含预期文件名的权重目录、显式 URL 或 `False`。其他规格若无对应权重映射，应使用
`pretrained=False` 构造结构，或先补充明确的权重来源。

## RT-DETR 接口

```python
from ultralytics import RTDETR

model = RTDETR("ultralytics/cfg/models/rt-detr/rtdetr-dinov3-ropevit.yaml")
model.train(data="coco.yaml", imgsz=640, freeze=1)
metrics = model.val(data="coco.yaml", imgsz=640)
results = model.predict("image.jpg", imgsz=640)
```

`rtdetr-dinov3-ropevit.yaml` 在 DINOv3-L 后接 RoPEViT 和 RT-DETR decoder；`rtdetr-dinov3-c3k2.yaml` 提供包含 C3k2 neck 的
版本。直接输入的高度和宽度应为 16 的倍数。

## NPU 支持

DINOv3 的 2D RoPE 会根据设备、dtype、张量形状和梯度状态自动选择安全实现。Ascend 快路径保持可训练性，
条件不满足时回退 PyTorch，不修改模型公开接口。

- [DINOv3 2D RoPE 910B2 报告](../npu-opt/优化记录/dinov3_rope/dinov3_rope_910b2_report.md)
- 上游代码及许可位于 `ultralytics/nn/modules/third_party/dinov3/`。
