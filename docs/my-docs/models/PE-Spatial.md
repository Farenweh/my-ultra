# PE-Spatial

## 模型概况

PE-Spatial 是强调空间表征的视觉 Transformer。本项目提供 T/S/B/L/G 五档结构、官方 checkpoint 加载和
检测器友好的空间特征输出。T/S/B 使用 patch 16，L/G 使用 patch 14；当前完整 NPU 实测规格为 L/14。

| 规格 | 默认输入 | 输出通道 | stride |
| --- | ---: | ---: | ---: |
| T | 512 | 192 | 16 |
| S | 512 | 384 | 16 |
| B | 512 | 768 | 16 |
| L | 448 | 1024 | 14 |
| G | 448 | 1536 | 14 |

## 直接使用 backbone

```python
import torch
from ultralytics.nn.modules import PESpatial

backbone = PESpatial("l", pretrained=True).eval().to("npu:0")
images = torch.randn(1, 3, 644, 644, device="npu:0")
features = backbone(images)  # [B, 1024, 46, 46]
```

`pretrained=True` 从固定的官方 Hugging Face 仓库加载权重；也可传本地 `.pt` checkpoint 或使用
`pretrained=False`。输入尺寸必须可被对应 patch stride 整除。

## YOLO 接口

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/rf-det/pe-spatial-yolo11.yaml")
model.train(data="coco.yaml", imgsz=644, freeze=1)
metrics = model.val(data="coco.yaml", imgsz=644)
results = model.predict("image.jpg", imgsz=644)
```

## NPU 支持

PE-Spatial 使用 interleave 形式的 2D RoPE，不能直接套用 DINOv3 的 half-layout 路径。当前实现会自动选择
保持数学语义的 NPU 算子或 PyTorch 回退，推理和训练均保留。

- [PE-Spatial-L/14 910B2 报告](../npu-opt/优化记录/pe_spatial/pe_spatial_l14_910b2_report.md)
- [结构化基准结果](../npu-opt/优化记录/pe_spatial/pe_spatial_l14_910b2_results.json)
