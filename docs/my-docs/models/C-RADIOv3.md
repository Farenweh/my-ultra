# C-RADIOv3

## 模型概况

C-RADIOv3 是 NVIDIA 的多分辨率视觉 backbone。本项目使用本地纯 PyTorch 实现和固定 revision 权重，
不依赖 `trust_remote_code`，支持 B、L、H、g 四档结构；各规格 patch stride 均为 16。当前完整实测为
C-RADIOv3-L/16，输出通道 1024。

## 直接使用 backbone

```python
import torch
from ultralytics.nn.modules import CRADIOv3

backbone = CRADIOv3("l", pretrained=True).eval().to("npu:0")
images = torch.randn(1, 3, 640, 640, device="npu:0")
features = backbone(images)  # [B, 1024, 40, 40]
```

`pretrained=True` 从固定 Hugging Face revision 下载 `model.safetensors`；也可传入本地 Safetensors/PyTorch
checkpoint 或使用 `pretrained=False`。wrapper 会执行模型需要的 CLIP mean/std 归一化。

## YOLO 接口

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/rf-det/c-radio-v3-yolo11.yaml")
model.train(data="coco.yaml", imgsz=640, freeze=1)
metrics = model.val(data="coco.yaml", imgsz=640)
results = model.predict("image.jpg", imgsz=640)
```

## 约束与 NPU 支持

- 输入必须是三通道浮点 BCHW、高宽为 16 的倍数，且不超过相应规格的最大分辨率。
- 冻结 backbone 训练时会启用确定性的 CPE 缓存；可训练路径不会复用失效缓存。
- 910B2 自动使用安全的 CPE 缓存、Fusion Attention v3 等路径，不满足条件时回退原生实现。

- [C-RADIOv3-L/16 910B2 报告](../npu-opt/优化记录/c_radio_v3/c_radio_v3_l16_910b2_report.md)
- [结构化基准结果](../npu-opt/优化记录/c_radio_v3/c_radio_v3_l16_910b2_results.json)
