# C-RADIOv4

## 模型概况

C-RADIOv4 延续 NVIDIA C-RADIO 多分辨率视觉表征。本项目在 C-RADIOv3 的安全加载和纯 PyTorch ViT 基础上
支持 SO400M 和 H 两档，patch stride 均为 16，不依赖 `trust_remote_code`。当前完整实测规格为
SO400M/16，输出通道 1152；H 输出通道 1280。

## 直接使用 backbone

```python
import torch
from ultralytics.nn.modules import CRADIOv4

backbone = CRADIOv4("so400m", pretrained=True).eval().to("npu:0")
images = torch.randn(1, 3, 640, 640, device="npu:0")
features = backbone(images)  # [B, 1152, 40, 40]
```

`pretrained=True` 使用固定 Hugging Face revision；也可传本地 Safetensors/PyTorch checkpoint 或使用
`pretrained=False`。所有 checkpoint 都执行严格键和形状校验。

## YOLO 接口

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/rf-det/c-radio-v4-yolo11.yaml")
model.train(data="coco.yaml", imgsz=640, freeze=1)
metrics = model.val(data="coco.yaml", imgsz=640)
results = model.predict("image.jpg", imgsz=640)
```

## 约束与 NPU 支持

- 输入必须是三通道浮点 BCHW、高宽为 16 的倍数，且不超过规格的最大分辨率。
- wrapper 自动执行 CLIP mean/std 归一化，并为冻结训练管理确定性 CPE 缓存。
- 910B2 已验证 BF16/FP16/FP32 前向和训练相关路径；Fusion Attention v3 不适用时自动回退。

- [C-RADIOv4-SO400M/16 910B2 报告](../npu-opt/优化记录/c_radio_v4/c_radio_v4_so400m_910b2_report.md)
- [结构化基准结果](../npu-opt/优化记录/c_radio_v4/c_radio_v4_so400m_910b2_results.json)
