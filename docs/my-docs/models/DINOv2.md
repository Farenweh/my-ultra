# DINOv2

## 模型概况

DINOv2 是 Meta 的自监督视觉 Transformer。本项目将其最后一层 patch token 转换为 NCHW 特征图，使其可作为
YOLO 或 RT-DETR 的单层视觉 backbone。支持 `s`、`b`、`l`、`g`，以及带 register token 的
`s_reg`、`b_reg`、`l_reg`、`g_reg`；patch stride 为 14。

## 直接使用 backbone

```python
import torch
from ultralytics.nn.modules import DINOv2

backbone = DINOv2("l", pretrained=True).eval().to("npu:0")
images = torch.randn(1, 3, 644, 644, device="npu:0")
features = backbone(images)  # [B, 1024, H/14, W/14]
```

`pretrained=False` 使用随机初始化；字符串可作为权重地址或路径传给 DINOv2 权重加载接口。输入高度和宽度必须
是 14 的倍数，`forward_sequence()` 可直接返回 NLC token。

## 检测接口

当前工作树提供 DINOv2 与 YOLO11、兼容旧 YOLOv8 head、C2f head 和 RT-DETR decoder 的配置：

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/rf-det/dinov2-yolo11.yaml")
model.train(data="coco.yaml", imgsz=644, freeze=1)
metrics = model.val(data="coco.yaml", imgsz=644)
results = model.predict("image.jpg", imgsz=644)
```

可用配置名包括 `dinov2-yolo11.yaml`、`dinov2-yolov8.yaml`、`dinov2-yolov8-c2f.yaml` 和
`dinov2-detr.yaml`。这些 YAML 当前是工作树文件，本页不会将它们纳入版本控制。

## 约束与 NPU 状态

- 检测输入会按真实特征步长处理；手工张量调用必须保证尺寸可被 14 整除。
- `freeze=1` 可冻结 DINOv2，仅训练检测 neck/head。
- 当前使用纯 PyTorch 路径，可在 CUDA、CPU 和 Ascend 上训练或推理；暂无独立的 DINOv2 910B2 基准报告。
- 上游代码及许可位于 `ultralytics/nn/modules/third_party/dinov2/`。
- [NPU 优化资料索引](../npu-opt/优化记录/README.md)
