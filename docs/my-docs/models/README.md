# 自定义模型索引

当前分支新增了一个完整视觉定位模型和六种可组合到检测器中的视觉 backbone。

| 模型 | 类型 | 主要入口 | 示例配置 | NPU 状态 |
| --- | --- | --- | --- | --- |
| [LocateAnything](./LocateAnything.md) | 视觉语言定位模型 | `from ultralytics import LocateAnything` | Python API | 8×910B2 推理、验证和训练路径 |
| [DINOv2](./DINOv2.md) | ViT backbone | `DINOv2` | `rf-det/dinov2-*.yaml` | 原生 PyTorch/NPU 路径 |
| [DINOv3](./DINOv3.md) | RoPE ViT backbone | `DINOv3ViT` | `rt-detr/rtdino*.yaml` | 2D RoPE 快路径已实测 |
| [SigLIP2](./SigLIP2.md) | NaFlex ViT backbone | `SigLIP2So400M` | `rf-det/siglip2-yolo11.yaml` | NPU 训推接口已覆盖 |
| [PE-Spatial](./PE-Spatial.md) | 空间视觉 backbone | `PESpatial` | `rf-det/pe-spatial-yolo11.yaml` | L/14 与 RoPE 快路径已实测 |
| [C-RADIOv3](./C-RADIOv3.md) | 多分辨率视觉 backbone | `CRADIOv3` | `rf-det/c-radio-v3-yolo11.yaml` | L/16 已在 910B2 实测 |
| [C-RADIOv4](./C-RADIOv4.md) | 多分辨率视觉 backbone | `CRADIOv4` | `rf-det/c-radio-v4-yolo11.yaml` | SO400M/16 已在 910B2 实测 |

## 通用检测接口

除 LocateAnything 外，其余模型都是检测器 backbone，可通过模型 YAML 使用统一接口：

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/rf-det/pe-spatial-yolo11.yaml")
model.train(data="coco.yaml", imgsz=644, freeze=1)
metrics = model.val(data="coco.yaml")
results = model.predict("image.jpg")
```

`freeze=1` 表示冻结模型 YAML 中的第 0 层视觉 backbone，只训练后续检测 neck/head。具体输入步长、权重和
可用规格请查看对应页面。
