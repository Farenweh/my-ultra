# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .fastsam import FastSAM
from .llm import LLM
from .nas import NAS
from .rtdetr import RTDETR
from .yolo import YOLO, YOLOE, YOLOWorld

__all__ = "LLM", "NAS", "RTDETR", "SAM", "YOLO", "YOLOE", "FastSAM", "YOLOWorld", "LocateAnything"


def __getattr__(name):
    """延迟导入具有重量级可选依赖的模型。"""
    if name == "SAM":
        # Scoped for import ultralytics speed: SAM pulls optional torchvision-heavy modules.
        from .sam import SAM

        return SAM
    if name == "LocateAnything":
        from .locateanything import LocateAnything

        return LocateAnything
    raise AttributeError(f"module {__name__} has no attribute {name}")
