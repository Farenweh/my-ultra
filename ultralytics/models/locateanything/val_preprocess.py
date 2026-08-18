# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""LocateAnything COCO验证专用预处理。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

PAPER_PROTOCOL = "paper"
LEGACY_PROTOCOL = "legacy"
PAPER_PROTOCOL_ID = "locateanything-paper-coco-v1"
LEGACY_PROTOCOL_ID = "locateanything-legacy-coco-v1"
PAPER_SHORT_SIDE = 840


def _detection_prompt(categories: list[str]) -> str:
    """使用LocateAnything论文中的多类别检测prompt。"""
    return "Locate all the instances that matches the following description: " + "</c>".join(categories) + "."


class LocateAnythingValPreprocessor:
    """COCO验证预处理器，支持论文协议和原有协议。"""

    def __init__(
        self,
        annotations: list[dict[str, Any]],
        categories: list[dict[str, Any]],
        *,
        protocol: str = PAPER_PROTOCOL,
    ) -> None:
        protocol = str(protocol).strip().lower()
        if protocol not in {PAPER_PROTOCOL, LEGACY_PROTOCOL}:
            raise ValueError(f"protocol必须是'paper'或'legacy'，得到{protocol!r}")
        self.protocol = protocol
        self.protocol_id = PAPER_PROTOCOL_ID if protocol == PAPER_PROTOCOL else LEGACY_PROTOCOL_ID
        self.short_side = PAPER_SHORT_SIDE if protocol == PAPER_PROTOCOL else None
        self.category_by_id = {
            int(category["id"]): str(category["name"])
            for category in sorted(categories, key=lambda item: int(item["id"]))
        }
        self.all_categories = list(self.category_by_id.values())
        positive_category_ids: dict[int, set[int]] = defaultdict(set)
        for annotation in annotations:
            category_id = int(annotation["category_id"])
            if category_id in self.category_by_id:
                positive_category_ids[int(annotation["image_id"])].add(category_id)
        self.positive_categories = {
            image_id: [self.category_by_id[category_id] for category_id in sorted(category_ids)]
            for image_id, category_ids in positive_category_ids.items()
        }

    def categories_for_image(self, image: dict[str, Any]) -> list[str]:
        """返回当前协议应放入prompt的类别。"""
        if self.protocol == LEGACY_PROTOCOL:
            return self.all_categories.copy()
        image_id = int(image["id"])
        return self.positive_categories.get(image_id, []).copy()

    def prepare(self, image: dict[str, Any]) -> tuple[str | Image.Image, str, dict[str, Any]]:
        """生成模型输入、逐样本prompt及坐标映射上下文。"""
        categories = self.categories_for_image(image)
        context = dict(image)
        if self.protocol == LEGACY_PROTOCOL:
            context["validation_preprocess"] = {
                "protocol": self.protocol,
                "protocol_id": self.protocol_id,
                "short_side": None,
                "interpolation": None,
                "original_size": [int(image["width"]), int(image["height"])],
                "resized_size": [int(image["width"]), int(image["height"])],
                "scale_factor": 1.0,
                "prompt_categories": categories,
            }
            return str(image["path"]), _detection_prompt(categories), context

        path = Path(image["path"])
        with Image.open(path) as source:
            source = source.convert("RGB")
            original_width, original_height = source.size
            scale_factor = PAPER_SHORT_SIDE / min(original_width, original_height)
            resized_width = int(original_width * scale_factor)
            resized_height = int(original_height * scale_factor)
            resized = source.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        context["validation_preprocess"] = {
            "protocol": self.protocol,
            "protocol_id": self.protocol_id,
            "short_side": PAPER_SHORT_SIDE,
            "interpolation": "bilinear",
            "original_size": [original_width, original_height],
            "resized_size": [resized_width, resized_height],
            "scale_factor": scale_factor,
            "prompt_categories": categories,
        }
        return resized, _detection_prompt(categories), context

    def box_to_original(self, xyxy: list[float] | tuple[float, ...], image: dict[str, Any]) -> list[float]:
        """按官方流程先在缩放图边界裁剪，再映射回原图。"""
        values = [float(value) for value in xyxy]
        metadata = image.get("validation_preprocess") or {}
        if metadata.get("protocol_id") != PAPER_PROTOCOL_ID:
            return values
        resized_width, resized_height = (int(value) for value in metadata["resized_size"])
        scale_factor = float(metadata["scale_factor"])
        maximums = (max(resized_width - 1, 0), max(resized_height - 1, 0))
        x1 = min(max(values[0], 0.0), maximums[0]) / scale_factor
        y1 = min(max(values[1], 0.0), maximums[1]) / scale_factor
        x2 = min(max(values[2], 0.0), maximums[0]) / scale_factor
        y2 = min(max(values[3], 0.0), maximums[1]) / scale_factor
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


__all__ = (
    "LEGACY_PROTOCOL",
    "LEGACY_PROTOCOL_ID",
    "PAPER_PROTOCOL",
    "PAPER_PROTOCOL_ID",
    "PAPER_SHORT_SIDE",
    "LocateAnythingValPreprocessor",
)
