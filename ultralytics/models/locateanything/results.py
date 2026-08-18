# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""LocateAnything结构化结果与可视化。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class GroundingBox:
    """一个不带伪造置信度的定位框。"""

    xyxy: tuple[float, float, float, float]
    xyxyn: tuple[float, float, float, float]
    label: str


@dataclass(frozen=True)
class GroundingPoint:
    """一个定位点。"""

    xy: tuple[float, float]
    xyn: tuple[float, float]
    label: str


class LocateAnythingResult:
    """保存单张图像的LocateAnything预测。"""

    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        *,
        boxes: list[GroundingBox] | None = None,
        points: list[GroundingPoint] | None = None,
        raw_output: str = "",
        parse_warnings: list[str] | None = None,
        stats: Any = None,
        speed: dict[str, float] | None = None,
    ) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.path = path
        self.boxes = boxes or []
        self.points = points or []
        self.labels = [x.label for x in (*self.boxes, *self.points)]
        self.raw_output = raw_output
        self.parse_warnings = parse_warnings or []
        self.stats = stats
        self.speed = speed or {}

    def __len__(self) -> int:
        """返回框与点的总数。"""
        return len(self.boxes) + len(self.points)

    def summary(self, normalize: bool = False) -> list[dict[str, Any]]:
        """返回便于序列化的定位摘要。"""
        records = []
        for box in self.boxes:
            records.append({"type": "box", "label": box.label, "xyxy": box.xyxyn if normalize else box.xyxy})
        for point in self.points:
            records.append({"type": "point", "label": point.label, "xy": point.xyn if normalize else point.xy})
        return records

    def to_json(self, normalize: bool = False) -> str:
        """将结果转换为JSON字符串。"""
        payload = {
            "path": self.path,
            "shape": {"height": self.orig_shape[0], "width": self.orig_shape[1]},
            "predictions": self.summary(normalize=normalize),
            "raw_output": self.raw_output,
            "parse_warnings": self.parse_warnings,
            "stats": self.stats,
            "speed": self.speed,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    def plot(
        self,
        *,
        boxes: bool = True,
        points: bool = True,
        labels: bool = True,
        line_width: int = 2,
    ) -> np.ndarray:
        """在原图副本上绘制定位框和点，返回BGR图像。"""
        image = self.orig_img.copy()
        palette = _label_colors(self.labels)
        if boxes:
            for item in self.boxes:
                color = palette[item.label]
                x1, y1, x2, y2 = (int(round(x)) for x in item.xyxy)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)
                if labels:
                    _draw_label(image, item.label, (x1, y1), color, line_width)
        if points:
            for item in self.points:
                color = palette[item.label]
                x, y = (int(round(v)) for v in item.xy)
                cv2.drawMarker(
                    image,
                    (x, y),
                    color,
                    cv2.MARKER_CROSS,
                    max(10, line_width * 6),
                    line_width,
                    cv2.LINE_AA,
                )
                if labels:
                    _draw_label(image, item.label, (x, y), color, line_width)
        return image

    def show(self, **kwargs: Any) -> None:
        """使用Pillow显示绘制后的结果。"""
        Image.fromarray(cv2.cvtColor(self.plot(**kwargs), cv2.COLOR_BGR2RGB)).show()

    def save(self, filename: str | Path | None = None, **kwargs: Any) -> str:
        """保存绘制后的图像并返回目标路径。"""
        source_name = Path(self.path).name if self.path else "image.jpg"
        target = Path(filename or f"results_{source_name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(target), self.plot(**kwargs)):
            raise OSError(f"无法保存LocateAnything结果到{target}")
        return str(target)


_TOKEN_RE = re.compile(r"<ref>(.*?)</ref>|<box>(.*?)</box>", re.DOTALL)
_COORD_RE = re.compile(r"<(-?\d+)>")


def parse_locate_output(
    output: str,
    image_shape: tuple[int, int],
    *,
    default_label: str = "target",
) -> tuple[list[GroundingBox], list[GroundingPoint], list[str]]:
    """将LocateAnything token输出解析为像素坐标。"""
    height, width = image_shape
    boxes: list[GroundingBox] = []
    points: list[GroundingPoint] = []
    warnings: list[str] = []
    active_label = default_label.strip() or "target"

    for match in _TOKEN_RE.finditer(output or ""):
        if match.group(1) is not None:
            active_label = re.sub(r"\s+", " ", match.group(1)).strip() or active_label
            continue
        body = (match.group(2) or "").strip()
        if body.lower() == "none":
            continue
        raw_coords = [int(x) for x in _COORD_RE.findall(body)]
        if len(raw_coords) not in {2, 4}:
            warnings.append(f"忽略无法解析的box token：{body!r}")
            continue
        clipped = [min(1000, max(0, x)) for x in raw_coords]
        if clipped != raw_coords:
            warnings.append(f"坐标{raw_coords}超出[0,1000]，已裁剪")
        if len(clipped) == 2:
            xn, yn = (x / 1000 for x in clipped)
            points.append(GroundingPoint((xn * width, yn * height), (xn, yn), active_label))
            continue
        x1n, y1n, x2n, y2n = (x / 1000 for x in clipped)
        if x2n < x1n or y2n < y1n:
            warnings.append(f"定位框{raw_coords}坐标倒置，已规范化")
            x1n, x2n = sorted((x1n, x2n))
            y1n, y2n = sorted((y1n, y2n))
        if x1n == x2n or y1n == y2n:
            warnings.append(f"忽略零面积定位框{raw_coords}")
            continue
        boxes.append(
            GroundingBox(
                (x1n * width, y1n * height, x2n * width, y2n * height),
                (x1n, y1n, x2n, y2n),
                active_label,
            )
        )
    if "<box>" in (output or "") and not boxes and not points and "<box>none</box>" not in output:
        warnings.append("模型输出包含box标记，但未得到有效坐标")
    return boxes, points, warnings


def _label_colors(labels: list[str]) -> dict[str, tuple[int, int, int]]:
    """为标签生成稳定的BGR颜色。"""
    colors: dict[str, tuple[int, int, int]] = {}
    for label in labels or ["target"]:
        seed = sum((i + 1) * ord(c) for i, c in enumerate(label))
        colors[label] = (64 + seed % 192, 64 + (seed // 7) % 192, 64 + (seed // 17) % 192)
    return colors


def _draw_label(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], width: int) -> None:
    """绘制带底色的短标签。"""
    x, y = origin
    scale = max(0.45, width * 0.25)
    size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, max(width - 1, 1))
    top = max(0, y - size[1] - baseline - 4)
    right = min(image.shape[1] - 1, x + size[0] + 4)
    cv2.rectangle(image, (max(x, 0), top), (right, max(y, size[1] + baseline + 4)), color, -1)
    cv2.putText(
        image,
        text,
        (max(x + 2, 0), max(y - baseline - 2, size[1] + 1)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        max(width - 1, 1),
        cv2.LINE_AA,
    )


__all__ = "GroundingBox", "GroundingPoint", "LocateAnythingResult", "parse_locate_output"
