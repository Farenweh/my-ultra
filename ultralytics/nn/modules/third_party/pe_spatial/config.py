# Copyright (c) Meta Platforms, Inc. and affiliates.
# 本文件基于 Apache-2.0 许可的 Perception Encoder 视觉配置修改。

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PESpatialConfig:
    """单个 PE-Spatial 视觉编码器的结构和官方权重信息。"""

    checkpoint: str
    image_size: int
    patch_size: int
    width: int
    layers: int
    heads: int
    mlp_ratio: float = 4.0
    use_cls_token: bool = True
    layer_scale: float | None = None


PE_SPATIAL_CONFIGS = {
    "t": PESpatialConfig("PE-Spatial-T16-512", 512, 16, 192, 12, 3),
    "s": PESpatialConfig("PE-Spatial-S16-512", 512, 16, 384, 12, 6),
    "b": PESpatialConfig("PE-Spatial-B16-512", 512, 16, 768, 12, 12),
    "l": PESpatialConfig("PE-Spatial-L14-448", 448, 14, 1024, 24, 16),
    "g": PESpatialConfig(
        "PE-Spatial-G14-448",
        448,
        14,
        1536,
        50,
        16,
        mlp_ratio=8960 / 1536,
        use_cls_token=False,
        layer_scale=0.1,
    ),
}
