from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CRADIOConfig:
    """C-RADIO视觉编码器的结构、权重和执行策略信息。"""

    repo_id: str
    revision: str
    width: int
    depth: int
    heads: int
    parameter_count: int
    layer_scale: float | None = None
    base_registers: int = 0
    mlp_hidden_dim: int | None = None
    family: str = "v3"
    patch_size: int = 16
    preferred_resolution: int = 512
    max_resolution: int = 2048
    prefix_tokens: int = 8

    @property
    def effective_mlp_hidden_dim(self) -> int:
        """返回checkpoint实际使用的MLP隐藏维度。"""
        return self.mlp_hidden_dim or self.width * 4


# 保留原名称兼容已有导入。
CRADIOv3Config = CRADIOConfig


CRADIO_V3_CONFIGS = {
    "b": CRADIOv3Config(
        "nvidia/C-RADIOv3-B",
        "44653a0482cf460bb4f12595fc3cc3dfecc403d1",
        768,
        12,
        12,
        98_254_854,
        layer_scale=1e-5,
        base_registers=4,
    ),
    "l": CRADIOv3Config(
        "nvidia/C-RADIOv3-L",
        "9d0413465e8a91e67bbf2c1ad342815478d1b906",
        1024,
        24,
        16,
        319_934_470,
        layer_scale=1e-5,
        base_registers=4,
    ),
    "h": CRADIOv3Config(
        "nvidia/C-RADIOv3-H",
        "d7fd0e2b0a1761f1af150582e06c41e9a99b0bf8",
        1280,
        32,
        16,
        651_642_886,
    ),
    "g": CRADIOv3Config(
        "nvidia/C-RADIOv3-g",
        "28e70735780c22cd7fca20f0c509cb9fc3893aeb",
        1536,
        40,
        24,
        1_159_618_566,
    ),
}
