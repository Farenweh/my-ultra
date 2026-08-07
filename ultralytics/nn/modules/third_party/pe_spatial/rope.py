# Copyright (c) Meta Platforms, Inc. and affiliates.
# 本文件基于 Apache-2.0 许可的 Perception Encoder 2D RoPE 实现修改。

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import torch
from torch import Tensor, nn

from ultralytics.utils.checks import IS_ASCEND


_VALID_ROPE_BACKENDS = frozenset({"auto", "rotary_mul", "manual"})
PE_SPATIAL_ROPE_BACKEND = os.getenv("PE_SPATIAL_ROPE_BACKEND", "auto").strip().lower()
if PE_SPATIAL_ROPE_BACKEND not in _VALID_ROPE_BACKENDS:
    raise ValueError(
        f"不支持PE_SPATIAL_ROPE_BACKEND={PE_SPATIAL_ROPE_BACKEND!r}；可选值为{sorted(_VALID_ROPE_BACKENDS)}"
    )

_torch_npu = None


def _get_torch_npu():
    """仅在真正选择 Ascend 融合算子时导入 torch_npu。"""
    global _torch_npu
    if _torch_npu is None:
        import torch_npu

        _torch_npu = torch_npu
    return _torch_npu


def _ascend_jit_compile_enabled() -> bool:
    """返回当前是否开启不支持 interleave RotaryMul 的旧 JIT 路径。"""
    try:
        return not torch.npu.is_jit_compile_false()
    except (AttributeError, RuntimeError):
        return os.getenv("USE_ASCEND_JIT_COMPILE", "0") == "1"


@lru_cache(maxsize=None)
def _npu_rotary_device_supported(device_index: int) -> bool:
    """缓存当前设备是否属于公开支持 interleave RotaryMul 的训练产品。"""
    try:
        name = torch.npu.get_device_name(device_index).upper()
    except (AttributeError, RuntimeError):
        return False
    return any(token in name for token in ("910B", "910_93", "ASCEND950", "ASCEND350"))


def rotate_interleaved(x: Tensor) -> Tensor:
    """按 GPT-J 相邻偶奇通道语义旋转最后一维。"""
    paired = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    first, second = paired.unbind(-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def apply_rope_manual(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """以官方 FP32 系数语义执行可微的 PyTorch RoPE。"""
    dtype = x.dtype
    x = x.to(dtype=sin.dtype)
    return ((x * cos) + (rotate_interleaved(x) * sin)).to(dtype=dtype)


def _supports_npu_rotary_mul(x: Tensor, sin: Tensor, cos: Tensor) -> bool:
    """检查 910B2 非 JIT interleave RotaryMul 的公开约束。"""
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if not IS_ASCEND or x.device.type != "npu" or x.ndim != 4:
        return False
    if x.dtype not in supported_dtypes or sin.dtype not in supported_dtypes or cos.dtype != sin.dtype:
        return False
    batch, sequence, heads, head_dim = x.shape
    if not batch or not sequence or head_dim >= 896 or head_dim % 2 or batch * heads >= 1000:
        return False
    if sin.shape != cos.shape or sin.shape != (1, sequence, 1, head_dim):
        return False
    if sin.device != x.device or cos.device != x.device or _ascend_jit_compile_enabled():
        return False
    if not _npu_rotary_device_supported(x.device.index or 0):
        return False
    try:
        return hasattr(_get_torch_npu(), "npu_rotary_mul")
    except (AttributeError, ImportError, RuntimeError):
        return False


def select_rope_backend(x: Tensor, sin: Tensor, cos: Tensor) -> Literal["rotary_mul", "manual"]:
    """根据策略和具体张量选择 PE-Spatial RoPE 后端。"""
    if PE_SPATIAL_ROPE_BACKEND == "manual":
        return "manual"
    supported = _supports_npu_rotary_mul(x, sin, cos)
    if PE_SPATIAL_ROPE_BACKEND == "rotary_mul" and not supported:
        raise RuntimeError(
            "PE_SPATIAL_ROPE_BACKEND=rotary_mul不支持当前设备、JIT模式、dtype或shape："
            f"x={tuple(x.shape)} {x.dtype} {x.device}, coeff={tuple(sin.shape)} {sin.dtype}"
        )
    return "rotary_mul" if supported else "manual"


def apply_rope(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """执行自动路由的 interleave RoPE。"""
    if select_rope_backend(x, sin, cos) == "manual":
        return apply_rope_manual(x, sin, cos)
    dtype = x.dtype
    x = _get_torch_npu().npu_rotary_mul(
        input=x.to(dtype=sin.dtype),
        r1=cos,
        r2=sin,
        rotary_mode="interleave",
    )
    return x.to(dtype=dtype)


class Rope2D(nn.Module):
    """生成并单条缓存 PE-Spatial 使用的确定性 2D RoPE 系数。"""

    def __init__(self, head_dim: int, use_cls_token: bool):
        super().__init__()
        if head_dim % 4:
            raise ValueError(f"PE-Spatial的head_dim必须能被4整除，但得到{head_dim}")
        axis_dim = head_dim // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))
        self.head_dim = head_dim
        self.use_cls_token = use_cls_token
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._frequency_cache = torch.empty(0)
        self._sin_cache = torch.empty(0)
        self._cos_cache = torch.empty(0)
        self._cache_key = None

    def __getstate__(self):
        """序列化和深拷贝时丢弃可重建缓存，避免污染 checkpoint。"""
        state = super().__getstate__()
        for name in ("_frequency_cache", "_sin_cache", "_cos_cache"):
            state[name] = torch.empty(0)
        state["_cache_key"] = None
        return state

    def _apply(self, fn):
        """设备或 dtype 变化时清空未注册缓存，避免跨设备持有旧张量。"""
        result = super()._apply(fn)
        self._frequency_cache = torch.empty(0)
        self._sin_cache = torch.empty(0)
        self._cos_cache = torch.empty(0)
        self._cache_key = None
        return result

    def _build_frequency(self, grid_h: int, grid_w: int, device: torch.device) -> Tensor:
        offset = int(self.use_cls_token)
        y = torch.arange(grid_h, device=device, dtype=torch.float32) + offset
        x = torch.arange(grid_w, device=device, dtype=torch.float32) + offset
        inv_freq = self.inv_freq.to(device=device)
        freq_y = (y[:, None] * inv_freq[None]).repeat_interleave(2, dim=-1)
        freq_x = (x[:, None] * inv_freq[None]).repeat_interleave(2, dim=-1)
        frequency = torch.cat(
            (
                freq_x[None].expand(grid_h, -1, -1),
                freq_y[:, None].expand(-1, grid_w, -1),
            ),
            dim=-1,
        ).reshape(grid_h * grid_w, self.head_dim)
        if self.use_cls_token:
            frequency = torch.cat((torch.zeros(1, self.head_dim, device=device), frequency), dim=0)
        return frequency

    def coefficients(self, grid_h: int, grid_w: int, device: torch.device) -> tuple[Tensor, Tensor]:
        """返回可广播到 BSND 的 sin/cos，并在尺寸变化时替换缓存。"""
        device = torch.device(device)
        key = (device.type, device.index, grid_h, grid_w, torch.float32)
        if key != self._cache_key:
            frequency = self._build_frequency(grid_h, grid_w, device)
            self._frequency_cache = frequency
            self._sin_cache = frequency.sin().view(1, -1, 1, self.head_dim)
            self._cos_cache = frequency.cos().view(1, -1, 1, self.head_dim)
            self._cache_key = key
        return self._sin_cache, self._cos_cache
