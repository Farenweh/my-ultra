from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ultralytics.utils.checks import IS_ASCEND


_VALID_ATTENTION_BACKENDS = frozenset({"auto", "fusion", "sdpa"})
CRADIO_V3_ATTENTION_BACKEND = os.getenv("CRADIO_V3_ATTENTION_BACKEND", "auto").strip().lower()
CRADIO_V4_ATTENTION_BACKEND = os.getenv("CRADIO_V4_ATTENTION_BACKEND", "auto").strip().lower()
for _name, _value in (
    ("CRADIO_V3_ATTENTION_BACKEND", CRADIO_V3_ATTENTION_BACKEND),
    ("CRADIO_V4_ATTENTION_BACKEND", CRADIO_V4_ATTENTION_BACKEND),
):
    if _value not in _VALID_ATTENTION_BACKENDS:
        raise ValueError(f"不支持{_name}={_value!r}；可选值为{sorted(_VALID_ATTENTION_BACKENDS)}")

_torch_npu = None


def _get_torch_npu():
    """仅在真正选择融合attention时导入torch_npu。"""
    global _torch_npu
    if _torch_npu is None:
        import torch_npu

        _torch_npu = torch_npu
    return _torch_npu


def _ascend_jit_compile_enabled() -> bool:
    try:
        return not torch.npu.is_jit_compile_false()
    except (AttributeError, RuntimeError):
        return os.getenv("USE_ASCEND_JIT_COMPILE", "0") == "1"


@lru_cache(maxsize=None)
def _npu_attention_device_supported(device_index: int) -> bool:
    try:
        name = torch.npu.get_device_name(device_index).upper()
    except (AttributeError, RuntimeError):
        return False
    return any(token in name for token in ("910B", "910_93", "ASCEND950", "ASCEND350"))


def _supports_npu_fusion_attention(q: Tensor, k: Tensor, v: Tensor, num_heads: int) -> bool:
    if not IS_ASCEND or q.device.type != "npu" or q.ndim != 4:
        return False
    if q.shape != k.shape or q.shape != v.shape or q.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype or k.device != q.device or v.device != q.device:
        return False
    batch, sequence, heads, head_dim = q.shape
    if not batch or not sequence or heads != num_heads or head_dim not in {64, 72, 80} or batch * heads >= 1000:
        return False
    if _ascend_jit_compile_enabled() or not _npu_attention_device_supported(q.device.index or 0):
        return False
    try:
        return hasattr(_get_torch_npu(), "npu_fusion_attention_v3")
    except (AttributeError, ImportError, RuntimeError):
        return False


def _attention_policy(family: str) -> tuple[str, str]:
    if family == "v3":
        return "CRADIO_V3_ATTENTION_BACKEND", CRADIO_V3_ATTENTION_BACKEND
    if family == "v4":
        return "CRADIO_V4_ATTENTION_BACKEND", CRADIO_V4_ATTENTION_BACKEND
    raise ValueError(f"不支持的C-RADIO family={family!r}")


def select_attention_backend(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    num_heads: int,
    family: str = "v3",
) -> Literal["fusion", "sdpa"]:
    env_name, policy = _attention_policy(family)
    if policy == "sdpa":
        return "sdpa"
    supported = _supports_npu_fusion_attention(q, k, v, num_heads)
    if policy == "fusion" and not supported:
        raise RuntimeError(
            f"{env_name}=fusion不支持当前设备、JIT模式、dtype或shape：q={tuple(q.shape)} {q.dtype} {q.device}"
        )
    return "fusion" if supported else "sdpa"


class Attention(nn.Module):
    """保持官方checkpoint键名并使用支持入图的Ascend Fusion Attention v3。"""

    def __init__(self, dim: int, num_heads: int, *, family: str = "v3", device=None, dtype=None):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim}不能被num_heads={num_heads}整除")
        factory = {"device": device, "dtype": dtype}
        self.num_heads = num_heads
        self.family = family
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True, **factory)
        self.proj = nn.Linear(dim, dim, bias=True, **factory)

    def forward(self, x: Tensor) -> Tensor:
        batch, sequence, channels = x.shape
        qkv = self.qkv(x).reshape(batch, sequence, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # BSND
        if select_attention_backend(q, k, v, self.num_heads, self.family) == "fusion":
            x = _get_torch_npu().npu_fusion_attention_v3(
                q,
                k,
                v,
                self.num_heads,
                "BSND",
                scale=self.scale,
                keep_prob=1.0,
            )[0]
        else:
            q, k, v = (value.transpose(1, 2) for value in (q, k, v))
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, scale=self.scale)
            x = x.transpose(1, 2)
        return self.proj(x.reshape(batch, sequence, channels))
