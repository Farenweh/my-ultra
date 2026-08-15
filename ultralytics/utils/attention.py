# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Attention helper utilities."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

import torch
import torch.nn.functional as F

from ultralytics.utils.checks import IS_ASCEND

_torch_npu = None


def _get_torch_npu():
    """Import torch_npu only when an Ascend-only operator is needed."""
    global _torch_npu
    if _torch_npu is None:
        import torch_npu

        _torch_npu = torch_npu
    return _torch_npu


def _ascend_jit_compile_enabled() -> bool:
    """Return whether the legacy Ascend JIT operator path is enabled."""
    try:
        return not torch.npu.is_jit_compile_false()
    except (AttributeError, RuntimeError):
        return os.getenv("USE_ASCEND_JIT_COMPILE", "0") == "1"


@lru_cache(maxsize=None)
def _npu_attention_device_name(device_index: int) -> str:
    """Cache the normalized Ascend product name used by attention capability checks."""
    try:
        return torch.npu.get_device_name(device_index).upper()
    except (AttributeError, RuntimeError):
        return ""


def _attention_shape(x: torch.Tensor, input_layout: str) -> tuple[int, int, int, int]:
    """Return B, S, N and D for a four-dimensional attention tensor."""
    if input_layout == "BSND":
        return x.shape
    batch, heads, sequence, head_dim = x.shape
    return batch, sequence, heads, head_dim


def _needs_attention_backward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> bool:
    """Return whether the attention result participates in an active autograd graph."""
    return torch.is_grad_enabled() and any(x.requires_grad for x in (query, key, value))


def select_npu_attention_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    input_layout: str,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Literal["training", "inference", "sdpa"]:
    """Select a documented Ascend fused-attention API or the portable SDPA fallback."""
    if (
        not IS_ASCEND
        or query.device.type != "npu"
        or input_layout not in {"BSND", "BNSD"}
        or query.ndim != 4
        or key.ndim != 4
        or value.ndim != 4
        or attn_mask is not None
        or dropout_p != 0.0
        or is_causal
    ):
        return "sdpa"
    if key.device != query.device or value.device != query.device:
        return "sdpa"
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return "sdpa"

    batch, query_sequence, query_heads, query_dim = _attention_shape(query, input_layout)
    key_batch, key_sequence, key_heads, key_dim = _attention_shape(key, input_layout)
    value_batch, value_sequence, value_heads, value_dim = _attention_shape(value, input_layout)
    if (
        not batch
        or not query_sequence
        or not key_sequence
        or query_heads != num_heads
        or key_heads != num_heads
        or value_heads != num_heads
        or key_batch != batch
        or value_batch != batch
        or value_sequence != key_sequence
        or key_dim != query_dim
        or value_dim > query_dim
        or num_heads > 256
        or batch * num_heads >= 1000
        or _ascend_jit_compile_enabled()
    ):
        return "sdpa"

    try:
        torch_npu = _get_torch_npu()
    except (AttributeError, ImportError, RuntimeError):
        return "sdpa"

    if _needs_attention_backward(query, key, value):
        device_name = _npu_attention_device_name(query.device.index or 0)
        if (
            "910B" not in device_name
            or query.dtype not in {torch.float16, torch.bfloat16, torch.float32}
            or not 1 <= query_dim <= 768
        ):
            return "sdpa"
        return "training" if hasattr(torch_npu, "npu_fusion_attention_v3") else "sdpa"

    # Packed QKV views need three materializations and are slower than SDPA in inference. RoPE-backed callers already
    # own contiguous Q/K storage, where materializing only V retains a measured end-to-end benefit.
    device_name = _npu_attention_device_name(query.device.index or 0)
    if (
        not any(token in device_name for token in ("910B", "910_93", "ASCEND950", "ASCEND350"))
        or query.dtype not in {torch.float16, torch.bfloat16}
        or query_dim > 512
        or query_dim % 16
        or not query.is_contiguous()
        or not key.is_contiguous()
    ):
        return "sdpa"
    return "inference" if hasattr(torch_npu, "npu_fused_infer_attention_score_v2") else "sdpa"


def _is_npu_nd_format(x: torch.Tensor) -> bool:
    """Return whether an NPU tensor is already represented as ND format."""
    torch_npu = _get_torch_npu()
    return torch_npu.get_npu_format(x) == torch_npu.Format.ND


def _npu_format_cast_to_nd_impl(x: torch.Tensor) -> torch.Tensor:
    """Cast an NPU tensor to ND format without relying on torch_npu autograd."""
    if _is_npu_nd_format(x):
        return x
    torch_npu = _get_torch_npu()
    return torch_npu.npu_format_cast(x, torch_npu.Format.ND)


class _NpuFormatCastToNd(torch.autograd.Function):
    """Identity for values, with forward and backward tensors represented as ND on Ascend."""

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return _npu_format_cast_to_nd_impl(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor]:
        return (_npu_format_cast_to_nd_impl(grad_output),)


def _npu_format_cast_to_nd(x: torch.Tensor) -> torch.Tensor:
    """Cast an NPU tensor to ND format for kernels that reject internal format."""
    return _NpuFormatCastToNd.apply(x)


def _npu_format_cast_to_nd_if_needed(x: torch.Tensor) -> torch.Tensor:
    """Cast only tensors that are not already ND to avoid unnecessary format conversion."""
    return x if _is_npu_nd_format(x) else _NpuFormatCastToNd.apply(x)


def npu_format_cast_to_nd_if_needed(x: torch.Tensor) -> torch.Tensor:
    """Cast an NPU tensor to ND only when needed, preserving an ND gradient in backward."""
    return _npu_format_cast_to_nd_if_needed(x)


def sdpa_with_npu_padding(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    guard_input_format: bool = True,
    guard_backward_format: bool = True,
) -> torch.Tensor:
    """Call PyTorch SDPA, padding the last dimension on NPU for Ascend kernel constraints."""
    value_dim = value.shape[-1]
    if IS_ASCEND and query.device.type == "npu":
        query_dim = query.shape[-1]
        key_dim = key.shape[-1]
        if guard_input_format:
            query = _npu_format_cast_to_nd_if_needed(query)
            key = _npu_format_cast_to_nd_if_needed(key)
            value = _npu_format_cast_to_nd_if_needed(value)
            if isinstance(attn_mask, torch.Tensor) and attn_mask.device.type == "npu":
                attn_mask = _npu_format_cast_to_nd_if_needed(attn_mask)
        target_dim = max(query_dim, key_dim, value_dim)
        target_dim = ((target_dim + 15) // 16) * 16
        if query_dim != target_dim:
            query = F.pad(query, (0, target_dim - query_dim))
        if key_dim != target_dim:
            key = F.pad(key, (0, target_dim - key_dim))
        if value_dim != target_dim:
            value = F.pad(value, (0, target_dim - value_dim))
        if scale is None:
            scale = query_dim**-0.5

    use_npu_format_guard = IS_ASCEND and query.device.type == "npu"
    out = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
    out = out[..., :value_dim]
    if use_npu_format_guard and guard_backward_format:
        out = _npu_format_cast_to_nd(out)
    return out


def sdpa_with_npu_fusion(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    num_heads: int,
    input_layout: Literal["BSND", "BNSD"],
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    allow_inference_fusion: bool = True,
) -> torch.Tensor:
    """Run fused Ascend attention when supported and otherwise preserve PyTorch SDPA semantics.

    The returned tensor uses the same layout as the inputs. Auto-routing is deliberately limited to the documented,
    performance-relevant vision-attention case without masks, causal attention, or dropout.
    """
    if input_layout not in {"BSND", "BNSD"}:
        raise ValueError(f"不支持attention输入布局{input_layout!r}；仅支持'BSND'或'BNSD'")
    if scale is None:
        scale = query.shape[-1] ** -0.5

    backend = select_npu_attention_backend(
        query,
        key,
        value,
        num_heads=num_heads,
        input_layout=input_layout,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
    )
    if backend == "inference" and not allow_inference_fusion:
        backend = "sdpa"
    if backend == "training":
        query = _npu_format_cast_to_nd_if_needed(query)
        key = _npu_format_cast_to_nd_if_needed(key)
        value = _npu_format_cast_to_nd_if_needed(value)
        return _get_torch_npu().npu_fusion_attention_v3(
            query,
            key,
            value,
            num_heads,
            input_layout,
            scale=scale,
            keep_prob=1.0,
        )[0]
    if backend == "inference":
        # The inference API explicitly rejects strided tensors. Q/K/V from a packed projection are normally views.
        query = _npu_format_cast_to_nd_if_needed(query.contiguous())
        key = _npu_format_cast_to_nd_if_needed(key.contiguous())
        value = _npu_format_cast_to_nd_if_needed(value.contiguous())
        return _get_torch_npu().npu_fused_infer_attention_score_v2(
            query,
            key,
            value,
            num_query_heads=num_heads,
            num_key_value_heads=num_heads,
            softmax_scale=scale,
            input_layout=input_layout,
        )[0]

    if input_layout == "BSND":
        query, key, value = (x.transpose(1, 2) for x in (query, key, value))
        out = sdpa_with_npu_padding(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )
        return out.transpose(1, 2)
    return sdpa_with_npu_padding(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )
