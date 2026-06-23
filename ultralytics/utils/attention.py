# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Attention helper utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.checks import IS_ASCEND

_torch_npu = None


def _get_torch_npu():
    """Import torch_npu only when an Ascend-only format cast is needed."""
    global _torch_npu
    if _torch_npu is None:
        import torch_npu

        _torch_npu = torch_npu
    return _torch_npu


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
