# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
import os
from functools import lru_cache
from typing import List, Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ultralytics.utils.checks import IS_ASCEND

from ..utils import cat_keep_shapes, uncat_with_shapes


_VALID_ROPE_BACKENDS = frozenset({"auto", "inference", "trainable", "manual"})
USE_DINOV3_ASCEND_ROPE = os.getenv("USE_DINOV3_ASCEND_ROPE", "1") == "1"
DINOV3_ROPE_BACKEND = os.getenv("DINOV3_ROPE_BACKEND", "auto").strip().lower()
if DINOV3_ROPE_BACKEND not in _VALID_ROPE_BACKENDS:
    raise ValueError(
        f"Unsupported DINOV3_ROPE_BACKEND={DINOV3_ROPE_BACKEND!r}; "
        f"expected one of {sorted(_VALID_ROPE_BACKENDS)}"
    )
if not USE_DINOV3_ASCEND_ROPE:
    DINOV3_ROPE_BACKEND = "manual"
USE_ASCEND_ROPE = IS_ASCEND and DINOV3_ROPE_BACKEND != "manual"
_torch_npu = None


def _get_torch_npu():
    """Import torch_npu only when an Ascend fused op is actually used."""
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
def _npu_apply_device_supported(device_index: int) -> bool:
    """Cache the stable device-name capability check outside the per-block hot path."""
    try:
        device_name = torch.npu.get_device_name(device_index).upper()
    except (AttributeError, RuntimeError):
        return False
    return any(token in device_name for token in ("910B", "910_93", "ASCEND950", "ASCEND350", "310P"))


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


def _prepend_identity_rope(coeff: Tensor, prefix: int, *, fill_value: float) -> Tensor:
    """Prepend identity-rotation coefficients along the sequence dimension."""
    if prefix == 0:
        return coeff
    prefix_shape = list(coeff.shape)
    prefix_shape[-2] = prefix
    identity = torch.full(prefix_shape, fill_value, dtype=coeff.dtype, device=coeff.device)
    return torch.cat((identity, coeff), dim=-2)


def _prepare_rope_bsnd(q: Tensor, rope: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
    """Convert DINOv3 RoPE coefficients to a full-sequence BSND-broadcastable layout."""
    sin, cos = rope
    if sin.shape != cos.shape:
        raise ValueError(f"sin/cos shape mismatch: {tuple(sin.shape)} != {tuple(cos.shape)}")
    if sin.dtype != cos.dtype:
        raise ValueError(f"sin/cos dtype mismatch: {sin.dtype} != {cos.dtype}")
    if sin.device != q.device or cos.device != q.device:
        raise ValueError(f"RoPE coefficients and Q must be on the same device: {sin.device}, {cos.device}, {q.device}")
    if sin.ndim not in {2, 3, 4}:
        raise ValueError(f"Unsupported RoPE coefficient rank {sin.ndim}; expected 2, 3, or 4")

    batch, sequence, heads, head_dim = q.shape
    if sin.shape[-1] != head_dim:
        raise ValueError(f"RoPE head dimension {sin.shape[-1]} does not match Q head dimension {head_dim}")
    prefix = sequence - sin.shape[-2]
    if prefix < 0:
        raise ValueError(f"RoPE sequence length {sin.shape[-2]} exceeds Q sequence length {sequence}")

    sin = _prepend_identity_rope(sin, prefix, fill_value=0.0)
    cos = _prepend_identity_rope(cos, prefix, fill_value=1.0)
    if sin.ndim == 2:  # [S, D] -> [1, S, 1, D]
        sin = sin.unsqueeze(0).unsqueeze(2)
        cos = cos.unsqueeze(0).unsqueeze(2)
    elif sin.ndim == 3:  # [N, S, D] -> [1, S, N, D]
        sin = sin.permute(1, 0, 2).unsqueeze(0)
        cos = cos.permute(1, 0, 2).unsqueeze(0)
    else:  # [B, N, S, D] -> [B, S, N, D]
        sin = sin.permute(0, 2, 1, 3)
        cos = cos.permute(0, 2, 1, 3)

    if sin.shape[0] not in {1, batch} or sin.shape[2] not in {1, heads}:
        raise ValueError(
            f"RoPE coefficients {tuple(sin.shape)} cannot broadcast to BSND input {tuple(q.shape)}"
        )
    return sin, cos


def _supports_npu_rotary_mul(q: Tensor, sin: Tensor, cos: Tensor) -> bool:
    """Check documented npu_rotary_mul constraints for a BSND half-mode call."""
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if q.ndim != 4 or q.dtype not in supported_dtypes or sin.dtype not in supported_dtypes:
        return False
    batch, sequence, heads, head_dim = q.shape
    if not sequence or head_dim >= 896 or head_dim % 2 or batch >= 1000 or heads >= 1000:
        return False
    if sin.shape != cos.shape or sin.dtype != cos.dtype or sin.shape[1] != sequence or sin.shape[-1] != head_dim:
        return False
    supported_coeff_layout = (
        (sin.shape[0] == 1 and sin.shape[2] == 1)
        or (sin.shape[0] == batch and sin.shape[2] == 1)
        or (sin.shape[0] == batch and sin.shape[2] == heads)
    )
    if not supported_coeff_layout:
        return False
    if (sin.requires_grad or cos.requires_grad) and batch * heads > 1024:
        return False
    if _ascend_jit_compile_enabled():
        broadcast_size = (batch if sin.shape[0] == 1 else 1) * (heads if sin.shape[2] == 1 else 1)
        return head_dim == 128 and broadcast_size <= 1024
    return True


def _supports_npu_apply_rotary_pos_emb(q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> bool:
    """Check the strict Atlas A2/A3 inference constraints for the fused Q/K RoPE operator."""
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float32}
    if q.shape != k.shape or q.ndim != 4 or q.dtype not in supported_dtypes or k.dtype not in supported_dtypes:
        return False
    batch, sequence, _, head_dim = q.shape
    if not batch or not sequence or head_dim not in {64, 128}:
        return False
    if sin.shape != cos.shape or sin.dtype != cos.dtype or sin.dtype not in supported_dtypes:
        return False
    if sin.shape[0] not in {1, batch} or sin.shape[1] != sequence or sin.shape[2] != 1 or sin.shape[3] != head_dim:
        return False
    try:
        torch_npu = _get_torch_npu()
        if not hasattr(torch_npu, "npu_apply_rotary_pos_emb"):
            return False
    except (AttributeError, RuntimeError):
        return False
    # Restrict auto-routing to product names whose public API explicitly supports this operator.
    return _npu_apply_device_supported(q.device.index or 0)


def _needs_rope_backward(q: Tensor, k: Tensor, sin: Tensor, cos: Tensor) -> bool:
    return torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, sin, cos))


class RoPEOnAscend:
    """Ascend RoPE implementations retained as a small compatibility surface for direct callers."""

    @staticmethod
    def _prepare_coeff(coeff: Tensor, x: Tensor, name: str) -> Tensor:
        if coeff.ndim == x.ndim:
            return coeff
        if coeff.ndim == 2 and x.ndim == 4:
            return coeff.unsqueeze(0).unsqueeze(0)
        raise ValueError(
            f"Unsupported {name} shape for npu_rotary_mul: coeff.shape={tuple(coeff.shape)}, x.shape={tuple(x.shape)}"
        )

    @staticmethod
    def apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
        if not USE_ASCEND_ROPE:
            raise RuntimeError("torch_npu is required for Ascend RoPE path")
        torch_npu = _get_torch_npu()
        cos_npu = RoPEOnAscend._prepare_coeff(cos, x, "cos")
        sin_npu = RoPEOnAscend._prepare_coeff(sin, x, "sin")
        y = torch_npu.npu_rotary_mul(input=x, r1=cos_npu, r2=sin_npu, rotary_mode="half")
        if y.shape != x.shape:
            raise RuntimeError(f"npu_rotary_mul returned unexpected shape {tuple(y.shape)} for input {tuple(x.shape)}")
        return y.to(dtype=x.dtype)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def select_rope_backend(
        self, q: Tensor, k: Tensor, sin: Tensor, cos: Tensor
    ) -> Literal["inference", "trainable", "manual"]:
        """Select a safe RoPE implementation for the concrete tensors."""
        policy = DINOV3_ROPE_BACKEND
        if policy == "manual" or not USE_ASCEND_ROPE or q.device.type != "npu":
            return "manual"

        needs_backward = _needs_rope_backward(q, k, sin, cos)
        inference_supported = not needs_backward and _supports_npu_apply_rotary_pos_emb(
            q, k, sin, cos
        )
        trainable_supported = _supports_npu_rotary_mul(q, sin, cos)

        if policy == "inference":
            if needs_backward:
                raise RuntimeError("DINOV3_ROPE_BACKEND=inference cannot be used when RoPE inputs require gradients")
            if not inference_supported:
                raise RuntimeError("DINOV3_ROPE_BACKEND=inference is unsupported for the current device/shape/dtype")
            return "inference"
        if policy == "trainable":
            if not trainable_supported:
                raise RuntimeError("DINOV3_ROPE_BACKEND=trainable is unsupported for the current JIT/shape/dtype")
            return "trainable"
        if inference_supported:
            return "inference"
        if trainable_supported:
            return "trainable"
        return "manual"

    @staticmethod
    def _apply_rope_manual_bsnd(
        q: Tensor, k: Tensor, sin: Tensor, cos: Tensor
    ) -> Tuple[Tensor, Tensor]:
        q_dtype, k_dtype = q.dtype, k.dtype
        q = rope_apply(q.to(dtype=sin.dtype), sin, cos).to(dtype=q_dtype)
        k = rope_apply(k.to(dtype=sin.dtype), sin, cos).to(dtype=k_dtype)
        return q, k

    @staticmethod
    def _apply_rope_trainable_bsnd(
        q: Tensor, k: Tensor, sin: Tensor, cos: Tensor
    ) -> Tuple[Tensor, Tensor]:
        torch_npu = _get_torch_npu()
        q_dtype, k_dtype = q.dtype, k.dtype
        q = torch_npu.npu_rotary_mul(
            input=q.to(dtype=sin.dtype), r1=cos, r2=sin, rotary_mode="half"
        ).to(dtype=q_dtype)
        k = torch_npu.npu_rotary_mul(
            input=k.to(dtype=sin.dtype), r1=cos, r2=sin, rotary_mode="half"
        ).to(dtype=k_dtype)
        return q, k

    @staticmethod
    def _apply_rope_inference_bsnd(
        q: Tensor, k: Tensor, sin: Tensor, cos: Tensor
    ) -> Tuple[Tensor, Tensor]:
        torch_npu = _get_torch_npu()
        q_dtype, k_dtype = q.dtype, k.dtype
        # The operator updates Q/K in place. Acquire owned contiguous storage without performing
        # a redundant clone when dtype conversion or contiguity conversion already allocated it.
        def owned_contiguous(x: Tensor) -> Tensor:
            if x.dtype != sin.dtype:
                return x.to(dtype=sin.dtype).contiguous()
            return x.clone() if x.is_contiguous() else x.contiguous()

        q = owned_contiguous(q)
        k = owned_contiguous(k)
        batch = q.shape[0]
        if sin.shape[0] == 1 and batch != 1:
            sin = sin.expand(batch, -1, -1, -1)
            cos = cos.expand(batch, -1, -1, -1)
        q, k = torch_npu.npu_apply_rotary_pos_emb(
            q, k, cos, sin, layout="BSND", rotary_mode="half"
        )
        return q.to(dtype=q_dtype), k.to(dtype=k_dtype)

    def apply_rope_bsnd(
        self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Apply RoPE while Q/K are still in BSND layout."""
        sin, cos = _prepare_rope_bsnd(q, rope)
        backend = self.select_rope_backend(q, k, sin, cos)
        if backend == "inference":
            return self._apply_rope_inference_bsnd(q, k, sin, cos)
        if backend == "trainable":
            return self._apply_rope_trainable_bsnd(q, k, sin, cos)
        return self._apply_rope_manual_bsnd(q, k, sin, cos)

    def apply_rope(
        self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        """Compatibility wrapper for callers that provide BNSD Q/K tensors."""
        q, k = self.apply_rope_bsnd(q.transpose(1, 2), k.transpose(1, 2), rope)
        return q.transpose(1, 2), k.transpose(1, 2)

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)  # BSND
        if rope is not None:
            q, k = self.apply_rope_bsnd(q, k, rope)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
