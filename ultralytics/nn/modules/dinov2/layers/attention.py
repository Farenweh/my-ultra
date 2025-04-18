# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging

# import os # No longer needed for XFORMERS check
# import warnings # No longer needed for XFORMERS warnings

import torch  # Ensure torch is imported
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # Import F

logger = logging.getLogger("dinov2")

# --- XFORMERS related code removed ---
# XFORMERS_ENABLED = ...
# try: ...
# except ImportError: ...
# XFORMERS_AVAILABLE = False # No longer needed


class Attention(nn.Module):
    """
    Attention module using torch.nn.functional.scaled_dot_product_attention
    for efficiency.
    """

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
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        # Removed self.scale as SDPA handles scaling internally

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # NOTE: sdpa uses attn_drop AFTER softmax but BEFORE multiplying by V
        #       which is the same as the original implementation's attn_drop placement.
        self.attn_drop = nn.Dropout(attn_drop)  # Keep dropout module definition if needed elsewhere, otherwise just use the rate
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self._attn_drop_rate = attn_drop  # Store the rate for SDPA

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (B, N, C)
            attn_bias (Tensor, optional): Attention bias (mask) to be added to
                                         attention scores before softmax. Defaults to None.
                                         Needed for compatibility with MemEffAttention.
        Returns:
            Tensor: Output tensor of shape (B, N, C)
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv shape: (3, B, H, N, D), where H is num_heads and D is head_dim
        q, k, v = qkv.unbind(0)  # q, k, v each shape: (B, H, N, D)

        # Use SDPA
        # attn_mask=attn_bias allows passing potential masks/biases
        # dropout_p is applied during training only
        # is_causal=False for standard ViT attention
        try:
            # Pass attn_bias directly if it's compatible with SDPA's expected format
            # (e.g., a boolean mask where True indicates masking, or a float additive mask)
            x = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_bias, dropout_p=self._attn_drop_rate if self.training else 0.0, is_causal=False
            )
        except RuntimeError as e:
            # Provide a more informative error if attn_bias format is wrong
            if attn_bias is not None and "mask" in str(e):
                logger.error(
                    f"Failed to apply attn_bias with shape {attn_bias.shape} in F.scaled_dot_product_attention. Ensure it's broadcastable to (B, H, N, N). Error: {e}"
                )
            raise e

        # x shape: (B, H, N, D)
        # Combine heads and reshape back to (B, N, C)
        x = x.transpose(1, 2).reshape(B, N, C)

        # Apply projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    """
    Memory-Efficient Attention layer.
    Now effectively inherits the SDPA implementation from the base Attention class.
    The primary reason for this class in the original code was the xformers path.
    Since we replaced the base implementation with an efficient one (SDPA),
    this class mainly serves to maintain the original structure and potentially
    handle `attn_bias` explicitly if needed (though the base class now handles it).
    """

    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        # Simply call the parent's forward method, which now uses SDPA
        # and accepts attn_bias.
        return super().forward(x, attn_bias=attn_bias)
