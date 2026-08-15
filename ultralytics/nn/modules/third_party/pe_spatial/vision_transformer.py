# Copyright (c) Meta Platforms, Inc. and affiliates.
# 本文件基于 Apache-2.0 许可的 Perception Encoder 视觉实现修改。

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from timm.layers import DropPath
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ultralytics.utils.attention import sdpa_with_npu_fusion

from .config import PESpatialConfig
from .rope import Rope2D, apply_rope


class LayerScale(nn.Module):
    """保持官方 checkpoint 键名的 LayerScale。"""

    def __init__(self, dim: int, init_values: float):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class SelfAttention(nn.Module):
    """兼容官方权重并保持 BSND RoPE 快路径的自注意力。"""

    def __init__(self, embed_dim: int, num_heads: int, rope: Rope2D | None):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError(f"embed_dim={embed_dim}不能被num_heads={num_heads}整除")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.zeros_(self.in_proj_bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None = None, attn_mask: Tensor | None = None) -> Tensor:
        batch, sequence, channels = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.reshape(batch, sequence, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)  # BSND
        if rope is not None:
            sin, cos = rope
            q = apply_rope(q, sin, cos)
            k = apply_rope(k, sin, cos)
        x = sdpa_with_npu_fusion(
            q,
            k,
            v,
            num_heads=self.num_heads,
            input_layout="BSND",
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scale,
        )
        x = x.reshape(batch, sequence, channels)
        return self.out_proj(x)


class ResidualAttentionBlock(nn.Module):
    """PE-Spatial Transformer block。"""

    def __init__(
        self,
        width: int,
        heads: int,
        mlp_ratio: float,
        rope: Rope2D,
        layer_scale: float | None,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.attn = SelfAttention(width, heads, rope)
        self.ls_1 = LayerScale(width, layer_scale) if layer_scale is not None else nn.Identity()
        self.ls_2 = LayerScale(width, layer_scale) if layer_scale is not None else nn.Identity()
        self.ln_1 = nn.LayerNorm(width, eps=1e-5)
        self.ln_2 = nn.LayerNorm(width, eps=1e-5)
        self.drop_path1 = DropPath(drop_path) if drop_path else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path else nn.Identity()
        mlp_width = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            OrderedDict(
                (
                    ("c_fc", nn.Linear(width, mlp_width)),
                    ("gelu", nn.GELU()),
                    ("c_proj", nn.Linear(mlp_width, width)),
                )
            )
        )

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], attn_mask: Tensor | None = None) -> Tensor:
        x = x + self.drop_path1(self.ls_1(self.attn(self.ln_1(x), rope=rope, attn_mask=attn_mask)))
        return x + self.drop_path2(self.ls_2(self.mlp(self.ln_2(x))))


class Transformer(nn.Module):
    """保持官方 transformer.resblocks checkpoint 命名的编码器堆叠。"""

    def __init__(self, config: PESpatialConfig, rope: Rope2D):
        super().__init__()
        self.grad_checkpointing = False
        self.resblocks = nn.ModuleList(
            ResidualAttentionBlock(
                config.width,
                config.heads,
                config.mlp_ratio,
                rope,
                config.layer_scale,
            )
            for _ in range(config.layers)
        )

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor], layer_idx: int = -1) -> Tensor:
        stop_idx = (len(self.resblocks) + layer_idx) % len(self.resblocks)
        for index, block in enumerate(self.resblocks):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(lambda value, module=block: module(value, rope), x, use_reentrant=False)
            else:
                x = block(x, rope)
            if index == stop_idx:
                break
        return x


class VisionTransformer(nn.Module):
    """PE-Spatial 纯视觉编码器，保持官方 checkpoint 结构并加入安全缓存。"""

    def __init__(self, config: PESpatialConfig, norm_layer: Callable = partial(nn.LayerNorm, eps=1e-5)):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.width = config.width
        self.layers = config.layers
        self.heads = config.heads
        self.image_size = config.image_size
        self.use_cls_token = config.use_cls_token
        self.conv1 = nn.Conv2d(3, config.width, config.patch_size, stride=config.patch_size, bias=False)
        self.rope = Rope2D(config.width // config.heads, config.use_cls_token)
        self.ln_pre = norm_layer(config.width)
        self.ln_post = nn.Identity()
        self.transformer = Transformer(config, self.rope)

        init_scale = config.width**-0.5
        if config.use_cls_token:
            self.class_embedding = nn.Parameter(init_scale * torch.randn(config.width))
        self.posemb_grid_size = config.image_size // config.patch_size
        self.positional_embedding = nn.Parameter(
            init_scale * torch.randn(int(config.use_cls_token) + self.posemb_grid_size**2, config.width)
        )
        self._pos_embed_cache = torch.empty(0)
        self._pos_embed_cache_key = None

    def _clear_position_cache(self) -> None:
        """清空可由位置参数重建的实例缓存。"""
        self._pos_embed_cache = torch.empty(0)
        self._pos_embed_cache_key = None

    def __getstate__(self):
        """序列化和深拷贝时不携带插值缓存。"""
        state = super().__getstate__()
        state["_pos_embed_cache"] = torch.empty(0)
        state["_pos_embed_cache_key"] = None
        return state

    def _apply(self, fn):
        """模型迁移设备或 dtype 时丢弃可重建的位置缓存。"""
        result = super()._apply(fn)
        self._clear_position_cache()
        return result

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """权重重载后丢弃可能引用旧位置参数的缓存。"""
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._clear_position_cache()

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.transformer.set_grad_checkpointing(enable)

    def _interpolate_positional_embedding(self, grid_h: int, grid_w: int) -> Tensor:
        if self.posemb_grid_size == grid_h == grid_w:
            return self.positional_embedding.unsqueeze(0)
        pos_embed = self.positional_embedding
        if self.use_cls_token:
            cls_embed, pos_embed = pos_embed[:1], pos_embed[1:]
        pos_embed = pos_embed.reshape(1, self.posemb_grid_size, self.posemb_grid_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(grid_h, grid_w), mode="bilinear", align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(-1, self.width)
        if self.use_cls_token:
            pos_embed = torch.cat((cls_embed, pos_embed), dim=0)
        return pos_embed.unsqueeze(0)

    def _sample_abs_posemb(self, grid_h: int, grid_w: int) -> Tensor:
        if self.positional_embedding.requires_grad:
            return self._interpolate_positional_embedding(grid_h, grid_w)
        parameter = self.positional_embedding
        key = (
            id(parameter),
            parameter.device.type,
            parameter.device.index,
            parameter.dtype,
            None if torch.is_inference(parameter) else parameter._version,
            grid_h,
            grid_w,
        )
        if key != self._pos_embed_cache_key:
            self._pos_embed_cache = self._interpolate_positional_embedding(grid_h, grid_w).detach()
            self._pos_embed_cache_key = key
        return self._pos_embed_cache

    def forward_features(
        self,
        x: Tensor,
        norm: bool = False,
        layer_idx: int = -1,
        strip_cls_token: bool = False,
    ) -> Tensor:
        batch, _, height, width = x.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        x = self.conv1(x).permute(0, 2, 3, 1).reshape(batch, -1, self.width)
        if self.use_cls_token:
            cls_token = self.class_embedding.view(1, 1, -1).expand(batch, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = self.ln_pre(x + self._sample_abs_posemb(grid_h, grid_w))
        rope = self.rope.coefficients(grid_h, grid_w, x.device)
        x = self.transformer(x, rope, layer_idx=layer_idx)
        if norm:
            x = self.ln_post(x)
        if strip_cls_token and self.use_cls_token:
            x = x[:, 1:]
        return x

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward_features(x, norm=True, **kwargs)
