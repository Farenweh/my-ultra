from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .attention import Attention
from .config import CRADIOv3Config
from .cpe import CPEPatchGenerator


class LayerScale(nn.Module):
    """使用官方safetensors中的grandma参数名。"""

    def __init__(self, dim: int, init_values: float, *, device=None, dtype=None):
        super().__init__()
        self.grandma = nn.Parameter(torch.full((dim,), init_values, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.grandma


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, *, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.fc1 = nn.Linear(dim, hidden_dim, **factory)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim, **factory)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, config: CRADIOv3Config, *, device=None, dtype=None):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.norm1 = nn.LayerNorm(config.width, eps=1e-6, **factory)
        self.attn = Attention(config.width, config.heads, **factory)
        self.ls1 = (
            LayerScale(config.width, config.layer_scale, **factory) if config.layer_scale is not None else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(config.width, eps=1e-6, **factory)
        self.mlp = Mlp(config.width, config.width * 4, **factory)
        self.ls2 = (
            LayerScale(config.width, config.layer_scale, **factory) if config.layer_scale is not None else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class VisionTransformer(nn.Module):
    """C-RADIOv3纯视觉编码器。"""

    def __init__(
        self,
        config: CRADIOv3Config,
        *,
        initialize: bool = True,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.config = config
        self.embed_dim = config.width
        self.patch_size = config.patch_size
        self.grad_checkpointing = False
        self.frozen_deterministic = False
        self.patch_generator = CPEPatchGenerator(
            config.width,
            patch_size=config.patch_size,
            max_resolution=config.max_resolution,
            prefix_tokens=config.prefix_tokens,
            device=device,
            dtype=dtype,
        )
        if config.base_registers:
            self.reg_token = nn.Parameter(
                torch.empty(1, config.base_registers, config.width, device=device, dtype=dtype)
            )
        self.blocks = nn.ModuleList(Block(config, device=device, dtype=dtype) for _ in range(config.depth))
        self.norm = nn.Identity()
        if initialize:
            self.reset_parameters(norm_layer)

    def reset_parameters(self, norm_layer: Callable[..., nn.Module] = nn.LayerNorm) -> None:
        del norm_layer
        self.patch_generator.reset_parameters()
        if hasattr(self, "reg_token"):
            nn.init.normal_(self.reg_token, std=self.config.width**-0.5)
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.patch_generator.embedder:
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    def forward_features(self, x: Tensor) -> Tensor:
        stochastic = self.training and not self.frozen_deterministic
        x = self.patch_generator(x, stochastic=stochastic)
        for block in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_features(x)
