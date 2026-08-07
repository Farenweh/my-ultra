from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class CPEPatchGenerator(nn.Module):
    """C-RADIOv3的patch投影、前缀token和裁剪位置编码。"""

    def __init__(
        self,
        width: int,
        *,
        patch_size: int = 16,
        max_resolution: int = 2048,
        prefix_tokens: int = 8,
        pos_dropout: float = 0.1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.width = width
        self.patch_size = patch_size
        self.max_grid = max_resolution // patch_size
        self.prefix_tokens = prefix_tokens
        self.pos_dropout = pos_dropout
        self.embedder = nn.Linear(3 * patch_size**2, width, bias=False, **factory)
        self.pos_embed = nn.Parameter(torch.empty(1, self.max_grid**2, width, **factory))
        self.cls_token = nn.Module()
        self.cls_token.token = nn.Parameter(torch.empty(prefix_tokens, width, **factory))
        self._base_grid_cache = torch.empty(0)
        self._base_grid_cache_key = None
        self._position_cache = torch.empty(0)
        self._position_cache_key = None

    def _clear_caches(self) -> None:
        """清空可由输入网格和位置参数重建的实例缓存。"""
        self._base_grid_cache = torch.empty(0)
        self._base_grid_cache_key = None
        self._position_cache = torch.empty(0)
        self._position_cache_key = None

    @property
    def num_skip(self) -> int:
        return self.prefix_tokens

    def __getstate__(self):
        state = super().__getstate__()
        state["_base_grid_cache"] = torch.empty(0)
        state["_base_grid_cache_key"] = None
        state["_position_cache"] = torch.empty(0)
        state["_position_cache_key"] = None
        return state

    def _apply(self, fn):
        result = super()._apply(fn)
        self._clear_caches()
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
        self._clear_caches()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.embedder.weight)
        nn.init.normal_(self.pos_embed, std=self.width**-0.5)
        nn.init.normal_(self.cls_token.token, std=self.width**-0.5)

    def _patches(self, x: Tensor) -> Tensor:
        patch = self.patch_size
        patches = x.unfold(2, patch, patch).unfold(3, patch, patch)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(x.shape[0], -1, 3 * patch**2)
        return self.embedder(patches)

    def _base_grid(self, grid_h: int, grid_w: int, device: torch.device) -> Tensor:
        key = (device.type, device.index, grid_h, grid_w, torch.float32)
        if key != self._base_grid_cache_key:
            x = torch.linspace(0, 1, steps=grid_w, device=device, dtype=torch.float32)
            y = torch.linspace(0, 1, steps=grid_h, device=device, dtype=torch.float32)
            grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
            self._base_grid_cache = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
            self._base_grid_cache_key = key
        return self._base_grid_cache

    def _stochastic_position(self, batch: int, grid_h: int, grid_w: int) -> Tensor:
        pos = self.pos_embed.reshape(1, self.max_grid, self.max_grid, self.width).permute(0, 3, 1, 2)
        min_scale = math.sqrt(0.1)
        scale = torch.rand(batch, 1, 1, device=pos.device) * (1 - min_scale) + min_scale
        aspect_min = math.log(3 / 4)
        aspect = torch.exp(torch.rand(batch, 1, 1, device=pos.device) * (-2 * aspect_min) + aspect_min)
        scale_xy = torch.stack((scale * aspect, scale / aspect), dim=-1).clamp_(0, 1)
        offset = torch.rand(batch, 1, 1, 2, device=pos.device) * (1 - scale_xy)
        grid = (self._base_grid(grid_h, grid_w, pos.device) * scale_xy + offset).mul(2).sub(1)
        pos = F.grid_sample(
            pos.float().expand(batch, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).to(dtype=self.pos_embed.dtype)
        return pos.flatten(2).transpose(1, 2)

    def _deterministic_position(self, grid_h: int, grid_w: int) -> Tensor:
        if grid_h == self.max_grid and grid_w == self.max_grid:
            return self.pos_embed
        pos = self.pos_embed.reshape(1, self.max_grid, self.max_grid, self.width).permute(0, 3, 1, 2)
        max_dim = max(grid_h, grid_w)
        pos = F.interpolate(pos.float(), size=(max_dim, max_dim), mode="bilinear", align_corners=False)
        pos = pos[..., :grid_h, :grid_w]
        return pos.to(dtype=self.pos_embed.dtype).flatten(2).transpose(1, 2)

    def _position(self, batch: int, grid_h: int, grid_w: int, stochastic: bool) -> Tensor:
        if grid_h == self.max_grid and grid_w == self.max_grid:
            return self.pos_embed
        if stochastic:
            return self._stochastic_position(batch, grid_h, grid_w)
        parameter = self.pos_embed
        can_cache = not torch.is_grad_enabled() or not parameter.requires_grad
        if not can_cache:
            return self._deterministic_position(grid_h, grid_w)
        key = (
            id(parameter),
            parameter.device.type,
            parameter.device.index,
            parameter.dtype,
            None if torch.is_inference(parameter) else parameter._version,
            grid_h,
            grid_w,
        )
        if key != self._position_cache_key:
            self._position_cache = self._deterministic_position(grid_h, grid_w).detach()
            self._position_cache_key = key
        return self._position_cache

    def forward(self, x: Tensor, *, stochastic: bool) -> Tensor:
        grid_h, grid_w = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        patches = self._patches(x)
        position = self._position(x.shape[0], grid_h, grid_w, stochastic)
        if stochastic and self.pos_dropout > 0:
            keep = torch.rand(x.shape[0], 1, 1, dtype=position.dtype, device=position.device) > self.pos_dropout
            position = torch.where(keep, position, 0)
        patches = patches + position
        token = self.cls_token.token.unsqueeze(0).expand(x.shape[0], -1, -1)
        return torch.cat((token, patches), dim=1)
