# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Callable, Tuple, Union

import torch.nn.functional as F
from torch import Tensor, nn


def project_nonoverlapping_patches(x: Tensor, projection: nn.Conv2d) -> Tensor:
    """将互不重叠的卷积 patch 等价展开为线性投影，保留原卷积参数及其梯度。"""
    batch, channels, height, width = x.shape
    ph, pw = projection.kernel_size
    grid_h, grid_w = height // ph, width // pw
    if grid_h < 1 or grid_w < 1:
        raise ValueError("输入图像的高宽不能小于 patch 的高宽。")
    # 与无 padding 的原卷积一致，忽略右侧和底部不足一个 patch 的像素。
    patches = x[..., : grid_h * ph, : grid_w * pw].reshape(batch, channels, grid_h, ph, grid_w, pw)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(batch, grid_h * grid_w, channels * ph * pw)
    return F.linear(patches, projection.weight.flatten(1), projection.bias)


class _PatchProjection(nn.Conv2d):
    """昇腾始终使用等价线性投影，保留 Conv2d 的参数和模块 hook。"""

    def _conv_forward(self, x: Tensor, weight: Tensor, bias: Tensor | None) -> Tensor:
        if x.device.type == "npu":
            ph, pw = self.kernel_size
            tokens = project_nonoverlapping_patches(x, self)
            return tokens.transpose(1, 2).reshape(x.shape[0], self.out_channels, x.shape[2] // ph, x.shape[3] // pw)
        return super()._conv_forward(x, weight, bias)


def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = _PatchProjection(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def __setstate__(self, state):
        """旧完整 checkpoint 的 Conv2d 原位升级，保留参数对象、状态与 hook。"""
        super().__setstate__(state)
        if type(self.proj) is nn.Conv2d:
            self.proj.__class__ = _PatchProjection

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
