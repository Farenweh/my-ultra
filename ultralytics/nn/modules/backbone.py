from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn

from .third_party.dinov2.hubconf import (
    dinov2_vitb14,
    dinov2_vitb14_reg,
    dinov2_vitg14,
    dinov2_vitg14_reg,
    dinov2_vitl14,
    dinov2_vitl14_reg,
    dinov2_vits14,
    dinov2_vits14_reg,
)
from .third_party.dinov3.hubconf import (
    dinov3_convnext_base,
    dinov3_convnext_large,
    dinov3_convnext_small,
    dinov3_convnext_tiny,
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vitl16plus,
    dinov3_vits16,
    dinov3_vits16plus,
)


def _is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in {"http", "https", "file"}


def resolve_dinov3_weights(pretrained: str, filename: str) -> str:
    """Resolve a DINOv3 local weights directory or explicit URL/path to a loadable path."""
    if _is_url(pretrained):
        return pretrained
    return str((Path(pretrained).expanduser() / filename).resolve())


class DINOv2(nn.Module):
    input_size_divisor = 14
    feature_stride = 14

    def __init__(self, scale: str, pretrained: str | bool = True):
        super().__init__()
        scaletable = {
            "s": dinov2_vits14,
            "b": dinov2_vitb14,
            "l": dinov2_vitl14,
            "g": dinov2_vitg14,
            "s_reg": dinov2_vits14_reg,
            "b_reg": dinov2_vitb14_reg,
            "l_reg": dinov2_vitl14_reg,
            "g_reg": dinov2_vitg14_reg,
        }
        self.scale = scale.lower()
        assert self.scale in scaletable.keys(), f"不存在的DINOv2架构预设，应该为{scaletable.keys()}"

        kwargs = {"pretrained": pretrained is not False}
        if isinstance(pretrained, str):
            kwargs["weights"] = pretrained
        self.model = scaletable[self.scale](**kwargs)

    def _check_input(self, x: torch.Tensor):
        if x.ndim != 4:
            raise AssertionError(f"DINOv2的输入形状应该为BCHW，但得到的是 {tuple(x.shape)}")
        h, w = x.shape[-2:]
        divisor = self.input_size_divisor
        if h % divisor or w % divisor:
            raise AssertionError(f"DINOv2要求输入的高度和宽度是{divisor}的倍数，但得到的是 {(h, w)}")

    def forward(self, x: torch.Tensor):
        self._check_input(x)
        return self.model.get_intermediate_layers(x, n=1, reshape=True)[0]

    def forward_sequence(self, x: torch.Tensor):
        self._check_input(x)
        return self.model.get_intermediate_layers(x, n=1, reshape=False)[0]

    @staticmethod
    def dims(scale: str):
        dimstable = {
            "s": 384,
            "b": 768,
            "l": 1024,
            "g": 1536,
            "s_reg": 384,
            "b_reg": 768,
            "l_reg": 1024,
            "g_reg": 1536,
        }
        return dimstable[scale.lower()]


class DINOv3ViT(nn.Module):
    input_size_divisor = 16
    feature_stride = 16

    def __init__(self, scale: str, pretrained: str | bool = "./weights", rescale_coords: None | int = None):
        super(DINOv3ViT, self).__init__()
        scaletable = {
            "s": dinov3_vits16,
            "sp": dinov3_vits16plus,
            "b": dinov3_vitb16,
            "l": dinov3_vitl16,
            "lp": dinov3_vitl16plus,
            "hp": dinov3_vith16plus,
            "7b": dinov3_vit7b16,
        }
        weightstable = {"l": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"}
        self.scale = scale.lower()
        assert self.scale in scaletable.keys(), f"不存在的DINOv3ViT架构预设，应该为{scaletable.keys()}"
        if pretrained is False:
            kv = {"pretrained": False, "weights": None}
        elif isinstance(pretrained, str):
            kv = {"pretrained": True, "weights": resolve_dinov3_weights(pretrained, weightstable[self.scale])}
        else:
            kv = {"pretrained": True}
        self.model = scaletable[self.scale](**kv)
        self.model.rope_embed.rescale_coords = rescale_coords

    def forward(self, x: torch.Tensor):
        result = self.model.get_intermediate_layers(x, n=1, reshape=True)[0]
        return result

    def forward_sequence(self, x: torch.Tensor):
        return self.model.get_intermediate_layers(x, n=1, reshape=False)[0]

    @staticmethod
    def dims(scale: str):
        dimstable = {
            "s": 384,
            "sp": 384,
            "b": 768,
            "l": 1024,
            "lp": 1024,
            "hp": 1280,
            "7b": 4096,
        }
        return dimstable[scale.lower()]
