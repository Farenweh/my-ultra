from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn

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


class DINOv3ViT(nn.Module):
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
