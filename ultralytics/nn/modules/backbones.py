import torch

from ultralytics import nn
from ultralytics.utils import LOGGER, colorstr
from .dinov2.vision_transformer import DinoVisionTransformer
from .dinov2.configs import dinov2_model_configs, dinov2_pretrained_urls


class DINOv2(DinoVisionTransformer):
    def __init__(self, scale: str, reg: bool, reshape: bool):
        self.scale = scale.lower()
        self.reg = reg
        self.reshape = reshape
        key = f"{self.scale}_reg" if self.reg else self.scale
        model_config = dinov2_model_configs[key]
        super().__init__(**model_config)
        weights_url = dinov2_pretrained_urls[key]
        weights = torch.hub.load_state_dict_from_url(weights_url, map_location="cpu")
        LOGGER.info(f"\nLoading {colorstr(f'DINOv2-{self.scale.upper()} REG={self.reg} RESHAPE={self.reshape}')} weights from {weights_url}\n")
        self.load_state_dict(weights)

    def forward(self, x):
        return self.get_intermediate_layers(x, 1, self.reshape)[0]

    @staticmethod
    def dims(scale: str):
        assert scale.lower() in {"s", "b", "l", "g"}, f"Invalid DINOv2 type {scale}, must be s, b, l, g"
        return dinov2_model_configs[scale.lower()]["embed_dim"]


class MoDINOv2(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass
