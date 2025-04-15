import torch
import torch.nn as nn
from transformers import AutoModel


class DINOv2(nn.Module):
    def __init__(self, scale: str, register: bool, reshape: bool):
        super().__init__()
        self.register = register
        self.reshape = reshape
        self.scale = scale.upper()

        scale_dict = {"S": "small", "B": "base", "L": "large", "G": "giant"}
        if register:
            self.hf_model_name = f"facebook/dinov2-with-registers-{scale_dict[self.scale]}"
        else:
            self.hf_model_name = f"facebook/dinov2-{scale_dict[self.scale]}"
        self.model = AutoModel.from_pretrained(self.hf_model_name)
        self.dim = self.model.config.hidden_size
        self.patch_size = self.model.config.patch_size

    def forward(self, x):
        b, _, h, w = x.shape
        x = self.model.forward(x).last_hidden_state[:, 1:, :]  # remove cls token
        if self.reshape:
            x = x.view(b, self.dim, h // self.patch_size, w // self.patch_size)
        return x

    @staticmethod
    def dims(scale: str):
        scale = scale.upper()
        dim_dict = {"S": 384, "B": 768, "L": 1024, "G": 1536}
        return dim_dict[scale]
