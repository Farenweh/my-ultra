# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import os

import torch
from torch import Tensor, nn
from ultralytics.utils.checks import IS_ASCEND


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        if IS_ASCEND and os.getenv("ASCEND_FUSED_RMSNORM", "1") == "1":
            return torch.npu.torch_npu.npu_rms_norm(x, self.weight, epsilon=self.eps)[0]

        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
