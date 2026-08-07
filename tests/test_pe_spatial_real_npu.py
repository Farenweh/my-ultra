from __future__ import annotations

import importlib.util
import os

import pytest
import torch

from ultralytics.nn.modules.backbone import PESpatial
from ultralytics.nn.modules.third_party.pe_spatial import rope


NPU_AVAILABLE = importlib.util.find_spec("torch_npu") is not None and hasattr(torch, "npu") and torch.npu.is_available()
RUN_REAL_MODEL = os.getenv("RUN_PE_SPATIAL_L_TESTS") == "1"


@pytest.mark.slow
@pytest.mark.skipif(not NPU_AVAILABLE or not RUN_REAL_MODEL, reason="需要显式启用PE-Spatial-L真实NPU测试")
def test_pe_spatial_l_official_checkpoint_forward_and_backward():
    """验证官方 L/14 权重、动态分辨率、AMP 和首层融合 RoPE 梯度。"""
    torch.npu.set_device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    model = PESpatial("l", pretrained=True).eval().requires_grad_(False).to("npu:0")

    for image_size, expected_grid in ((448, 32), (644, 46)):
        image = torch.rand(1, 3, image_size, image_size, device="npu:0")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with torch.inference_mode(), torch.autocast("npu", dtype=dtype, enabled=dtype != torch.float32):
                sequence = model.forward_sequence(image)
                spatial = model(image)
            assert spatial.shape == (1, 1024, expected_grid, expected_grid)
            assert torch.isfinite(spatial).all()
            torch.testing.assert_close(sequence, spatial.flatten(2).transpose(1, 2))

    model.train().requires_grad_(True)
    source = torch.rand(1, 3, 224, 224, device="npu:0")

    def run(policy: str):
        rope.PE_SPATIAL_ROPE_BACKEND = policy
        model.zero_grad(set_to_none=True)
        image = source.detach().clone().requires_grad_(True)
        with torch.autocast("npu", dtype=torch.float16):
            output = model.model.forward_features(
                image.mul(2).sub(1),
                layer_idx=0,
                strip_cls_token=True,
            )
            loss = output.float().square().mean()
        loss.backward()
        qkv_grad = model.model.transformer.resblocks[0].attn.in_proj_weight.grad
        return output.detach(), loss.detach(), image.grad.detach(), qkv_grad.detach().clone()

    manual = run("manual")
    fused = run("auto")
    for actual, expected in zip(fused, manual):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
