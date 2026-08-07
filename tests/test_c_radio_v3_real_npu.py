from __future__ import annotations

import importlib.util
import os

import pytest
import torch

from ultralytics.nn.modules.backbone import CRADIOv3
from ultralytics.nn.modules.third_party.c_radio_v3 import attention


NPU_AVAILABLE = importlib.util.find_spec("torch_npu") is not None and hasattr(torch, "npu") and torch.npu.is_available()
RUN_REAL_MODEL = os.getenv("RUN_C_RADIO_V3_L_TESTS") == "1"


@pytest.mark.slow
@pytest.mark.skipif(not NPU_AVAILABLE or not RUN_REAL_MODEL, reason="需要显式启用C-RADIOv3-L真实NPU测试")
def test_c_radio_v3_l_official_checkpoint_forward_and_backward(monkeypatch):
    """验证官方L/16权重、动态分辨率、AMP和首层融合attention梯度。"""
    torch.npu.set_device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    model = CRADIOv3("l", pretrained=True).eval().requires_grad_(False).to("npu:0")

    for height, width in ((512, 512), (640, 640), (800, 800), (384, 640)):
        image = torch.rand(1, 3, height, width, device="npu:0")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with torch.inference_mode(), torch.autocast("npu", dtype=dtype, enabled=dtype != torch.float32):
                sequence = model.forward_sequence(image)
                spatial = model(image)
            assert spatial.shape == (1, 1024, height // 16, width // 16)
            assert torch.isfinite(spatial).all()
            torch.testing.assert_close(sequence, spatial.flatten(2).transpose(1, 2))

    # 完整L模型的反向非常昂贵；首个block已经覆盖融合attention输出、输入梯度和QKV权重梯度。
    block = model.model.blocks[0].train().requires_grad_(True)
    source = torch.randn(1, 1032, 1024, device="npu:0", dtype=torch.float16)

    def run(policy: str):
        monkeypatch.setattr(attention, "CRADIO_V3_ATTENTION_BACKEND", policy)
        block.zero_grad(set_to_none=True)
        x = source.detach().clone().requires_grad_(True)
        output = block(x)
        loss = output.float().square().mean()
        loss.backward()
        return output.detach(), loss.detach(), x.grad.detach(), block.attn.qkv.weight.grad.detach().clone()

    sdpa = run("sdpa")
    fused = run("auto")
    for actual, expected in zip(fused, sdpa):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)
