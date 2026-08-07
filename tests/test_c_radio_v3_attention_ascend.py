from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import types

import pytest
import torch


NPU_AVAILABLE = importlib.util.find_spec("torch_npu") is not None and hasattr(torch, "npu") and torch.npu.is_available()


def test_c_radio_v3_attention_cpu_uses_sdpa(monkeypatch):
    from ultralytics.nn.modules.third_party.c_radio_v3 import attention

    monkeypatch.setattr(attention, "CRADIO_V3_ATTENTION_BACKEND", "auto")
    q = torch.randn(2, 17, 16, 64)
    assert attention.select_attention_backend(q, q, q, 16) == "sdpa"


def test_c_radio_v3_attention_strict_mode_reports_unsupported(monkeypatch):
    from ultralytics.nn.modules.third_party.c_radio_v3 import attention

    monkeypatch.setattr(attention, "CRADIO_V3_ATTENTION_BACKEND", "fusion")
    monkeypatch.setattr(attention, "_supports_npu_fusion_attention", lambda *args: False)
    q = torch.randn(2, 17, 16, 64)
    with pytest.raises(RuntimeError, match="fusion不支持"):
        attention.select_attention_backend(q, q, q, 16)


def test_c_radio_v3_attention_stub_preserves_bsnd(monkeypatch):
    from ultralytics.nn.modules.third_party.c_radio_v3 import attention

    calls = []

    def fusion(q, k, v, heads, layout, **kwargs):
        calls.append((q.shape, heads, layout, kwargs))
        output = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=kwargs["scale"]
        ).transpose(1, 2)
        return (output, None, None, None, 0, 0, 0)

    module = attention.Attention(1024, 16)
    monkeypatch.setattr(attention, "CRADIO_V3_ATTENTION_BACKEND", "auto")
    monkeypatch.setattr(attention, "_supports_npu_fusion_attention", lambda *args: True)
    monkeypatch.setattr(attention, "_get_torch_npu", lambda: types.SimpleNamespace(npu_fusion_attention=fusion))
    x = torch.randn(2, 17, 1024, requires_grad=True)
    output = module(x)
    output.square().mean().backward()

    assert calls == [(torch.Size([2, 17, 16, 64]), 16, "BSND", {"scale": 0.125, "keep_prob": 1.0})]
    assert x.grad is not None


def test_c_radio_v3_import_does_not_lock_visible_devices():
    if not NPU_AVAILABLE:
        pytest.skip("需要可用的Ascend NPU")
    script = """
import os
import torch
import ultralytics.nn.modules.third_party.c_radio_v3.attention

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
x = torch.ones(1, device="npu:0")
print(x.cpu().item())
"""
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=os.getcwd(), capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "1.0" in result.stdout


@pytest.mark.skipif(not NPU_AVAILABLE, reason="需要可用的Ascend NPU")
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
def test_c_radio_v3_real_npu_fusion_matches_sdpa_forward_backward(dtype, monkeypatch):
    from ultralytics.nn.modules.third_party.c_radio_v3 import attention

    torch.npu.set_device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    module = attention.Attention(1024, 16).to(device="npu:0", dtype=dtype)
    source = torch.randn(1, 1032, 1024, device="npu:0", dtype=dtype)

    def run(policy: str):
        monkeypatch.setattr(attention, "CRADIO_V3_ATTENTION_BACKEND", policy)
        module.zero_grad(set_to_none=True)
        x = source.detach().clone().requires_grad_(True)
        output = module(x)
        loss = output.float().square().mean()
        loss.backward()
        return output.detach(), loss.detach(), x.grad.detach(), module.qkv.weight.grad.detach().clone()

    sdpa = run("sdpa")
    fused = run("auto")
    for actual, expected in zip(fused, sdpa):
        torch.testing.assert_close(actual, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="需要可用的Ascend NPU")
def test_c_radio_v3_real_npu_fp32_and_jit_fall_back(monkeypatch):
    from ultralytics.nn.modules.third_party.c_radio_v3 import attention

    torch.npu.set_device("npu:0")
    q = torch.randn(1, 17, 16, 64, device="npu:0", dtype=torch.float32)
    monkeypatch.setattr(attention, "CRADIO_V3_ATTENTION_BACKEND", "auto")
    assert attention.select_attention_backend(q, q, q, 16) == "sdpa"

    q = q.half()
    monkeypatch.setattr(attention, "_ascend_jit_compile_enabled", lambda: True)
    assert attention.select_attention_backend(q, q, q, 16) == "sdpa"
