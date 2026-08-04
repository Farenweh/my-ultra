from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_attention_module(monkeypatch, *, is_ascend: bool, use_ascend_rope: bool, torch_npu_stub=None):
    # Ensure local external DINOv3 package is importable as `dinov3`.
    monkeypatch.syspath_prepend(str(_repo_root() / "ultralytics" / "nn" / "modules" / "third_party" / "dinov3"))

    import ultralytics.utils.checks as checks

    monkeypatch.setattr(checks, "IS_ASCEND", is_ascend, raising=True)
    monkeypatch.setenv("USE_DINOV3_ASCEND_ROPE", "1" if use_ascend_rope else "0")
    if torch_npu_stub is not None:
        monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_stub)

    # Re-import to pick up env + IS_ASCEND changes at module import time.
    sys.modules.pop("dinov3.layers.attention", None)
    mod = importlib.import_module("dinov3.layers.attention")
    return importlib.reload(mod)


def _manual_apply(module, q: torch.Tensor, k: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
    q_dtype = q.dtype
    k_dtype = k.dtype
    rope_dtype = sin.dtype

    q_cast = q.to(dtype=rope_dtype)
    k_cast = k.to(dtype=rope_dtype)
    prefix = q.shape[-2] - sin.shape[-2]

    q_prefix = q_cast[:, :, :prefix, :]
    k_prefix = k_cast[:, :, :prefix, :]

    q_suffix = module.rope_apply(q_cast[:, :, prefix:, :], sin, cos)
    k_suffix = module.rope_apply(k_cast[:, :, prefix:, :], sin, cos)

    q_out = torch.cat((q_prefix, q_suffix), dim=-2).to(dtype=q_dtype)
    k_out = torch.cat((k_prefix, k_suffix), dim=-2).to(dtype=k_dtype)
    return q_out, k_out


def test_dinov3_rope_non_ascend_uses_manual(monkeypatch):
    module = _load_attention_module(monkeypatch, is_ascend=False, use_ascend_rope=True)
    assert module.USE_ASCEND_ROPE is False

    attn = module.SelfAttention(dim=64, num_heads=4)
    q = torch.randn(2, 4, 10, 16, dtype=torch.float32)
    k = torch.randn(2, 4, 10, 16, dtype=torch.float32)
    sin = torch.randn(6, 16, dtype=torch.float16)
    cos = torch.randn(6, 16, dtype=torch.float16)

    q_out, k_out = attn.apply_rope(q, k, (sin, cos))
    q_expected, k_expected = _manual_apply(module, q, k, sin, cos)

    torch.testing.assert_close(q_out, q_expected)
    torch.testing.assert_close(k_out, k_expected)


def test_dinov3_rope_ascend_enabled_on_cpu_uses_manual(monkeypatch):
    calls = []

    def npu_rotary_mul(*, input, r1, r2, rotary_mode="half"):
        calls.append(
            {
                "input_shape": tuple(input.shape),
                "r1_shape": tuple(r1.shape),
                "r2_shape": tuple(r2.shape),
                "rotary_mode": rotary_mode,
            }
        )
        x1, x2 = torch.chunk(input, 2, dim=-1)
        return input * r1 + torch.cat((-x2, x1), dim=-1) * r2

    torch_npu_stub = types.SimpleNamespace(npu_rotary_mul=npu_rotary_mul)
    module = _load_attention_module(
        monkeypatch,
        is_ascend=True,
        use_ascend_rope=True,
        torch_npu_stub=torch_npu_stub,
    )
    assert module.USE_ASCEND_ROPE is True

    attn = module.SelfAttention(dim=64, num_heads=4)
    q = torch.randn(2, 4, 10, 16, dtype=torch.float32)
    k = torch.randn(2, 4, 10, 16, dtype=torch.float32)
    sin = torch.randn(6, 16, dtype=torch.float16)
    cos = torch.randn(6, 16, dtype=torch.float16)

    q_out, k_out = attn.apply_rope(q, k, (sin, cos))
    q_expected, k_expected = _manual_apply(module, q, k, sin, cos)

    assert calls == []

    torch.testing.assert_close(q_out, q_expected)
    torch.testing.assert_close(k_out, k_expected)


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu/NPU is not available",
)
def test_dinov3_attention_import_does_not_lock_visible_devices():
    script = """
import os
import torch
import ultralytics.nn.modules.third_party.dinov3.dinov3.layers.attention

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
x = torch.ones(1, device="npu:0")
print(x.cpu().item())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "1.0" in result.stdout


def test_dinov3_rope_ascend_disabled_by_env_uses_manual(monkeypatch):
    calls = []

    def npu_rotary_mul(*, input, r1, r2, rotary_mode="half"):
        calls.append(1)
        return input

    torch_npu_stub = types.SimpleNamespace(npu_rotary_mul=npu_rotary_mul)
    module = _load_attention_module(
        monkeypatch,
        is_ascend=True,
        use_ascend_rope=False,
        torch_npu_stub=torch_npu_stub,
    )
    assert module.USE_ASCEND_ROPE is False

    attn = module.SelfAttention(dim=64, num_heads=4)
    q = torch.randn(2, 4, 10, 16, dtype=torch.float32)
    k = torch.randn(2, 4, 10, 16, dtype=torch.float32)
    sin = torch.randn(6, 16, dtype=torch.float16)
    cos = torch.randn(6, 16, dtype=torch.float16)

    q_out, k_out = attn.apply_rope(q, k, (sin, cos))
    q_expected, k_expected = _manual_apply(module, q, k, sin, cos)

    assert len(calls) == 0
    torch.testing.assert_close(q_out, q_expected)
    torch.testing.assert_close(k_out, k_expected)


def test_rope_on_ascend_coeff_2d_to_4d_expansion(monkeypatch):
    def npu_rotary_mul(*, input, r1, r2, rotary_mode="half"):
        assert r1.shape == (1, 1, 5, 8)
        assert r2.shape == (1, 1, 5, 8)
        x1, x2 = torch.chunk(input, 2, dim=-1)
        return input * r1 + torch.cat((-x2, x1), dim=-1) * r2

    torch_npu_stub = types.SimpleNamespace(npu_rotary_mul=npu_rotary_mul)
    module = _load_attention_module(
        monkeypatch,
        is_ascend=True,
        use_ascend_rope=True,
        torch_npu_stub=torch_npu_stub,
    )

    x = torch.randn(2, 4, 5, 8, dtype=torch.float16)
    sin = torch.randn(5, 8, dtype=torch.float16)
    cos = torch.randn(5, 8, dtype=torch.float16)
    y = module.RoPEOnAscend.apply(x, sin, cos)

    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_prepare_rope_bsnd_adds_identity_prefix(monkeypatch):
    module = _load_attention_module(monkeypatch, is_ascend=False, use_ascend_rope=True)
    q = torch.randn(2, 10, 4, 16)
    sin = torch.randn(6, 16)
    cos = torch.randn(6, 16)

    sin_bsnd, cos_bsnd = module._prepare_rope_bsnd(q, (sin, cos))

    assert sin_bsnd.shape == (1, 10, 1, 16)
    assert cos_bsnd.shape == (1, 10, 1, 16)
    torch.testing.assert_close(sin_bsnd[:, :4], torch.zeros_like(sin_bsnd[:, :4]))
    torch.testing.assert_close(cos_bsnd[:, :4], torch.ones_like(cos_bsnd[:, :4]))
    torch.testing.assert_close(sin_bsnd[0, 4:, 0], sin)
    torch.testing.assert_close(cos_bsnd[0, 4:, 0], cos)


def test_trainable_bsnd_stub_uses_two_fused_calls_and_backward(monkeypatch):
    calls = []

    def npu_rotary_mul(*, input, r1, r2, rotary_mode="half"):
        calls.append((tuple(input.shape), tuple(r1.shape), tuple(r2.shape), rotary_mode))
        return module.rope_apply(input, r2, r1)

    torch_npu_stub = types.SimpleNamespace(npu_rotary_mul=npu_rotary_mul)
    module = _load_attention_module(
        monkeypatch,
        is_ascend=True,
        use_ascend_rope=True,
        torch_npu_stub=torch_npu_stub,
    )
    q = torch.randn(2, 9, 4, 16, requires_grad=True)
    k = torch.randn(2, 9, 4, 16, requires_grad=True)
    sin = torch.randn(1, 9, 1, 16)
    cos = torch.randn(1, 9, 1, 16)

    q_out, k_out = module.SelfAttention._apply_rope_trainable_bsnd(q, k, sin, cos)
    (q_out.square().mean() + k_out.square().mean()).backward()

    assert calls == [
        ((2, 9, 4, 16), (1, 9, 1, 16), (1, 9, 1, 16), "half"),
        ((2, 9, 4, 16), (1, 9, 1, 16), (1, 9, 1, 16), "half"),
    ]
    assert q.grad is not None
    assert k.grad is not None


def test_inference_bsnd_stub_fuses_q_and_k(monkeypatch):
    calls = []

    def npu_apply_rotary_pos_emb(q, k, cos, sin, *, layout="BSND", rotary_mode="half"):
        calls.append((tuple(q.shape), tuple(cos.shape), layout, rotary_mode))
        q.copy_(module.rope_apply(q, sin, cos))
        k.copy_(module.rope_apply(k, sin, cos))
        return q, k

    torch_npu_stub = types.SimpleNamespace(npu_apply_rotary_pos_emb=npu_apply_rotary_pos_emb)
    module = _load_attention_module(
        monkeypatch,
        is_ascend=True,
        use_ascend_rope=True,
        torch_npu_stub=torch_npu_stub,
    )
    q = torch.randn(2, 9, 4, 64)
    k = torch.randn(2, 9, 4, 64)
    sin = torch.randn(1, 9, 1, 64)
    cos = torch.randn(1, 9, 1, 64)
    q_expected = module.rope_apply(q, sin, cos)
    k_expected = module.rope_apply(k, sin, cos)

    q_out, k_out = module.SelfAttention._apply_rope_inference_bsnd(q, k, sin, cos)

    assert calls == [((2, 9, 4, 64), (2, 9, 1, 64), "BSND", "half")]
    torch.testing.assert_close(q_out, q_expected)
    torch.testing.assert_close(k_out, k_expected)


def test_rotary_mul_jit_constraint_rejects_d64(monkeypatch):
    module = _load_attention_module(monkeypatch, is_ascend=True, use_ascend_rope=True)
    monkeypatch.setattr(module, "_ascend_jit_compile_enabled", lambda: True)
    q64 = torch.randn(2, 9, 4, 64)
    sin64 = torch.randn(1, 9, 1, 64)
    cos64 = torch.randn(1, 9, 1, 64)
    q128 = torch.randn(2, 9, 4, 128)
    sin128 = torch.randn(1, 9, 1, 128)
    cos128 = torch.randn(1, 9, 1, 128)

    assert module._supports_npu_rotary_mul(q64, sin64, cos64) is False
    assert module._supports_npu_rotary_mul(q128, sin128, cos128) is True


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu/NPU is not available",
)
def test_dinov3_rope_real_npu_auto_routes_and_backward(monkeypatch):
    from ultralytics.nn.modules.third_party.dinov3.dinov3.layers import attention as module

    import torch_npu

    monkeypatch.setattr(module, "DINOV3_ROPE_BACKEND", "auto")
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    q = torch.randn(2, 13, 4, 64, device=device, requires_grad=True)
    k = torch.randn(2, 13, 4, 64, device=device, requires_grad=True)
    angles = torch.randn(8, 64, device=device)
    sin, cos = angles.sin(), angles.cos()
    sin_bsnd, cos_bsnd = module._prepare_rope_bsnd(q, (sin, cos))
    attn = module.SelfAttention(dim=256, num_heads=4).to(device)

    assert attn.select_rope_backend(q, k, sin_bsnd, cos_bsnd) == "trainable"
    q_out, k_out = attn.apply_rope_bsnd(q, k, (sin, cos))
    q_ref, k_ref = attn._apply_rope_manual_bsnd(q, k, sin_bsnd, cos_bsnd)
    torch.testing.assert_close(q_out, q_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k_out, k_ref, rtol=1e-5, atol=1e-5)

    q_grad = torch.randn_like(q_out)
    k_grad = torch.randn_like(k_out)
    q_expected = q.detach().clone().requires_grad_(True)
    k_expected = k.detach().clone().requires_grad_(True)
    q_ref, k_ref = attn._apply_rope_manual_bsnd(q_expected, k_expected, sin_bsnd, cos_bsnd)
    (q_ref * q_grad).sum().add((k_ref * k_grad).sum()).backward()
    (q_out * q_grad).sum().add((k_out * k_grad).sum()).backward()
    torch.testing.assert_close(q.grad, q_expected.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(k.grad, k_expected.grad, rtol=1e-5, atol=1e-5)

    with torch.inference_mode():
        q_infer = q.detach()
        k_infer = k.detach()
        assert attn.select_rope_backend(q_infer, k_infer, sin_bsnd, cos_bsnd) == "inference"
        q_out, k_out = attn.apply_rope_bsnd(q_infer, k_infer, (sin, cos))
        q_ref, k_ref = attn._apply_rope_manual_bsnd(q_infer, k_infer, sin_bsnd, cos_bsnd)
        torch.testing.assert_close(q_out, q_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(k_out, k_ref, rtol=1e-5, atol=1e-5)

    assert hasattr(torch_npu, "npu_apply_rotary_pos_emb")


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu/NPU is not available",
)
def test_dinov3_model_real_npu_inference_and_training_match_manual(monkeypatch):
    from ultralytics.nn.modules.third_party.dinov3.dinov3.layers import attention as attention_module
    from ultralytics.nn.modules.third_party.dinov3.dinov3.models.vision_transformer import DinoVisionTransformer

    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device("npu:0")
    torch.manual_seed(7)
    model = DinoVisionTransformer(
        img_size=32,
        patch_size=16,
        embed_dim=64,
        depth=2,
        num_heads=1,
        ffn_ratio=1.0,
        n_storage_tokens=2,
        pos_embed_rope_dtype="fp32",
    ).to(device)
    model.init_weights()
    x = torch.randn(1, 3, 32, 32, device=device)

    model.eval()
    with torch.inference_mode():
        monkeypatch.setattr(attention_module, "DINOV3_ROPE_BACKEND", "manual")
        expected = model.forward_features(x)["x_norm_patchtokens"].clone()
        monkeypatch.setattr(attention_module, "DINOV3_ROPE_BACKEND", "auto")
        actual = model.forward_features(x)["x_norm_patchtokens"]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    model.train()
    monkeypatch.setattr(attention_module, "DINOV3_ROPE_BACKEND", "manual")
    model.zero_grad(set_to_none=True)
    manual_loss = model.forward_features(x)["x_norm_patchtokens"].square().mean()
    manual_loss.backward()
    manual_grad = model.blocks[0].attn.qkv.weight.grad.clone()

    monkeypatch.setattr(attention_module, "DINOV3_ROPE_BACKEND", "auto")
    model.zero_grad(set_to_none=True)
    fused_loss = model.forward_features(x)["x_norm_patchtokens"].square().mean()
    fused_loss.backward()
    fused_grad = model.blocks[0].attn.qkv.weight.grad

    torch.testing.assert_close(fused_loss, manual_loss, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(fused_grad, manual_grad, rtol=1e-4, atol=1e-5)
