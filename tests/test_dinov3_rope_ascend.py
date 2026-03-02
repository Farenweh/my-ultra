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
