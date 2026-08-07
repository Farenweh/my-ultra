from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types

import pytest
import torch


NPU_AVAILABLE = importlib.util.find_spec("torch_npu") is not None and hasattr(torch, "npu") and torch.npu.is_available()


def _load_rope_module(monkeypatch, *, is_ascend: bool, backend: str, torch_npu_stub=None):
    import ultralytics.utils.checks as checks

    monkeypatch.setattr(checks, "IS_ASCEND", is_ascend, raising=True)
    monkeypatch.setenv("PE_SPATIAL_ROPE_BACKEND", backend)
    if torch_npu_stub is not None:
        monkeypatch.setitem(sys.modules, "torch_npu", torch_npu_stub)
    sys.modules.pop("ultralytics.nn.modules.third_party.pe_spatial.rope", None)
    module = importlib.import_module("ultralytics.nn.modules.third_party.pe_spatial.rope")
    return importlib.reload(module)


def _coefficients(sequence: int, head_dim: int, dtype=torch.float32):
    angles = torch.randn(1, sequence, 1, head_dim, dtype=dtype)
    return angles.sin(), angles.cos()


def test_pe_spatial_rope_non_ascend_uses_manual(monkeypatch):
    module = _load_rope_module(monkeypatch, is_ascend=False, backend="auto")
    x = torch.randn(2, 7, 4, 64)
    sin, cos = _coefficients(7, 64)

    assert module.select_rope_backend(x, sin, cos) == "manual"
    torch.testing.assert_close(module.apply_rope(x, sin, cos), module.apply_rope_manual(x, sin, cos))


def test_pe_spatial_rope_rejects_invalid_policy(monkeypatch):
    with pytest.raises(ValueError, match="PE_SPATIAL_ROPE_BACKEND"):
        _load_rope_module(monkeypatch, is_ascend=True, backend="invalid")


def test_pe_spatial_rope_stub_calls_interleave_rotary_mul(monkeypatch):
    calls = []

    def npu_rotary_mul(*, input, r1, r2, rotary_mode):
        calls.append((tuple(input.shape), tuple(r1.shape), tuple(r2.shape), rotary_mode))
        return input * r1 + module.rotate_interleaved(input) * r2

    module = _load_rope_module(
        monkeypatch,
        is_ascend=True,
        backend="auto",
        torch_npu_stub=types.SimpleNamespace(npu_rotary_mul=npu_rotary_mul),
    )
    monkeypatch.setattr(module, "_supports_npu_rotary_mul", lambda *args: True)
    x = torch.randn(2, 7, 4, 64, requires_grad=True)
    sin, cos = _coefficients(7, 64)

    actual = module.apply_rope(x, sin, cos)
    expected = module.apply_rope_manual(x, sin, cos)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()

    assert calls == [((2, 7, 4, 64), (1, 7, 1, 64), (1, 7, 1, 64), "interleave")]
    assert x.grad is not None


def test_pe_spatial_rope_strict_mode_reports_unsupported_tensor(monkeypatch):
    module = _load_rope_module(monkeypatch, is_ascend=True, backend="rotary_mul")
    monkeypatch.setattr(module, "_supports_npu_rotary_mul", lambda *args: False)
    x = torch.randn(2, 7, 4, 64)
    sin, cos = _coefficients(7, 64)

    with pytest.raises(RuntimeError, match="rotary_mul不支持"):
        module.apply_rope(x, sin, cos)


@pytest.mark.parametrize(
    ("shape", "supported"),
    (
        ((2, 7, 4, 64), True),
        ((2, 7, 4, 96), True),
        ((100, 7, 10, 64), False),
        ((2, 7, 4, 63), False),
    ),
)
def test_pe_spatial_rotary_mul_shape_constraints(monkeypatch, shape, supported):
    module = _load_rope_module(monkeypatch, is_ascend=True, backend="auto")
    monkeypatch.setattr(module, "IS_ASCEND", True)
    monkeypatch.setattr(module, "_ascend_jit_compile_enabled", lambda: False)
    monkeypatch.setattr(module, "_npu_rotary_device_supported", lambda index: True)
    monkeypatch.setattr(module, "_get_torch_npu", lambda: types.SimpleNamespace(npu_rotary_mul=lambda *a, **k: None))
    device = types.SimpleNamespace(type="npu", index=0)
    x = types.SimpleNamespace(device=device, ndim=4, dtype=torch.float32, shape=shape)
    coefficient_shape = (1, shape[1], 1, shape[3])
    sin = types.SimpleNamespace(device=device, dtype=torch.float32, shape=coefficient_shape)
    cos = types.SimpleNamespace(device=device, dtype=torch.float32, shape=coefficient_shape)

    assert module._supports_npu_rotary_mul(x, sin, cos) is supported


def test_pe_spatial_rotary_mul_jit_mode_falls_back(monkeypatch):
    module = _load_rope_module(monkeypatch, is_ascend=True, backend="auto")
    monkeypatch.setattr(module, "IS_ASCEND", True)
    monkeypatch.setattr(module, "_ascend_jit_compile_enabled", lambda: True)
    monkeypatch.setattr(module, "_npu_rotary_device_supported", lambda index: True)
    monkeypatch.setattr(module, "_get_torch_npu", lambda: types.SimpleNamespace(npu_rotary_mul=lambda *a, **k: None))
    device = types.SimpleNamespace(type="npu", index=0)
    x = types.SimpleNamespace(device=device, ndim=4, dtype=torch.float32, shape=(2, 7, 4, 64))
    sin = types.SimpleNamespace(device=device, dtype=torch.float32, shape=(1, 7, 1, 64))
    cos = types.SimpleNamespace(device=device, dtype=torch.float32, shape=(1, 7, 1, 64))

    assert not module._supports_npu_rotary_mul(x, sin, cos)


def test_pe_spatial_rope_import_does_not_lock_visible_devices():
    if not NPU_AVAILABLE:
        pytest.skip("需要可用的Ascend NPU")
    script = """
import os
import torch
import ultralytics.nn.modules.third_party.pe_spatial.rope

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
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
@pytest.mark.parametrize("head_dim", (64, 96))
def test_pe_spatial_rope_real_npu_forward_backward(dtype, head_dim, monkeypatch):
    from ultralytics.nn.modules.third_party.pe_spatial import rope as module

    torch.npu.set_device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    monkeypatch.setattr(module, "PE_SPATIAL_ROPE_BACKEND", "auto")
    q = torch.randn(2, 17, 4, head_dim, device="npu:0", dtype=dtype, requires_grad=True)
    q_reference = q.detach().clone().requires_grad_(True)
    angles = torch.randn(1, 17, 1, head_dim, device="npu:0", dtype=torch.float32)
    sin, cos = angles.sin(), angles.cos()
    grad = torch.randn_like(q)

    assert module.select_rope_backend(q, sin, cos) == "rotary_mul"
    actual = module.apply_rope(q, sin, cos)
    expected = module.apply_rope_manual(q_reference, sin, cos)
    (actual * grad).sum().backward()
    (expected * grad).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(q.grad, q_reference.grad, rtol=1e-5, atol=1e-5)
