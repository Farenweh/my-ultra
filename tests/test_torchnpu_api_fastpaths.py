from __future__ import annotations

import importlib.util
import types

import pytest
import torch
import torch.nn.functional as F

from ultralytics.models.sam.modules import utils as sam_utils
from ultralytics.models.sam.modules.blocks import RoPEAttention
from ultralytics.nn.modules import block as block_module
from ultralytics.utils import attention as attention_utils
from ultralytics.utils import npu as npu_utils


NPU_AVAILABLE = importlib.util.find_spec("torch_npu") is not None and hasattr(torch, "npu") and torch.npu.is_available()


def _bsnd_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        scale=scale,
    ).transpose(1, 2)


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    first, second = pairs.unbind(-1)
    return torch.stack((-second, first), dim=-1).flatten(-2)


def test_attention_training_stub_routes_to_fusion_v3(monkeypatch):
    calls = []

    def fusion(q, k, v, heads, layout, **kwargs):
        calls.append((heads, layout, kwargs))
        out = _bsnd_sdpa(q, k, v, kwargs["scale"])
        return (out, *[torch.empty(0) for _ in range(5)])

    monkeypatch.setattr(attention_utils, "select_npu_attention_backend", lambda *args, **kwargs: "training")
    monkeypatch.setattr(attention_utils, "_npu_format_cast_to_nd_if_needed", lambda x: x)
    monkeypatch.setattr(
        attention_utils,
        "_get_torch_npu",
        lambda: types.SimpleNamespace(npu_fusion_attention_v3=fusion),
    )
    q = torch.randn(2, 7, 4, 16, requires_grad=True)
    k = torch.randn(2, 7, 4, 16, requires_grad=True)
    v = torch.randn(2, 7, 4, 16, requires_grad=True)
    scale = 0.25

    actual = attention_utils.sdpa_with_npu_fusion(
        q,
        k,
        v,
        num_heads=4,
        input_layout="BSND",
        scale=scale,
    )
    expected = _bsnd_sdpa(q, k, v, scale)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()

    assert len(calls) == 1
    assert calls[0][0:2] == (4, "BSND")
    assert calls[0][2]["keep_prob"] == 1.0
    assert all(x.grad is not None for x in (q, k, v))


def test_attention_inference_stub_makes_packed_views_contiguous(monkeypatch):
    calls = []

    def inference(q, k, v, **kwargs):
        calls.append((q.is_contiguous(), k.is_contiguous(), v.is_contiguous(), kwargs))
        return _bsnd_sdpa(q, k, v, kwargs["softmax_scale"]), torch.empty(0)

    monkeypatch.setattr(attention_utils, "select_npu_attention_backend", lambda *args, **kwargs: "inference")
    monkeypatch.setattr(attention_utils, "_npu_format_cast_to_nd_if_needed", lambda x: x)
    monkeypatch.setattr(
        attention_utils,
        "_get_torch_npu",
        lambda: types.SimpleNamespace(npu_fused_infer_attention_score_v2=inference),
    )
    packed = torch.randn(2, 7, 3, 4, 16)
    q, k, v = packed.unbind(2)

    actual = attention_utils.sdpa_with_npu_fusion(
        q,
        k,
        v,
        num_heads=4,
        input_layout="BSND",
        scale=0.25,
    )
    expected = _bsnd_sdpa(q, k, v, 0.25)

    torch.testing.assert_close(actual, expected)
    assert calls[0][:3] == (True, True, True)
    assert calls[0][3]["num_query_heads"] == 4
    assert calls[0][3]["input_layout"] == "BSND"

    fallback = attention_utils.sdpa_with_npu_fusion(
        q,
        k,
        v,
        num_heads=4,
        input_layout="BSND",
        scale=0.25,
        allow_inference_fusion=False,
    )
    torch.testing.assert_close(fallback, expected)
    assert len(calls) == 1


def test_attention_cpu_and_unsupported_options_fall_back_to_sdpa():
    q = torch.randn(2, 4, 7, 16)
    k = torch.randn(2, 4, 7, 16)
    v = torch.randn(2, 4, 7, 16)

    actual = attention_utils.sdpa_with_npu_fusion(
        q,
        k,
        v,
        num_heads=4,
        input_layout="BNSD",
        dropout_p=0.0,
        is_causal=True,
        scale=0.25,
    )
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.25)

    torch.testing.assert_close(actual, expected)


def test_sam_rotary_stub_matches_complex_reference_with_repeated_keys(monkeypatch):
    calls = []

    def rotary_mul(*, input, r1, r2, rotary_mode):
        calls.append((tuple(input.shape), tuple(r1.shape), rotary_mode))
        return input * r1 + _rotate_interleaved(input) * r2

    monkeypatch.setattr(sam_utils, "_supports_npu_rotary_mul", lambda *args: True)
    monkeypatch.setattr(
        sam_utils,
        "_get_torch_npu",
        lambda: types.SimpleNamespace(npu_rotary_mul=rotary_mul),
    )
    q = torch.randn(2, 4, 5, 16, dtype=torch.float16, requires_grad=True)
    k = torch.randn(2, 4, 10, 16, dtype=torch.float16, requires_grad=True)
    angles = torch.randn(5, 8)
    freqs = torch.polar(torch.ones_like(angles), angles)

    actual_q, actual_k = sam_utils.apply_rotary_enc(q, k, freqs, repeat_freqs_k=True)
    expected_q, expected_k = sam_utils._apply_rotary_enc_manual(q, k, freqs, repeat_freqs_k=True)

    torch.testing.assert_close(actual_q, expected_q, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual_k, expected_k, rtol=2e-3, atol=2e-3)
    (actual_q.float().sum() + actual_k.float().sum()).backward()
    assert calls == [((2, 4, 5, 16), (1, 1, 5, 16), "interleave"), ((2, 4, 10, 16), (1, 1, 10, 16), "interleave")]
    assert q.grad is not None and k.grad is not None


def test_scatter_cpu_fallback_preserves_update_gradient():
    target = torch.zeros(3, 4, 2)
    indices = torch.tensor([[0, 1], [2, 3]])
    updates = torch.randn(2, 2, requires_grad=True)

    actual = npu_utils.scatter_nd_update_(target, indices, updates)
    actual.sum().backward()

    torch.testing.assert_close(actual[0, 1], updates[0])
    torch.testing.assert_close(actual[2, 3], updates[1])
    torch.testing.assert_close(updates.grad, torch.ones_like(updates))


def test_scatter_stub_routes_to_npu_api(monkeypatch):
    calls = []

    def scatter(target, indices, updates):
        calls.append(1)
        target[tuple(indices.unbind(-1))] = updates
        return target

    monkeypatch.setattr(npu_utils, "_supports_npu_scatter_nd_update", lambda *args: True)
    monkeypatch.setattr(
        npu_utils,
        "_get_torch_npu",
        lambda: types.SimpleNamespace(npu_scatter_nd_update_=scatter),
    )
    target = torch.zeros(3, 4)
    indices = torch.tensor([[0, 1], [2, 3]])
    updates = torch.tensor([2.0, 5.0])

    npu_utils.scatter_nd_update_(target, indices, updates)

    assert calls == [1]
    assert target[0, 1] == 2 and target[2, 3] == 5


def test_swiglu_stub_and_cpu_fallback_match_manual(monkeypatch):
    x = torch.randn(2, 5, 16, requires_grad=True)
    first, second = x.chunk(2, dim=-1)
    expected = F.silu(first) * second
    torch.testing.assert_close(npu_utils.swiglu_with_npu_fallback(x), expected)

    calls = []

    def swiglu(input, *, dim):
        calls.append(1)
        lhs, rhs = input.chunk(2, dim=dim)
        return F.silu(lhs) * rhs

    monkeypatch.setattr(npu_utils, "_supports_npu_swiglu", lambda *args: True)
    monkeypatch.setattr(
        npu_utils,
        "_get_torch_npu",
        lambda: types.SimpleNamespace(npu_swiglu=swiglu),
    )
    actual = npu_utils.swiglu_with_npu_fallback(x)

    assert calls == [1]
    torch.testing.assert_close(actual, expected)


def test_rms_norm_stub_and_cpu_fallback_match_manual(monkeypatch):
    x = torch.randn(2, 5, 16)
    weight = torch.randn(16)
    eps = 1e-5
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight
    torch.testing.assert_close(npu_utils.rms_norm_with_npu_fallback(x, weight, eps), expected)

    calls = []

    def rms_norm(input, gamma, *, epsilon):
        calls.append(1)
        out = input * torch.rsqrt(input.square().mean(-1, keepdim=True) + epsilon) * gamma
        return out, torch.empty(0)

    monkeypatch.setattr(npu_utils, "_supports_npu_rms_norm", lambda *args: True)
    monkeypatch.setattr(
        npu_utils,
        "_get_torch_npu",
        lambda: types.SimpleNamespace(npu_rms_norm=rms_norm),
    )
    actual = npu_utils.rms_norm_with_npu_fallback(x, weight, eps)

    assert calls == [1]
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="需要可用的Ascend NPU")
def test_attention_real_npu_training_and_inference_match_sdpa():
    torch.npu.set_device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    scale = 64**-0.5

    q = torch.randn(1, 33, 4, 64, device="npu:0", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 33, 4, 64, device="npu:0", dtype=torch.float16, requires_grad=True)
    v = torch.randn(1, 33, 4, 64, device="npu:0", dtype=torch.float16, requires_grad=True)
    q_ref, k_ref, v_ref = (x.detach().clone().requires_grad_(True) for x in (q, k, v))
    grad = torch.randn_like(q)

    assert (
        attention_utils.select_npu_attention_backend(
            q,
            k,
            v,
            num_heads=4,
            input_layout="BSND",
        )
        == "training"
    )
    actual = attention_utils.sdpa_with_npu_fusion(q, k, v, num_heads=4, input_layout="BSND", scale=scale)
    expected = _bsnd_sdpa(q_ref, k_ref, v_ref, scale)
    (actual * grad).sum().backward()
    (expected * grad).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    for actual_grad, expected_grad in ((q.grad, q_ref.grad), (k.grad, k_ref.grad), (v.grad, v_ref.grad)):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-2, atol=3e-2)

    with torch.no_grad():
        q_infer, k_infer, v_infer = (x.detach() for x in (q, k, v))
        assert (
            attention_utils.select_npu_attention_backend(
                q_infer,
                k_infer,
                v_infer,
                num_heads=4,
                input_layout="BSND",
            )
            == "inference"
        )
        actual = attention_utils.sdpa_with_npu_fusion(
            q_infer,
            k_infer,
            v_infer,
            num_heads=4,
            input_layout="BSND",
            scale=scale,
        )
        expected = _bsnd_sdpa(q_infer, k_infer, v_infer, scale)
        torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)

        packed = torch.randn(1, 33, 3, 4, 64, device="npu:0", dtype=torch.float16)
        packed_q, packed_k, packed_v = packed.unbind(2)
        assert (
            attention_utils.select_npu_attention_backend(
                packed_q,
                packed_k,
                packed_v,
                num_heads=4,
                input_layout="BSND",
            )
            == "sdpa"
        )
        assert (
            attention_utils.select_npu_attention_backend(
                packed_q.contiguous(),
                packed_k.contiguous(),
                packed_v,
                num_heads=4,
                input_layout="BSND",
            )
            == "inference"
        )


@pytest.mark.skipif(not NPU_AVAILABLE, reason="需要可用的Ascend NPU")
def test_sam_rotary_real_npu_forward_backward_matches_manual():
    torch.npu.set_device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    q = torch.randn(1, 4, 17, 64, device="npu:0", dtype=torch.float16, requires_grad=True)
    k = torch.randn(1, 4, 17, 64, device="npu:0", dtype=torch.float16, requires_grad=True)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    angles = torch.randn(17, 32)
    freqs = torch.polar(torch.ones_like(angles), angles).to("npu:0")
    grad_q, grad_k = torch.randn_like(q), torch.randn_like(k)

    assert sam_utils._supports_npu_rotary_mul(q, k, freqs, False)
    actual_q, actual_k = sam_utils.apply_rotary_enc(q, k, freqs)
    expected_q, expected_k = sam_utils._apply_rotary_enc_manual(q_ref, k_ref, freqs)
    (actual_q * grad_q + actual_k * grad_k).sum().backward()
    (expected_q * grad_q + expected_k * grad_k).sum().backward()

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(q.grad, q_ref.grad, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=2e-2, atol=2e-2)
    with torch.no_grad():
        assert not sam_utils._supports_npu_rotary_mul(q.detach(), k.detach(), freqs, False)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="需要可用的Ascend NPU")
def test_sam_rope_attention_real_npu_matches_fallback(monkeypatch):
    torch.npu.set_device("npu:0")
    torch.npu.set_compile_mode(jit_compile=False)
    module = RoPEAttention(embedding_dim=64, num_heads=4, feat_sizes=(4, 4)).to("npu:0").half().train()
    x = torch.randn(1, 16, 64, device="npu:0", dtype=torch.float16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    original_attention_select = attention_utils.select_npu_attention_backend
    original_rope_support = sam_utils._supports_npu_rotary_mul

    monkeypatch.setattr(attention_utils, "select_npu_attention_backend", lambda *args, **kwargs: "sdpa")
    monkeypatch.setattr(sam_utils, "_supports_npu_rotary_mul", lambda *args: False)
    expected = module(x_ref, x_ref, x_ref)
    monkeypatch.setattr(attention_utils, "select_npu_attention_backend", original_attention_select)
    monkeypatch.setattr(sam_utils, "_supports_npu_rotary_mul", original_rope_support)
    actual = module(x, x, x)
    grad = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * grad).sum(), x)[0]
    expected_grad = torch.autograd.grad((expected * grad).sum(), x_ref)[0]

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="需要可用的Ascend NPU")
def test_core_swiglu_real_npu_forward_backward_matches_manual(monkeypatch):
    import torch_npu

    torch.npu.set_device("npu:0")
    module = block_module.SwiGLUFFN(32, 16, e=4).to("npu:0").half()
    x = torch.randn(2, 11, 32, device="npu:0", dtype=torch.float16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)

    original_support = npu_utils._supports_npu_swiglu
    monkeypatch.setattr(npu_utils, "_supports_npu_swiglu", lambda *args: False)
    expected = module(x_ref)
    calls = []

    def swiglu(input, *, dim):
        calls.append(1)
        return torch_npu.npu_swiglu(input, dim=dim)

    monkeypatch.setattr(npu_utils, "_supports_npu_swiglu", original_support)
    monkeypatch.setattr(npu_utils, "_get_torch_npu", lambda: types.SimpleNamespace(npu_swiglu=swiglu))
    actual = module(x)
    grad = torch.randn_like(actual)
    actual_grad = torch.autograd.grad((actual * grad).sum(), x)[0]
    expected_grad = torch.autograd.grad((expected * grad).sum(), x_ref)[0]

    assert calls == [1]
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="需要可用的Ascend NPU")
def test_scatter_and_rms_norm_real_npu():
    torch.npu.set_device("npu:0")

    target = torch.zeros(3, 4, 5, device="npu:0")
    indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device="npu:0")
    updates = torch.randn(2, 5, device="npu:0")
    assert npu_utils._supports_npu_scatter_nd_update(target, indices, updates)
    npu_utils.scatter_nd_update_(target, indices, updates)
    torch.testing.assert_close(target[tuple(indices.unbind(-1))], updates)

    x = torch.randn(2, 17, 64, device="npu:0", dtype=torch.float16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight = torch.randn(64, device="npu:0", dtype=torch.float16, requires_grad=True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    grad = torch.randn_like(x)
    assert npu_utils._supports_npu_rms_norm(x, weight)
    actual = npu_utils.rms_norm_with_npu_fallback(x, weight, 1e-5)
    x_float = x_ref.float()
    expected = (x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + 1e-5)).to(x_ref.dtype) * weight_ref
    (actual * grad).sum().backward()
    (expected * grad).sum().backward()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(x.grad, x_ref.grad, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=5e-2, atol=5e-2)
