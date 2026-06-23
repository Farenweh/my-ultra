from __future__ import annotations

import torch
import pytest

from ultralytics.nn.modules import block as block_module
from ultralytics.nn.modules.block import AAttn, ImagePoolingAttn
from ultralytics.nn.modules.head import OBB
from ultralytics.nn.modules.third_party.dinov3.dinov3.layers.rms_norm import RMSNorm
from ultralytics.nn.modules.third_party.dinov3.dinov3.models import convnext as convnext_module
from ultralytics.utils.attention import sdpa_with_npu_padding


def test_image_pooling_attn_matmul_matches_einsum_reference():
    torch.manual_seed(0)
    module = ImagePoolingAttn(ec=16, ch=(8, 12), ct=32, nh=4, k=2, scale=True)
    x = [torch.randn(2, 8, 10, 10), torch.randn(2, 12, 5, 5)]
    text = torch.randn(2, 7, 32)

    actual = module(x, text)

    bs = x[0].shape[0]
    num_patches = module.k**2
    pooled = [
        pool(proj(feat)).view(bs, -1, num_patches) for feat, proj, pool in zip(x, module.projections, module.im_pools)
    ]
    pooled = torch.cat(pooled, dim=-1).transpose(1, 2)
    q = module.query(text).reshape(bs, -1, module.nh, module.hc)
    k = module.key(pooled).reshape(bs, -1, module.nh, module.hc)
    v = module.value(pooled).reshape(bs, -1, module.nh, module.hc)

    aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
    aw = (aw / (module.hc**0.5)).softmax(dim=-1)
    expected = torch.einsum("bmnk,bkmc->bnmc", aw, v)
    expected = module.proj(expected.reshape(bs, -1, module.ec))
    expected = expected * module.scale + text

    torch.testing.assert_close(actual, expected)


def test_dinov3_rmsnorm_matches_manual_reference():
    torch.manual_seed(1)
    module = RMSNorm(16, eps=1e-5)
    x = torch.randn(3, 5, 16, dtype=torch.float32)

    actual = module(x)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + module.eps) * module.weight

    torch.testing.assert_close(actual, expected)


def test_obb_postprocess_ne1_angle_idx_matches_repeat_reference():
    torch.manual_seed(4)
    head = OBB(nc=4, ne=1, ch=(8, 16, 32))
    head.max_det = 5
    preds = torch.randn(2, 11, 9)

    actual = head.postprocess(preds)

    boxes, scores, angle = preds.split([4, head.nc, head.ne], dim=-1)
    scores, conf, idx = head.get_topk_index(scores, head.max_det)
    expected_boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
    expected_angle = angle.gather(dim=1, index=idx.repeat(1, 1, head.ne))
    expected = torch.cat([expected_boxes, scores, conf, expected_angle], dim=-1)

    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is not available",
)
def test_sdpa_npu_padding_matches_manual_attention():
    torch.npu.set_device("npu:0")
    torch.manual_seed(5)
    q = torch.randn(2, 3, 5, 12, device="npu:0", dtype=torch.float16)
    k = torch.randn(2, 3, 7, 12, device="npu:0", dtype=torch.float16)
    v = torch.randn(2, 3, 7, 20, device="npu:0", dtype=torch.float16)
    scale = q.shape[-1] ** -0.5

    actual = sdpa_with_npu_padding(q, k, v, scale=scale)
    expected = ((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1) @ v

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is not available",
)
def test_sdpa_npu_padding_accepts_internal_format():
    import torch_npu

    torch.npu.set_device("npu:0")
    old_internal_format = getattr(torch.npu.config, "allow_internal_format", False)
    torch.npu.config.allow_internal_format = True
    try:
        module = block_module.Attention(dim=256, num_heads=4, attn_ratio=0.5).to("npu:0").half()
        x = torch.randn(2, 256, 40, 40, device="npu:0", dtype=torch.float16, requires_grad=True)
        qkv = module.qkv(x)
        assert torch_npu.get_npu_format(qkv) != torch_npu.Format.ND

        b, _, h, w = x.shape
        n = h * w
        q, k, v = qkv.view(b, module.num_heads, module.key_dim * 2 + module.head_dim, n).split(
            [module.key_dim, module.key_dim, module.head_dim], dim=2
        )
        actual = sdpa_with_npu_padding(
            q.transpose(-2, -1),
            k.transpose(-2, -1),
            v.transpose(-2, -1),
            scale=module.scale,
        )
        assert actual.shape == (b, module.num_heads, n, module.head_dim)
        y = module(x)
        assert y.shape == x.shape
        y.float().sum().backward()
        assert x.grad is not None
    finally:
        torch.npu.config.allow_internal_format = old_internal_format


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is not available",
)
def test_aattn_npu_sdpa_matches_manual_attention(monkeypatch):
    torch.npu.set_device("npu:0")
    torch.manual_seed(2)
    module = AAttn(dim=64, num_heads=2, area=1).to("npu:0").half().eval()
    x = torch.randn(2, 64, 8, 8, device="npu:0", dtype=torch.float16)

    monkeypatch.setattr(block_module, "IS_ASCEND", False)
    expected = module(x)
    monkeypatch.setattr(block_module, "IS_ASCEND", True)
    actual = module(x)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is not available",
)
def test_convnext_block_npu_contiguous_matches_original_branch(monkeypatch):
    torch.npu.set_device("npu:0")
    torch.manual_seed(3)
    module = convnext_module.Block(dim=32).to("npu:0").half().eval()
    module.norm.init_weights()
    x = torch.randn(2, 32, 8, 8, device="npu:0", dtype=torch.float16)

    monkeypatch.setattr(convnext_module, "IS_ASCEND", False)
    expected = module(x)
    monkeypatch.setattr(convnext_module, "IS_ASCEND", True)
    actual = module(x)

    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
