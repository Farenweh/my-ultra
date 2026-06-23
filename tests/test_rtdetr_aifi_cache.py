from __future__ import annotations

import pytest
import torch

from ultralytics.nn.modules.transformer import AIFI
from ultralytics.utils.patches import torch_load


def test_aifi_reuses_fixed_shape_pos_embed_cache(monkeypatch):
    module = AIFI(16, cm=32, num_heads=4).eval()
    calls = 0
    original = AIFI.build_2d_sincos_position_embedding

    def counted(w: int, h: int, embed_dim: int = 256, temperature: float = 10000.0, device=None):
        nonlocal calls
        calls += 1
        return original(w, h, embed_dim, temperature, device)

    monkeypatch.setattr(AIFI, "build_2d_sincos_position_embedding", staticmethod(counted))

    x = torch.randn(2, 16, 4, 4)
    with torch.no_grad():
        module(x)
        first_cached = module._cached_pos_embed
        module(x)

    assert calls == 1
    assert module._cached_pos_embed is first_cached

    with torch.no_grad():
        module(torch.randn(2, 16, 2, 2))

    assert calls == 2


def test_aifi_trace_does_not_populate_eager_cache():
    module = AIFI(16, cm=32, num_heads=4).eval()
    x = torch.randn(2, 16, 4, 4)

    traced = torch.jit.trace(module, x, check_trace=False)

    assert module._cached_pos_embed is None
    torch.testing.assert_close(traced(x), module(x))


def test_aifi_migrates_legacy_pos_embed_cache():
    module = AIFI(16, cm=32, num_heads=4).eval()
    x = torch.randn(2, 16, 4, 4)
    stale_cache = AIFI.build_2d_sincos_position_embedding(4, 4, 16).half()
    module._cached_pos_embed_key = (4, 4, 16, "cpu", None, torch.float32)
    module._cached_pos_embed = stale_cache

    with torch.no_grad():
        module(x)

    assert module._cached_pos_embed_key == (4, 4, 16)
    assert module._cached_pos_embed is not stale_cache
    assert module._cached_pos_embed.device == x.device
    assert module._cached_pos_embed.dtype == x.dtype


@pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="NPU is not available",
)
def test_aifi_checkpoint_cache_follows_npu_input(tmp_path):
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    module = AIFI(16, cm=32, num_heads=4).eval().to(device).half()
    x = torch.randn(2, 16, 4, 4, device=device, dtype=torch.float16)

    with torch.no_grad():
        module(x)

    checkpoint = tmp_path / "aifi.pt"
    torch.save(module, checkpoint)
    restored = torch_load(checkpoint, map_location="cpu").to(device).half()
    assert restored._cached_pos_embed.device.type == "cpu"

    with torch.no_grad():
        restored(x)
        restored_cache = restored._cached_pos_embed
        restored(x)

    assert restored_cache.device == x.device
    assert restored_cache.dtype == x.dtype
    assert restored._cached_pos_embed is restored_cache
