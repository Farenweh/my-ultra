from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules.backbone import DINOv3ViT, resolve_dinov3_weights
from ultralytics.nn.modules.third_party.dinov3.dinov3.layers.layer_scale import LayerScale


def test_dinov3_layer_scale_preserves_fp32_promotion():
    """LayerScale 默认使用非原地乘法，使 FP16 激活与 FP32 gamma 的结果提升为 FP32。"""
    layer = LayerScale(4)
    with torch.no_grad():
        layer.gamma.fill_(2.0)
    inputs = torch.ones(2, 4, dtype=torch.float16)

    outputs = layer(inputs)

    assert layer.inplace is False
    assert outputs.dtype == torch.float32
    assert torch.equal(outputs, torch.full_like(outputs, 2.0))
    assert torch.equal(inputs, torch.ones_like(inputs))


def test_resolve_dinov3_weights_uses_cwd_weights_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    resolved = resolve_dinov3_weights("./weights", "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")

    assert Path(resolved) == (tmp_path / "weights" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth").resolve()


def test_resolve_dinov3_weights_preserves_explicit_urls():
    resolved = resolve_dinov3_weights(
        "file:///tmp/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )

    assert resolved == "file:///tmp/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"


def test_dinov3_boolean_pretrained_uses_official_url(monkeypatch: pytest.MonkeyPatch):
    expected_state = DINOv3ViT("s", pretrained=False).model.state_dict()
    urls = []

    def fake_load_state_dict_from_url(url, *args, **kwargs):
        urls.append(url)
        return expected_state

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_load_state_dict_from_url)

    DINOv3ViT("s", pretrained=True)

    assert urls == ["https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"]
