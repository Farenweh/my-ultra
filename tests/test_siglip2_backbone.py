from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from ultralytics.nn.modules.backbone import SigLIP2So400M
from ultralytics.nn.tasks import DetectionModel


class _FakeSigLIP2Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.last_input = None
        self.calls = []

    def forward_intermediates(self, x, **kwargs):
        self.last_input = x.detach().clone()
        self.calls.append(kwargs)
        features = F.avg_pool2d(x, kernel_size=16, stride=16).mean(1, keepdim=True) * self.scale
        features = features.expand(-1, SigLIP2So400M.dims(), -1, -1)
        if kwargs["output_fmt"] == "NLC":
            features = features.flatten(2).transpose(1, 2)
        return [features]


@pytest.fixture
def fake_timm(monkeypatch: pytest.MonkeyPatch):
    import timm

    calls = []

    def create_model(name, **kwargs):
        calls.append((name, kwargs))
        return _FakeSigLIP2Model()

    monkeypatch.setattr(timm, "create_model", create_model)
    return calls


def test_timm_is_a_required_dependency():
    root = Path(__file__).resolve().parents[1]

    assert '"timm>=1.0.20"' in (root / "pyproject.toml").read_text()


def test_siglip2_weight_sources(fake_timm, tmp_path: Path):
    SigLIP2So400M(pretrained=True)
    SigLIP2So400M(pretrained=False)
    checkpoint = tmp_path / "model.safetensors"
    save_file({"scale": torch.tensor(2.0)}, checkpoint)
    local_model = SigLIP2So400M(pretrained=str(checkpoint))

    assert fake_timm[0] == (
        "naflexvit_so400m_patch16_siglip.v2_webli",
        {"pretrained": True, "num_classes": 0},
    )
    assert fake_timm[1] == (
        "naflexvit_so400m_patch16_siglip",
        {"pretrained": False, "num_classes": 0},
    )
    assert fake_timm[2][0] == "naflexvit_so400m_patch16_siglip"
    assert fake_timm[2][1] == {"pretrained": False, "num_classes": 0}
    assert local_model.model.scale.item() == 2.0


def test_siglip2_loads_pytorch_checkpoint_through_timm(fake_timm, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import timm

    calls = []
    monkeypatch.setattr(timm.models, "load_checkpoint", lambda model, path: calls.append((model, path)))
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    model = SigLIP2So400M(pretrained=str(checkpoint))

    assert calls == [(model.model, str(checkpoint.resolve()))]


def test_siglip2_rejects_missing_local_weights_and_invalid_arguments(fake_timm, tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="本地权重"):
        SigLIP2So400M(pretrained=str(tmp_path / "missing.safetensors"))
    with pytest.raises(TypeError, match="pretrained"):
        SigLIP2So400M(pretrained=1)
    for invalid in (True, 0, -1, 2.5):
        with pytest.raises(ValueError, match="max_num_patches"):
            SigLIP2So400M(pretrained=False, max_num_patches=invalid)


def test_siglip2_normalizes_input_and_preserves_actual_grid(fake_timm):
    model = SigLIP2So400M(pretrained=False).eval()
    x = torch.full((1, 3, 384, 640), 0.25)

    with torch.no_grad():
        spatial = model(x)
        sequence = model.forward_sequence(x)

    assert spatial.shape == (1, 1152, 24, 40)
    assert sequence.shape == (1, 960, 1152)
    assert torch.all(model.model.last_input == -0.5)
    assert model.model.calls == [
        {"indices": 1, "norm": True, "output_fmt": "NCHW", "intermediates_only": True},
        {"indices": 1, "norm": True, "output_fmt": "NLC", "intermediates_only": True},
    ]


def test_siglip2_default_patch_limit_covers_800_without_padding(fake_timm):
    model = SigLIP2So400M(pretrained=False).eval()

    with torch.no_grad():
        output_800 = model(torch.zeros(1, 3, 800, 800))
        output_640 = model(torch.zeros(1, 3, 640, 640))

    assert output_800.shape[-2:] == (50, 50)
    assert output_640.shape[-2:] == (40, 40)

    with pytest.raises(ValueError, match="2550.*2500"):
        model(torch.zeros(1, 3, 816, 800))

    unlimited = SigLIP2So400M(pretrained=False, max_num_patches=None).eval()
    with torch.no_grad():
        assert unlimited(torch.zeros(1, 3, 816, 800)).shape[-2:] == (51, 50)


@pytest.mark.parametrize(
    ("input_tensor", "error", "message"),
    [
        (torch.zeros(3, 640, 640), AssertionError, "BCHW"),
        (torch.zeros(1, 1, 640, 640), AssertionError, "三通道"),
        (torch.zeros(1, 3, 638, 640), AssertionError, "16"),
        (torch.zeros(1, 3, 640, 640, dtype=torch.uint8), TypeError, "浮点"),
    ],
)
def test_siglip2_input_validation(fake_timm, input_tensor, error, message):
    model = SigLIP2So400M(pretrained=False)

    with pytest.raises(error, match=message):
        model(input_tensor)


def test_yolo11_siglip2_yaml_channels_and_strides(fake_timm):
    cfg = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/rf-det/siglip2-yolo11.yaml"
    model = DetectionModel(cfg, ch=3, nc=80, verbose=False, summary=False)

    assert isinstance(model.model[0], SigLIP2So400M)
    assert model.model[0].max_num_patches == 2500
    assert model.model[1].i == 1
    assert model.stride.tolist() == [8.0, 16.0, 32.0]


def test_frozen_siglip2_backbone_keeps_detection_head_trainable(fake_timm):
    cfg = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/rf-det/siglip2-yolo11.yaml"
    model = DetectionModel(cfg, ch=3, nc=2, verbose=False, summary=False).train()
    for parameter in model.model[0].parameters():
        parameter.requires_grad_(False)

    output = model(torch.rand(1, 3, 64, 64))
    loss = output["boxes"].float().sum() + output["scores"].float().sum()
    loss.backward()

    assert all(parameter.grad is None for parameter in model.model[0].parameters())
    assert any(parameter.grad is not None for parameter in model.model[2].parameters())
