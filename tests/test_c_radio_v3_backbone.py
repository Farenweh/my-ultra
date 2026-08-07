from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from ultralytics.nn.modules import backbone as backbone_module
from ultralytics.nn.modules.backbone import CRADIOv3
from ultralytics.nn.modules.third_party.c_radio_v3 import CRADIO_V3_CONFIGS, CRADIOv3Config
from ultralytics.nn.modules.third_party.c_radio_v3.cpe import CPEPatchGenerator
from ultralytics.nn.modules.third_party.c_radio_v3.model import VisionTransformer
from ultralytics.nn.tasks import DetectionModel


EXPECTED_VARIANTS = {
    "b": ("nvidia/C-RADIOv3-B", 768, 12, 12, 98_254_854, 1e-5, 4),
    "l": ("nvidia/C-RADIOv3-L", 1024, 24, 16, 319_934_470, 1e-5, 4),
    "h": ("nvidia/C-RADIOv3-H", 1280, 32, 16, 651_642_886, None, 0),
    "g": ("nvidia/C-RADIOv3-g", 1536, 40, 24, 1_159_618_566, None, 0),
}


class _FakeCRADIOModel(nn.Module):
    def __init__(self, config: CRADIOv3Config, *, initialize=True, device=None, dtype=None):
        super().__init__()
        del initialize
        self.config = config
        self.scale = nn.Parameter(torch.ones((), device=device, dtype=dtype))
        self.patch_generator = nn.Module()
        self.patch_generator.num_skip = 8
        self.frozen_deterministic = False
        self.last_input = None

    def forward_features(self, x):
        self.last_input = x.detach().clone()
        feature = F.avg_pool2d(x, 16, 16).mean(1, keepdim=True).flatten(2).transpose(1, 2)
        feature = feature.expand(-1, -1, self.config.width) * self.scale
        prefix = feature.new_zeros(feature.shape[0], 8, feature.shape[-1])
        return torch.cat((prefix, feature), dim=1)


@pytest.fixture
def fake_c_radio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backbone_module, "CRADIOv3VisionTransformer", _FakeCRADIOModel)


def _fake_checkpoint(path: Path, *, value: float = 2.0, safetensors: bool = False) -> Path:
    state = {
        "radio_model.model.scale": torch.tensor(value),
        "radio_model.input_conditioner.norm_mean": torch.tensor([0.1, 0.2, 0.3]).view(3, 1, 1),
        "radio_model.input_conditioner.norm_std": torch.tensor([0.4, 0.5, 0.6]).view(3, 1, 1),
        "radio_model.summary_idxs": torch.tensor([0, 1, 2]),
    }
    if safetensors:
        save_file(state, path)
    else:
        torch.save({"state_dict": state}, path)
    return path


def test_c_radio_v3_official_variant_table():
    assert set(CRADIO_V3_CONFIGS) == set(EXPECTED_VARIANTS)
    for scale, expected in EXPECTED_VARIANTS.items():
        config = CRADIO_V3_CONFIGS[scale]
        actual = (
            config.repo_id,
            config.width,
            config.depth,
            config.heads,
            config.parameter_count,
            config.layer_scale,
            config.base_registers,
        )
        assert actual == expected
        assert CRADIOv3.dims(scale.upper()) == config.width
        assert CRADIOv3.stride(scale.upper()) == 16


def test_c_radio_v3_uses_official_layer_norm_epsilon():
    config = CRADIOv3Config("test", "test", 64, 1, 1, 0, layer_scale=1e-5, max_resolution=64)
    model = VisionTransformer(config, initialize=False)
    assert model.blocks[0].norm1.eps == 1e-6
    assert model.blocks[0].norm2.eps == 1e-6


def test_c_radio_v3_rejects_invalid_arguments(fake_c_radio, tmp_path: Path):
    with pytest.raises(ValueError, match="架构预设"):
        CRADIOv3("x", pretrained=False)
    with pytest.raises(TypeError, match="scale"):
        CRADIOv3(1, pretrained=False)
    with pytest.raises(TypeError, match="pretrained"):
        CRADIOv3("l", pretrained=1)
    with pytest.raises(FileNotFoundError, match="本地权重"):
        CRADIOv3("l", pretrained=str(tmp_path / "missing.safetensors"))


@pytest.mark.parametrize("suffix", (".safetensors", ".pt"))
def test_c_radio_v3_safe_local_weight_formats(fake_c_radio, tmp_path: Path, suffix: str):
    checkpoint = _fake_checkpoint(tmp_path / f"local{suffix}", safetensors=suffix == ".safetensors")
    model = CRADIOv3("l", pretrained=str(checkpoint))

    assert model.model.scale.item() == 2.0
    torch.testing.assert_close(model.norm_mean, torch.tensor([0.1, 0.2, 0.3]).view(3, 1, 1))
    torch.testing.assert_close(model.norm_std, torch.tensor([0.4, 0.5, 0.6]).view(3, 1, 1))


def test_c_radio_v3_remote_weights_are_revision_pinned(fake_c_radio, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    checkpoint = _fake_checkpoint(tmp_path / "model.safetensors", safetensors=True)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return checkpoint

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    CRADIOv3("l", pretrained=True)

    config = CRADIO_V3_CONFIGS["l"]
    assert calls == [{"repo_id": config.repo_id, "filename": "model.safetensors", "revision": config.revision}]


def test_c_radio_v3_checkpoint_mismatch_is_strict(fake_c_radio, tmp_path: Path):
    checkpoint = tmp_path / "bad.safetensors"
    save_file({"unexpected": torch.ones(1)}, checkpoint)
    with pytest.raises(RuntimeError, match="missing_keys=.*model.scale.*unexpected_keys=.*unexpected"):
        CRADIOv3("l", pretrained=str(checkpoint))


def test_c_radio_v3_normalizes_and_preserves_rectangular_grid(fake_c_radio):
    model = CRADIOv3("l", pretrained=False).eval()
    x = torch.full((1, 3, 384, 640), 0.5)
    sequence = model.forward_sequence(x)
    spatial = model(x)

    assert sequence.shape == (1, 960, 1024)
    assert spatial.shape == (1, 1024, 24, 40)
    expected = (x - model.norm_mean) / model.norm_std
    torch.testing.assert_close(model.model.last_input, expected)
    torch.testing.assert_close(sequence, spatial.flatten(2).transpose(1, 2))


@pytest.mark.parametrize(
    ("input_tensor", "error", "message"),
    (
        (torch.zeros(3, 32, 32), AssertionError, "BCHW"),
        (torch.zeros(1, 1, 32, 32), AssertionError, "三通道"),
        (torch.zeros(1, 3, 31, 32), AssertionError, "16"),
        (torch.zeros(1, 3, 2064, 32), ValueError, "2048"),
        (torch.zeros(1, 3, 32, 32, dtype=torch.uint8), TypeError, "浮点"),
    ),
)
def test_c_radio_v3_input_validation(fake_c_radio, input_tensor, error, message):
    model = CRADIOv3("l", pretrained=False)
    with pytest.raises(error, match=message):
        model(input_tensor)


def test_c_radio_v3_freeze_enables_deterministic_cpe(fake_c_radio):
    model = CRADIOv3("l", pretrained=False).train()
    assert not model.model.frozen_deterministic
    model.requires_grad_(False)
    model.train()
    assert model.model.frozen_deterministic
    model.eval()
    assert not model.model.frozen_deterministic


def test_cpe_stochastic_grid_and_deterministic_cache_are_safe():
    cpe = CPEPatchGenerator(64, patch_size=16, max_resolution=64, prefix_tokens=8)
    cpe.reset_parameters()
    x = torch.rand(2, 3, 32, 48)

    torch.manual_seed(0)
    stochastic_1 = cpe(x, stochastic=True)
    base_cache = cpe._base_grid_cache
    torch.manual_seed(1)
    stochastic_2 = cpe(x, stochastic=True)
    assert cpe._base_grid_cache is base_cache
    assert not torch.equal(stochastic_1, stochastic_2)

    cpe.pos_embed.requires_grad_(False)
    deterministic_1 = cpe(x, stochastic=False)
    position_cache = cpe._position_cache
    deterministic_2 = cpe(x, stochastic=False)
    torch.testing.assert_close(deterministic_1, deterministic_2)
    assert cpe._position_cache is position_cache
    assert not dict(cpe.named_buffers())

    copied = deepcopy(cpe)
    assert copied._base_grid_cache.numel() == 0
    assert copied._position_cache.numel() == 0
    cpe(torch.rand(1, 3, 48, 48), stochastic=False)
    assert cpe._position_cache is not position_cache

    position_cache = cpe._position_cache
    with torch.no_grad():
        cpe.pos_embed.add_(0.01)
        cpe(torch.rand(1, 3, 48, 48), stochastic=False)
    assert cpe._position_cache is not position_cache


def test_cpe_inference_tensor_position_cache_and_reload_are_safe():
    with torch.inference_mode():
        cpe = CPEPatchGenerator(64, patch_size=16, max_resolution=64, prefix_tokens=8)
        cpe.reset_parameters()
        cpe.pos_embed.requires_grad_(False)
        assert torch.is_inference(cpe.pos_embed)

        x = torch.rand(1, 3, 32, 48)
        first = cpe(x, stochastic=False)
        position_cache = cpe._position_cache
        second = cpe(x, stochastic=False)
        torch.testing.assert_close(first, second)
        assert cpe._position_cache is position_cache

        cpe(torch.rand(1, 3, 48, 48), stochastic=False)
        assert cpe._position_cache is not position_cache

        cpe.load_state_dict(cpe.state_dict())
        assert cpe._base_grid_cache.numel() == 0
        assert cpe._position_cache.numel() == 0
        assert cpe._base_grid_cache_key is None
        assert cpe._position_cache_key is None


def test_c_radio_v3_yolo11_l_channels_and_strides(fake_c_radio, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    checkpoint = _fake_checkpoint(tmp_path / "model.safetensors", safetensors=True)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: checkpoint)
    cfg = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/rf-det/c-radio-v3-yolo11.yaml"
    model = DetectionModel(cfg, ch=3, nc=80, verbose=False, summary=False)

    assert isinstance(model.model[0], CRADIOv3)
    assert model.model[0].scale == "l"
    assert model.model[0].feature_stride == 16
    assert model.stride.tolist() == [8.0, 16.0, 32.0]
