from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import save_file

from ultralytics.nn.modules import backbone as backbone_module
from ultralytics.nn.modules.backbone import CRADIOv4
from ultralytics.nn.modules.third_party.c_radio_v3.model import VisionTransformer
from ultralytics.nn.modules.third_party.c_radio_v3.cpe import CPEPatchGenerator
from ultralytics.nn.modules.third_party.c_radio_v4 import CRADIO_V4_CONFIGS, CRADIOConfig
from ultralytics.nn.tasks import DetectionModel


EXPECTED_VARIANTS = {
    "so400m": (
        "nvidia/C-RADIOv4-SO400M",
        "c0457f5dc26ca145f954cd4fc5bb6114e5705ad8",
        1152,
        27,
        16,
        4304,
        431_237_232,
    ),
    "h": (
        "nvidia/C-RADIOv4-H",
        "0057b339059c0b9e1b4ba996f975410ebbfdfcc8",
        1280,
        32,
        16,
        5120,
        651_645_440,
    ),
}


class _FakeCRADIOModel(nn.Module):
    def __init__(self, config: CRADIOConfig, *, initialize=True, device=None, dtype=None):
        super().__init__()
        del initialize
        self.config = config
        self.scale = nn.Parameter(torch.ones((), device=device, dtype=dtype))
        self.patch_generator = nn.Module()
        self.patch_generator.num_skip = config.prefix_tokens
        self.frozen_deterministic = False
        self.last_input = None

    def forward_features(self, x):
        self.last_input = x.detach().clone()
        feature = F.avg_pool2d(x, 16, 16).mean(1, keepdim=True).flatten(2).transpose(1, 2)
        feature = feature.expand(-1, -1, self.config.width) * self.scale
        prefix = feature.new_zeros(feature.shape[0], self.config.prefix_tokens, feature.shape[-1])
        return torch.cat((prefix, feature), dim=1)


@pytest.fixture
def fake_c_radio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backbone_module, "CRADIOv4VisionTransformer", _FakeCRADIOModel)


def _fake_checkpoint(path: Path, *, value: float = 2.0, safetensors: bool = False) -> Path:
    state = {
        "radio_model.model.scale": torch.tensor(value),
        "radio_model.input_conditioner.norm_mean": torch.tensor([0.1, 0.2, 0.3]).view(3, 1, 1),
        "radio_model.input_conditioner.norm_std": torch.tensor([0.4, 0.5, 0.6]).view(3, 1, 1),
        "radio_model.summary_idxs": torch.tensor([0, 1]),
    }
    if safetensors:
        save_file(state, path)
    else:
        torch.save({"weights": state}, path)
    return path


def test_c_radio_v4_official_variant_table_and_meta_parameter_counts():
    assert set(CRADIO_V4_CONFIGS) == set(EXPECTED_VARIANTS)
    for scale, expected in EXPECTED_VARIANTS.items():
        config = CRADIO_V4_CONFIGS[scale]
        actual = (
            config.repo_id,
            config.revision,
            config.width,
            config.depth,
            config.heads,
            config.effective_mlp_hidden_dim,
            config.parameter_count,
        )
        assert actual == expected
        assert config.family == "v4"
        assert config.prefix_tokens == 10
        assert config.layer_scale is None
        assert CRADIOv4.dims(scale.upper()) == config.width
        assert CRADIOv4.stride(scale.upper()) == 16

        model = VisionTransformer(config, initialize=False, device="meta")
        assert sum(parameter.numel() for parameter in model.parameters()) == config.parameter_count
        assert model.blocks[0].mlp.fc1.out_features == config.effective_mlp_hidden_dim
        assert model.patch_generator.cls_token.token.shape == (10, config.width)


def test_c_radio_v4_rejects_invalid_arguments(fake_c_radio, tmp_path: Path):
    with pytest.raises(ValueError, match="架构预设"):
        CRADIOv4("l", pretrained=False)
    with pytest.raises(TypeError, match="scale"):
        CRADIOv4(1, pretrained=False)
    with pytest.raises(TypeError, match="pretrained"):
        CRADIOv4("so400m", pretrained=1)
    with pytest.raises(FileNotFoundError, match="本地权重"):
        CRADIOv4("so400m", pretrained=str(tmp_path / "missing.safetensors"))


@pytest.mark.parametrize("suffix", (".safetensors", ".pt"))
@pytest.mark.parametrize("scale", ("so400m", "h"))
def test_c_radio_v4_safe_local_weight_formats(fake_c_radio, tmp_path: Path, suffix: str, scale: str):
    checkpoint = _fake_checkpoint(tmp_path / f"{scale}{suffix}", safetensors=suffix == ".safetensors")
    model = CRADIOv4(scale, pretrained=str(checkpoint))

    assert model.model.scale.item() == 2.0
    torch.testing.assert_close(model.norm_mean, torch.tensor([0.1, 0.2, 0.3]).view(3, 1, 1))
    torch.testing.assert_close(model.norm_std, torch.tensor([0.4, 0.5, 0.6]).view(3, 1, 1))


def test_c_radio_v4_remote_weights_are_revision_pinned(fake_c_radio, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    checkpoint = _fake_checkpoint(tmp_path / "model.safetensors", safetensors=True)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return checkpoint

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    CRADIOv4("so400m", pretrained=True)

    config = CRADIO_V4_CONFIGS["so400m"]
    assert calls == [{"repo_id": config.repo_id, "filename": "model.safetensors", "revision": config.revision}]


def test_c_radio_v4_checkpoint_mismatch_is_strict(fake_c_radio, tmp_path: Path):
    checkpoint = tmp_path / "bad.safetensors"
    save_file({"unexpected": torch.ones(1)}, checkpoint)
    with pytest.raises(RuntimeError, match="missing_keys=.*model.scale.*unexpected_keys=.*unexpected"):
        CRADIOv4("so400m", pretrained=str(checkpoint))


@pytest.mark.parametrize(
    ("height", "width", "tokens"),
    ((512, 512, 1024), (640, 640, 1600), (800, 800, 2500), (384, 640, 960)),
)
def test_c_radio_v4_normalizes_and_preserves_actual_grid(fake_c_radio, height: int, width: int, tokens: int):
    model = CRADIOv4("so400m", pretrained=False).eval()
    x = torch.full((1, 3, height, width), 0.5)
    sequence = model.forward_sequence(x)
    spatial = model(x)

    assert sequence.shape == (1, tokens, 1152)
    assert spatial.shape == (1, 1152, height // 16, width // 16)
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
def test_c_radio_v4_input_validation(fake_c_radio, input_tensor, error, message):
    with pytest.raises(error, match=message):
        CRADIOv4("so400m", pretrained=False)(input_tensor)


def test_c_radio_v4_freeze_enables_deterministic_cpe(fake_c_radio):
    model = CRADIOv4("so400m", pretrained=False).train()
    assert not model.model.frozen_deterministic
    model.requires_grad_(False)
    model.train()
    assert model.model.frozen_deterministic
    model.eval()
    assert not model.model.frozen_deterministic


def test_c_radio_v4_cpe_inference_cache_dynamic_shape_copy_and_reload_are_safe():
    with torch.inference_mode():
        cpe = CPEPatchGenerator(72, patch_size=16, max_resolution=64, prefix_tokens=10)
        cpe.reset_parameters()
        cpe.pos_embed.requires_grad_(False)
        assert torch.is_inference(cpe.pos_embed)

        x = torch.rand(1, 3, 32, 48)
        first = cpe(x, stochastic=False)
        position_cache = cpe._position_cache
        second = cpe(x, stochastic=False)
        assert first.shape == (1, 16, 72)
        torch.testing.assert_close(first, second)
        assert cpe._position_cache is position_cache

        cpe(torch.rand(1, 3, 48, 48), stochastic=False)
        assert cpe._position_cache is not position_cache

        copied = deepcopy(cpe)
        assert copied._base_grid_cache.numel() == 0
        assert copied._position_cache.numel() == 0

        cpe.load_state_dict(cpe.state_dict())
        assert cpe._base_grid_cache.numel() == 0
        assert cpe._position_cache.numel() == 0
        assert cpe._base_grid_cache_key is None
        assert cpe._position_cache_key is None


def test_c_radio_v4_yolo11_so400m_channels_and_strides(fake_c_radio, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    checkpoint = _fake_checkpoint(tmp_path / "model.safetensors", safetensors=True)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: checkpoint)
    cfg = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/rf-det/c-radio-v4-yolo11.yaml"
    model = DetectionModel(cfg, ch=3, nc=80, verbose=False, summary=False)

    assert isinstance(model.model[0], CRADIOv4)
    assert model.model[0].scale == "so400m"
    assert model.model[0].feature_stride == 16
    assert model.stride.tolist() == [8.0, 16.0, 32.0]
