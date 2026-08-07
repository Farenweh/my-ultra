from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import backbone as backbone_module
from ultralytics.nn.modules.backbone import PESpatial
from ultralytics.nn.modules.third_party.pe_spatial import PE_SPATIAL_CONFIGS, PESpatialConfig, VisionTransformer
from ultralytics.nn.modules.third_party.pe_spatial.rope import Rope2D, apply_rope_manual
from ultralytics.nn.tasks import DetectionModel, parse_model


EXPECTED_VARIANTS = {
    "t": ("PE-Spatial-T16-512", 192, 16, 512, 3),
    "s": ("PE-Spatial-S16-512", 384, 16, 512, 6),
    "b": ("PE-Spatial-B16-512", 768, 16, 512, 12),
    "l": ("PE-Spatial-L14-448", 1024, 14, 448, 16),
    "g": ("PE-Spatial-G14-448", 1536, 14, 448, 16),
}


class _FakePEModel(nn.Module):
    def __init__(self, config: PESpatialConfig):
        super().__init__()
        self.config = config
        self.scale = nn.Parameter(torch.ones(()))
        self.last_input = None

    def forward_features(self, x, *, norm=False, strip_cls_token=False, **kwargs):
        self.last_input = x.detach().clone()
        features = F.avg_pool2d(x, self.config.patch_size, self.config.patch_size).mean(1, keepdim=True)
        return features.flatten(2).transpose(1, 2).expand(-1, -1, self.config.width) * self.scale


@pytest.fixture
def fake_pe_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(backbone_module, "PESpatialVisionTransformer", _FakePEModel)


def test_pe_spatial_all_official_variants():
    assert set(PE_SPATIAL_CONFIGS) == set(EXPECTED_VARIANTS)
    for scale, (checkpoint, width, patch_size, image_size, heads) in EXPECTED_VARIANTS.items():
        config = PE_SPATIAL_CONFIGS[scale]
        assert (config.checkpoint, config.width, config.patch_size, config.image_size, config.heads) == (
            checkpoint,
            width,
            patch_size,
            image_size,
            heads,
        )
        assert PESpatial.dims(scale.upper()) == width
        assert PESpatial.stride(scale.upper()) == patch_size


def test_pe_spatial_all_variant_structures_build_on_meta_device():
    for scale, config in PE_SPATIAL_CONFIGS.items():
        with torch.device("meta"):
            model = VisionTransformer(config)
        assert model.conv1.weight.shape == (config.width, 3, config.patch_size, config.patch_size)
        assert len(model.transformer.resblocks) == config.layers
        block = model.transformer.resblocks[0]
        assert block.attn.in_proj_weight.shape == (config.width * 3, config.width)
        assert block.mlp.c_fc.out_features == int(config.width * config.mlp_ratio)
        assert hasattr(model, "class_embedding") is config.use_cls_token
        assert hasattr(block.ls_1, "gamma") is (scale == "g")


@pytest.mark.parametrize("scale", tuple(EXPECTED_VARIANTS))
def test_pe_spatial_parser_tracks_all_variant_strides(fake_pe_model, scale):
    model, _ = parse_model(
        {
            "nc": 1,
            "backbone": [[-1, 1, "PESpatial", [scale, False]]],
            "head": [],
        },
        ch=3,
        verbose=False,
    )

    expected_width, expected_stride = EXPECTED_VARIANTS[scale][1:3]
    assert model[0].dims(scale) == expected_width
    assert model.stride.tolist() == [float(expected_stride)]


def test_pe_spatial_rejects_invalid_scale_and_pretrained(fake_pe_model, tmp_path: Path):
    with pytest.raises(ValueError, match="架构预设"):
        PESpatial("x", pretrained=False)
    with pytest.raises(TypeError, match="scale"):
        PESpatial(1, pretrained=False)
    with pytest.raises(TypeError, match="pretrained"):
        PESpatial("t", pretrained=1)
    with pytest.raises(FileNotFoundError, match="本地权重"):
        PESpatial("t", pretrained=str(tmp_path / "missing.pt"))


@pytest.mark.parametrize("container", (None, "state_dict", "weights"))
@pytest.mark.parametrize("prefix", ("", "module.", "visual.", "module.visual."))
def test_pe_spatial_local_checkpoint_formats(fake_pe_model, tmp_path: Path, container: str | None, prefix: str):
    checkpoint = tmp_path / f"{container}-{prefix.replace('.', '-')}.pt"
    state = {f"{prefix}scale": torch.tensor(3.0)}
    torch.save({container: state} if container else state, checkpoint)

    model = PESpatial("t", pretrained=str(checkpoint))

    assert model.model.scale.item() == 3.0


def test_pe_spatial_remote_checkpoint_uses_official_huggingface_name(
    fake_pe_model, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    checkpoint = tmp_path / "official.pt"
    torch.save({"scale": torch.tensor(2.0)}, checkpoint)
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        return checkpoint

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    model = PESpatial("l", pretrained=True)

    assert calls == [{"repo_id": "facebook/PE-Spatial-L14-448", "filename": "PE-Spatial-L14-448.pt"}]
    assert model.model.scale.item() == 2.0


def test_pe_spatial_checkpoint_mismatch_is_strict(fake_pe_model, tmp_path: Path):
    checkpoint = tmp_path / "bad.pt"
    torch.save({"unexpected": torch.ones(1)}, checkpoint)

    with pytest.raises(RuntimeError, match="missing_keys=.*scale.*unexpected_keys=.*unexpected"):
        PESpatial("t", pretrained=str(checkpoint))


def test_pe_spatial_normalizes_and_preserves_rectangular_grid(fake_pe_model):
    model = PESpatial("t", pretrained=False).eval()
    x = torch.full((1, 3, 384, 640), 0.25)

    with torch.no_grad():
        sequence = model.forward_sequence(x)
        spatial = model(x)

    assert sequence.shape == (1, 960, 192)
    assert spatial.shape == (1, 192, 24, 40)
    assert torch.all(model.model.last_input == -0.5)
    torch.testing.assert_close(sequence, spatial.flatten(2).transpose(1, 2))


@pytest.mark.parametrize(
    ("input_tensor", "error", "message"),
    (
        (torch.zeros(3, 32, 32), AssertionError, "BCHW"),
        (torch.zeros(1, 1, 32, 32), AssertionError, "三通道"),
        (torch.zeros(1, 3, 31, 32), AssertionError, "16"),
        (torch.zeros(1, 3, 32, 32, dtype=torch.uint8), TypeError, "浮点"),
    ),
)
def test_pe_spatial_input_validation(fake_pe_model, input_tensor, error, message):
    model = PESpatial("t", pretrained=False)
    with pytest.raises(error, match=message):
        model(input_tensor)


def test_pe_spatial_rope_matches_official_interleave_formula_and_replaces_cache():
    rope = Rope2D(head_dim=64, use_cls_token=True)
    sin_1, cos_1 = rope.coefficients(2, 3, torch.device("cpu"))
    sin_2, cos_2 = rope.coefficients(2, 3, torch.device("cpu"))
    assert sin_1 is sin_2 and cos_1 is cos_2
    assert sin_1.shape == (1, 7, 1, 64)
    torch.testing.assert_close(sin_1[:, 0], torch.zeros_like(sin_1[:, 0]))
    torch.testing.assert_close(cos_1[:, 0], torch.ones_like(cos_1[:, 0]))

    x = torch.randn(2, 7, 4, 64)
    paired = x.float().reshape(2, 7, 4, 32, 2)
    first, second = paired.unbind(-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    expected = x.float() * cos_1 + rotated * sin_1
    torch.testing.assert_close(apply_rope_manual(x, sin_1, cos_1), expected)

    previous_frequency = rope._frequency_cache
    rope.coefficients(3, 3, torch.device("cpu"))
    assert rope._frequency_cache is not previous_frequency
    assert rope._cache_key[2:4] == (3, 3)
    assert not rope.state_dict()
    assert not dict(rope.named_buffers()).keys() - {"inv_freq"}


def test_pe_spatial_caches_are_instance_local_and_position_cache_is_safe():
    config = PESpatialConfig("test", 32, 16, 32, 2, 4)
    model = VisionTransformer(config).eval()
    model.positional_embedding.requires_grad_(False)
    x = torch.rand(1, 3, 32, 48)

    with torch.no_grad():
        first = model.forward_features(x, strip_cls_token=True)
        position_cache = model._pos_embed_cache
        sin_cache = model.rope._sin_cache
        second = model.forward_features(x, strip_cls_token=True)
    torch.testing.assert_close(first, second)
    assert model._pos_embed_cache is position_cache
    assert model.rope._sin_cache is sin_cache

    copied = deepcopy(model)
    assert copied._pos_embed_cache.numel() == 0
    assert copied.rope._sin_cache.numel() == 0
    assert copied._pos_embed_cache_key is None
    assert copied.rope._cache_key is None

    with torch.no_grad():
        model.forward_features(torch.rand(1, 3, 48, 48), strip_cls_token=True)
    assert model._pos_embed_cache is not position_cache

    position_cache = model._pos_embed_cache
    with torch.no_grad():
        model.positional_embedding.add_(0.01)
        model.forward_features(torch.rand(1, 3, 48, 48), strip_cls_token=True)
    assert model._pos_embed_cache is not position_cache
    assert model.rope._sin_cache is not sin_cache

    model.positional_embedding.requires_grad_(True)
    output = model.forward_features(torch.rand(1, 3, 32, 32), strip_cls_token=True)
    output.square().mean().backward()
    assert model.positional_embedding.grad is not None


def test_pe_spatial_inference_tensor_position_cache_and_reload_are_safe():
    config = PESpatialConfig("test", 32, 16, 32, 2, 4)
    with torch.inference_mode():
        model = VisionTransformer(config).eval()
        model.positional_embedding.requires_grad_(False)
        assert torch.is_inference(model.positional_embedding)

        x = torch.rand(1, 3, 32, 48)
        first = model.forward_features(x, strip_cls_token=True)
        position_cache = model._pos_embed_cache
        second = model.forward_features(x, strip_cls_token=True)
        torch.testing.assert_close(first, second)
        assert model._pos_embed_cache is position_cache

        model.forward_features(torch.rand(1, 3, 48, 48), strip_cls_token=True)
        assert model._pos_embed_cache is not position_cache

        model.load_state_dict(model.state_dict())
        assert model._pos_embed_cache.numel() == 0
        assert model._pos_embed_cache_key is None


def test_pe_spatial_yolo11_yaml_channels_and_strides(fake_pe_model, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    checkpoint = tmp_path / "l.pt"
    torch.save({"scale": torch.tensor(1.0)}, checkpoint)
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda **kwargs: checkpoint)
    cfg = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/rf-det/pe-spatial-yolo11.yaml"
    model = DetectionModel(cfg, ch=3, nc=80, verbose=False, summary=False)

    assert isinstance(model.model[0], PESpatial)
    assert model.model[0].scale == "l"
    assert model.model[0].feature_stride == 14
    assert model.model[1].i == 1
    assert model.stride.tolist() == [7.0, 14.0, 28.0]
