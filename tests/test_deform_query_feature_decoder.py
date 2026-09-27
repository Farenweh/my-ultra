"""DINOv3 查询式双尺度 RT-DETR 的离线回归测试。"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics.nn.modules.transformer import DeformableQueryFeatureDecoder
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import YAML


def test_query_feature_decoder_rectangular_forward_and_backward():
    torch.manual_seed(0)
    decoder = DeformableQueryFeatureDecoder([3, 12], 32, 2, 4, 64, 0.0, 2, 4)
    image = torch.randn(2, 3, 32, 48, requires_grad=True)
    memory = torch.randn(2, 12, 2, 3, requires_grad=True)

    output = decoder([image, memory])

    assert output.shape == (2, 32, 8, 12)
    assert torch.isfinite(output).all()
    reference = decoder._reference_points(8, 12, output)
    torch.testing.assert_close(reference[0, 0, 0], torch.tensor([0.5 / 12, 0.5 / 8]))
    torch.testing.assert_close(reference[0, -1, 0], torch.tensor([11.5 / 12, 7.5 / 8]))

    output[:, 0].square().mean().backward()
    assert image.grad is not None and image.grad.isfinite().all() and image.grad.abs().sum() > 0
    assert memory.grad is not None and memory.grad.isfinite().all() and memory.grad.abs().sum() > 0
    assert decoder.patch_embed.weight.grad is not None and decoder.patch_embed.weight.grad.abs().sum() > 0
    assert decoder.memory_proj[0].weight.grad is not None and decoder.memory_proj[0].weight.grad.abs().sum() > 0


def test_query_feature_decoder_defaults_and_misaligned_input():
    decoder = DeformableQueryFeatureDecoder([3, 16])
    assert decoder.hidden_dim == 256
    assert decoder.patch_size == 4
    assert len(decoder.layers) == 6

    small_decoder = DeformableQueryFeatureDecoder([3, 16], 32, 1, 4, 64)
    with pytest.raises(ValueError, match="16的倍数"):
        small_decoder([torch.randn(2, 3, 33, 48), torch.randn(2, 16, 2, 3)])


def test_query_geometry_inference_then_training_and_migration():
    """推理缓存能安全参与反向，精度和尺寸变化不会复用旧网格。"""
    decoder = DeformableQueryFeatureDecoder([3, 12], 32, 1, 4, 64, 0.0, 2, 4).eval()
    image, memory = torch.randn(2, 3, 32, 48), torch.randn(2, 12, 2, 3)
    with torch.inference_mode():
        decoder([image, memory])
    cache = decoder._query_geometry_cache
    assert not cache[1].is_inference() and not cache[2].is_inference()
    assert not decoder.layers[0].cross_attn._cached_offset_normalizer.is_inference()
    decoder.train()
    decoder([image, memory])[:, 0].square().mean().backward()
    assert decoder.patch_embed.weight.grad.isfinite().all()
    assert decoder.memory_proj[0].weight.grad.abs().sum() > 0
    assert decoder._query_geometry_cache is cache
    assert not any("cache" in key for key in decoder.state_dict())
    decoder.double()
    assert decoder._query_geometry_cache is None
    assert decoder.layers[0].self_attn._cached_offset_normalizer is None
    decoder([image.double(), memory.double()])
    assert decoder._query_geometry_cache[1].dtype == torch.float64
    decoder([torch.randn(2, 3, 48, 32).double(), torch.randn(2, 12, 3, 2).double()])
    assert decoder._query_geometry_cache[0][:2] == (12, 8)
    del decoder._query_geometry_cache  # 模拟旧完整模型 checkpoint 不含新字段。
    decoder([image.double(), memory.double()])




def test_qms_yaml_rtdetr_inference_and_training_without_pretrained_weights():
    config_path = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/rt-detr/rtdetr-dinov3-c3k2-qms.yaml"
    cfg = YAML.load(config_path)
    cfg["nc"] = 3
    cfg["backbone"][1][3] = ["s", False]
    cfg["head"][0][1] = 1
    cfg["head"][0][3] = [32, True]
    cfg["head"][1][3] = [32, 1, 4, 64, 0.0, 2, 4]
    cfg["head"][2][3] = ["nc", 32, 8, 2, 4, 1, 64]
    model = RTDETRDetectionModel(cfg, ch=3, nc=3, verbose=False, summary=False)
    model.nc = 3
    model.model[1].requires_grad_(False)
    model.model[-1].num_denoising = 4
    seen = []
    hooks = [
        model.model[3].register_forward_hook(lambda _, inputs, output: seen.append(("query", output.shape))),
        model.model[4].register_forward_hook(
            lambda _, inputs, output: seen.append(("head", [feat.shape for feat in inputs[0]]))
        ),
    ]
    image = torch.randn(1, 3, 32, 48)
    try:
        model.eval()
        with torch.no_grad():
            prediction = model(image)
        assert prediction[0].shape == (1, 8, 6)
        assert seen == [
            ("query", torch.Size((1, 32, 8, 12))),
            ("head", [torch.Size((1, 32, 8, 12)), torch.Size((1, 32, 2, 3))]),
        ]
        assert model.stride.tolist() == [16.0]

        model.train()
        batch = {
            "img": image,
            "batch_idx": torch.tensor([0.0]),
            "cls": torch.tensor([[1.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]),
        }
        loss, _ = model(batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert model.model[3].patch_embed.weight.grad is not None
        assert model.model[3].memory_proj[0].weight.grad is not None
        assert model.model[2].cv1.conv.weight.grad is not None
    finally:
        for hook in hooks:
            hook.remove()
