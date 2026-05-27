from __future__ import annotations

import importlib.util

import pytest
import torch

from ultralytics.nn.modules.transformer import DeformableTransformerEncoder, MSDeformAttn
from ultralytics.nn.tasks import RTDETRDetectionModel


def test_deformable_encoder_component_forward_and_backward():
    torch.manual_seed(0)
    hidden_dim = 16
    encoder = DeformableTransformerEncoder(
        [8, 16, 32],
        hidden_dim=hidden_dim,
        num_layers=2,
        n_heads=4,
        d_ffn=32,
        dropout=0.0,
        n_points=2,
    )
    inputs = [
        torch.randn(2, 8, 4, 4, requires_grad=True),
        torch.randn(2, 16, 2, 3, requires_grad=True),
        torch.randn(2, 32, 1, 2, requires_grad=True),
    ]

    outputs = encoder(inputs)

    assert isinstance(outputs, list)
    assert [tuple(output.shape) for output in outputs] == [
        (2, hidden_dim, 4, 4),
        (2, hidden_dim, 2, 3),
        (2, hidden_dim, 1, 2),
    ]
    assert all(torch.isfinite(output).all() for output in outputs)

    loss = sum(output.sum() for output in outputs)
    loss.backward()

    assert all(inp.grad is not None and torch.isfinite(inp.grad).all() for inp in inputs)
    parameter_grads = [p.grad for p in encoder.parameters() if p.requires_grad]
    assert any(grad is not None and torch.isfinite(grad).all() for grad in parameter_grads)


def test_deformable_encoder_defaults_to_six_layers():
    encoder = DeformableTransformerEncoder([8])

    assert encoder.hidden_dim == 256
    assert len(encoder.layers) == 6


def test_ms_deform_attn_accepts_decoder_and_encoder_reference_shapes():
    torch.manual_seed(0)
    batch_size = 2
    num_queries = 7
    hidden_dim = 16
    num_levels = 3
    attn = MSDeformAttn(d_model=hidden_dim, n_levels=num_levels, n_heads=4, n_points=2)
    query = torch.randn(batch_size, num_queries, hidden_dim)
    value_shapes = [[3, 3], [2, 2], [1, 1]]
    value = torch.randn(batch_size, sum(h * w for h, w in value_shapes), hidden_dim)

    decoder_ref_2d = torch.rand(batch_size, num_queries, 1, 2)
    decoder_ref_4d = torch.rand(batch_size, num_queries, 1, 4)
    encoder_ref_2d = torch.rand(batch_size, num_queries, num_levels, 2)

    assert attn(query, decoder_ref_2d, value, value_shapes).shape == (batch_size, num_queries, hidden_dim)
    assert attn(query, decoder_ref_4d, value, value_shapes).shape == (batch_size, num_queries, hidden_dim)
    assert attn(query, encoder_ref_2d, value, value_shapes).shape == (batch_size, num_queries, hidden_dim)

    with pytest.raises(ValueError, match="Reference points"):
        attn(query, torch.rand(batch_size, num_queries, 2, 2), value, value_shapes)


def test_deformable_encoder_to_rtdetr_decoder_yaml_smoke():
    cfg = {
        "nc": 3,
        "backbone": [
            [-1, 1, "Conv", [8, 3, 2]],
            [-1, 1, "Conv", [16, 3, 2]],
        ],
        "head": [
            [[0, 1], 1, "DeformableTransformerEncoder", [16, 1, 4, 32, 0.0, 2]],
            [-1, 1, "RTDETRDecoder", ["nc", 16, 5, 2, 4, 1, 32]],
        ],
    }
    model = RTDETRDetectionModel(cfg, ch=3, nc=3, verbose=False, summary=False, imgsz=32).eval()

    with torch.no_grad():
        output = model(torch.randn(1, 3, 32, 32))

    assert isinstance(model.model[-2], DeformableTransformerEncoder)
    assert isinstance(output, tuple)
    assert output[0].shape == (1, 5, 6)


def test_deformable_encoder_empty_yaml_args_use_default_depth():
    cfg = {
        "nc": 3,
        "backbone": [[-1, 1, "Conv", [8, 3, 2]]],
        "head": [
            [[0], 1, "DeformableTransformerEncoder", []],
            [-1, 1, "RTDETRDecoder", ["nc", 256, 5, 4, 8, 1, 1024]],
        ],
    }
    model = RTDETRDetectionModel(cfg, ch=3, nc=3, verbose=False, summary=False, imgsz=32).eval()

    encoder = model.model[-2]

    assert isinstance(encoder, DeformableTransformerEncoder)
    assert encoder.hidden_dim == 256
    assert len(encoder.layers) == 6


@pytest.mark.skipif(
    not hasattr(torch, "npu")
    or not torch.npu.is_available()
    or importlib.util.find_spec("mmcv") is None,
    reason="NPU and mmcv are required for the MSDA fast-path smoke test.",
)
def test_deformable_encoder_npu_smoke():
    torch.manual_seed(0)
    encoder = (
        DeformableTransformerEncoder(
            [8, 16],
            hidden_dim=128,
            num_layers=1,
            n_heads=4,
            d_ffn=128,
            dropout=0.0,
            n_points=4,
        )
        .eval()
        .to("npu")
    )
    inputs = [
        torch.randn(1, 8, 4, 4, device="npu"),
        torch.randn(1, 16, 4, 4, device="npu"),
    ]

    with torch.no_grad():
        outputs = encoder(inputs)

    assert [tuple(output.shape) for output in outputs] == [(1, 128, 4, 4), (1, 128, 4, 4)]
    assert all(output.device.type == "npu" and torch.isfinite(output).all() for output in outputs)
