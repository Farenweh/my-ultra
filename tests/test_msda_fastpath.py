from __future__ import annotations

import types

import pytest
import torch

from ultralytics.nn.modules import utils as module_utils


def _fake_value(dtype=torch.float16, device_type="npu"):
    return types.SimpleNamespace(dtype=dtype, device=types.SimpleNamespace(type=device_type))


def _fake_tensor(shape, dtype=torch.float16, device_type="npu"):
    return types.SimpleNamespace(shape=shape, dtype=dtype, device=types.SimpleNamespace(type=device_type))


@pytest.mark.parametrize(
    ("embed_dims", "expected"),
    [
        (31, "超出 [32, 256]"),
        (32, None),
        (33, "不能被 8 整除"),
        (40, None),
        (255, "不能被 8 整除"),
        (256, None),
        (257, "超出 [32, 256]"),
    ],
)
def test_msda_fastpath_embed_dims_boundaries(monkeypatch, embed_dims, expected):
    monkeypatch.setattr(module_utils, "IS_ASCEND", True)

    reason = module_utils._get_msda_fastpath_unavailable_reason(_fake_value(), embed_dims, 32, 4)

    if expected is None:
        assert reason is None
    else:
        assert expected in reason


@pytest.mark.parametrize(
    ("num_points", "expected"),
    [
        (3, "超出 [4, 8]"),
        (4, None),
        (5, None),
        (8, None),
        (9, "超出 [4, 8]"),
    ],
)
def test_msda_fastpath_num_points_boundaries(monkeypatch, num_points, expected):
    monkeypatch.setattr(module_utils, "IS_ASCEND", True)

    reason = module_utils._get_msda_fastpath_unavailable_reason(_fake_value(), 32, 32, num_points)

    if expected is None:
        assert reason is None
    else:
        assert expected in reason


def test_msda_fastpath_dtype_and_query_boundaries(monkeypatch):
    monkeypatch.setattr(module_utils, "IS_ASCEND", True)

    assert "dtype=torch.bfloat16" in module_utils._get_msda_fastpath_unavailable_reason(
        _fake_value(dtype=torch.bfloat16), 32, 32, 4
    )
    assert "num_queries=31" in module_utils._get_msda_fastpath_unavailable_reason(_fake_value(), 32, 31, 4)
    assert module_utils._get_msda_fastpath_unavailable_reason(_fake_value(), 32, 32, 4) is None


def test_msda_fastpath_warning_is_emitted_once(monkeypatch):
    warnings = []
    monkeypatch.setattr(module_utils, "_MSDA_FASTPATH_WARNING_EMITTED", False)
    monkeypatch.setattr(module_utils.LOGGER, "warning", warnings.append)

    module_utils._warn_msda_fastpath_unavailable("num_points=3 超出 [4, 8]")
    module_utils._warn_msda_fastpath_unavailable("embed_dims=33 不能被 8 整除")

    assert len(warnings) == 1
    assert "num_points=3" in warnings[0]
    assert "后续不再重复提示" in warnings[0]


def test_multi_scale_deformable_attn_rejects_ascend_bf16_grid_sample(monkeypatch):
    monkeypatch.setattr(module_utils, "IS_ASCEND", True)
    value = _fake_tensor((1, 20, 4, 32), dtype=torch.bfloat16)
    sampling_locations = _fake_tensor((1, 32, 4, 8, 2), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="F.grid_sample.*BF16.*FP16.*FP32"):
        module_utils.multi_scale_deformable_attn_pytorch(
            value,
            [(4, 4), (2, 2)],
            sampling_locations,
            _fake_tensor((1, 32, 4, 8), dtype=torch.bfloat16),
        )


def _run_msda_npu(embed_dims=32, num_queries=32, num_points=4):
    bs, num_heads = 1, 4
    value_shapes = [(4, 4), (2, 2)]
    num_keys = sum(h * w for h, w in value_shapes)
    value = torch.randn(bs, num_keys, num_heads, embed_dims, device="npu", dtype=torch.float16, requires_grad=True)
    sampling_locations = torch.rand(
        bs, num_queries, num_heads, len(value_shapes) * num_points, 2, device="npu", dtype=torch.float16
    )
    attention_weights = torch.rand(
        bs, num_queries, num_heads, len(value_shapes) * num_points, device="npu", dtype=torch.float16
    )

    output = module_utils.multi_scale_deformable_attn_pytorch(
        value, value_shapes, sampling_locations, attention_weights
    )
    output.float().sum().backward()
    torch.npu.synchronize()
    return output


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU is not available")
def test_msda_fastpath_embed_dims_not_divisible_by_8_falls_back_on_npu(monkeypatch):
    monkeypatch.setattr(module_utils, "IS_ASCEND", True)
    warnings = []
    monkeypatch.setattr(module_utils, "_MSDA_FASTPATH_WARNING_EMITTED", False)
    monkeypatch.setattr(module_utils.LOGGER, "warning", warnings.append)

    output = _run_msda_npu(embed_dims=33)

    assert output.shape == (1, 32, 132)
    assert len(warnings) == 1
    assert "embed_dims=33" in warnings[0]


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(), reason="NPU is not available")
def test_msda_fastpath_missing_mmcv_warns_once_on_npu(monkeypatch):
    monkeypatch.setattr(module_utils, "IS_ASCEND", True)
    warnings = []
    monkeypatch.setattr(module_utils, "_MSDA_FASTPATH_WARNING_EMITTED", False)
    monkeypatch.setattr(module_utils.LOGGER, "warning", warnings.append)
    monkeypatch.setattr(module_utils, "_get_msda_fastpath_function", lambda: None)

    _run_msda_npu()
    _run_msda_npu()

    assert len(warnings) == 1
    assert "MMCV MultiScaleDeformableAttnFunction 导入失败" in warnings[0]
