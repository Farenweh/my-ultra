from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from ultralytics.nn.modules.backbone import DINOv2
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils.checks import check_imgsz


def test_dinov2_dims_all_official_variants():
    assert [DINOv2.dims(s) for s in ("s", "b", "l", "g", "s_reg", "b_reg", "l_reg", "g_reg")] == [
        384,
        768,
        1024,
        1536,
        384,
        768,
        1024,
        1536,
    ]


def test_dinov2_forward_shapes_and_input_assertion():
    model = DINOv2("s", pretrained=False).eval()

    with torch.no_grad():
        assert model(torch.randn(1, 3, 224, 224)).shape == (1, 384, 16, 16)
        assert model(torch.randn(1, 3, 644, 644)).shape == (1, 384, 46, 46)

    with pytest.raises(AssertionError, match="14"):
        model(torch.randn(1, 3, 640, 640))


def test_dinov2_default_weights_uses_official_url(monkeypatch: pytest.MonkeyPatch):
    expected_state = DINOv2("s", pretrained=False).model.state_dict()
    urls = []

    def fake_load_state_dict_from_url(url, *args, **kwargs):
        urls.append(url)
        return expected_state

    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", fake_load_state_dict_from_url)

    DINOv2("s")

    assert urls == ["https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth"]


def test_dinov2_rtdetr_stride_tracks_following_downsample():
    cfg = {
        "nc": 80,
        "backbone": [
            [-1, 1, "DINOv2", ["s", False]],
            [-1, 1, "Conv", [384, 3, 2]],
        ],
        "head": [[[1], 1, "RTDETRDecoder", ["nc"]]],
    }

    model = RTDETRDetectionModel(cfg, ch=3, nc=80, verbose=False)

    assert model.stride.tolist() == [28.0]
    assert model.model.stride.tolist() == [28.0]
    assert check_imgsz(640, stride=model.stride) == 644


def test_validator_init_preserves_dinov2_stride_aligned_imgsz(tmp_path):
    from ultralytics.engine.validator import BaseValidator

    validator = BaseValidator(save_dir=tmp_path, args={"imgsz": 644})

    assert validator.args.imgsz == 644


def test_dinov2_third_party_has_no_xformers_references():
    root = Path(__file__).resolve().parents[1] / "ultralytics/nn/modules/third_party/dinov2"
    forbidden = ("xformers", "XFORMERS", "MemEffAttention", "NestedTensorBlock")

    for path in root.rglob("*.py"):
        text = path.read_text()
        assert not any(term in text for term in forbidden), path


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu/NPU is not available",
)
def test_dinov2_swiglu_import_does_not_lock_visible_devices():
    script = """
import os
import torch
import ultralytics.nn.modules.third_party.dinov2.dinov2.layers.swiglu_ffn

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
x = torch.ones(1, device="npu:0")
print(x.cpu().item())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "1.0" in result.stdout


@pytest.mark.skipif(
    importlib.util.find_spec("torch_npu") is None or not hasattr(torch, "npu") or not torch.npu.is_available(),
    reason="torch_npu/NPU is not available",
)
def test_dinov2_npu_forward_and_swiglu():
    import torch.nn.functional as F
    import torch_npu

    device = torch.device("npu:0")
    torch.npu.set_device(device)

    model = DINOv2("s", pretrained=False).eval().to(device)
    with torch.no_grad():
        y = model(torch.randn(1, 3, 224, 224, device=device))
    torch.npu.synchronize()
    assert y.device.type == "npu"
    assert y.shape == (1, 384, 16, 16)
    assert torch.isfinite(y).all()

    x = torch.randn(2, 4, 8, device=device)
    fused = torch_npu.npu_swiglu(x, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    torch.testing.assert_close(fused, F.silu(x1) * x2)
