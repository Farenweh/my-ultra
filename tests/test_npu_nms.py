from __future__ import annotations

import types

import pytest
import torch

import ultralytics.models.yolo.detect.val as detect_val
import ultralytics.nn.autobackend as autobackend_module
import ultralytics.utils.nms as nms_module


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


def _nms_prediction(batch=3, device="cpu"):
    prediction = torch.zeros(batch, 5, 3, device=device)
    prediction[:, :4] = torch.tensor(
        [[10.0, 10.5, 30.0], [10.0, 10.5, 30.0], [4.0, 4.0, 2.0], [4.0, 4.0, 2.0]], device=device
    )
    prediction[:, 4] = torch.tensor([0.9, 0.8, 0.7], device=device)
    return prediction


@pytest.mark.parametrize(("max_time_img", "expected"), [(None, [3, 3, 3]), (0.0, [3, 0, 0])])
def test_nms_none_disables_batch_timeout(monkeypatch, max_time_img, expected):
    ticks = iter([0.0, 100.0, 200.0, 300.0])
    monkeypatch.setattr(nms_module.time, "time", lambda: next(ticks))
    monkeypatch.setattr(
        nms_module.TorchNMS,
        "nms",
        staticmethod(lambda boxes, _scores, _iou: torch.arange(len(boxes), device=boxes.device)),
    )

    outputs = nms_module.non_max_suppression(
        _nms_prediction(), conf_thres=0.1, iou_thres=0.5, nc=1, max_time_img=max_time_img
    )

    assert [len(output) for output in outputs] == expected


def test_detection_validator_disables_nms_timeout(monkeypatch):
    captured = {}
    validator = object.__new__(detect_val.DetectionValidator)
    validator.args = types.SimpleNamespace(
        conf=0.001,
        iou=0.7,
        task="detect",
        single_cls=False,
        agnostic_nms=False,
        max_det=300,
    )
    validator.nc = 80
    validator.end2end = False

    def fake_nms(preds, *args, **kwargs):
        captured.update(kwargs)
        return [torch.zeros(0, 6, device=preds.device) for _ in range(len(preds))]

    monkeypatch.setattr(detect_val.nms, "non_max_suppression", fake_nms)
    validator.postprocess(torch.zeros(2, 84, 10))

    assert captured["max_time_img"] is None


def test_autobackend_warmup_disables_nms_timeout(monkeypatch):
    original_rand = torch.rand
    captured = {}
    backend = object.__new__(autobackend_module.AutoBackend)
    object.__setattr__(backend, "end2end", True)
    object.__setattr__(backend, "format", "pt")
    object.__setattr__(backend, "device", types.SimpleNamespace(type="npu"))
    object.__setattr__(backend, "fp16", False)

    monkeypatch.setattr(autobackend_module.AutoBackend, "forward", lambda self, image: image)
    monkeypatch.setattr(autobackend_module.torch, "rand", lambda *shape, **kwargs: original_rand(*shape))
    monkeypatch.setattr(nms_module, "non_max_suppression", lambda *_args, **kwargs: captured.update(kwargs))

    backend.warmup(im=torch.zeros(1, 3, 32, 32))

    assert captured["max_time_img"] is None


def test_exact_nms_cpu_matches_torchvision_and_preserves_empty_device():
    from torchvision.ops import nms

    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [0.5, 0.5, 4.5, 4.5], [8.0, 8.0, 10.0, 10.0], [8.0, 8.0, 9.0, 9.0]])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])

    assert torch.equal(nms_module.TorchNMS.nms(boxes, scores, 0.5), nms(boxes, scores, 0.5))
    empty = nms_module.TorchNMS.nms(boxes[:0], scores[:0], 0.5)
    assert empty.dtype == torch.int64 and empty.device == boxes.device and empty.shape == (0,)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="需要可用的Ascend NPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_exact_nms_npu_uses_fp32_cpu_kernel_and_returns_npu_indices(monkeypatch, dtype):
    import torchvision

    boxes = torch.tensor([[0.0, 0.0, 4.0, 4.0], [0.5, 0.5, 4.5, 4.5], [8.0, 8.0, 10.0, 10.0], [8.0, 8.0, 9.0, 9.0]])
    scores = torch.tensor([0.9, 0.8, 0.7, 0.6])
    expected = torchvision.ops.nms(boxes, scores, 0.5)
    original_nms = torchvision.ops.nms
    captured = {}

    def checked_cpu_nms(cpu_boxes, cpu_scores, threshold):
        captured.update(device=cpu_boxes.device.type, boxes_dtype=cpu_boxes.dtype, scores_dtype=cpu_scores.dtype)
        return original_nms(cpu_boxes, cpu_scores, threshold)

    monkeypatch.setattr(torchvision.ops, "nms", checked_cpu_nms)
    actual = nms_module.TorchNMS.nms(boxes.to("npu:0", dtype=dtype), scores.to("npu:0", dtype=dtype), 0.5)

    assert captured == {"device": "cpu", "boxes_dtype": torch.float32, "scores_dtype": torch.float32}
    assert actual.device.type == "npu" and actual.dtype == torch.int64
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="需要可用的Ascend NPU")
def test_detection_nms_npu_matches_cpu_with_original_indices():
    prediction = _nms_prediction(batch=2)
    expected, expected_indices = nms_module.non_max_suppression(
        prediction.clone(), conf_thres=0.1, iou_thres=0.5, nc=1, max_time_img=None, return_idxs=True
    )
    actual, actual_indices = nms_module.non_max_suppression(
        prediction.npu(), conf_thres=0.1, iou_thres=0.5, nc=1, max_time_img=None, return_idxs=True
    )

    for cpu_output, npu_output, cpu_indices, npu_indices in zip(expected, actual, expected_indices, actual_indices):
        torch.testing.assert_close(npu_output.cpu(), cpu_output)
        assert npu_indices.device.type == "npu"
        assert torch.equal(npu_indices.cpu(), cpu_indices)
