from __future__ import annotations

import pytest
import torch

from ultralytics.utils.metrics import batch_probiou, box_iou, probiou


NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()


def _box_iou_reference(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    return intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + eps)


def _boxes(dtype=torch.float32, device="cpu"):
    return torch.tensor(
        [[0.0, 0.0, 4.0, 4.0], [0.5, 0.5, 4.5, 4.5], [8.0, 8.0, 10.0, 10.0], [2.0, 2.0, 2.0, 3.0]],
        dtype=dtype,
        device=device,
    )


def _covariance_reference(boxes: torch.Tensor):
    gaussian_boxes = torch.cat((boxes[:, 2:4].square() / 12, boxes[:, 4:]), dim=-1)
    a, b, angle = gaussian_boxes.split(1, dim=-1)
    cos, sin = angle.cos(), angle.sin()
    return a * cos.square() + b * sin.square(), a * sin.square() + b * cos.square(), (a - b) * cos * sin


def _probiou_reference(obb1: torch.Tensor, obb2: torch.Tensor, eps: float = 1e-7):
    x1, y1 = obb1[..., :2].split(1, dim=-1)
    x2, y2 = obb2[..., :2].split(1, dim=-1)
    a1, b1, c1 = _covariance_reference(obb1)
    a2, b2, c2 = _covariance_reference(obb2)
    denominator = (a1 + a2) * (b1 + b2) - (c1 + c2).square() + eps
    t1 = ((a1 + a2) * (y1 - y2).square() + (b1 + b2) * (x1 - x2).square()) / denominator * 0.25
    t2 = (c1 + c2) * (x2 - x1) * (y1 - y2) / denominator * 0.5
    determinant = (a1 + a2) * (b1 + b2) - (c1 + c2).square()
    determinant1 = (a1 * b1 - c1.square()).clamp(0)
    determinant2 = (a2 * b2 - c2.square()).clamp(0)
    t3 = (determinant / (4 * (determinant1 * determinant2).sqrt() + eps) + eps).log() * 0.5
    distance = (t1 + t2 + t3).clamp(eps, 100.0)
    return 1 - (1.0 - (-distance).exp() + eps).sqrt()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_box_iou_matches_reference_for_supported_dtypes_and_empty_inputs(dtype):
    boxes1, boxes2 = _boxes(dtype), _boxes(dtype).flip(0)
    actual = box_iou(boxes1, boxes2)
    expected = _box_iou_reference(boxes1, boxes2)

    assert torch.equal(actual, expected)
    assert box_iou(boxes1[:0], boxes2).shape == (0, len(boxes2))
    assert box_iou(boxes1, boxes2[:0]).shape == (len(boxes1), 0)


def test_box_iou_preserves_autograd():
    boxes1, boxes2 = _boxes().requires_grad_(True), _boxes().flip(0).requires_grad_(True)
    reference1, reference2 = boxes1.detach().clone().requires_grad_(True), boxes2.detach().clone().requires_grad_(True)

    box_iou(boxes1, boxes2).sum().backward()
    _box_iou_reference(reference1, reference2).sum().backward()

    torch.testing.assert_close(boxes1.grad, reference1.grad)
    torch.testing.assert_close(boxes2.grad, reference2.grad)


def test_probiou_values_and_gradients_match_reference():
    obb1 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.2], [3.0, 1.0, 2.0, 5.0, -0.4]], requires_grad=True)
    obb2 = torch.tensor([[1.5, 2.5, 3.5, 4.5, 0.1], [2.0, 1.0, 2.5, 4.0, -0.2]], requires_grad=True)
    reference1 = obb1.detach().clone().requires_grad_(True)
    reference2 = obb2.detach().clone().requires_grad_(True)

    aligned = probiou(obb1, obb2)
    pairwise = batch_probiou(obb1, obb2)
    (aligned.sum() + pairwise.sum()).backward()
    expected_aligned = _probiou_reference(reference1, reference2)
    expected_pairwise = torch.stack(
        [torch.cat([_probiou_reference(box1[None], box2[None]) for box2 in reference2]) for box1 in reference1]
    )
    (expected_aligned.sum() + expected_pairwise.sum()).backward()

    assert aligned.shape == (2, 1) and pairwise.shape == (2, 2)
    torch.testing.assert_close(aligned, expected_aligned, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(pairwise, expected_pairwise.squeeze(-1), rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(obb1.grad, reference1.grad, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(obb2.grad, reference2.grad, rtol=5e-5, atol=5e-5)


@pytest.mark.skipif(not NPU_AVAILABLE, reason="需要可用的Ascend NPU")
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(torch.float32, 2e-4, 2e-4), (torch.float16, 3e-3, 3e-3), (torch.bfloat16, 3e-2, 3e-2)],
)
def test_iou_and_probiou_npu_match_cpu(dtype, rtol, atol):
    boxes1, boxes2 = _boxes(dtype), _boxes(dtype).flip(0)
    torch.testing.assert_close(box_iou(boxes1.npu(), boxes2.npu()).cpu(), box_iou(boxes1, boxes2))

    obb1 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.2], [3.0, 1.0, 2.0, 5.0, -0.4]], dtype=dtype)
    obb2 = torch.tensor([[1.5, 2.5, 3.5, 4.5, 0.1], [2.0, 1.0, 2.5, 4.0, -0.2]], dtype=dtype)
    torch.testing.assert_close(
        probiou(obb1.npu(), obb2.npu(), CIoU=True).cpu(), probiou(obb1, obb2, CIoU=True), rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        batch_probiou(obb1.npu(), obb2.npu()).cpu(), batch_probiou(obb1, obb2), rtol=rtol, atol=atol
    )
