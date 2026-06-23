from __future__ import annotations

import torch

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.ops import xywh2xyxy


def test_v8_detection_loss_preprocess_handles_sparse_batch_indices():
    loss = object.__new__(v8DetectionLoss)
    loss.device = torch.device("cpu")
    scale = torch.tensor([100.0, 100.0, 100.0, 100.0])
    targets = torch.tensor(
        [
            [0.0, 1.0, 0.50, 0.50, 0.20, 0.20],
            [2.0, 3.0, 0.25, 0.25, 0.10, 0.10],
            [2.0, 4.0, 0.75, 0.75, 0.20, 0.20],
        ]
    )

    out = loss.preprocess(targets, batch_size=4, scale_tensor=scale)

    assert out.shape == (4, 2, 5)
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert torch.equal(out[3], torch.zeros_like(out[3]))
    assert out[0, 0, 0] == 1
    assert torch.allclose(out[0, 0, 1:5], xywh2xyxy(targets[0, 2:6] * scale))
    assert torch.equal(out[0, 1], torch.zeros_like(out[0, 1]))
    assert torch.equal(out[2, :, 0], torch.tensor([3.0, 4.0]))
    assert torch.allclose(out[2, 1, 1:5], xywh2xyxy(targets[2, 2:6] * scale))


def test_v8_detection_loss_preprocess_empty_targets():
    loss = object.__new__(v8DetectionLoss)
    loss.device = torch.device("cpu")

    out = loss.preprocess(torch.zeros(0, 6), batch_size=3, scale_tensor=torch.ones(4))

    assert out.shape == (3, 0, 5)
