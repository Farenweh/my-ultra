from __future__ import annotations

import torch

from ultralytics.utils.nms import non_max_suppression


def test_detect_nms_single_label_outputs_and_optional_indices():
    prediction = torch.zeros(1, 7, 3)
    prediction[0, :4] = torch.tensor(
        [
            [10.0, 30.0, 50.0],
            [10.0, 30.0, 50.0],
            [4.0, 4.0, 4.0],
            [4.0, 4.0, 4.0],
        ]
    )
    prediction[0, 4:] = torch.tensor(
        [
            [0.90, 0.10, 0.20],
            [0.05, 0.80, 0.15],
            [0.10, 0.20, 0.70],
        ]
    )

    output = non_max_suppression(prediction.clone(), conf_thres=0.25, iou_thres=0.5, nc=3, max_det=10)

    assert len(output) == 1
    assert output[0].shape == (3, 6)
    assert torch.equal(output[0][:, 5].sort().values, torch.tensor([0.0, 1.0, 2.0]))

    output_with_idx, keep = non_max_suppression(
        prediction.clone(), conf_thres=0.25, iou_thres=0.5, nc=3, max_det=10, return_idxs=True
    )

    assert torch.equal(output_with_idx[0], output[0])
    assert torch.equal(keep[0].sort().values, torch.tensor([0, 1, 2]))
