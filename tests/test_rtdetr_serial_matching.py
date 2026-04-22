"""使用真实检测 Matcher 验证串行路径，避免上游接口更新后遗留掩码参数。"""

import pytest
import torch

from ultralytics.models.utils.loss import DETRLoss


@pytest.mark.parametrize("use_uni_match", [False, True])
@pytest.mark.parametrize("aux_loss", [False, True])
@pytest.mark.parametrize("with_gt", [False, True])
def test_serial_detection_loss_preserves_values_and_gradients(use_uni_match, aux_loss, with_gt):
    """主层、辅助层及统一匹配的损失和梯度应与 legacy 路径一致。"""
    torch.manual_seed(8)
    boxes = torch.rand(3, 2, 8, 4)
    scores = torch.randn(3, 2, 8, 3)
    batch = {
        "cls": torch.tensor([0, 2]) if with_gt else torch.empty(0, dtype=torch.long),
        "bboxes": torch.tensor([[0.4, 0.5, 0.2, 0.3], [0.7, 0.6, 0.1, 0.2]]) if with_gt else torch.empty(0, 4),
        "gt_groups": [2, 0] if with_gt else [0, 0],
    }
    results = []
    for legacy in (False, True):
        criterion = DETRLoss(nc=3, aux_loss=aux_loss, use_uni_match=use_uni_match)
        criterion._use_legacy_loss = legacy
        predicted_boxes = boxes.clone().requires_grad_()
        predicted_scores = scores.clone().requires_grad_()
        losses = criterion(predicted_boxes, predicted_scores, batch)
        # 早期历史的空目标回归分支没有梯度；比较两条路径一致性即可。
        gradients = torch.autograd.grad(sum(losses.values()), (predicted_boxes, predicted_scores), allow_unused=True)
        assert all(torch.isfinite(value).all() for value in losses.values())
        assert all(gradient is None or torch.isfinite(gradient).all() for gradient in gradients)
        if with_gt:
            assert all(gradient is not None for gradient in gradients)
        results.append((losses, gradients))
    torch.testing.assert_close(results[0][0], results[1][0], atol=1e-6, rtol=1e-5)
    for index, (actual, expected) in enumerate(zip(results[0][1], results[1][1])):
        # 原始历史中空目标框的 None 与零梯度均表示无回归贡献；后续 DDP 测试检查图连通性。
        if actual is None:
            assert not with_gt and index == 0
            actual = torch.zeros_like(boxes)
        if expected is None:
            assert not with_gt and index == 0
            expected = torch.zeros_like(boxes)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
