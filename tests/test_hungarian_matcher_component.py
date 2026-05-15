from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.models.utils.ops import HungarianMatcher
from ultralytics.utils.metrics import bbox_iou


def reference_hungarian_matcher(
    matcher: HungarianMatcher,
    pred_bboxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_bboxes: torch.Tensor,
    gt_cls: torch.Tensor,
    gt_groups: list[int],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    bs, nq, nc = pred_scores.shape

    if sum(gt_groups) == 0:
        return [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)) for _ in range(bs)]

    pred_scores = pred_scores.detach().view(-1, nc)
    pred_scores = pred_scores.sigmoid() if matcher.use_fl else F.softmax(pred_scores, dim=-1)
    pred_bboxes = pred_bboxes.detach().view(-1, 4)

    pred_scores = pred_scores[:, gt_cls]
    if matcher.use_fl:
        neg_cost_class = (1 - matcher.alpha) * (pred_scores**matcher.gamma) * (-(1 - pred_scores + 1e-8).log())
        pos_cost_class = matcher.alpha * ((1 - pred_scores) ** matcher.gamma) * (-(pred_scores + 1e-8).log())
        cost_class = pos_cost_class - neg_cost_class
    else:
        cost_class = -pred_scores

    cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)
    cost_giou = 1.0 - bbox_iou(pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True).squeeze(-1)
    cost = (
        matcher.cost_gain["class"] * cost_class
        + matcher.cost_gain["bbox"] * cost_bbox
        + matcher.cost_gain["giou"] * cost_giou
    )
    cost[cost.isnan() | cost.isinf()] = 0.0

    cost_np = cost.view(bs, nq, -1).cpu().numpy()
    gt_groups_np = np.asarray(gt_groups, dtype=np.int64)
    gt_offsets_np = np.zeros(bs, dtype=np.int64)
    if bs > 1:
        gt_offsets_np[1:] = np.cumsum(gt_groups_np[:-1], dtype=np.int64)

    indices = []
    for bi in range(bs):
        start = int(gt_offsets_np[bi])
        end = start + int(gt_groups_np[bi])
        if end == start:
            empty = np.empty(0, dtype=np.int64)
            indices.append((empty, empty))
        else:
            indices.append(linear_sum_assignment(cost_np[bi, :, start:end]))

    match_lengths = [len(src_idx) for src_idx, _ in indices]
    total_matches = sum(match_lengths)
    if total_matches == 0:
        return [(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)) for _ in range(bs)]

    src_np = np.concatenate([src_idx.astype(np.int64, copy=False) for src_idx, _ in indices], axis=0)
    dst_np = np.concatenate(
        [(dst_idx.astype(np.int64, copy=False) + gt_offsets_np[bi]) for bi, (_, dst_idx) in enumerate(indices)],
        axis=0,
    )
    src_all = torch.from_numpy(src_np)
    dst_all = torch.from_numpy(dst_np)
    return list(zip(src_all.split(match_lengths), dst_all.split(match_lengths)))


@torch.no_grad()
def test_hungarian_matcher_matches_reference_focal():
    torch.manual_seed(0)
    matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2}, use_fl=True)
    bs, nq, nc = 3, 32, 10
    gt_groups = [4, 0, 3]
    total_gt = sum(gt_groups)

    pred_bboxes = torch.rand(bs, nq, 4)
    pred_scores = torch.randn(bs, nq, nc)
    gt_bboxes = torch.rand(total_gt, 4)
    gt_cls = torch.randint(0, nc, (total_gt,), dtype=torch.long)

    expected = reference_hungarian_matcher(matcher, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)
    actual = matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)

    assert len(expected) == len(actual)
    assert all(torch.equal(src_a, src_b) and torch.equal(dst_a, dst_b) for (src_a, dst_a), (src_b, dst_b) in zip(expected, actual))


@torch.no_grad()
def test_hungarian_matcher_matches_reference_softmax():
    torch.manual_seed(1)
    matcher = HungarianMatcher(cost_gain={"class": 2, "bbox": 5, "giou": 2}, use_fl=False)
    bs, nq, nc = 2, 24, 6
    gt_groups = [2, 5]
    total_gt = sum(gt_groups)

    pred_bboxes = torch.rand(bs, nq, 4)
    pred_scores = torch.randn(bs, nq, nc)
    gt_bboxes = torch.rand(total_gt, 4)
    gt_cls = torch.randint(0, nc, (total_gt,), dtype=torch.long)

    expected = reference_hungarian_matcher(matcher, pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)
    actual = matcher(pred_bboxes, pred_scores, gt_bboxes, gt_cls, gt_groups)

    assert len(expected) == len(actual)
    assert all(torch.equal(src_a, src_b) and torch.equal(dst_a, dst_b) for (src_a, dst_a), (src_b, dst_b) in zip(expected, actual))
