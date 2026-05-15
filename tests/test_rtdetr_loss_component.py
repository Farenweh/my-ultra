from __future__ import annotations

import torch

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils.ops import get_cdn_group


def test_get_cdn_group_preserves_output_dtypes():
    torch.manual_seed(0)
    batch = {
        "cls": torch.randint(0, 10, (5,), dtype=torch.long),
        "bboxes": torch.rand(5, 4, dtype=torch.float16),
        "batch_idx": torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
        "gt_groups": [2, 3],
    }
    class_embed = torch.randn(10, 16, dtype=torch.float32)

    padding_cls, padding_bbox, attn_mask, dn_meta = get_cdn_group(
        batch=batch,
        num_classes=10,
        num_queries=32,
        class_embed=class_embed,
        num_dn=20,
        training=True,
    )

    assert padding_cls is not None and padding_bbox is not None and attn_mask is not None and dn_meta is not None
    assert padding_cls.dtype == class_embed.dtype
    assert padding_bbox.dtype == batch["bboxes"].dtype
    assert attn_mask.dtype == torch.bool


def test_rtdetr_loss_component_forward():
    torch.manual_seed(0)
    bs = 2
    nc = 30
    hidden_dim = 64
    num_queries = 64
    num_dn = 40
    decoder_layers = 3
    gt_groups = [3, 2]
    total_gt = sum(gt_groups)

    batch = {
        "cls": torch.randint(0, nc, (total_gt,), dtype=torch.long),
        "bboxes": torch.rand(total_gt, 4),
        "batch_idx": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
        "gt_groups": gt_groups,
    }
    class_embed = torch.randn(nc, hidden_dim)
    _, _, _, dn_meta = get_cdn_group(
        batch=batch,
        num_classes=nc,
        num_queries=num_queries,
        class_embed=class_embed,
        num_dn=num_dn,
        training=True,
    )
    assert dn_meta is not None
    dn_q = dn_meta["dn_num_split"][0]

    main_layers = decoder_layers + 1  # encoder + decoder outputs
    pred_bboxes = torch.rand(main_layers, bs, num_queries, 4)
    pred_scores = torch.randn(main_layers, bs, num_queries, nc)
    dn_bboxes = torch.rand(decoder_layers, bs, dn_q, 4)
    dn_scores = torch.randn(decoder_layers, bs, dn_q, nc)

    criterion = RTDETRDetectionLoss(nc=nc, use_vfl=True)
    losses = criterion(
        preds=(pred_bboxes, pred_scores),
        batch=batch,
        dn_bboxes=dn_bboxes,
        dn_scores=dn_scores,
        dn_meta=dn_meta,
    )

    assert "loss_class" in losses and "loss_bbox" in losses and "loss_giou" in losses
    assert "loss_class_dn" in losses and "loss_bbox_dn" in losses and "loss_giou_dn" in losses
    assert all(torch.isfinite(v).item() for v in losses.values())
