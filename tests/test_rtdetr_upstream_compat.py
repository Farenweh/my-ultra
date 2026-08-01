"""验证本地 RT-DETR 优化与上游动态输入、去噪目标截断兼容。"""

import pytest
import torch

from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.utils.ops import get_cdn_group
from ultralytics.nn.modules.head import RTDETRDecoder
from ultralytics.nn.modules.transformer import AIFI


@pytest.mark.parametrize("learnt_query", (False, True))
@torch.no_grad()
def test_gather_query_selection_preserves_dynamic_small_inputs(learnt_query):
    """查询不足时限制 top-k，导出后仍可切换输入大小及 batch。"""
    head = RTDETRDecoder(
        nc=2, ch=(16,), hd=32, nq=8, nh=4, ndl=1, d_ffn=64, learnt_init_query=learnt_query
    ).eval()
    head.export, head.dynamic, head.format = True, True, "torchscript"
    traced = torch.jit.trace(head, ([torch.randn(1, 16, 2, 2)],), check_trace=False)
    for batch, h, w in ((1, 1, 1), (2, 2, 2), (1, 3, 4)):
        inputs = [torch.randn(batch, 16, h, w)]
        expected = head(inputs)
        actual = traced(inputs)
        assert actual.shape == (batch, min(8, h * w * 2), 6)
        torch.testing.assert_close(actual, expected)


@torch.no_grad()
def test_cached_aifi_preserves_dynamic_trace():
    """位置编码缓存不固定导出图的空间尺寸。"""
    aifi = AIFI(16, cm=32, num_heads=4).eval()
    aifi(torch.randn(1, 16, 2, 2))
    traced = torch.jit.trace(aifi, torch.randn(1, 16, 2, 2), check_trace=False)
    inputs = torch.randn(2, 16, 3, 4)
    torch.testing.assert_close(traced(inputs), aifi(inputs))


@pytest.mark.parametrize("batched", (False, True))
def test_denoising_cap_keeps_original_target_mapping(batched):
    """批量索引与 NPU scatter 路径保留上游截断目标映射及 loss 梯度。"""
    groups = [5, 0, 7]
    batch = {
        "cls": torch.arange(12) % 2,
        "bboxes": torch.rand(12, 4),
        "batch_idx": torch.tensor([0] * 5 + [2] * 7),
        "gt_groups": groups,
    }
    _, boxes, _, meta = get_cdn_group(
        batch, 2, 8, torch.randn(2, 16), num_dn=3, cls_noise_ratio=0, box_noise_scale=0, training=True
    )
    assert meta["dn_num_split"] == [6, 8]
    offsets = (0, 5, 5)
    for image, (original, positive) in enumerate(zip(meta["dn_gt_idx"], meta["dn_pos_idx"])):
        assert original.numel() == min(groups[image], 3)
        assert ((original >= offsets[image]) & (original < offsets[image] + groups[image])).all()
        torch.testing.assert_close(boxes[image, positive], batch["bboxes"][original])
    loss_fn = RTDETRDetectionLoss(nc=2)
    loss_fn._use_legacy_loss = not batched
    pred_boxes = torch.rand(2, 3, 8, 4, requires_grad=True)
    pred_scores = torch.randn(2, 3, 8, 2, requires_grad=True)
    dn_boxes = torch.rand(2, 3, 6, 4, requires_grad=True)
    dn_scores = torch.randn(2, 3, 6, 2, requires_grad=True)
    losses = loss_fn((pred_boxes, pred_scores), batch, dn_bboxes=dn_boxes, dn_scores=dn_scores, dn_meta=meta)
    sum(losses.values()).backward()
    for tensor in (pred_boxes, pred_scores, dn_boxes, dn_scores):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
