from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tests.rtdetr_helpers import detection_batch, run_gloo_workers, tiny_rtdetr
from ultralytics.models.utils.loss import RTDETRDetectionLoss


@pytest.mark.parametrize("box_value", (0.2, float("nan")))
def test_empty_rtdetr_loss_preserves_legacy_values_and_gradients(box_value):
    """空目标回归损失保留零梯度，且预测框含 NaN 时仍为有限零值。"""
    scores = torch.randn(3, 2, 8, 2)
    targets = {"cls": torch.empty(0, dtype=torch.long), "bboxes": torch.empty(0, 4), "gt_groups": [0, 0]}
    results = []
    for legacy in (True, False):
        boxes = torch.full((3, 2, 8, 4), box_value, requires_grad=True)
        logits = scores.clone().requires_grad_()
        criterion = RTDETRDetectionLoss(nc=2, use_vfl=True)
        criterion._use_legacy_loss = legacy
        losses = criterion((boxes, logits), targets)
        gradients = torch.autograd.grad(sum(losses.values()), (boxes, logits))
        assert all(torch.isfinite(value) for value in losses.values())
        assert torch.count_nonzero(gradients[0]) == 0
        results.append((losses, gradients))
    torch.testing.assert_close(results[0], results[1])


def test_empty_rtdetr_model_gives_every_parameter_a_gradient():
    """完整模型的回归头和去噪 embedding 在空目标时也应参与反向。"""
    model = tiny_rtdetr().train()
    model(detection_batch(empty=True))[0].backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.isfinite().all(), name
        if "bbox_head" in name or "denoising_class_embed" in name:
            assert torch.count_nonzero(parameter.grad) == 0, name


def _empty_ddp_worker(rank, world_size, init_method):
    """交替执行有目标、全空和单个 rank 空目标的同步训练步。"""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        model = DistributedDataParallel(tiny_rtdetr().train(), find_unused_parameters=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        for empty in (False, True, rank == 0, False):
            optimizer.zero_grad(set_to_none=True)
            loss = model(detection_batch(empty=empty))[0]
            assert loss.isfinite()
            loss.backward()
            assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())
            optimizer.step()
    finally:
        dist.destroy_process_group()


def test_rtdetr_ddp_continues_after_empty_batches(tmp_path):
    """两进程 reducer 在空目标后必须继续接受后续前向和反向。"""
    run_gloo_workers(_empty_ddp_worker, tmp_path)
