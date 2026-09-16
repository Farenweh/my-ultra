from contextlib import nullcontext
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tests.rtdetr_helpers import run_gloo_workers
from ultralytics.engine import trainer as trainer_module
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.rtdetr.train import RTDETRTrainer
from ultralytics.models.utils.loss import RTDETRDetectionLoss
from ultralytics.models.yolo.detect.train import DetectionTrainer


@pytest.mark.parametrize("rank,world_size", ((-1, 1), (0, 1), (0, 2), (1, 4)))
def test_trainer_loss_scale_contract(monkeypatch, rank, world_size):
    """RT-DETR 保留均值损失尺度，普通检测维持原 DDP 求和补偿。"""
    monkeypatch.setattr(trainer_module, "RANK", rank)
    for trainer_class in (BaseTrainer, DetectionTrainer, RTDETRTrainer):
        trainer = object.__new__(trainer_class)
        trainer.world_size = world_size
        expected = 1 if trainer_class is RTDETRTrainer or rank == -1 else world_size
        assert trainer._get_ddp_loss_scale() == expected


class _PredictionLoss(torch.nn.Module):
    """共享预测参数隔离 BN 和去噪随机性，仅验证真实检测损失的缩放。"""

    def __init__(self):
        super().__init__()
        self.predictions = torch.nn.Parameter(torch.randn(2, 1, 8, 6))
        self.criterion = RTDETRDetectionLoss(nc=2, use_vfl=True)

    def forward(self, batch_size):
        predictions = self.predictions.repeat(1, batch_size, 1, 1)
        targets = {
            "cls": torch.zeros(batch_size, dtype=torch.long),
            "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3]]).repeat(batch_size, 1),
            "gt_groups": [1] * batch_size,
        }
        return sum(self.criterion((predictions[..., :4].sigmoid(), predictions[..., 4:]), targets).values())


def _loss_scale_worker(rank, world_size, init_method):
    """真实 DDP 梯度平均及 no_sync 累积均应与单进程控制组一致。"""
    torch.set_num_threads(1)
    torch.manual_seed(41)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        distributed = DistributedDataParallel(_PredictionLoss())
        reference = _PredictionLoss()
        reference.load_state_dict(distributed.module.state_dict())
        trainer = object.__new__(RTDETRTrainer)
        trainer.world_size = world_size
        trainer_module.RANK = rank
        for accumulation in (1, 3):
            distributed.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            for step in range(accumulation):
                with distributed.no_sync() if step < accumulation - 1 else nullcontext():
                    (distributed(2) * trainer._get_ddp_loss_scale()).backward()
                reference(2 * world_size).backward()
            torch.testing.assert_close(distributed.module.predictions.grad, reference.predictions.grad)
    finally:
        dist.destroy_process_group()


def test_rtdetr_ddp_gradient_matches_single_process(tmp_path):
    """相同目标密度下，两 rank 不再产生额外倍增的梯度。"""
    run_gloo_workers(_loss_scale_worker, tmp_path)
