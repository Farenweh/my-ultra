"""使用torchrun验证两卡冻结C-RADIOv4-SO400M骨干网络的检测训练。"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ultralytics.nn.tasks import DetectionModel


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")
    dist.init_process_group(backend="hccl")
    try:
        cfg = Path(__file__).resolve().parents[3] / "ultralytics/cfg/models/rf-det/c-radio-v4-yolo11.yaml"
        model = DetectionModel(cfg, ch=3, nc=2, verbose=False, summary=False)
        model.model[0].requires_grad_(False)
        model.args = SimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
        model = model.to(device).train()
        model.criterion = model.init_criterion()
        assert model.model[0].model.frozen_deterministic
        dynamic_cache_names = {"_base_grid_cache", "_position_cache"}
        assert not dynamic_cache_names & dict(model.named_buffers()).keys()
        ddp = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=True)
        optimizer = torch.optim.SGD(
            (parameter for parameter in ddp.parameters() if parameter.requires_grad),
            lr=1e-4,
            momentum=0.9,
            foreach=False,
        )
        batch = {
            "batch_idx": torch.tensor([0], device=device, dtype=torch.long),
            "cls": torch.zeros((1, 1), device=device),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=device),
        }
        optimizer.zero_grad(set_to_none=True)
        image = torch.rand(1, 3, 640, 640, device=device)
        with torch.autocast("npu", dtype=torch.float16):
            predictions = ddp(image)
            loss, _ = model.loss(batch, predictions)
            total = loss.sum()
        total.backward()
        optimizer.step()
        reduced = total.detach().clone()
        dist.all_reduce(reduced)
        reduced /= dist.get_world_size()
        assert torch.isfinite(reduced)
        assert all(parameter.grad is None for parameter in model.model[0].parameters())
        if dist.get_rank() == 0:
            print(f"两卡C-RADIOv4 DDP smoke通过，平均loss={float(reduced):.6f}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
