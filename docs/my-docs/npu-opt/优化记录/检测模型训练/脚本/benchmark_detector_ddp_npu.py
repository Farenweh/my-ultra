# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""在昇腾NPU上测量YOLO11-L DDP训练的find_unused_parameters开销。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析两卡DDP基准参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=32, help="每个rank的batch")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--find-unused-parameters", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict | None:
    """运行DDP训练基准，仅在rank 0返回结果。"""
    os.environ.setdefault("TASK_QUEUE_ENABLE", "2")

    import torch
    import torch.distributed as dist
    import torch_npu
    from torch.nn.parallel import DistributedDataParallel as DDP

    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.torch_utils import ModelEMA

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(local_rank)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = True
    dist.init_process_group(backend="hccl")
    device = torch.device(f"npu:{local_rank}")
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed + rank)

    model = YOLO("yolo11l.yaml", verbose=False).model
    model.args = DEFAULT_CFG
    model.nc = model.model[-1].nc
    model = model.to(device).train()
    model = DDP(
        model,
        device_ids=[local_rank],
        broadcast_buffers=True,
        find_unused_parameters=args.find_unused_parameters,
        gradient_as_bucket_view=False,
    )
    optimizer = torch_npu.optim.NpuFusedSGD(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    scaler = torch_npu.npu.amp.GradScaler()
    ema = ModelEMA(model.module, batch_scale=world_size)

    generator = torch.Generator().manual_seed(args.seed + rank)
    images = [
        (torch.randn(args.batch, 3, args.imgsz, args.imgsz, generator=generator) + offset).to(device)
        for offset in (0.0, 0.125)
    ]
    boxes_per_image = 4
    batch_idx = torch.arange(args.batch).repeat_interleave(boxes_per_image)
    classes = torch.arange(args.batch * boxes_per_image).remainder(model.module.nc).view(-1, 1).float()
    centers = torch.rand(args.batch * boxes_per_image, 2, generator=generator) * 0.6 + 0.2
    sizes = torch.rand(args.batch * boxes_per_image, 2, generator=generator) * 0.15 + 0.05
    batch = {
        "batch_idx": batch_idx.to(device),
        "cls": classes.to(device),
        "bboxes": torch.cat((centers, sizes), dim=1).to(device),
    }

    def train_step(step: int):
        optimizer.zero_grad()
        runtime_batch = {**batch, "img": images[step % len(images)]}
        with torch.autocast("npu", dtype=torch.float16):
            loss, _ = model(runtime_batch)
            loss = loss.sum()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        optimizer.clip_grad_norm_fused_(max_norm=10, norm_type=2)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model.module)

    torch.npu.synchronize(device)
    first_start = time.perf_counter()
    train_step(0)
    torch.npu.synchronize(device)
    first_step_ms = (time.perf_counter() - first_start) * 1000

    for step in range(1, args.warmup_steps + 1):
        train_step(step)
    torch.npu.synchronize(device)

    block_ms = []
    for repeat in range(args.repeats):
        torch.npu.synchronize(device)
        start = time.perf_counter()
        for index in range(args.measure_steps):
            train_step(args.warmup_steps + 1 + repeat * args.measure_steps + index)
        torch.npu.synchronize(device)
        elapsed = torch.tensor([(time.perf_counter() - start) * 1000 / args.measure_steps], device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        block_ms.append(float(elapsed.cpu()))

    first_step = torch.tensor([first_step_ms], device=device)
    dist.all_reduce(first_step, op=dist.ReduceOp.MAX)
    result = None
    if rank == 0:
        median_ms = statistics.median(block_ms)
        result = {
            "状态": "成功",
            "world_size": world_size,
            "每rank_batch": args.batch,
            "全局batch": args.batch * world_size,
            "imgsz": args.imgsz,
            "find_unused_parameters": args.find_unused_parameters,
            "first_step_ms": float(first_step.cpu()),
            "median_step_ms": median_ms,
            "mean_step_ms": statistics.mean(block_ms),
            "全局吞吐_images_per_s": args.batch * world_size / (median_ms / 1000),
            "计时块_ms": block_ms,
        }
        text = json.dumps(result, ensure_ascii=False, indent=2)
        print(text)
        if args.output:
            args.output.write_text(text + "\n")

    dist.barrier()
    dist.destroy_process_group()
    return result


if __name__ == "__main__":
    main(parse_args())
