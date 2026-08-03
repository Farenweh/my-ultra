# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""测量 YOLO11-L 和 RT-DETR-L 在昇腾 NPU 上的完整训练 step。"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析基准参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("yolo11l", "rtdetr-l"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--task-queue", choices=(0, 1, 2), type=int, default=2)
    parser.add_argument("--fused-optimizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused-grad-clip", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batched-hungarian", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--internal-format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit-compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--profile-phases", action="store_true")
    parser.add_argument("--ready-file", type=Path, help="预热结束后创建该文件，供外部利用率采样器同步")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict:
    """运行完整训练 step 基准并返回统计结果。"""
    # TorchNPU 在导入阶段初始化任务队列，因此必须在导入 torch 或 Ultralytics 前设置。
    os.environ["TASK_QUEUE_ENABLE"] = str(args.task_queue)
    os.environ["USE_BATCHED_HUNGARIAN"] = "1" if args.batched_hungarian else "0"

    import torch
    import torch_npu
    import ultralytics
    from ultralytics import RTDETR, YOLO
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils.torch_utils import ModelEMA

    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    torch.npu.manual_seed(args.seed)
    torch.npu.set_compile_mode(jit_compile=args.jit_compile)
    torch.npu.config.allow_internal_format = args.internal_format
    device = torch.device(f"npu:{args.device}")

    wrapper = YOLO("yolo11l.yaml", verbose=False) if args.model == "yolo11l" else RTDETR("rtdetr-l.yaml")
    model = wrapper.model
    model.args = DEFAULT_CFG
    model.nc = model.model[-1].nc
    model = model.to(device).train()

    optimizer_cls = torch_npu.optim.NpuFusedSGD if args.fused_optimizer else torch.optim.SGD
    optimizer = optimizer_cls(model.parameters(), lr=1e-3, momentum=0.9, nesterov=True)
    if args.fused_grad_clip and not hasattr(optimizer, "clip_grad_norm_fused_"):
        raise RuntimeError("融合梯度裁剪要求使用 TorchNPU 融合优化器")
    ema = ModelEMA(model, batch_scale=1.0) if args.ema else None
    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.dtype)
    scaler = torch_npu.npu.amp.GradScaler(enabled=args.dtype == "fp16")

    generator = torch.Generator().manual_seed(args.seed + 1)
    images = [
        (torch.randn(args.batch, 3, args.imgsz, args.imgsz, generator=generator) + offset).to(device)
        for offset in (0.0, 0.125)
    ]
    boxes_per_image = 4
    batch_idx = torch.arange(args.batch).repeat_interleave(boxes_per_image)
    classes = torch.arange(args.batch * boxes_per_image).remainder(model.nc).view(-1, 1).float()
    centers = torch.rand(args.batch * boxes_per_image, 2, generator=generator) * 0.6 + 0.2
    sizes = torch.rand(args.batch * boxes_per_image, 2, generator=generator) * 0.15 + 0.05
    batch = {
        "batch_idx": batch_idx.to(device),
        "cls": classes.to(device),
        "bboxes": torch.cat((centers, sizes), dim=1).to(device),
    }

    def autocast_context():
        return torch.autocast("npu", dtype=amp_dtype, enabled=amp_dtype is not None)

    def forward_loss(step: int):
        runtime_batch = {**batch, "img": images[step % len(images)]}
        with autocast_context():
            loss, _ = model(runtime_batch)
            return loss.sum()

    def clip_gradients():
        if args.fused_grad_clip:
            optimizer.clip_grad_norm_fused_(max_norm=10, norm_type=2)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0, foreach=False)

    def train_step(step: int):
        optimizer.zero_grad()
        loss = forward_loss(step)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_gradients()
        scaler.step(optimizer)
        scaler.update()
        if ema:
            ema.update(model)
        return loss.detach()

    torch.npu.reset_peak_memory_stats(device)
    torch.npu.synchronize(device)
    first_start = time.perf_counter()
    first_loss = train_step(0)
    torch.npu.synchronize(device)
    first_step_ms = (time.perf_counter() - first_start) * 1000

    for step in range(1, args.warmup_steps + 1):
        train_step(step)
    torch.npu.synchronize(device)
    if args.ready_file:
        args.ready_file.touch()

    block_ms = []
    final_loss = first_loss
    for repeat in range(args.repeats):
        torch.npu.synchronize(device)
        start = time.perf_counter()
        for index in range(args.measure_steps):
            final_loss = train_step(args.warmup_steps + 1 + repeat * args.measure_steps + index)
        torch.npu.synchronize(device)
        block_ms.append((time.perf_counter() - start) * 1000 / args.measure_steps)

    phase_ms = None
    if args.profile_phases:
        phase_values = {name: [] for name in ("zero_grad", "forward_loss", "backward", "unscale_clip", "step", "ema")}

        def record(name: str, fn):
            torch.npu.synchronize(device)
            start = time.perf_counter()
            value = fn()
            torch.npu.synchronize(device)
            phase_values[name].append((time.perf_counter() - start) * 1000)
            return value

        for step in range(5):
            record("zero_grad", optimizer.zero_grad)
            loss = record("forward_loss", lambda: forward_loss(step))
            record("backward", lambda: scaler.scale(loss).backward())
            record("unscale_clip", lambda: (scaler.unscale_(optimizer), clip_gradients()))
            record("step", lambda: (scaler.step(optimizer), scaler.update()))
            if ema:
                record("ema", lambda: ema.update(model))
        phase_ms = {name: statistics.median(values) if values else 0.0 for name, values in phase_values.items()}

    return {
        "状态": "成功",
        "模型": args.model,
        "Ultralytics源码": str(Path(ultralytics.__file__).resolve()),
        "源码git_head": subprocess.run(
            ["git", "-C", str(Path(ultralytics.__file__).resolve().parent.parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "基准脚本": str(Path(__file__).resolve()),
        "Python": platform.python_version(),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_git": torch_npu.version.git_version,
        "参数量": sum(parameter.numel() for parameter in model.parameters()),
        "逻辑device": args.device,
        "ASCEND_RT_VISIBLE_DEVICES": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "batch": args.batch,
        "imgsz": args.imgsz,
        "dtype": args.dtype,
        "TASK_QUEUE_ENABLE": args.task_queue,
        "融合优化器": args.fused_optimizer,
        "融合梯度裁剪": args.fused_grad_clip,
        "跨层批量Hungarian": args.batched_hungarian,
        "internal_format": args.internal_format,
        "jit_compile": args.jit_compile,
        "EMA": args.ema,
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "repeats": args.repeats,
        "first_step_ms": first_step_ms,
        "median_step_ms": statistics.median(block_ms),
        "mean_step_ms": statistics.mean(block_ms),
        "min_step_ms": min(block_ms),
        "max_step_ms": max(block_ms),
        "吞吐_images_per_s": args.batch / (statistics.median(block_ms) / 1000),
        "最终损失": float(final_loss.cpu()),
        "峰值已分配显存_MiB": torch.npu.max_memory_allocated(device) / 2**20,
        "峰值已保留显存_MiB": torch.npu.max_memory_reserved(device) / 2**20,
        "阶段中位时延_ms": phase_ms,
        "计时块_ms": block_ms,
    }


if __name__ == "__main__":
    parsed = parse_args()
    result = main(parsed)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if parsed.output:
        parsed.output.write_text(text + "\n")
