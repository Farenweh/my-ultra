# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""在昇腾 NPU 上测试 YOLO11-L/X 固定形状训练的 NPUGraph 性能。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析基准参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=("eager", "npugraphs", "make_graphed_callables"), required=True)
    parser.add_argument("--scale", choices=("l", "x"), default="l")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--dtype", choices=("fp32", "amp_bf16"), default="amp_bf16")
    parser.add_argument("--task-queue", choices=(0, 1, 2), type=int, default=1)
    fullgraph_group = parser.add_mutually_exclusive_group()
    fullgraph_group.add_argument("--fullgraph", dest="fullgraph", action="store_true")
    fullgraph_group.add_argument("--no-fullgraph", dest="fullgraph", action="store_false")
    parser.set_defaults(fullgraph=False)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--state-output", type=Path, help="正确性比较时保存模型、梯度、优化器和 loss 轨迹")
    parser.add_argument("--memory-usage-output-dir", type=Path, help="导出 CANN 各组件当前/峰值显存 CSV")
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict:
    """运行固定形状、真实检测损失和 SGD 更新的训练基准。"""
    # TorchNPU 在导入阶段初始化任务队列，因此必须在导入 torch 或 Ultralytics 前设置。
    os.environ["TASK_QUEUE_ENABLE"] = str(args.task_queue)
    if args.memory_usage_output_dir:
        args.memory_usage_output_dir.mkdir(parents=True, exist_ok=True)
        os.environ["OOM_SNAPSHOT_PATH"] = str(args.memory_usage_output_dir.resolve())

    import torch
    import torch_npu
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG

    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    torch.npu.manual_seed(args.seed)
    device = torch.device(f"npu:{args.device}")
    amp_dtype = torch.bfloat16 if args.dtype == "amp_bf16" else None

    model = YOLO(f"yolo11{args.scale}.yaml", verbose=False).model
    model.args = DEFAULT_CFG
    model = model.to(device).train()
    model.criterion = model.init_criterion()
    initial_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, foreach=False)

    generator = torch.Generator().manual_seed(args.seed + 1)
    images = [
        (torch.randn(args.batch, 3, args.imgsz, args.imgsz, generator=generator) + offset).to(device)
        for offset in (0.0, 0.125)
    ]
    batch = {
        "batch_idx": torch.arange(args.batch, device=device, dtype=torch.long),
        "cls": torch.zeros((args.batch, 1), device=device),
        "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]] * args.batch, device=device),
    }

    def autocast_context():
        return nullcontext() if amp_dtype is None else torch.autocast("npu", dtype=amp_dtype, cache_enabled=False)

    setup_start = time.perf_counter()
    if args.scheme == "make_graphed_callables":
        with autocast_context():
            callable_model = torch_npu.npu.make_graphed_callables(model, (images[0].clone(),))
        model.load_state_dict(initial_state)
        optimizer.zero_grad(set_to_none=True)
    elif args.scheme == "npugraphs":
        callable_model = torch.compile(model, backend="npugraphs", fullgraph=args.fullgraph, dynamic=False)
    else:
        callable_model = model
    torch.npu.synchronize(device)
    setup_ms = (time.perf_counter() - setup_start) * 1000

    loss_trace_device = []

    def train_step(step: int):
        optimizer.zero_grad(set_to_none=True)
        if args.scheme == "npugraphs":
            torch.compiler.npugraph_mark_step_begin()
        with autocast_context():
            preds = callable_model(images[step % len(images)])
            loss, _ = model.loss(batch, preds)
            total = loss.sum()
        total.backward()
        optimizer.step()
        if args.state_output:
            # 图输出可能复用静态存储；异步克隆每个标量，将 D2H 延迟到计时结束后。
            loss_trace_device.append(total.detach().clone())
        return total.detach()

    allocated_before_steps = torch.npu.memory_allocated(device)
    reserved_before_steps = torch.npu.memory_reserved(device)
    torch.npu.reset_peak_memory_stats(device)
    torch.npu.synchronize(device)
    first_start = time.perf_counter()
    first_loss = train_step(0)
    torch.npu.synchronize(device)
    first_step_ms = (time.perf_counter() - first_start) * 1000

    for step in range(1, args.warmup_steps + 1):
        train_step(step)
    torch.npu.synchronize(device)

    block_ms = []
    final_loss = first_loss
    for repeat in range(args.repeats):
        torch.npu.synchronize(device)
        start = time.perf_counter()
        for index in range(args.measure_steps):
            final_loss = train_step(args.warmup_steps + 1 + repeat * args.measure_steps + index)
        torch.npu.synchronize(device)
        block_ms.append((time.perf_counter() - start) * 1000 / args.measure_steps)

    counters = {
        group: {str(key): int(value) for key, value in counter.items()}
        for group, counter in torch._dynamo.utils.counters.items()
        if counter
    }
    result = {
        "状态": "成功",
        "方案": args.scheme,
        "模型": f"YOLO11-{args.scale.upper()}",
        "参数量": sum(parameter.numel() for parameter in model.parameters()),
        "批量": args.batch,
        "图像尺寸": args.imgsz,
        "精度": args.dtype,
        "TASK_QUEUE_ENABLE": args.task_queue,
        "fullgraph": args.fullgraph,
        "setup_ms": setup_ms,
        "first_step_ms": first_step_ms,
        "median_step_ms": statistics.median(block_ms),
        "mean_step_ms": statistics.mean(block_ms),
        "min_step_ms": min(block_ms),
        "max_step_ms": max(block_ms),
        "最终损失": float(final_loss.cpu()),
        "步骤前已分配显存_MiB": allocated_before_steps / 2**20,
        "步骤前已保留显存_MiB": reserved_before_steps / 2**20,
        "峰值已分配显存_MiB": torch.npu.max_memory_allocated(device) / 2**20,
        "峰值已保留显存_MiB": torch.npu.max_memory_reserved(device) / 2**20,
        "结束时已分配显存_MiB": torch.npu.memory_allocated(device) / 2**20,
        "结束时已保留显存_MiB": torch.npu.memory_reserved(device) / 2**20,
        "计时块_ms": block_ms,
        "dynamo_counters": counters,
    }
    if args.memory_usage_output_dir:
        torch_npu._C._npu_saveDevMemUsageInfo(args.device)
    if args.state_output:
        loss_trace = [float(loss.cpu()) for loss in loss_trace_device]

        def cpu_tree(value):
            """递归复制状态到 CPU。"""
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().clone()
            if isinstance(value, dict):
                return {key: cpu_tree(item) for key, item in value.items()}
            if isinstance(value, list):
                return [cpu_tree(item) for item in value]
            if isinstance(value, tuple):
                return tuple(cpu_tree(item) for item in value)
            return value

        state = {
            "losses": loss_trace,
            "model": cpu_tree(model.state_dict()),
            "grads": {name: cpu_tree(parameter.grad) for name, parameter in model.named_parameters()},
            "optimizer": cpu_tree(optimizer.state_dict()),
        }
        torch.save(state, args.state_output)
        result["loss_trace"] = loss_trace
        result["计时说明"] = "state-output 每 step 增加异步 clone，仅用于正确性对照，不与正式性能结果比较"
    return result


if __name__ == "__main__":
    parsed = parse_args()
    output = main(parsed)
    text = json.dumps(output, ensure_ascii=False, indent=2)
    print(text)
    if parsed.output:
        parsed.output.write_text(text + "\n")
