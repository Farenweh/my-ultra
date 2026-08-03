#!/usr/bin/env python3
"""测量单个 Eager 或 Eager-DVM Conv-BN-ReLU 完整训练 step。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=("fp32", "amp_bf16"), default="amp_bf16")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


class ConvBNReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, value):
        return self.relu(self.bn(self.conv(value)))


def main():
    args = parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    device = torch.device("npu:0")
    model = ConvBNReLU().to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, foreach=False)
    generator = torch.Generator().manual_seed(args.seed + 1)
    inputs = [torch.randn(32, 64, 56, 56, generator=generator).to(device) for _ in range(2)]

    def autocast_context():
        if args.dtype == "amp_bf16":
            return torch.autocast("npu", dtype=torch.bfloat16, cache_enabled=False)
        return nullcontext()

    def step(index):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            output = model(inputs[index % len(inputs)])
            loss = output.float().square().mean()
        loss.backward()
        optimizer.step()
        return loss, output

    torch.npu.reset_peak_memory_stats()
    torch.npu.synchronize()
    start = time.perf_counter()
    first_loss, first_output = step(0)
    torch.npu.synchronize()
    first_step_ms = (time.perf_counter() - start) * 1000

    for index in range(args.warmup_steps):
        step(index)
    torch.npu.synchronize()

    samples_ms = []
    last_loss, last_output = first_loss, first_output
    for repeat in range(args.repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        for index in range(args.measure_steps):
            last_loss, last_output = step(index + repeat * args.measure_steps)
        torch.npu.synchronize()
        samples_ms.append((time.perf_counter() - start) * 1000 / args.measure_steps)

    result = {
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_git": torch_npu.version.git_version,
        "device": torch.npu.get_device_name(0),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE"),
        "lazy_fusion": os.environ.get("TORCH_NPU_LAZY_FUSION"),
        "dvm_binding": hasattr(torch_npu._C, "dvm"),
        "dtype": args.dtype,
        "shape": [32, 64, 56, 56],
        "first_step_ms": first_step_ms,
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "last_loss": float(last_loss.detach().cpu()),
        "output_dtype": str(last_output.dtype),
        "peak_memory_mib": torch.npu.max_memory_allocated() / 1024**2,
    }
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized, flush=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
