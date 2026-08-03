#!/usr/bin/env python3
"""测量 Transformer/ViT FFN+LayerNorm 的 Eager DVM 完整训练 step。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_npu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("vit_gelu", "transformer_swiglu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-output", type=Path)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


class VitGeluMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fc1 = nn.Linear(768, 3072, bias=True)
        self.fc2 = nn.Linear(3072, 768, bias=True)

    def forward(self, value):
        residual = value
        value = self.norm(value)
        value = self.fc2(F.gelu(self.fc1(value), approximate="tanh"))
        return residual + value


class TransformerSwiGlu(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(1024)
        self.gate = nn.Linear(1024, 2816, bias=False)
        self.up = nn.Linear(1024, 2816, bias=False)
        self.down = nn.Linear(2816, 1024, bias=False)

    def forward(self, value):
        residual = value
        value = self.norm(value)
        value = self.down(F.silu(self.gate(value)) * self.up(value))
        return residual + value


def main():
    args = parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(args.seed)
    torch.npu.manual_seed(args.seed)
    device = torch.device("npu:0")
    if args.variant == "vit_gelu":
        model = VitGeluMlp().to(device).train()
        shape = (32, 196, 768)
    else:
        model = TransformerSwiGlu().to(device).train()
        shape = (4, 512, 1024)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, foreach=False)
    generator = torch.Generator().manual_seed(args.seed + 1)
    inputs = [torch.randn(shape, generator=generator).to(device) for _ in range(2)]

    def step(index):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("npu", dtype=torch.bfloat16, cache_enabled=False):
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
        "variant": args.variant,
        "shape": list(shape),
        "dtype": "amp_bf16",
        "optimizer": "SGD(momentum=0.9, foreach=False)",
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
    if args.state_output:
        names = {parameter: name for name, parameter in model.named_parameters()}
        torch.save(
            {
                "loss": last_loss.detach().cpu(),
                "state": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "gradients": {
                    name: None if parameter.grad is None else parameter.grad.detach().cpu()
                    for name, parameter in model.named_parameters()
                },
                "optimizer": {
                    names[parameter]: {
                        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                        for key, value in state.items()
                    }
                    for parameter, state in optimizer.state.items()
                },
            },
            args.state_output,
        )
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized, flush=True)
    args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
