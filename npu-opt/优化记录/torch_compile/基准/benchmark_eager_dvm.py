# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""测量 Conv 或 FFN+LayerNorm 完整训练 step 的 Eager DVM 性能。"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from contextlib import nullcontext
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """在导入 TorchNPU 前解析并固定 DVM 环境。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=("conv_bn_relu", "vit_gelu", "transformer_swiglu"), required=True)
    parser.add_argument("--lazy-fusion", choices=("off", "on"), required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--task-queue", choices=(0, 1), type=int, default=1)
    parser.add_argument("--dtype", choices=("fp32", "amp_bf16"), default="amp_bf16")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-output", type=Path)
    return parser.parse_args()


def main(args: argparse.Namespace) -> dict:
    """运行一个独立 Eager/DVM 训练进程。"""
    os.environ["TASK_QUEUE_ENABLE"] = str(args.task_queue)
    os.environ["TORCH_NPU_LAZY_FUSION"] = "True" if args.lazy_fusion == "on" else "False"

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch_npu

    dvm_binding = hasattr(torch_npu._C, "dvm")
    if args.lazy_fusion == "on" and not dvm_binding:
        raise RuntimeError("当前 torch_npu wheel 未包含 Eager DVM binding，不能执行 --lazy-fusion on")

    class ConvBNReLU(nn.Module):
        """标准卷积训练块。"""

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(64, 64, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=False)

        def forward(self, value):
            return self.relu(self.bn(self.conv(value)))

    class VitGeluMlp(nn.Module):
        """ViT GELU MLP。"""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(768)
            self.fc1 = nn.Linear(768, 3072)
            self.fc2 = nn.Linear(3072, 768)

        def forward(self, value):
            return value + self.fc2(F.gelu(self.fc1(self.norm(value)), approximate="tanh"))

    class TransformerSwiGlu(nn.Module):
        """Transformer SwiGLU FFN。"""

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(1024)
            self.gate = nn.Linear(1024, 2816, bias=False)
            self.up = nn.Linear(1024, 2816, bias=False)
            self.down = nn.Linear(2816, 1024, bias=False)

        def forward(self, value):
            normalized = self.norm(value)
            return value + self.down(F.silu(self.gate(normalized)) * self.up(normalized))

    configs = {
        "conv_bn_relu": (ConvBNReLU, (32, 64, 56, 56), 1e-3),
        "vit_gelu": (VitGeluMlp, (32, 196, 768), 1e-4),
        "transformer_swiglu": (TransformerSwiGlu, (4, 512, 1024), 1e-4),
    }
    model_cls, shape, lr = configs[args.workload]
    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    torch.npu.manual_seed(args.seed)
    device = torch.device(f"npu:{args.device}")
    model = model_cls().to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, foreach=False)
    generator = torch.Generator().manual_seed(args.seed + 1)
    inputs = [torch.randn(shape, generator=generator).to(device) for _ in range(2)]

    def autocast_context():
        return (
            torch.autocast("npu", dtype=torch.bfloat16, cache_enabled=False)
            if args.dtype == "amp_bf16"
            else nullcontext()
        )

    def train_step(index: int):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            output = model(inputs[index % len(inputs)])
            loss = output.float().square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach(), output.detach()

    torch.npu.reset_peak_memory_stats(device)
    torch.npu.synchronize(device)
    start = time.perf_counter()
    first_loss, first_output = train_step(0)
    torch.npu.synchronize(device)
    first_step_ms = (time.perf_counter() - start) * 1000

    for index in range(1, args.warmup_steps + 1):
        train_step(index)
    torch.npu.synchronize(device)

    block_ms = []
    final_loss, final_output = first_loss, first_output
    for repeat in range(args.repeats):
        torch.npu.synchronize(device)
        start = time.perf_counter()
        for index in range(args.measure_steps):
            step = args.warmup_steps + 1 + repeat * args.measure_steps + index
            final_loss, final_output = train_step(step)
        torch.npu.synchronize(device)
        block_ms.append((time.perf_counter() - start) * 1000 / args.measure_steps)

    result = {
        "状态": "成功",
        "workload": args.workload,
        "shape": shape,
        "dtype": args.dtype,
        "lazy_fusion": args.lazy_fusion,
        "TASK_QUEUE_ENABLE": args.task_queue,
        "逻辑device": args.device,
        "ASCEND_RT_VISIBLE_DEVICES": os.getenv("ASCEND_RT_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_git": torch_npu.version.git_version,
        "dvm_binding": dvm_binding,
        "DVM实际启用": args.lazy_fusion == "on" and dvm_binding,
        "seed": args.seed,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "repeats": args.repeats,
        "first_step_ms": first_step_ms,
        "median_step_ms": statistics.median(block_ms),
        "mean_step_ms": statistics.fmean(block_ms),
        "min_step_ms": min(block_ms),
        "max_step_ms": max(block_ms),
        "计时块_ms": block_ms,
        "最终损失": float(final_loss.cpu()),
        "输出dtype": str(final_output.dtype),
        "峰值已分配显存_MiB": torch.npu.max_memory_allocated(device) / 2**20,
    }
    if args.state_output:
        parameter_names = {parameter: name for name, parameter in model.named_parameters()}
        torch.save(
            {
                "loss": final_loss.cpu(),
                "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "grads": {
                    name: None if parameter.grad is None else parameter.grad.detach().cpu()
                    for name, parameter in model.named_parameters()
                },
                "optimizer": {
                    parameter_names[parameter]: {
                        key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                        for key, value in state.items()
                    }
                    for parameter, state in optimizer.state.items()
                },
            },
            args.state_output,
        )
    return result


if __name__ == "__main__":
    parsed = parse_args()
    output = main(parsed)
    serialized = json.dumps(output, ensure_ascii=False, indent=2)
    print(serialized)
    parsed.output.write_text(serialized + "\n")
