"""在 Ascend NPU 上比较 DINOv3 RoPE 原实现与自动路由实现。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import types
from pathlib import Path

import torch

from ultralytics.nn.modules.third_party.dinov3.dinov3.hub.backbones import (
    dinov3_vitb16,
    dinov3_vitl16,
    dinov3_vits16,
)
from ultralytics.nn.modules.third_party.dinov3.dinov3.layers import attention as attention_module


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _baseline_rope_bsnd(
    q: torch.Tensor,
    k: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """复现优化前的 BNSD 切片、双 RotaryMul 和 prefix 拼接。"""
    import torch_npu

    q_dtype, k_dtype = q.dtype, k.dtype
    q = q.transpose(1, 2).to(dtype=sin.dtype)
    k = k.transpose(1, 2).to(dtype=sin.dtype)
    prefix = q.shape[-2] - sin.shape[-2]
    sin = sin.view(1, 1, sin.shape[-2], sin.shape[-1])
    cos = cos.view(1, 1, cos.shape[-2], cos.shape[-1])
    q_rotated = torch_npu.npu_rotary_mul(input=q[:, :, prefix:], r1=cos, r2=sin, rotary_mode="half")
    k_rotated = torch_npu.npu_rotary_mul(input=k[:, :, prefix:], r1=cos, r2=sin, rotary_mode="half")
    q = torch.cat((q[:, :, :prefix], q_rotated), dim=2).to(dtype=q_dtype)
    k = torch.cat((k[:, :, :prefix], k_rotated), dim=2).to(dtype=k_dtype)
    return q.transpose(1, 2), k.transpose(1, 2)


def _synchronize() -> None:
    torch.npu.synchronize()


def _measure(fn, *, warmup: int, iterations: int, repeats: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    _synchronize()
    samples = []
    peaks = []
    for _ in range(repeats):
        torch.npu.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        _synchronize()
        samples.append((time.perf_counter() - start) * 1000 / iterations)
        peaks.append(torch.npu.max_memory_allocated() / 1024**2)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "peak_allocated_mib": statistics.median(peaks),
    }


def _speedup(baseline: dict[str, float], optimized: dict[str, float]) -> float:
    return baseline["median_ms"] / optimized["median_ms"]


def benchmark_rope_subgraph(args) -> dict:
    device = torch.device("npu:0")
    batch, sequence, heads, head_dim, prefix = (
        args.batch,
        args.sequence,
        args.heads,
        args.head_dim,
        args.prefix,
    )
    qkv = torch.randn(batch, sequence, 3, heads, head_dim, dtype=torch.float16, device=device)
    q, k, _ = qkv.unbind(2)
    angles = torch.randn(sequence - prefix, head_dim, dtype=torch.float32, device=device)
    sin, cos = angles.sin(), angles.cos()
    attn = attention_module.SelfAttention(dim=heads * head_dim, num_heads=heads).to(device)

    def baseline_forward():
        return _baseline_rope_bsnd(q, k, sin, cos)

    def optimized_inference():
        return attn.apply_rope_bsnd(q, k, (sin, cos))

    old_policy = attention_module.DINOV3_ROPE_BACKEND
    with torch.inference_mode():
        attention_module.DINOV3_ROPE_BACKEND = "auto"
        baseline_infer = _measure(
            baseline_forward, warmup=args.warmup, iterations=args.iterations, repeats=args.repeats
        )
        optimized_infer = _measure(
            optimized_inference, warmup=args.warmup, iterations=args.iterations, repeats=args.repeats
        )

    q_base = q.detach().float()
    k_base = k.detach().float()

    def baseline_training():
        q_train = q_base.detach().requires_grad_(True)
        k_train = k_base.detach().requires_grad_(True)
        q_out, k_out = _baseline_rope_bsnd(q_train, k_train, sin, cos)
        (q_out.square().mean() + k_out.square().mean()).backward()

    def optimized_training():
        q_train = q_base.detach().requires_grad_(True)
        k_train = k_base.detach().requires_grad_(True)
        q_out, k_out = attn.apply_rope_bsnd(q_train, k_train, (sin, cos))
        (q_out.square().mean() + k_out.square().mean()).backward()

    attention_module.DINOV3_ROPE_BACKEND = "auto"
    baseline_train = _measure(
        baseline_training,
        warmup=max(2, args.warmup // 4),
        iterations=max(5, args.iterations // 5),
        repeats=args.repeats,
    )
    optimized_train = _measure(
        optimized_training,
        warmup=max(2, args.warmup // 4),
        iterations=max(5, args.iterations // 5),
        repeats=args.repeats,
    )
    attention_module.DINOV3_ROPE_BACKEND = old_policy

    return {
        "shape": {"B": batch, "S": sequence, "N": heads, "D": head_dim, "prefix": prefix},
        "inference": {
            "baseline": baseline_infer,
            "optimized": optimized_infer,
            "speedup": _speedup(baseline_infer, optimized_infer),
        },
        "training_forward_backward": {
            "baseline": baseline_train,
            "optimized": optimized_train,
            "speedup": _speedup(baseline_train, optimized_train),
        },
    }


def _build_model(variant: str):
    builders = {"s": dinov3_vits16, "b": dinov3_vitb16, "l": dinov3_vitl16}
    model = builders[variant](pretrained=False, weights=None)
    # 与项目 DINOv3ViT wrapper 的默认行为一致。
    model.rope_embed.rescale_coords = None
    return model


def _install_baseline_model_rope(model) -> tuple[list, object]:
    saved_methods = []
    prefix = model.n_storage_tokens + 1

    def baseline_apply(self, q, k, rope):
        sin, cos = rope
        # 优化后的生成器已经带 identity prefix；基线复现旧实现时去掉它。
        return _baseline_rope_bsnd(q, k, sin[..., prefix:, :], cos[..., prefix:, :])

    for block in model.blocks:
        saved_methods.append(block.attn.apply_rope_bsnd)
        block.attn.apply_rope_bsnd = types.MethodType(baseline_apply, block.attn)
    saved_stochastic = model._rope_is_stochastic
    # 旧实现在 eval 和确定性训练中也会逐 block 重算 sin/cos。
    model._rope_is_stochastic = types.MethodType(lambda self: True, model)
    return saved_methods, saved_stochastic


def _restore_optimized_model_rope(model, saved_methods, saved_stochastic) -> None:
    for block, method in zip(model.blocks, saved_methods):
        block.attn.apply_rope_bsnd = method
    model._rope_is_stochastic = saved_stochastic


def benchmark_full_model(args) -> dict:
    device = torch.device("npu:0")
    model = _build_model(args.variant).to(device)
    image = torch.randn(args.full_batch, 3, args.image_size, args.image_size, device=device)
    old_policy = attention_module.DINOV3_ROPE_BACKEND

    def inference_step():
        with torch.autocast(device_type="npu", dtype=torch.float16):
            return model.forward_features(image)["x_norm_patchtokens"]

    model.eval()
    saved_methods, saved_stochastic = _install_baseline_model_rope(model)
    with torch.inference_mode():
        baseline_infer = _measure(
            inference_step,
            warmup=args.full_warmup,
            iterations=args.full_iterations,
            repeats=args.repeats,
        )
    _restore_optimized_model_rope(model, saved_methods, saved_stochastic)
    attention_module.DINOV3_ROPE_BACKEND = "auto"
    with torch.inference_mode():
        optimized_infer = _measure(
            inference_step,
            warmup=args.full_warmup,
            iterations=args.full_iterations,
            repeats=args.repeats,
        )

    training_result = None
    if not args.skip_full_training:

        def training_step():
            model.zero_grad(set_to_none=True)
            with torch.autocast(device_type="npu", dtype=torch.float16):
                output = model.forward_features(image)["x_norm_patchtokens"]
                loss = output.float().square().mean()
            loss.backward()

        model.train()
        saved_methods, saved_stochastic = _install_baseline_model_rope(model)
        baseline_train = _measure(
            training_step,
            warmup=1,
            iterations=args.full_train_iterations,
            repeats=args.repeats,
        )
        _restore_optimized_model_rope(model, saved_methods, saved_stochastic)
        attention_module.DINOV3_ROPE_BACKEND = "auto"
        optimized_train = _measure(
            training_step,
            warmup=1,
            iterations=args.full_train_iterations,
            repeats=args.repeats,
        )
        training_result = {
            "baseline": baseline_train,
            "optimized": optimized_train,
            "speedup": _speedup(baseline_train, optimized_train),
        }
    attention_module.DINOV3_ROPE_BACKEND = old_policy

    return {
        "variant": args.variant,
        "image_shape": [args.full_batch, 3, args.image_size, args.image_size],
        "inference": {
            "baseline": baseline_infer,
            "optimized": optimized_infer,
            "speedup": _speedup(baseline_infer, optimized_infer),
        },
        "training_step": training_result,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--sequence", type=int, default=1605)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--prefix", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--full-warmup", type=int, default=3)
    parser.add_argument("--full-iterations", type=int, default=10)
    parser.add_argument("--full-train-iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--variant", choices=("s", "b", "l"), default="l")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--full-batch", type=int, default=1)
    parser.add_argument("--skip-full-model", action="store_true")
    parser.add_argument("--skip-full-training", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    result = {
        "environment": {
            "device": torch.npu.get_device_name(args.device),
            "torch": torch.__version__,
            "torch_npu": __import__("torch_npu").__version__,
            "jit_compile": False,
        },
        "rope_subgraph": benchmark_rope_subgraph(args),
    }
    if not args.skip_full_model:
        result["full_model"] = benchmark_full_model(args)
    output = json.dumps(result, ensure_ascii=False, indent=2)
    print(output)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
