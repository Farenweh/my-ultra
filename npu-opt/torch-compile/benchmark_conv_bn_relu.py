"""在昇腾 NPU 上测试不同 torch.compile/图化方案的 Conv-BN-ReLU 训练性能。"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path


SCHEMES = (
    "eager",
    "aot_eager",
    "inductor",
    "inductor_reduce_overhead",
    "inductor_mlir",
    "inductor_dvm",
    "npugraphs",
    "npugraph_ex",
    "torchair_ge",
    "make_graphed_callables",
)


def parse_args() -> argparse.Namespace:
    """在导入 torch 前解析命令行参数，使 backend 环境变量生效。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=SCHEMES, required=True)
    parser.add_argument("--device", type=int, default=0, help="NPU 逻辑设备索引")
    parser.add_argument("--shape", default="32,64,56,56", help="N,C,H,W")
    parser.add_argument("--out-channels", type=int, default=64)
    parser.add_argument("--input-variants", type=int, default=1, help="计时时轮换的同 shape NPU 输入张量数")
    parser.add_argument("--input-offset", type=float, default=0.0, help="第 i 个输入额外加 i 倍该偏移")
    parser.add_argument(
        "--reuse-capture-input",
        action="store_true",
        help="仅用于静态缓冲上限测试：将捕获样例张量直接作为唯一运行时输入",
    )
    parser.add_argument("--dtype", choices=("fp32", "amp_fp16", "amp_bf16"), default="amp_bf16")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--correctness-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=10)
    zero_grad_group = parser.add_mutually_exclusive_group()
    zero_grad_group.add_argument("--zero-grad-set-to-none", dest="zero_grad_set_to_none", action="store_true")
    zero_grad_group.add_argument("--no-zero-grad-set-to-none", dest="zero_grad_set_to_none", action="store_false")
    parser.set_defaults(zero_grad_set_to_none=True)
    fullgraph_group = parser.add_mutually_exclusive_group()
    fullgraph_group.add_argument("--fullgraph", dest="fullgraph", action="store_true")
    fullgraph_group.add_argument("--no-fullgraph", dest="fullgraph", action="store_false")
    parser.set_defaults(fullgraph=True)
    parser.add_argument(
        "--cantsplit-compat",
        action="store_true",
        help="仅用于验证 torch_npu 2.12 无参 CantSplit 与 torch 2.12 两参接口不匹配的进程内 workaround",
    )
    parser.add_argument("--phase", choices=("correctness", "performance", "both"), default="both")
    parser.add_argument("--atol", type=float, help="正确性判据绝对容差；默认按 dtype 选择")
    parser.add_argument("--rtol", type=float, help="正确性判据相对容差；默认按 dtype 选择")
    parser.add_argument("--state-atol", type=float, help="FP32 训练状态判据绝对容差；默认按 dtype 选择")
    parser.add_argument("--state-rtol", type=float, help="FP32 训练状态判据相对容差；默认按 dtype 选择")
    parser.add_argument(
        "--performance-order",
        choices=("eager_first", "candidate_first"),
        default="eager_first",
        help="配对性能阶段的执行顺序，多次复测应同时覆盖 AB/BA",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def distribution_version(name: str) -> str | None:
    """返回已安装 distribution 版本，未安装时返回 None。"""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def percentile(values: list[float], q: float) -> float:
    """返回非空列表线性插值后的分位数。"""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def timing_summary(block_ms: list[float], batch_size: int) -> dict:
    """汇总以每训练步毫秒表示的同步计时 block。"""
    mean = statistics.fmean(block_ms)
    std = statistics.stdev(block_ms) if len(block_ms) > 1 else 0.0
    median = statistics.median(block_ms)
    return {
        "block_ms_per_step": block_ms,
        "mean_ms_per_step": mean,
        "median_ms_per_step": median,
        "p25_ms_per_step": percentile(block_ms, 0.25),
        "p75_ms_per_step": percentile(block_ms, 0.75),
        "p90_ms_per_step": percentile(block_ms, 0.90),
        "cv": std / mean if mean else 0.0,
        "images_per_second": batch_size * 1000.0 / median,
    }


def clone_state(module) -> dict:
    """将模块状态字典克隆到 CPU。"""
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def clone_gradients(module) -> dict:
    """将参数梯度克隆到 CPU。"""
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }


def clone_optimizer_state(optimizer, module) -> dict:
    """用稳定参数名克隆 optimizer tensor 状态，避免使用进程内 ID。"""
    names = {parameter: name for name, parameter in module.named_parameters()}
    state = {}
    for parameter, values in optimizer.state.items():
        state[names[parameter]] = {
            key: value.detach().cpu().clone() if hasattr(value, "detach") else value for key, value in values.items()
        }
    return state


def tensor_difference(reference, candidate, atol: float, rtol: float) -> dict:
    """计算两个 tensor 的有限性、绝对误差、相对误差和 RMSE。"""
    import torch

    if reference is None or candidate is None:
        return {"same_presence": reference is candidate}
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {
            "same_presence": True,
            "same_shape": reference.shape == candidate.shape,
            "same_dtype": reference.dtype == candidate.dtype,
        }
    if not (reference.dtype.is_floating_point or reference.dtype.is_complex):
        return {
            "same_presence": True,
            "exact": bool(torch.equal(reference, candidate)),
            "reference": reference.tolist() if reference.numel() <= 8 else None,
            "candidate": candidate.tolist() if candidate.numel() <= 8 else None,
        }
    ref = reference.float()
    cand = candidate.float()
    absolute = (ref - cand).abs()
    relative = absolute / ref.abs().clamp_min(1e-12)
    return {
        "same_presence": True,
        "finite": bool(torch.isfinite(cand).all()),
        "close": bool(torch.allclose(ref, cand, atol=atol, rtol=rtol)),
        "max_abs": float(absolute.max()),
        "max_rel": float(relative.max()),
        "rmse": float(torch.sqrt(torch.mean((ref - cand).square()))),
    }


def nested_differences(reference: dict, candidate: dict, atol: float, rtol: float) -> dict:
    """比较平铺 tensor 字典或只嵌套一层的 optimizer 字典。"""
    differences = {}
    for name in sorted(reference.keys() | candidate.keys()):
        if name not in reference or name not in candidate:
            differences[name] = {"missing": "reference" if name not in reference else "candidate"}
            continue
        ref, cand = reference[name], candidate[name]
        if isinstance(ref, dict) and isinstance(cand, dict):
            differences[name] = {
                key: tensor_difference(ref.get(key), cand.get(key), atol, rtol)
                for key in sorted(ref.keys() | cand.keys())
            }
        else:
            differences[name] = tensor_difference(ref, cand, atol, rtol)
    return differences


def worst_float_difference(differences: dict) -> tuple[float, float]:
    """返回嵌套差异映射中最大的绝对误差和 RMSE。"""
    max_abs = 0.0
    max_rmse = 0.0
    stack = [differences]
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            if "max_abs" in value:
                if not math.isfinite(value["max_abs"]) or not math.isfinite(value["rmse"]):
                    return math.inf, math.inf
                max_abs = max(max_abs, value["max_abs"])
                max_rmse = max(max_rmse, value["rmse"])
            else:
                stack.extend(value.values())
    return max_abs, max_rmse


def difference_issues(differences: dict, prefix: str = "") -> list[str]:
    """收集缺失、结构不匹配、非有限或非精确整数差异。"""
    issues = []
    for name, value in differences.items():
        path = f"{prefix}.{name}" if prefix else str(name)
        if not isinstance(value, dict):
            continue
        if "missing" in value:
            issues.append(f"{path}: missing from {value['missing']}")
            continue
        for key in ("same_presence", "same_shape", "same_dtype", "finite", "close", "exact"):
            if key in value and value[key] is not True:
                issues.append(f"{path}: {key}={value[key]!r}")
        if "max_abs" in value and (
            not math.isfinite(value["max_abs"]) or not math.isfinite(value.get("rmse", math.nan))
        ):
            issues.append(f"{path}: non-finite difference")
        elif "max_abs" not in value and "exact" not in value:
            issues.extend(difference_issues(value, path))
    return issues


def dynamo_counters() -> dict:
    """将 TorchDynamo counters 转为可 JSON 序列化的字典。"""
    import torch

    return {
        group: {str(key): int(value) for key, value in counter.items()}
        for group, counter in torch._dynamo.utils.counters.items()
        if counter
    }


def compile_model(model, sample_input, scheme: str, fullgraph: bool):
    """使用指定图或编译 backend 包装模型。"""
    import torch
    import torch_npu

    if scheme == "eager":
        return model
    if scheme == "make_graphed_callables":
        return torch_npu.npu.make_graphed_callables(model, (sample_input,))

    kwargs = {"fullgraph": fullgraph, "dynamic": False}
    if scheme == "aot_eager":
        kwargs["backend"] = "aot_eager"
    elif scheme == "inductor":
        kwargs["backend"] = "inductor"
    elif scheme == "inductor_reduce_overhead":
        kwargs.update(backend="inductor", mode="reduce-overhead")
    elif scheme == "inductor_mlir":
        kwargs.update(backend="inductor", options={"npu_backend": "mlir"})
    elif scheme == "inductor_dvm":
        kwargs.update(backend="inductor", options={"npu_backend": "dvm"})
    elif scheme in {"npugraphs", "npugraph_ex"}:
        kwargs["backend"] = scheme
    elif scheme == "torchair_ge":
        import torchair

        kwargs["backend"] = torchair.get_npu_backend(compiler_config=torchair.CompilerConfig())
    else:
        raise ValueError(f"未知方案：{scheme}")
    return torch.compile(model, **kwargs)


def main(args: argparse.Namespace) -> dict:
    """运行正确性和/或配对性能测量。"""
    if args.scheme == "inductor_mlir":
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "mlir"
    elif args.scheme == "inductor_dvm":
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "dvm"
    elif args.scheme.startswith("inductor"):
        os.environ["TORCHINDUCTOR_NPU_BACKEND"] = "default"

    import torch
    import torch.nn as nn
    import torch_npu

    try:
        import triton

        triton_runtime = getattr(triton, "__version__", None)
    except ImportError:
        triton_runtime = None

    if args.cantsplit_compat:
        from torch._inductor.codegen.simd import CantSplit

        original_cantsplit_init = CantSplit.__init__

        def compatible_cantsplit_init(self, expr=None, remaining=None):
            original_cantsplit_init(self, expr, remaining)

        CantSplit.__init__ = compatible_cantsplit_init

    class ConvBNReLU(nn.Module):
        """标准的未融合训练块。"""

        def __init__(self, in_channels: int, out_channels: int):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=False)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    shape = tuple(int(value) for value in args.shape.split(","))
    if len(shape) != 4 or any(value <= 0 for value in shape):
        raise ValueError(f"--shape 必须是四个正整数，当前为 {args.shape!r}")
    if shape[1] <= 0 or args.out_channels <= 0:
        raise ValueError("输入和输出通道数必须为正整数")
    if args.input_variants <= 0:
        raise ValueError("--input-variants 必须为正整数")
    if args.reuse_capture_input and args.input_variants != 1:
        raise ValueError("--reuse-capture-input 只能与 --input-variants 1 一起使用")
    if args.device < 0:
        raise ValueError("--device 不能为负数")
    if args.phase in {"correctness", "both"} and args.correctness_steps <= 0:
        raise ValueError("--correctness-steps 必须为正整数")
    if args.phase in {"performance", "both"} and args.warmup_steps < 0:
        raise ValueError("--warmup-steps 不能为负数")
    if args.phase in {"performance", "both"} and args.measure_steps <= 0:
        raise ValueError("--measure-steps 必须为正整数")
    if args.phase in {"performance", "both"} and args.repeats <= 0:
        raise ValueError("--repeats 必须为正整数")
    if not math.isfinite(args.input_offset):
        raise ValueError("--input-offset 必须是有限数")
    if not math.isfinite(args.lr) or args.lr < 0:
        raise ValueError("--lr 必须是有限非负数")
    if not math.isfinite(args.momentum) or args.momentum < 0:
        raise ValueError("--momentum 必须是有限非负数")
    for name in ("atol", "rtol", "state_atol", "state_rtol"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"--{name.replace('_', '-')} 必须是有限非负数")

    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    torch.npu.manual_seed(args.seed)
    device = torch.device(f"npu:{args.device}")
    amp_dtype = {"amp_fp16": torch.float16, "amp_bf16": torch.bfloat16}.get(args.dtype)
    default_tolerances = {"fp32": (1e-5, 1e-4), "amp_fp16": (1e-4, 1e-3), "amp_bf16": (1e-4, 1e-3)}
    default_state_tolerances = {
        "fp32": (1e-5, 1e-4),
        "amp_fp16": (1e-5, 1e-3),
        "amp_bf16": (1e-4, 1e-2),
    }
    default_atol, default_rtol = default_tolerances[args.dtype]
    default_state_atol, default_state_rtol = default_state_tolerances[args.dtype]
    atol = default_atol if args.atol is None else args.atol
    rtol = default_rtol if args.rtol is None else args.rtol
    state_atol = default_state_atol if args.state_atol is None else args.state_atol
    state_rtol = default_state_rtol if args.state_rtol is None else args.state_rtol
    expected_output_dtype = str(amp_dtype or torch.float32)

    master = ConvBNReLU(shape[1], args.out_channels)
    initial_state = clone_state(master)
    cpu_generator = torch.Generator().manual_seed(args.seed + 1)
    runtime_inputs = [
        (torch.randn(shape, generator=cpu_generator) + index * args.input_offset).to(device)
        for index in range(args.input_variants)
    ]
    sample_input = runtime_inputs[0] if args.reuse_capture_input else runtime_inputs[0].clone()

    def autocast_context():
        return nullcontext() if amp_dtype is None else torch.autocast("npu", dtype=amp_dtype, cache_enabled=False)

    def wrap_model(model, scheme: str):
        context = autocast_context() if scheme == "make_graphed_callables" else nullcontext()
        with context:
            return compile_model(model, sample_input, scheme, args.fullgraph)

    def make_model():
        model = ConvBNReLU(shape[1], args.out_channels).to(device)
        model.load_state_dict(initial_state)
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0, foreach=False
        )
        return model, optimizer

    def release_npu_objects():
        """确定性回收图闭包的循环引用，并清理已经释放的 NPU allocation。"""
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize(device)

    def mark_step(scheme: str):
        if scheme in {"inductor_reduce_overhead", "npugraphs"} and hasattr(torch.compiler, "npugraph_mark_step_begin"):
            torch.compiler.npugraph_mark_step_begin()

    def train_step(callable_model, optimizer, scheme: str, step: int):
        optimizer.zero_grad(set_to_none=args.zero_grad_set_to_none)
        mark_step(scheme)
        with autocast_context():
            output = callable_model(runtime_inputs[step % len(runtime_inputs)])
            loss = output.float().square().mean()
        loss.backward()
        optimizer.step()
        return loss.detach(), str(output.dtype)

    def run_correctness(scheme: str) -> dict:
        model, optimizer = make_model()
        torch.npu.synchronize(device)
        setup_start = time.perf_counter()
        callable_model = wrap_model(model, scheme)
        if scheme == "make_graphed_callables":
            model.load_state_dict(initial_state)
            optimizer.zero_grad(set_to_none=True)
        torch.npu.synchronize(device)
        setup_ms = (time.perf_counter() - setup_start) * 1000.0

        losses = []
        output_dtypes = []
        first_step_ms = None
        for step in range(args.correctness_steps):
            torch.npu.synchronize(device)
            start = time.perf_counter()
            loss, output_dtype = train_step(callable_model, optimizer, scheme, step)
            torch.npu.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if step == 0:
                first_step_ms = elapsed_ms
            losses.append(float(loss.cpu()))
            output_dtypes.append(output_dtype)

        result = {
            "setup_ms": setup_ms,
            "first_train_step_ms": first_step_ms,
            "losses": losses,
            "output_dtypes": output_dtypes,
            "all_losses_finite": all(math.isfinite(loss) for loss in losses),
            "state": clone_state(model),
            "gradients": clone_gradients(model),
            "optimizer_state": clone_optimizer_state(optimizer, model),
            "dynamo_counters": dynamo_counters(),
        }
        del callable_model, optimizer, model
        release_npu_objects()
        return result

    def strip_tensors(run: dict) -> dict:
        return {key: value for key, value in run.items() if key not in {"state", "gradients", "optimizer_state"}}

    def run_performance(scheme: str) -> dict:
        release_npu_objects()
        torch.npu.reset_peak_memory_stats(device)
        model, optimizer = make_model()
        torch.npu.synchronize(device)
        setup_start = time.perf_counter()
        callable_model = wrap_model(model, scheme)
        if scheme == "make_graphed_callables":
            model.load_state_dict(initial_state)
            optimizer.zero_grad(set_to_none=True)
        torch.npu.synchronize(device)
        setup_ms = (time.perf_counter() - setup_start) * 1000.0

        torch.npu.synchronize(device)
        first_start = time.perf_counter()
        first_loss, first_output_dtype = train_step(callable_model, optimizer, scheme, 0)
        torch.npu.synchronize(device)
        first_step_ms = (time.perf_counter() - first_start) * 1000.0

        step = 1
        for _ in range(args.warmup_steps):
            train_step(callable_model, optimizer, scheme, step)
            step += 1
        torch.npu.synchronize(device)

        blocks = []
        last_loss = first_loss
        last_output_dtype = first_output_dtype
        for _ in range(args.repeats):
            torch.npu.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.measure_steps):
                last_loss, last_output_dtype = train_step(callable_model, optimizer, scheme, step)
                step += 1
            torch.npu.synchronize(device)
            blocks.append((time.perf_counter() - start) * 1000.0 / args.measure_steps)

        result = {
            "setup_ms": setup_ms,
            "first_train_step_ms": first_step_ms,
            "first_loss": float(first_loss.cpu()),
            "last_loss": float(last_loss.cpu()),
            "first_output_dtype": first_output_dtype,
            "last_output_dtype": last_output_dtype,
            "timing": timing_summary(blocks, shape[0]),
            "peak_memory_mib": torch.npu.max_memory_allocated(device) / 1024**2,
            "dynamo_counters": dynamo_counters(),
        }
        del callable_model, optimizer, model
        release_npu_objects()
        return result

    result = {
        "status": "ok",
        "scheme": args.scheme,
        "phase": args.phase,
        "config": {
            "logical_device": args.device,
            "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            "shape": shape,
            "out_channels": args.out_channels,
            "input_variants": args.input_variants,
            "input_offset": args.input_offset,
            "reuse_capture_input": args.reuse_capture_input,
            "dtype": args.dtype,
            "seed": args.seed,
            "lr": args.lr,
            "momentum": args.momentum,
            "correctness_steps": args.correctness_steps,
            "warmup_steps": args.warmup_steps,
            "measure_steps": args.measure_steps,
            "repeats": args.repeats,
            "zero_grad_set_to_none": args.zero_grad_set_to_none,
            "fullgraph": args.fullgraph,
            "cantsplit_compat": args.cantsplit_compat,
            "performance_order": args.performance_order,
            "correctness_atol": atol,
            "correctness_rtol": rtol,
            "state_atol": state_atol,
            "state_rtol": state_rtol,
            "expected_output_dtype": expected_output_dtype,
        },
        "environment": {
            "python": sys.version.replace("\n", " "),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "triton_runtime": triton_runtime,
            "triton_distribution": distribution_version("triton"),
            "triton_ascend_distribution": distribution_version("triton_ascend"),
            "device_name": torch.npu.get_device_name(args.device),
            "cann": os.environ.get("ASCEND_HOME_PATH"),
            "torchinductor_npu_backend": os.environ.get("TORCHINDUCTOR_NPU_BACKEND"),
            "torchinductor_cache_dir": os.environ.get("TORCHINDUCTOR_CACHE_DIR"),
            "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            "npu_inductor_fallback_list": os.environ.get("NPU_INDUCTOR_FALLBACK_LIST"),
        },
    }

    if args.phase in {"correctness", "both"}:
        eager = run_correctness("eager")
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        candidate = run_correctness(args.scheme)
        state_diff = nested_differences(eager["state"], candidate["state"], state_atol, state_rtol)
        gradient_diff = nested_differences(eager["gradients"], candidate["gradients"], state_atol, state_rtol)
        optimizer_diff = nested_differences(
            eager["optimizer_state"], candidate["optimizer_state"], state_atol, state_rtol
        )
        state_max_abs, state_max_rmse = worst_float_difference(state_diff)
        gradient_max_abs, gradient_max_rmse = worst_float_difference(gradient_diff)
        optimizer_max_abs, optimizer_max_rmse = worst_float_difference(optimizer_diff)
        state_issues = difference_issues(state_diff)
        gradient_issues = difference_issues(gradient_diff)
        optimizer_issues = difference_issues(optimizer_diff)
        loss_abs = max(abs(a - b) for a, b in zip(eager["losses"], candidate["losses"]))
        losses_close = all(
            math.isclose(a, b, abs_tol=atol, rel_tol=rtol) for a, b in zip(eager["losses"], candidate["losses"])
        )
        output_dtypes_match = (
            all(dtype == expected_output_dtype for dtype in eager["output_dtypes"])
            and eager["output_dtypes"] == candidate["output_dtypes"]
        )
        correctness_pass = (
            eager["all_losses_finite"]
            and candidate["all_losses_finite"]
            and losses_close
            and output_dtypes_match
            and not (state_issues or gradient_issues or optimizer_issues)
        )
        result["correctness"] = {
            "eager": strip_tensors(eager),
            "candidate": strip_tensors(candidate),
            "loss_max_abs": loss_abs,
            "losses_close": losses_close,
            "output_dtypes_match": output_dtypes_match,
            "state_max_abs": state_max_abs,
            "state_max_rmse": state_max_rmse,
            "gradient_max_abs": gradient_max_abs,
            "gradient_max_rmse": gradient_max_rmse,
            "optimizer_max_abs": optimizer_max_abs,
            "optimizer_max_rmse": optimizer_max_rmse,
            "all_tensor_checks_pass": not (state_issues or gradient_issues or optimizer_issues),
            "correctness_pass": correctness_pass,
            "state_issues": state_issues,
            "gradient_issues": gradient_issues,
            "optimizer_issues": optimizer_issues,
            "state_differences": state_diff,
            "gradient_differences": gradient_diff,
            "optimizer_differences": optimizer_diff,
        }

    if args.phase in {"performance", "both"}:
        performance_runs = {}
        for label, scheme in (
            (("eager", "eager"), ("candidate", args.scheme))
            if args.performance_order == "eager_first"
            else (("candidate", args.scheme), ("eager", "eager"))
        ):
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            performance_runs[label] = run_performance(scheme)
        eager_perf = performance_runs["eager"]
        candidate_perf = performance_runs["candidate"]
        eager_ms = eager_perf["timing"]["median_ms_per_step"]
        candidate_ms = candidate_perf["timing"]["median_ms_per_step"]
        observed_overhead_ms = max(
            0.0,
            candidate_perf["setup_ms"]
            + candidate_perf["first_train_step_ms"]
            - eager_perf["setup_ms"]
            - eager_perf["first_train_step_ms"],
        )
        saving_ms = eager_ms - candidate_ms
        result["performance"] = {
            "eager": eager_perf,
            "candidate": candidate_perf,
            "speedup": eager_ms / candidate_ms,
            "observed_setup_first_step_overhead_ms": observed_overhead_ms,
            "observed_break_even_steps": observed_overhead_ms / saving_ms if saving_ms > 0 else None,
            "setup_semantics": "phase=both 时为同进程正确性检查后的再次捕获，不代表全新进程冷启动",
        }

    return result


def write_result(result: dict, output: Path | None) -> None:
    """打印结果，并按需保存为 JSON。"""
    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    print(serialized, flush=True)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    parsed_output = None
    try:
        parsed_args = parse_args()
        parsed_output = parsed_args.output
        write_result(main(parsed_args), parsed_output)
    except Exception as error:
        failure = {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        try:
            write_result(failure, parsed_output)
        except Exception:
            print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr, flush=True)
        raise
