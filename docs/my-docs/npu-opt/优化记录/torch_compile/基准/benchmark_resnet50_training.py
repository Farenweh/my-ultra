"""测试 ResNet-50 的 NPU 编译训练，可选择将标准 SGD 编成独立更新图。"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import inspect
import json
import math
import os
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

from benchmark_conv_bn_relu import clone_gradients, clone_optimizer_state, clone_state, nested_differences


def parse_args():
    """在导入框架前设置设备执行选项。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scheme", choices=("inductor_dvm", "npugraphs", "npugraph_ex", "eager"), required=True)
    parser.add_argument("--dtype", choices=("fp32", "amp_bf16"), default="amp_bf16")
    parser.add_argument("--phase", choices=("correctness", "performance", "both"), default="both")
    parser.add_argument("--order", choices=("eager_first", "candidate_first"), default="eager_first")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--task-queue", type=int, choices=(1, 2), default=1)
    parser.add_argument("--correctness-steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--steps-per-block", type=int, default=20)
    parser.add_argument("--stage-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--deterministic", action="store_true", help="正确性诊断时启用确定性算子")
    parser.add_argument("--optimizer-mode", choices=("outside", "direct", "inner"), default="outside",
                        help="图外更新、直接编译标准 step、或显式 no_grad 下编译原始 SGD 实现")
    parser.add_argument("--baseline", choices=("eager", "backend_outside"), default="eager",
                        help="性能基线使用全 eager，或相同网络后端加图外优化器")
    parser.add_argument("--network-eager", action="store_true", help="仅隔离测试优化器，网络和反向保持 eager")
    parser.add_argument("--preinit-momentum", action="store_true", help="两侧均预分配零 momentum，保持本测试 dampening=0 的更新语义")
    parser.add_argument("--audit-replay", action="store_true", help="仅在正确性阶段统计各区域的 NPUGraph.replay 实际调用")
    parser.add_argument("--static-optimizer-grads", action="store_true",
                        help="首步反向后将梯度标为静态地址；正确性阶段逐步验证地址稳定")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def summarize(values):
    """汇总重复测量，不把首次编译耗时混入稳态。"""
    mean = statistics.fmean(values)
    return {"均值": mean, "中位数": statistics.median(values), "最小": min(values), "最大": max(values),
            "CV": statistics.stdev(values) / mean if len(values) > 1 and mean else 0.0, "样本": values}


def main(args, result):
    """执行同一设备上的正确性比较和两侧串行性能测量。"""
    os.environ["TASK_QUEUE_ENABLE"] = str(args.task_queue)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ.pop("TORCHINDUCTOR_NPU_BACKEND", None)

    import torch
    import torch_npu
    from torch._dynamo.utils import counters
    from torch._dynamo.backends.registry import lookup_backend
    from torchvision.models import resnet50

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.npu.set_device(args.device)
    torch.npu.manual_seed(args.seed)
    torch._dynamo.config.suppress_errors = False
    torch.use_deterministic_algorithms(args.deterministic)
    device = torch.device(f"npu:{args.device}")

    class ClassificationLoss(torch.nn.Module):
        """把 ResNet-50 和 FP32 交叉熵一起交给编译器。"""

        def __init__(self):
            super().__init__()
            self.network = resnet50(weights=None, num_classes=1000)

        def forward(self, images, targets):
            return torch.nn.functional.cross_entropy(self.network(images).float(), targets)

    master = ClassificationLoss()
    initial_state = clone_state(master)
    result["环境"] = {
        "Python": sys.version.split()[0], "PyTorch": torch.__version__, "TorchNPU": torch_npu.__version__,
        "torchvision": importlib.metadata.version("torchvision"), "NPU": torch.npu.get_device_name(args.device),
        "设备": args.device, "CANN路径": os.environ.get("ASCEND_HOME_PATH"), "TASK_QUEUE_ENABLE": args.task_queue,
    }
    result["配置"] = {
        "模型": "torchvision ResNet-50，随机初始化，1000类", "参数量": sum(p.numel() for p in master.parameters()),
        "输入形状": [args.batch, 3, args.size, args.size], "精度": args.dtype, "输入数量": 2,
        "损失": "FP32 cross_entropy，图外" if args.network_eager else "FP32 cross_entropy，随网络一起入图",
        "优化器": "SGD(lr=0.001,momentum=0.9,foreach=False)",
        "优化器模式": args.optimizer_mode, "性能基线": args.baseline, "网络保持eager": args.network_eager,
        "预分配momentum": args.preinit_momentum,
        "优化器fullgraph": args.optimizer_mode == "inner",
        "标记梯度静态地址": args.static_optimizer_grads,
        "优化器边界": "独立优化器图；zero_grad 和 backward 调用在 Python 外层",
        "fullgraph": True, "dynamic": False, "顺序": args.order,
        "预热步数": args.warmup, "测量块数": args.blocks, "每块步数": args.steps_per_block,
        "阶段测量步数": args.stage_steps, "随机种子": args.seed,
        "确定性算子": args.deterministic,
    }
    del master
    generator = torch.Generator().manual_seed(args.seed + 1)
    batches = [(torch.randn(args.batch, 3, args.size, args.size, generator=generator).to(device),
                torch.randint(1000, (args.batch,), generator=generator).to(device)) for _ in range(2)]
    replay_audit = {"区域": "其他", "计数": {}}

    def context():
        return torch.autocast("npu", dtype=torch.bfloat16, cache_enabled=False) if args.dtype == "amp_bf16" else nullcontext()

    def release():
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()

    def build(scheme, optimizer_mode=None):
        result["阶段"] = f"{scheme}:构造模型"
        torch._dynamo.reset()
        counters.clear()
        release()
        module = ClassificationLoss().to(device).train()
        module.load_state_dict(initial_state)
        optimizer = torch.optim.SGD(module.parameters(), lr=0.001, momentum=0.9, foreach=False)
        if args.preinit_momentum:
            for p in module.parameters():
                optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p)
        torch.npu.synchronize()
        tick = time.perf_counter()
        if scheme == "eager" or args.network_eager:
            compiled = module
        elif scheme == "inductor_dvm":
            import torch_npu._inductor

            compiled = torch.compile(module, backend="inductor", options={"npu_backend": "dvm"},
                                     fullgraph=True, dynamic=False)
        else:
            compiled = torch.compile(module, backend=scheme, fullgraph=True, dynamic=False)
        mode = "outside" if scheme == "eager" else (optimizer_mode or args.optimizer_mode)
        optimizer._benchmark_graphs = []
        optimizer._benchmark_mode = mode
        optimizer._benchmark_static_grads = False
        optimizer._benchmark_step = optimizer.step
        if mode != "outside":
            if scheme == "inductor_dvm":
                import torch_npu._inductor

            actual_backend = lookup_backend("inductor" if scheme == "inductor_dvm" else scheme)

            def audited_backend(gm, inputs, **kwargs):
                nodes = [str(node.target) for node in gm.graph.nodes if node.op.startswith("call_")]
                optimizer._benchmark_graphs.append({"节点数": len(nodes), "算子": nodes})
                if scheme == "inductor_dvm":
                    kwargs["config_patches"] = kwargs.pop("options", {})
                return actual_backend(gm, inputs, **kwargs)

            function = optimizer.step
            if mode == "inner":
                raw_sgd_step = inspect.unwrap(torch.optim.SGD.step)

                @torch.no_grad()
                def function():
                    return raw_sgd_step(optimizer)

            optimizer._benchmark_step = torch.compile(
                function, backend=audited_backend, fullgraph=mode == "inner", dynamic=False,
                options={"npu_backend": "dvm"} if scheme == "inductor_dvm" else None,
            )
        torch.npu.synchronize()
        return module, optimizer, compiled, (time.perf_counter() - tick) * 1000

    def step(compiled, optimizer, scheme, index, events=None, inspect=False):
        optimizer.zero_grad(set_to_none=True)
        if scheme == "npugraphs":
            torch.compiler.npugraph_mark_step_begin()
        if events:
            events[0].record()
        replay_audit["区域"] = "前向含损失"
        with context():
            loss = compiled(*batches[index % 2])
        if events:
            events[1].record()
        replay_audit["区域"] = "反向"
        loss.backward()
        if events:
            events[2].record()
        replay_audit["区域"] = "优化器"
        if args.static_optimizer_grads and optimizer._benchmark_mode != "outside" and not optimizer._benchmark_static_grads:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    torch._dynamo.mark_static_address(p.grad)
            optimizer._benchmark_static_grads = True
        optimizer._benchmark_step()
        replay_audit["区域"] = "其他"
        if events:
            events[3].record()
        value = float(loss.detach().cpu()) if inspect else None
        del loss
        return value

    def stats():
        return {group: {str(k): int(v) for k, v in counter.items()} for group, counter in counters.items() if counter}

    def correctness(scheme):
        module, optimizer, compiled, setup = build(scheme)
        replay_audit["计数"] = {}
        original_replay = torch.npu.NPUGraph.replay
        if args.audit_replay:
            def counted_replay(graph):
                region = replay_audit["区域"]
                replay_audit["计数"][region] = replay_audit["计数"].get(region, 0) + 1
                return original_replay(graph)

            torch.npu.NPUGraph.replay = counted_replay
        losses = []
        gradient_addresses = []
        first_ms = None
        for index in range(args.correctness_steps):
            result["阶段"] = f"{scheme}:正确性第{index + 1}步"
            tick = time.perf_counter()
            losses.append(step(compiled, optimizer, scheme, index, inspect=True))
            gradient_addresses.append([p.grad.data_ptr() for p in module.parameters()])
            torch.npu.synchronize()
            if first_ms is None:
                first_ms = (time.perf_counter() - tick) * 1000
            print(f"正确性：{scheme} 第{index + 1}步，loss={losses[-1]:.7f}", flush=True)
        snapshots = {"状态": clone_state(module), "梯度": clone_gradients(module),
                     "优化器": clone_optimizer_state(optimizer, module)}
        torch.npu.NPUGraph.replay = original_replay
        values = {"损失": losses, "包装毫秒": setup, "首次训练步毫秒": first_ms, "Dynamo计数": stats(),
                  "优化器模式": optimizer._benchmark_mode, "优化器捕获图": optimizer._benchmark_graphs,
                  "NPUGraph回放计数": dict(replay_audit["计数"]),
                  "梯度地址逐步相同": all(row == gradient_addresses[0] for row in gradient_addresses),
                  "BN计数": sorted({int(v.cpu()) for k, v in module.named_buffers() if "num_batches_tracked" in k})}
        del compiled, optimizer, module
        torch._dynamo.reset()
        release()
        return snapshots, values

    if args.phase in {"correctness", "both"}:
        reference, ref_info = correctness("eager")
        candidate, cand_info = correctness(args.scheme)
        atol, rtol = (1e-5, 1e-3) if args.dtype == "fp32" else (1e-4, 1e-2)
        differences = {}
        issues = []
        for category in reference:
            diff = nested_differences(reference[category], candidate[category], atol, rtol)
            differences[category] = diff

            def walk(value, path):
                if isinstance(value, dict):
                    if any(value.get(k) is False for k in ["finite", "close", "exact", "same_presence", "same_shape", "same_dtype"]) or "missing" in value:
                        issues.append(path)
                    for key, item in value.items():
                        if isinstance(item, dict):
                            walk(item, f"{path}/{key}")

            walk(diff, category)
        loss_atol, loss_rtol = (1e-5, 1e-4) if args.dtype == "fp32" else (1e-3, 5e-3)
        losses_close = all(math.isclose(x, y, abs_tol=loss_atol, rel_tol=loss_rtol)
                           for x, y in zip(ref_info["损失"], cand_info["损失"]))
        finite = all(math.isfinite(v) for v in cand_info["损失"])
        if not losses_close or not finite:
            issues.append("损失")
        result["正确性"] = {"状态": "通过" if not issues else "数值超差", "eager": ref_info, "候选": cand_info,
                          "状态容差": {"atol": atol, "rtol": rtol}, "损失容差": {"atol": loss_atol, "rtol": loss_rtol},
                          "超差项": issues, "详细差异": differences}
        print(f"正确性完成：{args.scheme}，{result['正确性']['状态']}，超差项 {len(issues)}", flush=True)
        del reference, candidate
        release()

    def performance(scheme, optimizer_mode=None):
        module, optimizer, compiled, setup = build(scheme, optimizer_mode)
        torch.npu.reset_peak_memory_stats()
        result["阶段"] = f"{scheme}:首次训练步"
        tick = time.perf_counter()
        first_loss = step(compiled, optimizer, scheme, 0, inspect=True)
        torch.npu.synchronize()
        first_ms = (time.perf_counter() - tick) * 1000
        for index in range(args.warmup):
            step(compiled, optimizer, scheme, index + 1)
        torch.npu.synchronize()
        warm_stats = stats()
        print(f"预热完成：{scheme}，首次训练步 {first_ms:.1f} ms", flush=True)
        blocks = []
        index = args.warmup + 1
        for block in range(args.blocks):
            result["阶段"] = f"{scheme}:整体计时块{block + 1}"
            torch.npu.synchronize()
            tick = time.perf_counter()
            for _ in range(args.steps_per_block):
                step(compiled, optimizer, scheme, index)
                index += 1
            torch.npu.synchronize()
            blocks.append((time.perf_counter() - tick) * 1000 / args.steps_per_block)
            print(f"整体计时：{scheme} 第{block + 1}块，{blocks[-1]:.4f} ms/step", flush=True)
        primary_stats = stats()
        # 单独使用事件测量前向、反向和优化器；事件不进入上面的整体墙钟计时。
        events = [[torch.npu.Event(enable_timing=True) for _ in range(4)] for _ in range(args.stage_steps)]
        for row in events:
            step(compiled, optimizer, scheme, index, events=row)
            index += 1
        torch.npu.synchronize()
        stages = {"前向含损失毫秒": [], "反向毫秒": [], "优化器毫秒": [], "前反向毫秒": []}
        for row in events:
            stages["前向含损失毫秒"].append(row[0].elapsed_time(row[1]))
            stages["反向毫秒"].append(row[1].elapsed_time(row[2]))
            stages["优化器毫秒"].append(row[2].elapsed_time(row[3]))
            stages["前反向毫秒"].append(row[0].elapsed_time(row[2]))
        final_loss = step(compiled, optimizer, scheme, index, inspect=True)
        finite = math.isfinite(final_loss) and all(p.grad is not None and bool(p.grad.isfinite().all().cpu()) for p in module.parameters())
        if not finite:
            raise AssertionError(f"{scheme} 在性能测量结束后出现非有限损失或梯度")
        information = {"包装毫秒": setup, "首次训练步毫秒": first_ms, "首次损失": first_loss, "最终损失": final_loss,
                       "整体每步毫秒": summarize(blocks), "吞吐张每秒": args.batch * 1000 / statistics.median(blocks),
                       "阶段事件计时": {key: summarize(values) for key, values in stages.items()},
                       "峰值已分配MiB": torch.npu.max_memory_allocated() / 2**20,
                       "峰值已保留MiB": torch.npu.max_memory_reserved() / 2**20,
                       "优化器模式": optimizer._benchmark_mode, "优化器捕获图": optimizer._benchmark_graphs,
                       "预热后计数": warm_stats, "主计时后计数": primary_stats, "最终计数": stats()}
        del events, compiled, optimizer, module
        torch._dynamo.reset()
        release()
        return information

    if args.phase in {"performance", "both"}:
        baseline_key = "backend_outside" if args.baseline == "backend_outside" else "eager"
        order = ["eager"] if args.scheme == "eager" else (
            [baseline_key, args.scheme] if args.order == "eager_first" else [args.scheme, baseline_key]
        )
        measurements = {}
        for scheme in order:
            measurements[scheme] = performance(args.scheme, "outside") if scheme == "backend_outside" else performance(scheme)
        result["性能"] = measurements
        baseline, candidate = measurements[baseline_key], measurements[args.scheme]
        result["加速比"] = {"整体训练步": baseline["整体每步毫秒"]["中位数"] / candidate["整体每步毫秒"]["中位数"]}
        for key in ["前向含损失毫秒", "反向毫秒", "前反向毫秒", "优化器毫秒"]:
            result["加速比"][key] = baseline["阶段事件计时"][key]["中位数"] / candidate["阶段事件计时"][key]["中位数"]
    result.update({"状态": "完成", "阶段": "完成"})


if __name__ == "__main__":
    arguments = parse_args()
    outcome = {"状态": "失败", "阶段": "导入", "后端": arguments.scheme, "精度": arguments.dtype,
               "命令": sys.argv, "阶段选择": arguments.phase}
    started = time.perf_counter()
    try:
        main(arguments, outcome)
    except Exception as error:
        outcome.update({"错误类型": type(error).__name__, "错误": str(error), "堆栈": traceback.format_exc()})
        print(f"基准失败：{outcome['阶段']}\n{traceback.format_exc()}", flush=True)
    outcome["总耗时秒"] = time.perf_counter() - started
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"结果已保存：{arguments.output}", flush=True)
