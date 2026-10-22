"""用最小训练代码检查当前 NPU 编译后端，并与 eager 逐步比较。"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


MODELS = ("linear", "conv_bn_relu", "transformer")
BACKENDS = (
    "raw_eager", "eager", "aot_eager", "inductor", "inductor_reduce_overhead",
    "inductor_mlir", "inductor_dvm", "npu", "npugraphs", "npugraph_ex",
)


def parse_args():
    """矩阵调度器只使用标准库，子进程独立初始化编译器和 NPU。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all", help="最小模型")
    parser.add_argument("--backend", nargs="+", choices=BACKENDS, default=list(BACKENDS), help="编译后端")
    parser.add_argument("--dtype", choices=("fp32", "amp_bf16"), default="fp32", help="训练精度")
    parser.add_argument("--steps", type=int, default=6, help="每项训练步数，包含捕获及重放")
    parser.add_argument("--dynamic", action="store_true", help="启用符号动态形状编译")
    parser.add_argument("--vary-shapes", action="store_true", help="轮换三组输入形状，检查首次遇到和再次使用时的行为")
    parser.add_argument("--devices", default="0,1,2", help="空闲 NPU 逻辑编号，每卡串行测试")
    parser.add_argument("--timeout", type=int, default=240, help="单项子进程超时秒数")
    parser.add_argument("--output", type=Path, required=True, help="结果目录")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def save_json(path, data):
    """写入可直接查看的中文结果。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def worker(args):
    """前向、反向、SGD 更新都在 NPU 上执行，检查操作在图外。"""
    result = {"模型": args.model, "后端": args.backend[0], "精度": args.dtype, "状态": "失败", "阶段": "导入"}
    started = time.perf_counter()
    try:
        import copy
        import importlib.metadata
        from contextlib import nullcontext

        import torch
        import torch_npu
        from torch._dynamo.utils import counters

        torch.set_num_threads(1)
        torch.npu.set_device(int(args.devices))
        torch.manual_seed(20261020)
        torch._dynamo.config.suppress_errors = False
        counters.clear()
        backend = args.backend[0]
        result["环境"] = {
            "Python": sys.version.split()[0], "PyTorch": torch.__version__, "TorchNPU": torch_npu.__version__,
            "PyTorch提交": torch.version.git_version, "NPU": torch.npu.get_device_name(), "设备": args.devices,
            "CANN路径": os.environ.get("ASCEND_HOME_PATH"), "TASK_QUEUE_ENABLE": os.environ.get("TASK_QUEUE_ENABLE"),
            "编译后端": torch._dynamo.list_backends(),
            "版本": {},
        }
        for package in ("triton", "triton-ascend", "torch-mlir"):
            try:
                result["环境"]["版本"][package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                result["环境"]["版本"][package] = None

        if args.model == "linear":
            original = torch.nn.Linear(16, 8)
            shape, out_shape = (4, 16), (4, 8)
        elif args.model == "conv_bn_relu":
            original = torch.nn.Sequential(
                torch.nn.Conv2d(3, 8, 3, padding=1, bias=False),
                torch.nn.BatchNorm2d(8), torch.nn.ReLU(),
            )
            shape, out_shape = (2, 3, 8, 8), (2, 8, 8, 8)
        else:
            original = torch.nn.TransformerEncoderLayer(
                d_model=32, nhead=4, dim_feedforward=64, dropout=0.0, batch_first=True,
            )
            shape = out_shape = (2, 8, 32)

        result.update({"输入形状": list(shape), "参数量": sum(p.numel() for p in original.parameters()),
                       "fullgraph": True, "dynamic": args.dynamic, "优化器": "SGD(lr=0.01, foreach=False)",
                       "容差": {"atol": 1e-5 if args.dtype == "fp32" else 1e-4,
                                "rtol": 1e-4 if args.dtype == "fp32" else 1e-2}})
        reference = copy.deepcopy(original).npu().train()
        candidate = copy.deepcopy(original).npu().train()
        initial = {key: value.detach().cpu().clone() for key, value in candidate.named_parameters()}
        # 每步使用不同数据和地址，验证重放确实接收新输入。
        batches = []
        for step in range(args.steps):
            current_shape, current_out_shape = shape, out_shape
            if args.vary_shapes:
                index = step % 3
                if args.model == "linear":
                    current_shape = ((4, 6, 3)[index], 16)
                    current_out_shape = (current_shape[0], 8)
                elif args.model == "conv_bn_relu":
                    current_shape = (2 + index, 3, 8 + 2 * index, 10 + 2 * index)
                    current_out_shape = (current_shape[0], 8, *current_shape[2:])
                else:
                    current_shape = current_out_shape = (2 + index, (8, 12, 6)[index], 32)
            batches.append((torch.randn(current_shape).npu() + step * 0.1, torch.randn(current_out_shape).npu()))
        result["每步输入形状"] = [list(inputs.shape) for inputs, _ in batches]

        def run(module, train_model, compiled):
            optimizer = torch.optim.SGD(module.parameters(), lr=0.01, foreach=False)
            snapshots, times = [], []
            for step, (inputs, target) in enumerate(batches):
                result["完成步数"] = step
                result["阶段"] = ("编译训练" if compiled else "eager对照") + ":前向"
                if compiled and hasattr(torch.compiler, "npugraph_mark_step_begin"):
                    torch.compiler.npugraph_mark_step_begin()
                optimizer.zero_grad(set_to_none=True)
                torch.npu.synchronize()
                tick = time.perf_counter()
                context = torch.autocast("npu", dtype=torch.bfloat16) if args.dtype == "amp_bf16" else nullcontext()
                with context:
                    output = train_model(inputs)
                    loss = (output.float() - target).square().mean()
                result["阶段"] = ("编译训练" if compiled else "eager对照") + ":反向"
                loss.backward()
                result["阶段"] = ("编译训练" if compiled else "eager对照") + ":更新"
                optimizer.step()
                torch.npu.synchronize()
                times.append((time.perf_counter() - tick) * 1000)
                if compiled:
                    result.setdefault("逐步编译计数", []).append({key: dict(value) for key, value in counters.items() if value})
                gradients = {name: p.grad.detach().cpu().clone() for name, p in module.named_parameters()}
                snapshot = {
                    "输出": output.detach().float().cpu().clone(), "loss": loss.detach().cpu().clone(),
                    "梯度": gradients, "状态": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
                }
                for tensor in [snapshot["输出"], snapshot["loss"], *gradients.values(), *snapshot["状态"].values()]:
                    if not torch.isfinite(tensor).all():
                        raise AssertionError("输出、梯度或训练状态出现非有限值")
                snapshots.append(snapshot)
                result.setdefault("编译梯度函数" if compiled else "eager梯度函数", type(output.grad_fn).__name__)
                del output, loss, snapshot, gradients
            return snapshots, times

        expected, eager_times = run(reference, reference, False)
        counters.clear()
        result["完成步数"] = 0
        result["阶段"] = "创建编译包装"
        options = {}
        compile_backend = backend
        if backend.startswith("inductor_"):
            compile_backend = "inductor"
            if backend == "inductor_reduce_overhead":
                options["mode"] = "reduce-overhead"
            else:
                # 在读取 NPU 扩展配置前完成惰性 Inductor 初始化。
                import torch_npu._inductor
                options["options"] = {"npu_backend": backend.rsplit("_", 1)[1]}
        compiled = candidate if backend == "raw_eager" else torch.compile(
            candidate, backend=compile_backend, fullgraph=True, dynamic=args.dynamic, **options,
        )
        actual, candidate_times = run(candidate, compiled, backend != "raw_eager")
        result["阶段"] = "逐步数值对照"
        differences = {key: 0.0 for key in ("输出", "loss", "梯度", "状态")}
        failures = []
        for step, (left, right) in enumerate(zip(expected, actual)):
            for category in differences:
                left_items = left[category] if isinstance(left[category], dict) else {category: left[category]}
                right_items = right[category] if isinstance(right[category], dict) else {category: right[category]}
                for name, tensor in left_items.items():
                    other = right_items[name]
                    differences[category] = max(differences[category], (tensor.float() - other.float()).abs().max().item())
                    close = torch.allclose(tensor, other, **result["容差"]) if tensor.is_floating_point() else torch.equal(tensor, other)
                    if not close:
                        failures.append(f"第{step + 1}步 {category}/{name}")
        changed = [name for name, p in candidate.named_parameters() if not torch.equal(initial[name], p.detach().cpu())]
        if not changed:
            raise AssertionError("SGD 没有更新任何参数")
        stats = {key: dict(value) for key, value in counters.items() if value}
        if backend != "raw_eager" and stats.get("stats", {}).get("unique_graphs", 0) == 0:
            raise AssertionError("没有捕获任何 Dynamo 图")
        result.update({
            "状态": "通过" if not failures else "数值不一致", "阶段": "完成", "完成步数": args.steps,
            "最大绝对误差": differences, "超差项": failures, "更新参数数量": len(changed),
            "eager每步毫秒": eager_times, "候选每步毫秒": candidate_times,
            "eager损失": [float(s["loss"]) for s in expected], "候选损失": [float(s["loss"]) for s in actual],
            "Dynamo计数": stats,
            "BN计数": {name: int(value.cpu()) for name, value in candidate.named_buffers() if "num_batches_tracked" in name},
        })
    except Exception as error:
        result.update({"错误类型": type(error).__name__, "错误": str(error), "堆栈": traceback.format_exc()})
        if "counters" in locals():
            result["Dynamo计数"] = {key: dict(value) for key, value in counters.items() if value}
        print(f"测试失败：{result['模型']} / {result['后端']} / {result['阶段']}\n{traceback.format_exc()}", flush=True)
    result["总耗时秒"] = time.perf_counter() - started
    save_json(args.output / "result.json", result)
    print(f"测试结束：{result['模型']} / {result['后端']} / {result['状态']}", flush=True)


def run_matrix(args):
    """每卡串行调度独立子进程，故障和超时不影响其他案例。"""
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = MODELS if args.model == "all" else (args.model,)
    devices = [value.strip() for value in args.devices.split(",")]
    cases = [(model, backend) for backend in args.backend for model in models]
    partitions = [cases[i::len(devices)] for i in range(len(devices))]

    def run_partition(device, tasks):
        results = []
        for model, backend in tasks:
            work = output / f"{model}__{backend}__{args.dtype}"
            work.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--model", model,
                       "--backend", backend, "--dtype", args.dtype, "--steps", str(args.steps),
                       "--devices", device, "--output", str(work)]
            if args.dynamic:
                command.append("--dynamic")
            if args.vary_shapes:
                command.append("--vary-shapes")
            env = dict(os.environ, TASK_QUEUE_ENABLE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            # 避免已有环境变量改变默认后端含义，保留调用者的缓存路径。
            env.pop("TORCHINDUCTOR_NPU_BACKEND", None)
            path = work / "result.json"
            path.unlink(missing_ok=True)
            with (work / "run.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=work, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    process.wait(timeout=args.timeout)
                    status = "进程异常"
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    status = "超时"
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
            else:
                row = {"模型": model, "后端": backend, "精度": args.dtype, "状态": status, "退出码": process.returncode}
            row["日志"] = str(work / "run.log")
            results.append(row)
            print(f"完成：NPU{device} {model} / {backend} / {row['状态']}", flush=True)
        return results

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(run_partition, device, tasks) for device, tasks in zip(devices, partitions)]
        results = [row for future in futures for row in future.result()]
    report = {"时间": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "说明": "单卡最小训练可用性测试；并行案例的时延不用于性能比较。", "结果": results}
    save_json(output / "summary.json", report)
    print(f"矩阵完成，共 {len(results)} 项，汇总：{output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.steps < 3:
        raise SystemExit("至少运行 3 步，以覆盖图捕获和重放")
    worker(arguments) if arguments.worker else run_matrix(arguments)
