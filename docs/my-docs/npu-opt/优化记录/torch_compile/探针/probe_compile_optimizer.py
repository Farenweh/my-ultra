"""检查三个 NPU 后端能否捕获标准 SGD 更新，并核对参数和 momentum。"""

import argparse
import inspect
import json
import os
import time
import traceback
from pathlib import Path


def main(args, result):
    """用每步不同的梯度检查优化器图，避免只捕获而没有实际更新。"""
    os.environ["TASK_QUEUE_ENABLE"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    import torch_npu
    from torch._dynamo.backends.registry import lookup_backend
    from torch._dynamo.utils import counters

    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    torch.manual_seed(20261020)
    torch._dynamo.config.suppress_errors = False
    device = f"npu:{args.device}"
    reference = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.Linear(16, 4)).to(device)
    candidate = torch.nn.Sequential(torch.nn.Linear(32, 16), torch.nn.Linear(16, 4)).to(device)
    candidate.load_state_dict(reference.state_dict())
    opt_ref = torch.optim.SGD(reference.parameters(), lr=0.001, momentum=0.9, foreach=False)
    opt = torch.optim.SGD(candidate.parameters(), lr=0.001, momentum=0.9, foreach=False)
    if args.preinit_momentum:
        for optimizer in (opt_ref, opt):
            for p in optimizer.param_groups[0]["params"]:
                optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p)
    options = {}
    if args.backend == "inductor_dvm":
        import torch_npu._inductor

        backend = lookup_backend("inductor")
        options = {"npu_backend": "dvm"}
    else:
        backend = lookup_backend(args.backend)
    graphs = []
    result["捕获图"] = graphs

    def audited(gm, inputs, **kwargs):
        nodes = [str(node.target) for node in gm.graph.nodes if node.op.startswith("call_")]
        graphs.append({"节点数": len(nodes), "算子": nodes})
        if args.backend == "inductor_dvm":
            kwargs["config_patches"] = kwargs.pop("options", {})
        return backend(gm, inputs, **kwargs)

    if args.mode == "direct":
        function = opt.step
    else:
        # 原始 SGD 实现保持不变，仅显式提供其装饰器原本设置的 no_grad 上下文。
        raw_sgd_step = inspect.unwrap(torch.optim.SGD.step)

        @torch.no_grad()
        def function():
            return raw_sgd_step(opt)

    compiled = torch.compile(function, backend=audited, options=options or None,
                             fullgraph=args.fullgraph, dynamic=False)
    records = []
    result["逐步结果"] = records
    generator = torch.Generator().manual_seed(20261021)
    for step in range(8):
        if args.backend == "npugraphs":
            torch.compiler.npugraph_mark_step_begin()
        for p, q in zip(reference.parameters(), candidate.parameters()):
            gradient = torch.randn(p.shape, generator=generator).to(device) * (step + 1)
            p.grad = gradient.clone()
            q.grad = gradient.clone()
        opt_ref.step()
        compiled()
        torch.npu.synchronize()
        max_parameter = max(float((p - q).abs().max().cpu()) for p, q in zip(reference.parameters(), candidate.parameters()))
        max_momentum = max(float((opt_ref.state[p]["momentum_buffer"] - opt.state[q]["momentum_buffer"]).abs().max().cpu())
                           for p, q in zip(reference.parameters(), candidate.parameters()))
        records.append({"步": step + 1, "参数最大绝对误差": max_parameter, "momentum最大绝对误差": max_momentum,
                        "累计图数": len(graphs)})
        print(f"优化器探针：{args.backend}，第 {step + 1} 步，参数误差 {max_parameter:.8g}，momentum误差 {max_momentum:.8g}，图数 {len(graphs)}", flush=True)
    result.update({"状态": "通过" if all(x["参数最大绝对误差"] < 1e-6 and x["momentum最大绝对误差"] < 1e-5 for x in records) else "数值超差",
                   "逐步结果": records, "捕获图": graphs,
                   "Dynamo计数": {g: {str(k): int(v) for k, v in c.items()} for g, c in counters.items() if c}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("inductor_dvm", "npugraphs", "npugraph_ex"), required=True)
    parser.add_argument("--mode", choices=("direct", "inner"), default="direct")
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--preinit-momentum", action="store_true", help="两侧均预分配零 momentum，适用于本测试的 dampening=0")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    outcome = {"状态": "失败", "配置": vars(arguments) | {"output": str(arguments.output)}}
    started = time.perf_counter()
    try:
        main(arguments, outcome)
    except Exception as error:
        outcome.update({"错误类型": type(error).__name__, "错误": str(error), "堆栈": traceback.format_exc()})
        print(f"优化器探针失败：{traceback.format_exc()}", flush=True)
    outcome["总耗时秒"] = time.perf_counter() - started
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(outcome, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"结果已保存：{arguments.output}", flush=True)
