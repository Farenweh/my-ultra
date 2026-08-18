#!/usr/bin/env python3
"""在隔离子进程比较 Eager DVM 关闭/开启时的 Conv-BN-ReLU 训练。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--activation-probe", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def has_binary_marker(torch_npu) -> bool:
    library = Path(torch_npu.__file__).resolve().parent / "lib" / "libtorch_npu.so"
    return library.exists() and b"TORCH_NPU_LAZY_FUSION" in library.read_bytes()


def run_worker(output: Path, activation_probe: bool) -> None:
    import torch
    import torch.nn as nn
    import torch_npu

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    metadata = {
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "env": os.environ.get("TORCH_NPU_LAZY_FUSION"),
        "task_queue_enable": os.environ.get("TASK_QUEUE_ENABLE"),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "torch_npu_git": getattr(torch_npu.version, "git_version", None),
        "device": torch.npu.get_device_name(0),
        "c_has_dvm": hasattr(torch_npu._C, "dvm"),
        "binary_has_lazy_fusion_marker": has_binary_marker(torch_npu),
    }

    if activation_probe:
        x = torch.randn(64, 128, device=device, requires_grad=True)
        y = torch.relu(torch.sigmoid((x + 1.25) * 0.75))
        loss = y.square().mean()
        loss.backward()
        torch.npu.synchronize()
        output.write_text(
            json.dumps(
                {
                    "metadata": metadata,
                    "loss": float(loss.detach().cpu()),
                    "gradient_max": float(x.grad.detach().abs().max().cpu()),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return

    class ConvBNReLU(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(16, 16, 3, padding=1, bias=False)
            self.bn = nn.BatchNorm2d(16)
            self.relu = nn.ReLU(inplace=False)

        def forward(self, value):
            return self.relu(self.bn(self.conv(value)))

    seed = 20260730
    torch.manual_seed(seed)
    torch.npu.manual_seed(seed)
    model = ConvBNReLU().to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, foreach=False)
    generator = torch.Generator().manual_seed(seed + 1)
    inputs = [torch.randn(4, 16, 32, 32, generator=generator).to(device) for _ in range(2)]

    losses, outputs, output_dtypes = [], [], []
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        value = model(inputs[step % len(inputs)])
        loss = value.float().square().mean()
        loss.backward()
        optimizer.step()
        torch.npu.synchronize()
        losses.append(float(loss.detach().cpu()))
        outputs.append(value.detach().cpu().clone())
        output_dtypes.append(str(value.dtype))

    names = {parameter: name for name, parameter in model.named_parameters()}
    optimizer_state = {
        names[parameter]: {
            key: item.detach().cpu().clone() if hasattr(item, "detach") else item for key, item in values.items()
        }
        for parameter, values in optimizer.state.items()
    }
    torch.save(
        {
            "metadata": metadata,
            "losses": losses,
            "outputs": outputs,
            "output_dtypes": output_dtypes,
            "state": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
            "gradients": {
                name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
                for name, parameter in model.named_parameters()
            },
            "optimizer_state": optimizer_state,
        },
        output,
    )


def tensor_difference(reference, candidate, atol=1e-5, rtol=1e-4):
    import torch

    if reference is None or candidate is None:
        same = reference is candidate
        return {"same_presence": same, "pass": same}
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {
            "same_shape": reference.shape == candidate.shape,
            "same_dtype": reference.dtype == candidate.dtype,
            "pass": False,
        }
    if not reference.dtype.is_floating_point:
        exact = bool(torch.equal(reference, candidate))
        return {"exact": exact, "reference": reference.tolist(), "candidate": candidate.tolist(), "pass": exact}
    ref, cand = reference.float(), candidate.float()
    absolute = (ref - cand).abs()
    return {
        "finite": bool(torch.isfinite(ref).all() and torch.isfinite(cand).all()),
        "max_abs": float(absolute.max()),
        "rmse": float(torch.sqrt(torch.mean((ref - cand).square()))),
        "pass": bool(torch.allclose(ref, cand, atol=atol, rtol=rtol)),
    }


def compare_mapping(reference, candidate):
    result = {}
    for name in sorted(reference.keys() | candidate.keys()):
        if name not in reference or name not in candidate:
            result[name] = {"missing": True, "pass": False}
        elif isinstance(reference[name], dict):
            result[name] = compare_mapping(reference[name], candidate[name])
        else:
            result[name] = tensor_difference(reference[name], candidate[name])
    return result


def collect_checks(value):
    if isinstance(value, dict):
        if "pass" in value:
            return [bool(value["pass"])]
        return [check for item in value.values() for check in collect_checks(item)]
    if isinstance(value, (list, tuple)):
        return [check for item in value for check in collect_checks(item)]
    return []


def child(script: Path, output: Path, env: dict[str, str], activation=False):
    command = [sys.executable, str(script), "--worker", "--output", str(output)]
    if activation:
        command.append("--activation-probe")
    return command, subprocess.run(command, env=env, capture_output=True, text=True, timeout=180)


def run_driver(output: Path) -> None:
    import torch

    script = Path(__file__).resolve()
    base_env = os.environ.copy()
    base_env["ASCEND_RT_VISIBLE_DEVICES"] = "6"
    base_env.pop("TORCH_NPU_LAZY_FUSION", None)
    on_env = base_env.copy()
    on_env["TORCH_NPU_LAZY_FUSION"] = "True"

    with tempfile.TemporaryDirectory(prefix="eager-dvm-probe-", dir="/tmp") as directory:
        temp = Path(directory)
        dump_dir = temp / "dump"
        dump_dir.mkdir()
        activation_env = base_env.copy()
        activation_env["TORCH_NPU_LAZY_FUSION"] = f"True dump_as_text dump_dir={dump_dir}"

        off_command, off_run = child(script, temp / "off.pt", base_env)
        on_command, on_run = child(script, temp / "on.pt", on_env)
        activation_command, activation_run = child(script, temp / "activation.json", activation_env, True)
        runs = {"off": off_run, "on": on_run, "activation": activation_run}
        failed = {name: run.returncode for name, run in runs.items() if run.returncode != 0}
        if failed:
            result = {
                "status": "error",
                "failed_returncodes": failed,
                "stderr": {name: run.stderr for name, run in runs.items()},
            }
        else:
            off = torch.load(temp / "off.pt", weights_only=False)
            on = torch.load(temp / "on.pt", weights_only=False)
            output_differences = [tensor_difference(a, b) for a, b in zip(off["outputs"], on["outputs"])]
            state_differences = compare_mapping(off["state"], on["state"])
            gradient_differences = compare_mapping(off["gradients"], on["gradients"])
            optimizer_differences = compare_mapping(off["optimizer_state"], on["optimizer_state"])
            loss_max_abs = max(abs(a - b) for a, b in zip(off["losses"], on["losses"]))
            dump_files = sorted(path.name for path in dump_dir.iterdir())
            marker = bool(on["metadata"]["binary_has_lazy_fusion_marker"])
            checks = collect_checks(
                [output_differences, state_differences, gradient_differences, optimizer_differences]
            )
            result = {
                "status": "ok",
                "feature_available": marker and bool(dump_files),
                "commands": {
                    "off": off_command,
                    "on": on_command,
                    "activation": activation_command,
                },
                "activation_evidence": {
                    "binary_has_lazy_fusion_marker": marker,
                    "c_has_dvm": on["metadata"]["c_has_dvm"],
                    "dump_files": dump_files,
                    "activation_probe": json.loads((temp / "activation.json").read_text(encoding="utf-8")),
                },
                "environment": on["metadata"],
                "off_losses": off["losses"],
                "on_losses": on["losses"],
                "loss_max_abs": loss_max_abs,
                "output_dtypes_match": off["output_dtypes"] == on["output_dtypes"],
                "output_differences": output_differences,
                "state_differences": state_differences,
                "gradient_differences": gradient_differences,
                "optimizer_differences": optimizer_differences,
                "correctness_pass": loss_max_abs <= 1e-5 and all(checks),
                "child_stderr": {name: run.stderr for name, run in runs.items() if run.stderr},
            }
        serialized = json.dumps(result, ensure_ascii=False, indent=2)
        print(serialized, flush=True)
        output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    args = parse_args()
    if args.worker:
        run_worker(args.output, args.activation_probe)
    else:
        run_driver(args.output)
