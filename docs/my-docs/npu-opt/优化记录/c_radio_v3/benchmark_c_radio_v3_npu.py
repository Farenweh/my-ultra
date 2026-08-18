"""在Ascend NPU上验证并分解C-RADIOv3-L的CPE缓存和融合attention。"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import threading
import time
import types
from pathlib import Path

import torch

from ultralytics.nn.modules.backbone import CRADIOv3
from ultralytics.nn.modules.third_party.c_radio_v3 import attention as attention_module


MODES = ("official_sdpa", "cpe_cache", "fusion", "full")


class NpuUtilizationSampler:
    """使用常驻npu-smi watch进程采样，避免反复创建进程干扰基准。"""

    def __init__(self, device: int):
        self.device = device
        self.samples = []
        self._process = None
        self._thread = None

    def _read(self) -> None:
        for line in self._process.stdout:
            values = re.match(r"\s*\d+\s+\d+\s+(\d+)\s+(\d+)\s*$", line)
            if values:
                self.samples.append((int(values.group(1)), int(values.group(2))))

    def __enter__(self):
        self._process = subprocess.Popen(
            ["npu-smi", "info", "watch", "-i", str(self.device), "-c", "0", "-d", "1", "-s", "an"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._process.terminate()
        try:
            self._process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=1)
        self._thread.join(timeout=2)

    def result(self) -> dict[str, float | int | None]:
        if not self.samples:
            return {"samples": 0, "aicore_mean": None, "npu_mean": None}
        return {
            "samples": len(self.samples),
            "aicore_mean": statistics.fmean(value[0] for value in self.samples),
            "npu_mean": statistics.fmean(value[1] for value in self.samples),
        }


class ModeController:
    """在不改动模型权重的前提下切换四个消融实现。"""

    def __init__(self, model: CRADIOv3):
        self.cpe = model.model.patch_generator
        self.original_base_grid = self.cpe._base_grid
        self.original_position = self.cpe._position

    def _uncached_base_grid(self, cpe, grid_h: int, grid_w: int, device: torch.device):
        cpe._base_grid_cache_key = None
        return self.original_base_grid(grid_h, grid_w, device)

    def _uncached_position(self, cpe, batch: int, grid_h: int, grid_w: int, stochastic: bool):
        cpe._position_cache_key = None
        return self.original_position(batch, grid_h, grid_w, stochastic)

    def configure(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"未知模式{mode!r}")
        self.cpe._base_grid_cache_key = None
        self.cpe._position_cache_key = None
        use_cache = mode in {"cpe_cache", "full"}
        self.cpe._base_grid = (
            self.original_base_grid if use_cache else types.MethodType(self._uncached_base_grid, self.cpe)
        )
        self.cpe._position = (
            self.original_position if use_cache else types.MethodType(self._uncached_position, self.cpe)
        )
        attention_module.CRADIO_V3_ATTENTION_BACKEND = "auto" if mode in {"fusion", "full"} else "sdpa"

    def restore(self) -> None:
        self.cpe._base_grid = self.original_base_grid
        self.cpe._position = self.original_position
        attention_module.CRADIO_V3_ATTENTION_BACKEND = "auto"


def synchronize() -> None:
    torch.npu.synchronize()


def measure(controller, mode: str, step, *, warmup: int, iterations: int, device_index: int) -> dict:
    controller.configure(mode)
    for _ in range(warmup):
        step()
    synchronize()
    torch.npu.reset_peak_memory_stats()
    with NpuUtilizationSampler(device_index) as sampler:
        start = time.perf_counter()
        for _ in range(iterations):
            step()
        synchronize()
        elapsed = time.perf_counter() - start
    return {
        "milliseconds_per_step": elapsed * 1000 / iterations,
        "peak_allocated_mib": torch.npu.max_memory_allocated() / 1024**2,
        "utilization": sampler.result(),
    }


def measure_repeated(controller, step, args, *, training: bool) -> dict:
    samples = {mode: [] for mode in MODES}
    warmup = args.train_warmup if training else args.warmup
    iterations = args.train_iterations if training else args.iterations
    for repeat in range(args.repeats):
        order = MODES if repeat % 2 == 0 else tuple(reversed(MODES))
        for mode in order:
            samples[mode].append(
                measure(controller, mode, step, warmup=warmup, iterations=iterations, device_index=args.device)
            )
    output = {}
    for mode, values in samples.items():
        output[mode] = {
            "median_ms": statistics.median(item["milliseconds_per_step"] for item in values),
            "runs": values,
        }
    original_median = output["official_sdpa"]["median_ms"]
    for value in output.values():
        value["speedup_vs_official"] = original_median / value["median_ms"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="", help="空值表示从固定Hugging Face revision加载官方L权重")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--image-sizes", type=int, nargs="+", default=(512, 640, 800))
    parser.add_argument("--training-sizes", type=int, nargs="+", default=(512, 640))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--train-warmup", type=int, default=1)
    parser.add_argument("--train-iterations", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--output", type=Path, help="可选JSON输出路径；省略时只打印到stdout")
    args = parser.parse_args()

    import torch_npu

    torch.npu.set_device(args.device)
    torch.npu.set_compile_mode(jit_compile=False)
    device = torch.device(f"npu:{args.device}")
    model = CRADIOv3("l", pretrained=args.checkpoint or True).to(device)
    controller = ModeController(model)
    result = {
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(args.device),
            "amp": "fp16",
            "jit_compile": False,
            "attention": "npu_fusion_attention_v3",
        },
        "model": {
            "variant": "C-RADIOv3-L/16",
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "inference": {},
        "training": {},
    }
    try:
        for image_size in args.image_sizes:
            image = torch.rand(1, 3, image_size, image_size, device=device)
            model.eval().requires_grad_(False)

            def inference_step():
                with torch.inference_mode(), torch.autocast("npu", dtype=torch.float16):
                    return model(image)

            result["inference"][str(image_size)] = measure_repeated(controller, inference_step, args, training=False)

        for image_size in () if args.skip_training else args.training_sizes:
            image = torch.rand(1, 3, image_size, image_size, device=device)
            model.train().requires_grad_(True)

            def training_step():
                model.zero_grad(set_to_none=True)
                with torch.autocast("npu", dtype=torch.float16):
                    loss = model(image).float().square().mean()
                loss.backward()

            result["training"][str(image_size)] = measure_repeated(controller, training_step, args, training=True)
    finally:
        controller.restore()
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{payload}\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
