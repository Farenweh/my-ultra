"""在 Ascend NPU 上验证并分解 PE-Spatial-L/14 的 RoPE 与缓存优化。"""

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

from ultralytics.nn.modules.backbone import PESpatial
from ultralytics.nn.modules.third_party.pe_spatial import rope as rope_module
from ultralytics.nn.modules.third_party.pe_spatial import vision_transformer as vision_module


MODES = ("original_manual", "cached_manual", "rotary_mul", "full")


class NpuUtilizationSampler:
    """使用一个常驻 npu-smi watch 进程采样，避免反复创建进程干扰基准。"""

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
    """在不修改模型权重的情况下切换四个消融实现。"""

    def __init__(self, model: PESpatial):
        self.model = model.model
        self.original_apply = vision_module.apply_rope
        self.original_sample = self.model._sample_abs_posemb
        self.original_coefficients = self.model.rope.coefficients

        def recompute_trigonometry(x, sin, cos):
            frequency = self.model.rope._frequency_cache
            head_dim = frequency.shape[-1]
            sin = frequency.sin().view(1, -1, 1, head_dim)
            cos = frequency.cos().view(1, -1, 1, head_dim)
            return rope_module.apply_rope_manual(x, sin, cos)

        self.recompute_trigonometry = recompute_trigonometry

    def _frequency_only(self, rope, grid_h: int, grid_w: int, device: torch.device):
        """复现官方只缓存frequency、在每次Q/K RoPE中计算sin/cos的行为。"""
        device = torch.device(device)
        key = (device.type, device.index, grid_h, grid_w, torch.float32)
        if key != rope._cache_key:
            rope._frequency_cache = rope._build_frequency(grid_h, grid_w, device)
            rope._sin_cache = torch.empty(0)
            rope._cos_cache = torch.empty(0)
            rope._cache_key = key
        return None, None

    def _uncached_position(self, _, grid_h: int, grid_w: int):
        return self.model._interpolate_positional_embedding(grid_h, grid_w)

    def configure(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"未知模式{mode!r}")
        self.model.rope._cache_key = None
        self.model._pos_embed_cache_key = None
        if mode == "original_manual":
            vision_module.apply_rope = self.recompute_trigonometry
            self.model.rope.coefficients = types.MethodType(self._frequency_only, self.model.rope)
        else:
            vision_module.apply_rope = self.original_apply
            self.model.rope.coefficients = self.original_coefficients
            rope_module.PE_SPATIAL_ROPE_BACKEND = "manual" if mode == "cached_manual" else "auto"
        if mode == "full":
            self.model._sample_abs_posemb = self.original_sample
        else:
            self.model._sample_abs_posemb = types.MethodType(self._uncached_position, self.model)

    def restore(self) -> None:
        vision_module.apply_rope = self.original_apply
        self.model.rope.coefficients = self.original_coefficients
        self.model._sample_abs_posemb = self.original_sample


def synchronize() -> None:
    torch.npu.synchronize()


def measure(
    controller: ModeController,
    mode: str,
    step,
    *,
    warmup: int,
    iterations: int,
    device_index: int,
) -> dict:
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
                measure(
                    controller,
                    mode,
                    step,
                    warmup=warmup,
                    iterations=iterations,
                    device_index=args.device,
                )
            )
    output = {}
    for mode, values in samples.items():
        median_ms = statistics.median(item["milliseconds_per_step"] for item in values)
        output[mode] = {
            "median_ms": median_ms,
            "speedup_vs_original": samples["original_manual"][0]["milliseconds_per_step"] / median_ms,
            "runs": values,
        }
    original_median = output["original_manual"]["median_ms"]
    for value in output.values():
        value["speedup_vs_original"] = original_median / value["median_ms"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="", help="空值表示从Hugging Face缓存或下载官方L/14权重")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--image-sizes", type=int, nargs="+", default=(448, 644))
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
    model = PESpatial("l", pretrained=args.checkpoint or True).to(device)
    controller = ModeController(model)
    result = {
        "environment": {
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(args.device),
            "amp": "fp16",
            "jit_compile": False,
        },
        "model": {
            "variant": "PE-Spatial-L14-448",
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

            if not args.skip_training:
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
