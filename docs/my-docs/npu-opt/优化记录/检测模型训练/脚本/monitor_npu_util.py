# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""在训练基准预热结束后，用 npu-smi 按秒采样 AICore、带宽和 NPU 利用率。"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """解析监控参数和待运行命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("缺少待运行命令")
    if args.ready_file.exists():
        parser.error(f"ready file 已存在：{args.ready_file}")
    return args


def summarize(values: list[int]) -> dict:
    """汇总整数采样序列。"""
    if not values:
        return {"样本数": 0, "均值": None, "中位数": None, "最小值": None, "最大值": None}
    return {
        "样本数": len(values),
        "均值": statistics.mean(values),
        "中位数": statistics.median(values),
        "最小值": min(values),
        "最大值": max(values),
    }


def main(args: argparse.Namespace) -> int:
    """运行命令，并只在 ready marker 之后采集稳态设备利用率。"""
    # 编译日志可能超过 pipe 容量；临时文件可避免等待 ready marker 时子进程因无人读取 stdout 而阻塞。
    with tempfile.TemporaryFile(mode="w+t") as process_log, tempfile.TemporaryFile(mode="w+t") as monitor_log:
        process = subprocess.Popen(args.command, stdout=process_log, stderr=subprocess.STDOUT, text=True)
        deadline = time.monotonic() + 600
        while process.poll() is None and not args.ready_file.exists() and time.monotonic() < deadline:
            time.sleep(0.25)

        monitor = None
        if args.ready_file.exists() and process.poll() is None:
            monitor = subprocess.Popen(
                [
                    "npu-smi",
                    "info",
                    "watch",
                    "-i",
                    str(args.physical_device),
                    "-c",
                    "0",
                    "-d",
                    "1",
                    "-s",
                    "anmb",
                ],
                stdout=monitor_log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        process.wait()
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)
        process_log.seek(0)
        process_output = process_log.read()
        monitor_log.seek(0)
        monitor_output = monitor_log.read()
    print(process_output, end="")

    samples = {"AICore_%": [], "HBM_%": [], "HBM带宽_%": [], "NPU利用率_%": []}
    for line in monitor_output.splitlines()[1:]:
        fields = line.split()
        if len(fields) < 6 or not all(field.isdigit() for field in fields[:6]):
            continue
        samples["AICore_%"].append(int(fields[2]))
        samples["HBM_%"].append(int(fields[3]))
        samples["HBM带宽_%"].append(int(fields[4]))
        samples["NPU利用率_%"].append(int(fields[5]))

    # 计算停止后、HBM 释放前，npu-smi 可能额外输出几个尾部样本。
    while samples["AICore_%"] and samples["AICore_%"][-1] == 0 and samples["NPU利用率_%"][-1] < 20:
        for values in samples.values():
            values.pop()

    result = {
        "状态码": process.returncode,
        "物理NPU": args.physical_device,
        "命令": args.command,
        "采样": {name: summarize(values) for name, values in samples.items()},
        "原始采样": samples,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    args.output.write_text(text + "\n")
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main(parse_args()))
