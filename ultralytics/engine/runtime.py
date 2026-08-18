# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""训练器共享的轻量运行设施。"""

from __future__ import annotations

import inspect
import os
from datetime import timedelta
from typing import Any

import torch


class CallbackHost:
    """为不同训练循环提供一致的callback管理接口。"""

    callbacks: dict[str, list]

    def setup_callbacks(self, callback_map: dict[str, list] | None = None) -> dict[str, list]:
        """安装显式callback映射或Ultralytics默认事件映射。"""
        if callback_map is None:
            from ultralytics.utils import callbacks

            callback_map = callbacks.get_default_callbacks()
        self.callbacks = callback_map
        return callback_map

    def add_integration_callbacks(self) -> None:
        """按Ultralytics现有幂等规则注册日志和平台集成callback。"""
        from ultralytics.utils import callbacks

        callbacks.add_integration_callbacks(self)

    def add_callback(self, event: str, callback) -> None:
        """向事件追加一个callback。"""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback) -> None:
        """用单个callback替换事件的现有callback。"""
        self.callbacks[event] = [callback]

    def clear_callback(self, event: str) -> None:
        """清空指定事件的callback。"""
        self.callbacks[event] = []

    def run_callbacks(self, event: str) -> None:
        """依次运行指定事件已注册的callback。"""
        for callback in self.callbacks.get(event, []):
            callback(self)


def initialize_distributed_runtime(
    *,
    device_type: str,
    device_spec: str,
    local_rank: int,
    rank: int,
    world_size: int,
    dist_module: Any,
    accelerator_resolver,
    is_ascend: bool,
    timeout_seconds: int = 10800,
) -> tuple[torch.device, Any, str]:
    """初始化CUDA、Ascend或XPU分布式运行环境并返回当前设备。"""
    devices = device_spec.split(":", 1)[-1].split(",")
    if not 0 <= local_rank < len(devices):
        raise ValueError(f"LOCAL_RANK={local_rank}超出设备列表{device_spec!r}的范围")
    index = int(devices[local_rank])
    device = torch.device(device_type, index)
    accelerator = accelerator_resolver(device)
    accelerator.set_device(index)

    if device_type == "cuda":
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    elif device_type == "npu":
        os.environ["TORCH_HCCL_BLOCKING_WAIT"] = "1"
        os.environ.setdefault("HCCL_CONNECT_TIMEOUT", "1800")
    elif device_type == "xpu" and not (hasattr(dist_module, "is_xccl_available") and dist_module.is_xccl_available()):
        raise RuntimeError("Multi-XPU training requires XCCL, which is not available in this PyTorch build.")

    backend = (
        "hccl"
        if device_type == "npu" or is_ascend
        else "xccl"
        if device_type == "xpu"
        else "nccl"
        if dist_module.is_nccl_available()
        else "gloo"
    )
    init_kwargs = {
        "backend": backend,
        "timeout": timedelta(seconds=timeout_seconds),
        "rank": rank,
        "world_size": world_size,
    }
    parameters = inspect.signature(dist_module.init_process_group).parameters
    accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    if device_type != "cpu" and ("device_id" in parameters or accepts_kwargs):
        init_kwargs["device_id"] = device
    dist_module.init_process_group(
        **init_kwargs,
    )
    return device, accelerator, backend
