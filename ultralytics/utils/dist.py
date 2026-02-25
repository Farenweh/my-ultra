# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import shutil
import socket
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .torch_utils import TORCH_1_9

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer


@dataclass
class K8sLaunchConfig:
    """Normalized K8s task rendezvous metadata."""

    master_addr: str
    master_port: int
    nnodes: int
    node_rank: int


def _env_int(name: str, default: int | None = None) -> int | None:
    """Parse an integer environment variable and surface a clear K8s-env error when malformed."""
    value = os.getenv(name)
    if value in {None, ""}:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"K8s 任务环境非法：{name}={value!r} 不是整数。") from exc


def find_free_network_port() -> int:
    """Find a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.

    Returns:
        (int): The available network port number.

    Notes:
        Candidates are drawn below the default OS ephemeral floor (32768 on Linux, 49152 on macOS and Windows)
        because the port is released here and rebound later by the DDP subprocess. An ephemeral port can be handed to
        any outbound connection in that window, which surfaces as an EADDRINUSE rendezvous failure at launch.
    """
    import random

    # SystemRandom as init_seeds() seeds the global RNG earlier in this process, which would hand every concurrent
    # DDP launch on a host the same candidate list
    for port in random.SystemRandom().sample(range(10000, 32768), 10):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue  # in use by an explicit listener, try the next candidate
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))  # no non-ephemeral candidate available, fall back to an ephemeral port
        return s.getsockname()[1]


def is_k8s_training_enabled() -> bool:
    """Return True when the current process was launched through the explicit K8s entrypoint."""
    return os.getenv("K8S_TRAINING") == "1"


def is_k8s_distributed_parent() -> bool:
    """Return True when running as the K8s-managed parent launcher process."""
    return is_k8s_training_enabled() and _env_int("LOCAL_RANK", -1) == -1


def normalize_k8s_launch_config(nproc_per_node: int) -> K8sLaunchConfig | None:
    """Normalize K8s pod-level rendezvous envs into a torchrun-compatible multi-node config."""
    if not is_k8s_distributed_parent():
        return None
    if nproc_per_node < 1:
        raise ValueError("任务提交分布式状态下至少需要检测到一张可用加速卡。")

    required = ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    if any(os.getenv(key) in {None, ""} for key in required):
        raise ValueError("当前环境不存在 K8s 任务提交父进程所需的 rendezvous 环境变量。")

    node_rank = _env_int("RANK")
    world_size = _env_int("WORLD_SIZE")
    master_port = _env_int("MASTER_PORT")
    if node_rank is None or world_size is None or master_port is None:
        raise ValueError("当前环境不存在 K8s 任务提交父进程所需的 rendezvous 环境变量。")

    nnodes = world_size - 1
    master_addr = os.environ["MASTER_ADDR"]
    if nnodes < 1:
        raise ValueError(f"K8s 任务环境非法：WORLD_SIZE={os.environ['WORLD_SIZE']} 无法推导有效节点数。")
    if not 0 <= node_rank < nnodes:
        raise ValueError(f"K8s 任务环境非法：RANK={node_rank} 不在 [0, {nnodes}) 范围内。")

    return K8sLaunchConfig(
        master_addr=master_addr,
        master_port=master_port,
        nnodes=nnodes,
        node_rank=node_rank,
    )


def generate_ddp_file(trainer: BaseTrainer, dist_dir: str | Path | None = None) -> Path:
    """Generate a DDP (Distributed Data Parallel) file for multi-GPU training.

    This function creates a temporary Python file that enables distributed training across multiple GPUs. The file
    contains the necessary configuration to initialize the trainer in a distributed environment.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer containing training configuration and arguments.
            Must have args attribute and be a class instance.

    Returns:
        (Path): Path to the generated temporary DDP file.

    Notes:
        The generated file is saved in the trainer's shared DDP directory and includes:
        - Trainer class import
        - Configuration overrides from the trainer arguments
        - Model path configuration
        - Training initialization code
    """
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)

    # Serialize augmentations to JSON-safe dicts to avoid NameError in DDP subprocess
    overrides = vars(trainer.args).copy()
    overrides["save_dir"] = str(Path(trainer.save_dir).resolve())
    if overrides.get("augmentations") is not None:
        import albumentations as A

        overrides["augmentations"] = [A.to_dict(t) for t in overrides["augmentations"]]

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
from pathlib import Path, PosixPath  # For model arguments stored as Path instead of str
overrides = {overrides}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    # Deserialize augmentations from dicts back to Albumentations transform objects
    if overrides.get("augmentations") is not None:
        import albumentations as A
        overrides["augmentations"] = [A.from_dict(t) for t in overrides["augmentations"]]

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.args.model = "{getattr(trainer.hub_session, "model_url", trainer.args.model)}"
    results = trainer.train()
"""
    dist_dir = Path(dist_dir or Path(trainer.save_dir) / ".dist")
    dist_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=dist_dir,
        delete=False,
    ) as file:
        file.write(content)
    return Path(file.name)


def build_torchrun_command(
    *,
    runner: str | Path,
    nproc_per_node: int,
    master_port: int,
    nnodes: int = 1,
    node_rank: int = 0,
    master_addr: str | None = None,
) -> list[str]:
    """Build a torchrun command."""
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    cmd = [sys.executable, "-m", dist_cmd, "--nproc_per_node", f"{nproc_per_node}"]
    if nnodes > 1:
        if not master_addr:
            raise ValueError("'master_addr' is required when nnodes > 1.")
        cmd.extend(
            [
                "--nnodes",
                f"{nnodes}",
                "--node_rank",
                f"{node_rank}",
                "--master_addr",
                master_addr,
                "--master_port",
                f"{master_port}",
            ]
        )
    else:
        cmd.extend(["--master_port", f"{master_port}"])
    cmd.append(str(runner))
    return cmd


def generate_k8s_ddp_command(trainer: BaseTrainer) -> tuple[list[str], str]:
    """Generate a multi-node torchrun command from K8s-provided pod-level envs."""
    file = generate_ddp_file(trainer)
    config = normalize_k8s_launch_config(getattr(trainer, "local_world_size", trainer.world_size))
    if config is None:
        raise ValueError("当前环境不存在 K8s 任务提交父进程所需的 rendezvous 环境变量。")
    cmd = build_torchrun_command(
        runner=file,
        nproc_per_node=trainer.local_world_size,
        master_port=config.master_port,
        nnodes=config.nnodes,
        node_rank=config.node_rank,
        master_addr=config.master_addr,
    )
    return cmd, str(file)


def generate_ddp_command(trainer: BaseTrainer) -> tuple[list[str], str]:
    """Generate command for distributed training.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer containing configuration for distributed training.

    Returns:
        cmd (list[str]): The command to execute for distributed training.
        file (str): Path to the temporary file created for DDP training.
    """
    file = generate_ddp_file(trainer)
    port = find_free_network_port()
    cmd = build_torchrun_command(
        runner=file,
        nproc_per_node=getattr(trainer, "local_world_size", trainer.world_size),
        master_port=port,
    )
    return cmd, str(file)


def ddp_cleanup(trainer: BaseTrainer, file: str | Path | list[str | Path] | tuple[str | Path, ...] | set[str | Path]) -> None:
    """Delete temporary file if created during distributed data parallel (DDP) training.

    This function checks if the provided file contains the trainer's ID in its name, indicating it was created as a
    temporary file for DDP training, and deletes it if so.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer used for distributed training.
        file (str | Path | list[str | Path]): Path or paths that might need to be deleted.

    Examples:
        >>> trainer = YOLOTrainer()
        >>> file = "/tmp/ddp_temp_123456789.py"
        >>> ddp_cleanup(trainer, file)
    """
    files = file if isinstance(file, (list, tuple, set)) else [file]
    for candidate in files:
        if not candidate:
            continue
        path = Path(candidate)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            continue
        if path.exists() and f"{id(trainer)}.py" in path.name:
            path.unlink()
