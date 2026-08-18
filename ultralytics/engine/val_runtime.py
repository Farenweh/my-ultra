# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""独立多卡验证的共享运行时。

这里只负责设备、进程、任务调度和结果回传，不引入任何 YOLO 任务语义。
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import signal
import subprocess
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.distributed as dist

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.engine.runtime import initialize_distributed_runtime
from ultralytics.utils import LOGGER
from ultralytics.utils.checks import IS_ASCEND
from ultralytics.utils.dist import (
    build_torchrun_command,
    find_free_network_port,
    is_k8s_distributed_parent,
    normalize_k8s_launch_config,
)
from ultralytics.utils.torch_utils import get_torch_device_backend, parse_device

if TYPE_CHECKING:
    from ultralytics.engine.model import Model


_WORKER_ENV = "ULTRALYTICS_DISTRIBUTED_VAL_WORKER"
_STORE_PORT_ENV = "ULTRALYTICS_VAL_STORE_PORT"
_ACTIVE_CONTEXT: DistributedValContext | None = None


def _device_request(device: Any) -> tuple[str, list[int]]:
    """将多设备请求解析为加速器类型和本机设备号。"""
    value = parse_device(device)
    if value.startswith("npu"):
        device_type, _, indices = value.partition(":")
    elif value.startswith("xpu"):
        raise ValueError("独立多卡验证首版仅支持CUDA和Ascend NPU，不支持XPU")
    else:
        device_type, indices = ("npu" if IS_ASCEND else "cuda"), value
    if not indices or indices in {"cpu", "mps"}:
        return indices or device_type, []
    try:
        device_ids = [int(item) for item in indices.split(",")]
    except ValueError as error:
        raise ValueError(f"无法解析多卡验证设备：device={device!r}") from error
    if any(index < 0 for index in device_ids) or len(device_ids) != len(set(device_ids)):
        raise ValueError(f"多卡验证设备必须是互不重复的非负整数，得到{device_ids}")
    return device_type, device_ids


def _visible_devices() -> tuple[str, list[int]]:
    """返回K8S父进程当前可见的所有本地加速器。"""
    if IS_ASCEND and hasattr(torch, "npu") and torch.npu.is_available():
        return "npu", list(range(torch.npu.device_count()))
    if torch.cuda.is_available():
        return "cuda", list(range(torch.cuda.device_count()))
    raise ValueError("多节点验证父进程未检测到可用的CUDA或Ascend设备")


def _validate_global_batch(batch: Any, world_size: int) -> tuple[int, int]:
    """验证严格全局batch并返回(global_batch, local_batch)。"""
    if isinstance(batch, bool) or not isinstance(batch, int):
        raise TypeError(f"分布式验证batch必须是整数，得到{batch!r}")
    if batch < world_size or batch % world_size:
        raise ValueError(
            f"分布式验证使用全局batch：要求batch >= world_size且能整除world_size，"
            f"得到batch={batch}, world_size={world_size}"
        )
    return batch, batch // world_size


def _is_native_model(owner: Model) -> bool:
    """判断当前包装器是否持有可快照的原生PyTorch模型。"""
    return isinstance(getattr(owner, "model", None), torch.nn.Module)


def _validate_callbacks(callbacks: dict[str, list[Callable]]) -> None:
    """在启动worker前给出可理解的callback序列化错误。"""
    for event, functions in callbacks.items():
        for function in functions:
            module = getattr(function, "__module__", "")
            qualname = getattr(function, "__qualname__", "")
            if module == "__main__" or "<locals>" in qualname or getattr(function, "__name__", "") == "<lambda>":
                raise TypeError(
                    f"多卡验证callback {event}:{function!r} 无法在torchrun worker中导入；"
                    "请将它定义为可导入模块的顶层函数"
                )
    try:
        pickle.dumps(callbacks, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        raise TypeError(f"多卡验证callback无法序列化：{error}") from error


def _model_device(model: torch.nn.Module) -> torch.device:
    """返回模型当前设备，无参数模型按CPU处理。"""
    value = next(model.parameters(), None)
    if value is None:
        value = next(model.buffers(), None)
    return value.device if value is not None else torch.device("cpu")


def _save_model_snapshot(owner: Model, path: Path) -> torch.device:
    """保存当前内存模型的FP32快照，并返回原设备。"""
    if not _is_native_model(owner):
        raise TypeError(
            f"多卡验证仅支持原生PyTorch .pt/YAML/内存模型；当前模型类型为{type(getattr(owner, 'model', None)).__name__}"
        )
    original_device = _model_device(owner.model)
    owner.model.to("cpu")
    snapshot = deepcopy(owner.model).float().eval()
    train_args = dict(getattr(snapshot, "args", {}) or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": snapshot, "train_args": train_args}, path)
    del snapshot
    return original_device


def _restore_model_device(owner: Model, device: torch.device) -> None:
    """将父进程模型恢复到启动前设备。"""
    if _is_native_model(owner):
        owner.model.to(device)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """以原子替换写入JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class DynamicBatchScheduler:
    """通过TCPStore跨进程、跨节点原子领取batch序号。"""

    def __init__(self, store: dist.Store, total_samples: int, batch_size: int, namespace: str) -> None:
        self.store = store
        self.total_samples = int(total_samples)
        self.batch_size = int(batch_size)
        self.total_batches = (self.total_samples + self.batch_size - 1) // self.batch_size
        self.key = f"{namespace}/next_batch"
        self._initial_batch_id: int | None = None
        self._initial_claimed = True

    def initialize(self, rank: int, world_size: int | None = None) -> None:
        """由rank 0初始化游标，并为每个rank预留第一个batch。"""
        world_size = int(world_size if world_size is not None else dist.get_world_size())
        if rank == 0:
            self.store.set(self.key, str(min(world_size, self.total_batches)))
        self._initial_batch_id = rank if rank < self.total_batches else None
        self._initial_claimed = False
        dist.barrier()

    def claim(self) -> tuple[int, list[int]] | None:
        """领取下一个batch的全局索引。"""
        if not self._initial_claimed:
            self._initial_claimed = True
            if self._initial_batch_id is not None:
                batch_id = self._initial_batch_id
                start = batch_id * self.batch_size
                end = min(start + self.batch_size, self.total_samples)
                return batch_id, list(range(start, end))
        batch_id = int(self.store.add(self.key, 1)) - 1
        if batch_id >= self.total_batches:
            return None
        start = batch_id * self.batch_size
        end = min(start + self.batch_size, self.total_samples)
        return batch_id, list(range(start, end))


class DynamicBatchSampler(torch.utils.data.Sampler[list[int]]):
    """每次从全局调度器领取一个完整batch。"""

    def __init__(self, scheduler: DynamicBatchScheduler, context: DistributedValContext) -> None:
        self.scheduler = scheduler
        self.context = context

    def __iter__(self):
        while claimed := self.scheduler.claim():
            batch_id, indices = claimed
            self.context.claimed_batches.append(batch_id)
            self.context.claimed_indices.extend(indices)
            yield indices

    def __len__(self) -> int:
        return self.scheduler.total_batches


@dataclass
class DistributedValContext:
    """当前验证worker的分布式状态。"""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    global_batch: int
    local_batch: int
    save_dir: Path
    metrics_path: Path
    progress_path: Path
    store: dist.Store
    namespace: str
    started_at: float
    processed: int = 0

    def __post_init__(self) -> None:
        self.scheduler: DynamicBatchScheduler | None = None
        self.claimed_batches: list[int] = []
        self.claimed_indices: list[int] = []

    def make_scheduler(self, dataset_len: int) -> DynamicBatchScheduler:
        """为已构建的数据集创建或复用调度器。"""
        if self.scheduler is None:
            self.scheduler = DynamicBatchScheduler(self.store, dataset_len, self.local_batch, self.namespace)
            self.scheduler.initialize(self.rank, self.world_size)
            if self.rank == 0:
                _atomic_json(self.save_dir / ".dist" / "dataset.json", {"total_samples": int(dataset_len)})
        elif self.scheduler.total_samples != dataset_len:
            raise RuntimeError("同一验证worker不能对两个不同长度的主数据集复用动态调度器")
        return self.scheduler

    def report_batch(self, count: int) -> None:
        """报告已完成的图片数，供父进程无同步监控。"""
        self.processed += int(count)
        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        _atomic_json(
            self.progress_path,
            {
                "rank": self.rank,
                "processed": self.processed,
                "elapsed": elapsed,
                "images_per_second": self.processed / elapsed,
            },
        )

    def begin_validation(self) -> None:
        """在模型warmup和数据集初始化完成后重置吞吐计时。"""
        self.started_at = time.perf_counter()
        self.processed = 0
        self.progress_path.unlink(missing_ok=True)

    def aggregate_speed(self, profiles: tuple[Any, ...]) -> dict[str, float]:
        """汇总四阶段设备时间与全局墙钟吞吐。"""
        values = torch.tensor(
            [*(float(profile.t) for profile in profiles), float(self.processed)],
            dtype=torch.float32,
            device=self.device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        wall = torch.tensor(time.perf_counter() - self.started_at, dtype=torch.float32, device=self.device)
        dist.all_reduce(wall, op=dist.ReduceOp.MAX)
        count = max(float(values[-1].item()), 1.0)
        names = ("preprocess", "inference", "loss", "postprocess")
        speed = {name: float(values[index].item()) / count * 1e3 for index, name in enumerate(names)}
        speed["images_per_second"] = float(values[-1].item()) / max(float(wall.item()), 1e-9)
        return speed


def get_distributed_val_context() -> DistributedValContext | None:
    """返回当前worker的独立分布式验证上下文。"""
    return _ACTIVE_CONTEXT


def broadcast_val_tensor(value: torch.Tensor | None, src: int = 0) -> torch.Tensor:
    """在独立分布式验证worker中广播动态形状Tensor。"""
    context = get_distributed_val_context()
    if context is None:
        if value is None:
            raise ValueError("非分布式调用broadcast_val_tensor时value不能为None")
        return value
    metadata = [(tuple(value.shape), value.dtype) if context.rank == src and value is not None else None]
    dist.broadcast_object_list(metadata, src=src, device=context.device)
    shape, dtype = metadata[0]
    if context.rank != src:
        value = torch.empty(shape, dtype=dtype, device=context.device)
    else:
        value = value.to(context.device)
    dist.broadcast(value, src=src)
    return value


def _create_store(config: dict[str, Any], rank: int, world_size: int) -> dist.Store:
    """创建专用于动态任务的TCPStore。"""
    return dist.TCPStore(
        config["store_host"],
        int(config["store_port"]),
        world_size,
        rank == 0,
        timedelta(seconds=1800),
        True,
    )


def create_k8s_parent_store(k8s) -> dist.Store:
    """为K8S各节点父进程创建一次性TCPStore，避免共享manifest陈旧竞态。"""
    default_port = k8s.master_port + 1
    port = int(os.getenv("ULTRALYTICS_VAL_PARENT_STORE_PORT", str(default_port)))
    if not 1 <= port <= 65535:
        raise ValueError(f"K8S验证父进程TCPStore端口非法：{port}")
    return dist.TCPStore(
        k8s.master_addr,
        port,
        k8s.nnodes,
        k8s.node_rank == 0,
        timedelta(seconds=1800),
        True,
    )


def _worker_context(config: dict[str, Any]) -> DistributedValContext:
    """初始化验证worker的设备、process group和调度store。"""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != int(config["world_size"]):
        raise RuntimeError(f"验证worker WORLD_SIZE={world_size}与配置{config['world_size']}不一致")
    device, _, _ = initialize_distributed_runtime(
        device_type=config["device_type"],
        device_spec=",".join(str(value) for value in config["device_ids"]),
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        dist_module=dist,
        accelerator_resolver=get_torch_device_backend,
        is_ascend=config["device_type"] == "npu",
    )
    store = _create_store(config, rank, world_size)
    dist.barrier()
    save_dir = Path(config["save_dir"])
    return DistributedValContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        global_batch=int(config["global_batch"]),
        local_batch=int(config["local_batch"]),
        save_dir=save_dir,
        metrics_path=Path(config["metrics_path"]),
        progress_path=save_dir / ".dist" / f"progress.rank{rank}.json",
        store=store,
        namespace=config["launch_id"],
        started_at=time.perf_counter(),
    )


def _finish_worker(context: DistributedValContext, metrics: Any) -> Any:
    """保存rank 0指标，并让所有worker返回同一份metrics。"""
    dist.barrier()
    if context.rank == 0:
        torch.save(metrics, context.metrics_path)
    dist.barrier()
    result = torch.load(context.metrics_path, map_location="cpu", weights_only=False)
    dist.barrier()
    return result


def _worker_failure(context: DistributedValContext | None, error: BaseException) -> None:
    """写入worker失败记录。"""
    if context is not None:
        _atomic_json(
            context.save_dir / ".dist" / f"failure.rank{context.rank}.json",
            {"rank": context.rank, "type": type(error).__name__, "message": str(error)},
        )


def _read_worker_failures(save_dir: Path) -> list[str]:
    """读取worker失败记录，供父进程返回可操作的错误。"""
    failures = []
    for path in sorted((save_dir / ".dist").glob("failure.rank*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        failures.append(f"rank={record.get('rank', '?')} {record.get('type', 'Error')}: {record.get('message', '')}")
    return failures


def _distributed_oom_error(context: DistributedValContext, error: BaseException) -> RuntimeError | None:
    """将CUDA/NPU OOM补充为包含全局和本地batch的错误。"""
    if "out of memory" not in str(error).lower() and "oom" not in type(error).__name__.lower():
        return None
    memory = ""
    try:
        accelerator = get_torch_device_backend(context.device)
        free, total = accelerator.mem_get_info(context.device.index)
        memory = f"，可用/总显存={free / 2**30:.2f}/{total / 2**30:.2f} GiB"
    except (AttributeError, RuntimeError, TypeError):
        pass
    return RuntimeError(
        f"分布式验证OOM：global_batch={context.global_batch}, local_batch={context.local_batch}, "
        f"rank={context.rank}, device={context.device}{memory}。不会自动降低batch。"
    )


def _run_worker(owner: Model, args: dict[str, Any], direct: Callable[[dict[str, Any]], Any]) -> Any:
    """在torchrun worker中执行原有Validator并回传指标。"""
    global _ACTIVE_CONTEXT
    config = _load_worker_config()
    context = _worker_context(config)
    _ACTIVE_CONTEXT = context
    effective = dict(args)
    effective["batch"] = context.local_batch
    effective["device"] = config["device_argument"]
    effective["save_dir"] = str(context.save_dir)
    try:
        metrics = direct(effective)
        metrics = _finish_worker(context, metrics)
        owner.metrics = metrics
        return metrics
    except BaseException as error:
        _worker_failure(context, error)
        if oom_error := _distributed_oom_error(context, error):
            raise oom_error from error
        raise
    finally:
        _ACTIVE_CONTEXT = None
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _load_worker_config() -> dict[str, Any]:
    """读取当前worker的pickle配置。"""
    path = os.getenv("ULTRALYTICS_DISTRIBUTED_VAL_CONFIG")
    if not path:
        raise RuntimeError("分布式验证worker缺少ULTRALYTICS_DISTRIBUTED_VAL_CONFIG")
    with Path(path).open("rb") as file:
        return pickle.load(file)


def _prepare_external_worker_config(owner: Model, args: dict[str, Any]) -> None:
    """为用户手动torchrun启动的进程构造无嵌套worker配置。"""
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1"))
    if args.get("device") in {None, "", "none"}:
        device_type, device_ids = _visible_devices()
    else:
        device_type, device_ids = _device_request(args["device"])
    if len(device_ids) != local_world_size:
        raise ValueError(
            f"手动torchrun验证需要device列表长度等于LOCAL_WORLD_SIZE，"
            f"得到device={device_ids}, LOCAL_WORLD_SIZE={local_world_size}"
        )
    global_batch, local_batch = _validate_global_batch(args.get("batch"), world_size)
    effective_args = dict(args)
    effective_args["exist_ok"] = True
    save_dir = get_save_dir(get_cfg(overrides=effective_args))
    save_dir.mkdir(parents=True, exist_ok=True)
    master_port = int(os.environ["MASTER_PORT"])
    default_store_port = master_port + 1 if master_port < 65535 else master_port - 1
    store_port = int(os.getenv(_STORE_PORT_ENV, "0")) or default_store_port
    if not 1 <= store_port <= 65535 or store_port == master_port:
        raise ValueError(
            f"{_STORE_PORT_ENV}={store_port}非法或与torchrun MASTER_PORT={master_port}冲突，请指定其他端口"
        )
    config = {
        "device_type": device_type,
        "device_ids": device_ids,
        "device_argument": f"{device_type}:" + ",".join(str(value) for value in device_ids),
        "world_size": world_size,
        "global_batch": global_batch,
        "local_batch": local_batch,
        "save_dir": str(save_dir),
        "metrics_path": str(save_dir / ".dist" / "metrics.pt"),
        "store_host": os.environ["MASTER_ADDR"],
        "store_port": store_port,
        "launch_id": hashlib.sha256(
            f"external|{os.environ['MASTER_ADDR']}|{os.environ['MASTER_PORT']}".encode()
        ).hexdigest()[:16],
    }
    config_path = save_dir / ".dist" / f"external_config.rank{rank}.pkl"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("wb") as file:
        pickle.dump(config, file, protocol=pickle.HIGHEST_PROTOCOL)
    os.environ[_WORKER_ENV] = "1"
    os.environ["ULTRALYTICS_DISTRIBUTED_VAL_CONFIG"] = str(config_path)


def _monitor_process(
    process: subprocess.Popen,
    save_dir: Path,
    world_size: int,
    total: int | None,
    *,
    log_progress: bool = True,
) -> None:
    """监控worker进程并周期性打印全局进度。"""
    last_log = 0.0
    while process.poll() is None:
        now = time.monotonic()
        if log_progress and now - last_log >= 5.0:
            if total is None:
                try:
                    total = int(
                        json.loads((save_dir / ".dist" / "dataset.json").read_text(encoding="utf-8"))["total_samples"]
                    )
                except (FileNotFoundError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
                    pass
            processed = 0
            elapsed = 0.0
            for rank in range(world_size):
                path = save_dir / ".dist" / f"progress.rank{rank}.json"
                if not path.is_file():
                    continue
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                processed += int(record.get("processed", 0))
                elapsed = max(elapsed, float(record.get("elapsed", 0.0)))
            speed = processed / elapsed if elapsed else 0.0
            suffix = f"/{total}" if total else ""
            progress = f"（{processed / total:.1%}）" if total else ""
            eta = f"，ETA {(total - processed) / speed:.1f}s" if total and speed and processed < total else ""
            LOGGER.info(f"分布式验证进度：{processed}{suffix}{progress}，global {speed:.2f} images/s{eta}")
            last_log = now
        time.sleep(0.5)


def _runner_file(dist_dir: Path, config_path: Path) -> Path:
    """生成最小torchrun入口。"""
    dist_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        prefix="_distributed_val_",
        dir=dist_dir,
        delete=False,
        encoding="utf-8",
    ) as file:
        file.write(
            "from ultralytics.engine.val_runtime import distributed_val_from_config\n"
            f"distributed_val_from_config({str(config_path)!r})\n"
        )
        return Path(file.name)


def _launch_config(
    owner: Model,
    args: dict[str, Any],
    special: dict[str, Any],
    device_type: str,
    device_ids: list[int],
    world_size: int,
    local_batch: int,
    save_dir: Path,
    store_host: str,
    store_port: int,
) -> dict[str, Any]:
    """构造可由所有节点读取的worker配置。"""
    dist_dir = save_dir / ".dist"
    snapshot_path = dist_dir / "model.snapshot.pt"
    metrics_path = dist_dir / "metrics.pt"
    return {
        "wrapper_module": owner.__class__.__module__,
        "wrapper_name": owner.__class__.__name__,
        "snapshot_path": str(snapshot_path),
        "callbacks": owner.callbacks,
        "args": args,
        "special": special,
        "device_type": device_type,
        "device_ids": device_ids,
        "device_argument": f"{device_type}:" + ",".join(str(value) for value in device_ids),
        "world_size": world_size,
        "global_batch": int(args["batch"]),
        "local_batch": local_batch,
        "save_dir": str(save_dir),
        "metrics_path": str(metrics_path),
        "store_host": store_host,
        "store_port": store_port,
        "launch_id": hashlib.sha256(f"{save_dir}|{store_port}".encode()).hexdigest()[:16],
    }


def _read_dataset_total(args: dict[str, Any]) -> int | None:
    """父进程无需触发数据下载，无法快速确定时返回None。"""
    return None


def _launch(
    owner: Model,
    args: dict[str, Any],
    special: dict[str, Any],
    device_type: str,
    device_ids: list[int],
) -> Any:
    """由普通或K8S父进程启动分布式验证。"""
    k8s = normalize_k8s_launch_config(len(device_ids)) if is_k8s_distributed_parent() else None
    nnodes = k8s.nnodes if k8s else 1
    node_rank = k8s.node_rank if k8s else 0
    world_size = len(device_ids) * nnodes
    _, local_batch = _validate_global_batch(args.get("batch"), world_size)
    _validate_callbacks(owner.callbacks)

    parent_store = create_k8s_parent_store(k8s) if k8s else None
    config_path: Path
    original_device = _model_device(owner.model) if _is_native_model(owner) else torch.device("cpu")
    success = False
    try:
        if node_rank == 0:
            cfg = get_cfg(overrides=args)
            save_dir = get_save_dir(cfg)
            save_dir.mkdir(parents=True, exist_ok=True)
            dist_dir = save_dir / ".dist"
            dist_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = dist_dir / "model.snapshot.pt"
            original_device = _save_model_snapshot(owner, snapshot_path)
            store_port = int(os.getenv(_STORE_PORT_ENV, "0")) or find_free_network_port()
            parent_store_port = (
                int(os.getenv("ULTRALYTICS_VAL_PARENT_STORE_PORT", str(k8s.master_port + 1))) if k8s else None
            )
            if k8s and store_port in {k8s.master_port, parent_store_port}:
                raise ValueError(
                    f"{_STORE_PORT_ENV}={store_port}与torchrun或父进程协调端口冲突，请为动态调度指定其他端口"
                )
            store_host = k8s.master_addr if k8s else "127.0.0.1"
            args = dict(args)
            args["save_dir"] = str(save_dir)
            config = _launch_config(
                owner,
                args,
                special,
                device_type,
                device_ids,
                world_size,
                local_batch,
                save_dir,
                store_host,
                store_port,
            )
            config_path = dist_dir / "config.pkl"
            with config_path.open("wb") as file:
                pickle.dump(config, file, protocol=pickle.HIGHEST_PROTOCOL)
            for pattern in ("progress.rank*.json", "failure.rank*.json", "metrics.pt", "dataset.json"):
                for path in dist_dir.glob(pattern):
                    path.unlink(missing_ok=True)
            if parent_store is not None:
                parent_store.set("config_path", str(config_path))
        else:
            config_path = Path(parent_store.get("config_path").decode())
            config = _load_pickle(config_path)
            save_dir = Path(config["save_dir"])
            owner.model.to("cpu")

        runner = _runner_file(save_dir / ".dist", config_path)
        torchrun_port = k8s.master_port if k8s else find_free_network_port()
        while not k8s and torchrun_port == int(config["store_port"]):
            torchrun_port = find_free_network_port()
        command = build_torchrun_command(
            runner=runner,
            nproc_per_node=len(device_ids),
            master_port=torchrun_port,
            nnodes=nnodes,
            node_rank=node_rank,
            master_addr=k8s.master_addr if k8s else None,
        )
        LOGGER.info("分布式验证启动命令：" + " ".join(command))
        environment = os.environ.copy()
        environment[_WORKER_ENV] = "1"
        environment["ULTRALYTICS_DISTRIBUTED_VAL_CONFIG"] = str(config_path)
        process = subprocess.Popen(command, env=environment)
        try:
            _monitor_process(
                process,
                save_dir,
                world_size,
                _read_dataset_total(args),
                log_progress=node_rank == 0,
            )
            return_code = process.wait()
        except KeyboardInterrupt:
            LOGGER.warning("收到中断信号，正在停止分布式验证worker…")
            process.send_signal(signal.SIGINT)
            process.wait()
            raise
        if return_code:
            failures = _read_worker_failures(save_dir)
            called_process_error = subprocess.CalledProcessError(return_code, command)
            if failures:
                raise RuntimeError("分布式验证worker失败：" + "；".join(failures)) from called_process_error
            raise called_process_error
        metrics_path = Path(config["metrics_path"])
        if not metrics_path.is_file():
            raise RuntimeError(f"分布式验证未生成metrics文件：{metrics_path}")
        metrics = torch.load(metrics_path, map_location="cpu", weights_only=False)
        owner.metrics = metrics
        success = True
        if parent_store is not None:
            parent_store.set(f"parent_done/{node_rank}", "1")
            if node_rank == 0:
                parent_store.wait([f"parent_done/{rank}" for rank in range(nnodes)])
        return metrics
    finally:
        _restore_model_device(owner, original_device)
        if success and node_rank == 0:
            for candidate in (save_dir / ".dist").glob("_distributed_val_*.py"):
                candidate.unlink(missing_ok=True)
            Path(config.get("snapshot_path", "")).unlink(missing_ok=True)
            config_path.unlink(missing_ok=True)


def _load_pickle(path: str | Path) -> Any:
    """读取可信的内部pickle文件。"""
    with Path(path).open("rb") as file:
        return pickle.load(file)


def run_or_launch_distributed_validation(
    owner: Model,
    args: dict[str, Any],
    direct: Callable[[dict[str, Any]], Any],
    *,
    special: dict[str, Any] | None = None,
    custom_validator: Any = None,
) -> Any:
    """直接验证、父进程启动和worker执行的统一入口。"""
    special = special or {}
    if os.getenv(_WORKER_ENV) == "1":
        return _run_worker(owner, args, direct)
    if int(os.getenv("LOCAL_RANK", "-1")) >= 0:
        _prepare_external_worker_config(owner, args)
        return _run_worker(owner, args, direct)

    if is_k8s_distributed_parent():
        if args.get("device") not in {None, "", "none"}:
            raise ValueError("K8S多节点验证不应手动设置device，请使用device=None自动选择本地可见设备")
        device_type, device_ids = _visible_devices()
        should_launch = True
    else:
        device_type, device_ids = _device_request(args.get("device"))
        should_launch = len(device_ids) > 1
    if not should_launch:
        return direct(args)
    if custom_validator is not None:
        raise TypeError("自动多卡验证不支持传入validator实例，请使用模型默认Validator")
    return _launch(owner, args, special, device_type, device_ids)


def distributed_val_from_config(config_path: str | Path) -> Any:
    """torchrun临时入口：重建包装器并调用原生val()。"""
    os.environ[_WORKER_ENV] = "1"
    os.environ["ULTRALYTICS_DISTRIBUTED_VAL_CONFIG"] = str(config_path)
    config = _load_pickle(config_path)
    module = importlib.import_module(config["wrapper_module"])
    wrapper = getattr(module, config["wrapper_name"])
    owner = wrapper(config["snapshot_path"])
    owner.callbacks = config["callbacks"]
    kwargs = dict(config["args"])
    kwargs.pop("model", None)
    kwargs.pop("mode", None)
    return owner.val(**config.get("special", {}), **kwargs)


__all__ = (
    "broadcast_val_tensor",
    "create_k8s_parent_store",
    "DynamicBatchSampler",
    "DistributedValContext",
    "get_distributed_val_context",
    "run_or_launch_distributed_validation",
)
