# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from copy import deepcopy
from numbers import Number
from pathlib import Path

import torch

from ultralytics.utils import LOGGER, RANK, SETTINGS, TESTS_RUNNING, colorstr, torch_utils
from ultralytics.utils.torch_utils import smart_inference_mode

WRITER = None  # TensorBoard SummaryWriter instance
PREFIX = colorstr("TensorBoard: ")
_MODEL_INFO_LOGGED = False
_PROCESSED_PLOTS = {}
_MAX_SAMPLE_IMAGES = 6
_TEXT_VALUE_LIMIT = 1000
_HPARAM_VALUE_LIMIT = 250
_TENSORBOARD_PROCESS = None
_TENSORBOARD_PORT = None
_TENSORBOARD_LOGDIR = None
_TENSORBOARD_HOST = "0.0.0.0"
_TENSORBOARD_CHECK_HOST = "127.0.0.1"
_TENSORBOARD_DEFAULT_PORT = 6006
_TENSORBOARD_MAX_PORT = 6020
_TENSORBOARD_START_TIMEOUT = 5.0
_TENSORBOARD_REQUEST_TIMEOUT = 1.0
_TENSORBOARD_RELOAD_INTERVAL = 5

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["tensorboard"] is True  # verify integration is enabled

    # Imports below only required if TensorBoard enabled
    from torch.utils.tensorboard import SummaryWriter

except (ImportError, AssertionError, TypeError, AttributeError):
    # TypeError for handling 'Descriptors cannot not be created directly.' protobuf errors in Windows
    # AttributeError: module 'tensorflow' has no attribute 'io' if 'tensorflow' not installed
    SummaryWriter = None


def _asdict(x) -> dict:
    """Return a plain dictionary for mappings and namespace-like objects."""
    if isinstance(x, Mapping):
        return dict(x)
    return vars(x) if hasattr(x, "__dict__") else {}


def _stringify(value, limit: int = _TEXT_VALUE_LIMIT) -> str:
    """Convert values to compact strings for TensorBoard text/hparams."""
    if isinstance(value, Path):
        text = str(value)
    else:
        text = str(value) if isinstance(value, (str, Number, bool, type(None))) else repr(value)
    text = text.replace("\n", "\\n").replace("`", "'")
    return text if len(text) <= limit else f"{text[: limit - 3]}..."


def _format_markdown_table(items: Mapping) -> str:
    """Format a mapping as a compact Markdown table for TensorBoard text."""
    rows = ["| Key | Value |", "| --- | --- |"]
    for k, v in sorted(items.items(), key=lambda x: str(x[0])):
        rows.append(f"| `{_stringify(k)}` | `{_stringify(v)}` |")
    return "\n".join(rows)


def _scalar_value(value):
    """Convert a scalar-like object to a TensorBoard-compatible numeric value."""
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "numel") and value.numel() != 1:
        return None
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Number):
        return float(value)
    return None


def _log_scalars(scalars: dict, step: int = 0) -> None:
    """Log scalar values to TensorBoard.

    Args:
        scalars (dict): Dictionary of scalar values to log to TensorBoard. Keys are scalar names and values are the
            corresponding scalar values.
        step (int): Global step value to record with the scalar values. Used for x-axis in TensorBoard graphs.

    Examples:
        Log training metrics
        >>> metrics = {"loss": 0.5, "accuracy": 0.95}
        >>> _log_scalars(metrics, step=100)
    """
    if WRITER and scalars:
        for k, v in scalars.items():
            v = _scalar_value(v)
            if v is not None:
                WRITER.add_scalar(k, v, step)


def _clean_hparam(value):
    """Convert an argparse value to a TensorBoard hparams-compatible scalar."""
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bool):
        return value
    if isinstance(value, Number):
        value = float(value)
        return value if math.isfinite(value) else _stringify(value, _HPARAM_VALUE_LIMIT)
    if isinstance(value, str):
        return value
    return _stringify(value, _HPARAM_VALUE_LIMIT)


def _sanitize_hparams(args) -> dict:
    """Return safely stringified hyperparameters for TensorBoard hparams."""
    return {str(k): _clean_hparam(v) for k, v in _asdict(args).items()}


def _final_metrics(trainer) -> dict:
    """Collect final numeric metrics for TensorBoard hparams."""
    metrics = {}
    if getattr(trainer, "tloss", None) is not None:
        metrics.update(trainer.label_loss_items(trainer.tloss, prefix="train"))
    metrics.update(getattr(trainer, "metrics", None) or {})
    metrics.update(getattr(trainer, "lr", None) or {})
    fitness_items = (
        ("fitness", getattr(trainer, "fitness", None)),
        ("fitness/best", getattr(trainer, "best_fitness", None)),
    )
    for name, value in fitness_items:
        if value is not None:
            metrics[name] = value
    return {k: v for k, v in ((k, _scalar_value(v)) for k, v in metrics.items()) if v is not None}


def _dataset_summary(trainer) -> dict:
    """Return a compact dataset summary suitable for TensorBoard text."""
    data = getattr(trainer, "data", None)
    if not isinstance(data, Mapping):
        return {"data": _stringify(data)}

    preferred = ("path", "train", "val", "test", "nc", "channels", "names", "kpt_shape")
    summary = {k: data[k] for k in preferred if k in data}
    summary["keys"] = sorted(data.keys(), key=str)
    return summary


def _path_status(path) -> str:
    """Return a path string with existence status."""
    try:
        path = Path(path)
        return f"{path} ({'exists' if path.exists() else 'missing'})"
    except TypeError:
        return _stringify(path)


def _artifact_summary(trainer) -> dict:
    """Return paths to local run artifacts without copying large files."""
    save_dir = Path(getattr(trainer, "save_dir", ""))
    artifacts = {
        "save_dir": save_dir,
        "weights_dir": getattr(trainer, "wdir", save_dir / "weights"),
        "best": getattr(trainer, "best", save_dir / "weights" / "best.pt"),
        "last": getattr(trainer, "last", save_dir / "weights" / "last.pt"),
        "args": save_dir / "args.yaml",
        "results": getattr(trainer, "csv", save_dir / "results.csv"),
    }
    return {k: _path_status(v) for k, v in artifacts.items()}


def _log_run_metadata(trainer, step: int = 0) -> None:
    """Log run arguments, dataset summary, and artifact paths as TensorBoard text."""
    if not WRITER:
        return
    try:
        WRITER.add_text("run/args", _format_markdown_table(_asdict(trainer.args)), step)
        WRITER.add_text("run/data", _format_markdown_table(_dataset_summary(trainer)), step)
        WRITER.add_text("run/artifacts", _format_markdown_table(_artifact_summary(trainer)), step)
    except Exception as e:
        LOGGER.warning(f"{PREFIX}failed to log run metadata: {e}")


def _log_model_info(trainer, step: int) -> None:
    """Log static model information once per run."""
    global _MODEL_INFO_LOGGED
    if _MODEL_INFO_LOGGED:
        return
    _MODEL_INFO_LOGGED = True
    try:
        _log_scalars(torch_utils.model_info_for_loggers(trainer), step)
    except Exception as e:
        LOGGER.warning(f"{PREFIX}failed to log model info: {e}")


def _log_hparams(trainer) -> None:
    """Log hparams and final metrics for TensorBoard's HParams plugin."""
    if not WRITER or not hasattr(WRITER, "add_hparams"):
        return
    hparams, metrics = _sanitize_hparams(trainer.args), _final_metrics(trainer)
    if not hparams or not metrics:
        return
    try:
        try:
            WRITER.add_hparams(hparams, metrics, run_name="hparams")
        except TypeError:
            WRITER.add_hparams(hparams, metrics)
    except Exception as e:
        LOGGER.warning(f"{PREFIX}failed to log hparams: {e}")


def _clean_tag(tag: str) -> str:
    """Normalize a TensorBoard tag while preserving hierarchy."""
    return str(tag).replace("\\", "/").replace(" ", "_")


def _log_image(path: Path, tag: str, step: int, timestamp=None) -> None:
    """Log a generated image file to TensorBoard without creating new plots."""
    if not WRITER:
        return
    path = Path(path)
    if not path.exists() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        return
    timestamp = timestamp if timestamp is not None else path.stat().st_mtime
    if _PROCESSED_PLOTS.get(path) == timestamp:
        return
    try:
        import numpy as np
        from PIL import Image

        with Image.open(path) as im:
            image = np.asarray(im.convert("RGB"))
        WRITER.add_image(_clean_tag(tag), image, step, dataformats="HWC")
        _PROCESSED_PLOTS[path] = timestamp
    except Exception as e:
        LOGGER.warning(f"{PREFIX}failed to log image {path}: {e}")


def _log_plots(plots: dict, step: int, prefix: str) -> None:
    """Log existing plot files, capping batch sample images to limit IO."""
    batch_count = 0
    for name, params in sorted((plots or {}).items(), key=lambda x: str(x[0])):
        path = Path(name)
        is_sample = path.name.startswith(("train_batch", "val_batch"))
        if is_sample:
            if batch_count >= _MAX_SAMPLE_IMAGES:
                continue
            batch_count += 1
        timestamp = params.get("timestamp") if isinstance(params, Mapping) else None
        _log_image(path, f"{prefix}/{path.stem}", step, timestamp)


def _log_sample_images(trainer, step: int) -> None:
    """Log a small number of saved train/val batch images if they were generated."""
    save_dir = Path(getattr(trainer, "save_dir", ""))
    for pattern, prefix in (("train_batch*.jpg", "samples/train"), ("val_batch*.jpg", "samples/val")):
        for path in sorted(save_dir.glob(pattern))[:_MAX_SAMPLE_IMAGES]:
            if path not in _PROCESSED_PLOTS:
                _log_image(path, f"{prefix}/{path.stem}", step)


def _flush_writer() -> None:
    """Flush TensorBoard events if supported by the writer."""
    if WRITER and hasattr(WRITER, "flush"):
        try:
            WRITER.flush()
        except Exception as e:
            LOGGER.warning(f"{PREFIX}failed to flush writer: {e}")


def _close_writer() -> None:
    """Close the TensorBoard writer and reset per-run state."""
    global WRITER, _MODEL_INFO_LOGGED, _PROCESSED_PLOTS
    if WRITER and hasattr(WRITER, "close"):
        try:
            WRITER.close()
        except Exception as e:
            LOGGER.warning(f"{PREFIX}failed to close writer: {e}")
    WRITER = None
    _MODEL_INFO_LOGGED = False
    _PROCESSED_PLOTS = {}


def _normalize_logdir(logdir) -> str:
    """返回可比较的 TensorBoard 日志路径。"""
    if not logdir:
        return ""
    try:
        return str(Path(str(logdir)).expanduser().resolve())
    except (OSError, TypeError, ValueError):
        return str(logdir)


def _tensorboard_environment(port: int, timeout: float = _TENSORBOARD_REQUEST_TIMEOUT) -> dict | None:
    """读取指定端口 TensorBoard 的环境信息。"""
    url = f"http://{_TENSORBOARD_CHECK_HOST}:{port}/data/environment"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            data = json.load(response)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, urllib.error.URLError, TimeoutError):
        return None


def _environment_logdir(environment: dict | None) -> str:
    """从 TensorBoard 环境信息中提取 logdir。"""
    if not isinstance(environment, dict):
        return ""
    debug = environment.get("debug")
    flags = debug.get("flags", {}) if isinstance(debug, dict) else {}
    for value in (environment.get("data_location"), flags.get("logdir")):
        if value:
            return str(value)
    return ""


def _logdir_matches(actual, expected) -> bool:
    """判断两个 logdir 是否指向同一目录。"""
    return _normalize_logdir(actual) == _normalize_logdir(expected)


def _port_is_free(port: int) -> bool:
    """判断端口是否可以用于启动新的 TensorBoard。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("", port))
            return True
        except OSError:
            return False


def _is_tensorboard_args(args: list[str]) -> bool:
    """判断进程命令是否像 TensorBoard。"""
    if not args:
        return False
    command = " ".join(args).lower()
    return "tensorboard" in command


def _args_use_port(args: list[str], port: int) -> bool:
    """判断 TensorBoard 命令是否使用目标端口。"""
    explicit_port = None
    for i, arg in enumerate(args):
        if arg == "--port" and i + 1 < len(args):
            explicit_port = args[i + 1]
            break
        if arg.startswith("--port="):
            explicit_port = arg.split("=", 1)[1]
            break
    return explicit_port == str(port) if explicit_port is not None else port == _TENSORBOARD_DEFAULT_PORT


def _visible_tensorboard_pids(port: int) -> list[int]:
    """返回当前命名空间内可见且可能占用目标端口的 TensorBoard 进程。"""
    proc_root = Path("/proc")
    if not proc_root.exists():
        return []

    pids = []
    current_pid = os.getpid()
    for proc_dir in proc_root.iterdir():
        if not proc_dir.name.isdigit():
            continue
        pid = int(proc_dir.name)
        if pid == current_pid:
            continue
        try:
            raw = (proc_dir / "cmdline").read_bytes().rstrip(b"\0")
        except OSError:
            continue
        if not raw:
            continue
        args = [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]
        if _is_tensorboard_args(args) and _args_use_port(args, port):
            pids.append(pid)
    return pids


def _terminate_visible_tensorboard(port: int, timeout: float = 3.0) -> list[int]:
    """终止当前命名空间内可见的目标端口 TensorBoard 进程。"""
    pids = _visible_tensorboard_pids(port)
    for pid in pids:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGTERM)

    deadline = time.time() + timeout
    while pids and time.time() < deadline:
        if _port_is_free(port):
            break
        time.sleep(0.1)

    if pids and not _port_is_free(port):
        for pid in pids:
            with contextlib.suppress(OSError):
                os.kill(pid, signal.SIGKILL)
        deadline = time.time() + 1.0
        while time.time() < deadline and not _port_is_free(port):
            time.sleep(0.1)
    return pids


def _stop_tensorboard_process() -> None:
    """关闭本训练进程启动的 TensorBoard 子进程。"""
    global _TENSORBOARD_PROCESS, _TENSORBOARD_PORT, _TENSORBOARD_LOGDIR
    process = _TENSORBOARD_PROCESS
    if process is not None and process.poll() is None:
        with contextlib.suppress(OSError):
            process.terminate()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(OSError):
                process.kill()
            with contextlib.suppress(Exception):
                process.wait(timeout=1)
    _TENSORBOARD_PROCESS = None
    _TENSORBOARD_PORT = None
    _TENSORBOARD_LOGDIR = None


def _start_tensorboard_server(logdir: Path, port: int) -> str:
    """使用当前 Python 环境启动 TensorBoard 服务。"""
    global _TENSORBOARD_PROCESS, _TENSORBOARD_PORT, _TENSORBOARD_LOGDIR
    _stop_tensorboard_process()

    command = [
        sys.executable,
        "-m",
        "ultralytics.utils.tensorboard_launcher",
        "--logdir",
        str(logdir),
        "--host",
        _TENSORBOARD_HOST,
        "--port",
        str(port),
        "--reload_interval",
        str(_TENSORBOARD_RELOAD_INTERVAL),
        "--reload_multifile",
        "true",
    ]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _TENSORBOARD_PROCESS = process
    _TENSORBOARD_PORT = port
    _TENSORBOARD_LOGDIR = str(logdir)

    deadline = time.time() + _TENSORBOARD_START_TIMEOUT
    while time.time() < deadline:
        if process.poll() is not None:
            _TENSORBOARD_PROCESS = None
            _TENSORBOARD_PORT = None
            _TENSORBOARD_LOGDIR = None
            raise RuntimeError(f"TensorBoard 进程提前退出，命令：{' '.join(command)}")
        environment = _tensorboard_environment(port)
        if _logdir_matches(_environment_logdir(environment), logdir):
            return f"http://localhost:{port}/"
        time.sleep(0.25)

    LOGGER.warning(f"{PREFIX}TensorBoard 已启动但暂未完成环境确认，端口={port}，logdir={logdir}")
    return f"http://localhost:{port}/"


def _ensure_tensorboard_server(logdir: Path) -> None:
    """确保 TensorBoard 服务读取当前训练目录。"""
    expected_logdir = Path(logdir).expanduser().resolve()
    for port in range(_TENSORBOARD_DEFAULT_PORT, _TENSORBOARD_MAX_PORT + 1):
        environment = _tensorboard_environment(port)
        actual_logdir = _environment_logdir(environment)

        if environment is not None:
            if actual_logdir and _logdir_matches(actual_logdir, expected_logdir):
                LOGGER.info(f"{PREFIX}TensorBoard 已指向当前训练目录：{expected_logdir}，访问 http://localhost:{port}/")
                return

            LOGGER.warning(
                f"{PREFIX}端口 {port} 当前 TensorBoard logdir={actual_logdir or '<unknown>'}，"
                f"将尝试重启到 {expected_logdir}"
            )
            killed_pids = _terminate_visible_tensorboard(port)
            if killed_pids:
                LOGGER.info(f"{PREFIX}已终止端口 {port} 的可见 TensorBoard 进程：{killed_pids}")

            if _port_is_free(port):
                url = _start_tensorboard_server(expected_logdir, port)
                LOGGER.info(f"{PREFIX}已在端口 {port} 重启 TensorBoard，logdir={expected_logdir}，访问 {url}")
                return

            LOGGER.warning(f"{PREFIX}端口 {port} 的 TensorBoard 无法释放，继续查找其他端口。")
            continue

        if _port_is_free(port):
            url = _start_tensorboard_server(expected_logdir, port)
            LOGGER.info(f"{PREFIX}已启动 TensorBoard，logdir={expected_logdir}，访问 {url}")
            return

        LOGGER.warning(f"{PREFIX}端口 {port} 已被非 TensorBoard 服务占用，跳过。")

    raise RuntimeError(f"未找到可用端口启动 TensorBoard，logdir={expected_logdir}")


@smart_inference_mode()
def _log_tensorboard_graph(trainer) -> None:
    """Log model graph to TensorBoard.

    This function attempts to visualize the model architecture in TensorBoard by tracing the model with a dummy input
    tensor. It first tries a simple method suitable for YOLO models, and if that fails, falls back to a more complex
    approach for models like RTDETR that may require special handling.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer object containing the model to visualize. Must
            have attributes model and args with imgsz.

    Notes:
        This function requires TensorBoard integration to be enabled and the global WRITER to be initialized.
        It handles potential warnings from the PyTorch JIT tracer and attempts to gracefully handle different
        model architectures.
    """
    # Input image
    imgsz = trainer.args.imgsz
    ch = trainer.data.get("channels", 3)
    imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz

    # Try simple method first (YOLO)
    try:
        model = deepcopy(torch_utils.unwrap_model(trainer.model))
        p = next(model.parameters())  # for device, type
        im = torch.zeros((1, ch, *imgsz), device=p.device, dtype=p.dtype)  # input image (must be zeros, not empty)
        model.eval()  # 仅跟踪副本，避免改变训练模型状态或留下 inference tensor 缓存
        WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])
        LOGGER.info(f"{PREFIX}model graph visualization added ✅")
        return
    except Exception as e1:
        # Fallback to TorchScript export steps (RTDETR)
        try:
            model = deepcopy(torch_utils.unwrap_model(trainer.model))
            p = next(model.parameters())
            im = torch.zeros((1, ch, *imgsz), device=p.device, dtype=p.dtype)
            model.eval()
            model = model.fuse(verbose=False)
            for m in model.modules():
                if hasattr(m, "export"):  # Detect, RTDETRDecoder (Segment and Pose use Detect base class)
                    m.export = True
                    m.format = "torchscript"
            model(im)  # dry run
            WRITER.add_graph(torch.jit.trace(model, im, strict=False), [])
            LOGGER.info(f"{PREFIX}model graph visualization added ✅")
        except Exception as e2:
            LOGGER.warning(f"{PREFIX}TensorBoard graph visualization failure: {e1} -> {e2}")


def on_pretrain_routine_start(trainer) -> None:
    """Initialize TensorBoard logging with SummaryWriter."""
    if RANK not in {-1, 0}:
        return
    if SummaryWriter:
        try:
            global WRITER, _MODEL_INFO_LOGGED, _PROCESSED_PLOTS
            _close_writer()
            _MODEL_INFO_LOGGED = False
            _PROCESSED_PLOTS = {}
            WRITER = SummaryWriter(str(trainer.save_dir))
        except Exception as e:
            LOGGER.warning(f"{PREFIX}TensorBoard not initialized correctly, not logging this run. {e}")
            return
        try:
            _ensure_tensorboard_server(Path(trainer.save_dir))
        except Exception as e:
            LOGGER.warning(
                f"{PREFIX}TensorBoard 服务未自动启动，请手动运行 'tensorboard --logdir {trainer.save_dir}'。错误：{e}"
            )


def on_train_start(trainer) -> None:
    """Log TensorBoard graph and static run metadata."""
    if WRITER:
        _log_run_metadata(trainer)
        _log_tensorboard_graph(trainer)


def on_train_epoch_end(trainer) -> None:
    """Log scalar statistics at the end of a training epoch."""
    _log_scalars(trainer.label_loss_items(trainer.tloss, prefix="train"), trainer.epoch + 1)
    _log_scalars(trainer.lr, trainer.epoch + 1)


def on_fit_epoch_end(trainer) -> None:
    """Log epoch metrics at end of training epoch."""
    step = trainer.epoch + 1
    _log_scalars(trainer.metrics, step)
    scalars = {}
    if getattr(trainer, "epoch_time", None) is not None:
        scalars["time/epoch"] = trainer.epoch_time
    if getattr(trainer, "train_time_start", None) is not None:
        scalars["time/elapsed"] = time.time() - trainer.train_time_start
    if getattr(trainer, "fitness", None) is not None:
        scalars["fitness"] = trainer.fitness
    if getattr(trainer, "best_fitness", None) is not None:
        scalars["fitness/best"] = trainer.best_fitness
    _log_scalars(scalars, step)
    if trainer.epoch == 0:
        _log_model_info(trainer, step)


def on_train_end(trainer) -> None:
    """Log final metadata, hparams, existing images, and flush TensorBoard events."""
    step = trainer.epoch + 1
    _log_run_metadata(trainer, step)
    _log_hparams(trainer)
    _log_plots(getattr(trainer, "plots", {}), step, "plots/train")
    _log_plots(getattr(getattr(trainer, "validator", None), "plots", {}), step, "plots/val")
    _log_sample_images(trainer, step)
    _flush_writer()


def teardown(trainer) -> None:
    """Close TensorBoard writer at the end of a run."""
    _close_writer()
    _stop_tensorboard_process()


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_start": on_train_start,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_epoch_end": on_train_epoch_end,
        "on_train_end": on_train_end,
        "teardown": teardown,
    }
    if SummaryWriter
    else {}
)
