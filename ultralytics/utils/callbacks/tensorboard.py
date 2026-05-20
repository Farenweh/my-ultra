# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import math
import time
from collections.abc import Mapping
from numbers import Number
from pathlib import Path

from ultralytics.utils import LOGGER, RANK, SETTINGS, TESTS_RUNNING, colorstr, torch_utils
from ultralytics.utils.torch_utils import smart_inference_mode

WRITER = None  # TensorBoard SummaryWriter instance
PREFIX = colorstr("TensorBoard: ")
_MODEL_INFO_LOGGED = False
_PROCESSED_PLOTS = {}
_MAX_SAMPLE_IMAGES = 6
_TEXT_VALUE_LIMIT = 1000
_HPARAM_VALUE_LIMIT = 250

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["tensorboard"] is True  # verify integration is enabled

    # Imports below only required if TensorBoard enabled
    from copy import deepcopy

    import torch
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
    p = next(trainer.model.parameters())  # for device, type
    im = torch.zeros((1, ch, *imgsz), device=p.device, dtype=p.dtype)  # input image (must be zeros, not empty)

    # Try simple method first (YOLO)
    try:
        trainer.model.eval()  # place in .eval() mode to avoid BatchNorm statistics changes
        WRITER.add_graph(torch.jit.trace(torch_utils.unwrap_model(trainer.model), im, strict=False), [])
        LOGGER.info(f"{PREFIX}model graph visualization added ✅")
        return
    except Exception as e1:
        # Fallback to TorchScript export steps (RTDETR)
        try:
            model = deepcopy(torch_utils.unwrap_model(trainer.model))
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
            LOGGER.info(f"{PREFIX}Start with 'tensorboard --logdir {trainer.save_dir}', view at http://localhost:6006/")
        except Exception as e:
            LOGGER.warning(f"{PREFIX}TensorBoard not initialized correctly, not logging this run. {e}")


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
