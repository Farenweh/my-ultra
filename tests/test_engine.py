# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import sys
import threading
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tests import MODEL, SOURCE, TASK_MODEL_DATA
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine import trainer as trainer_module
from ultralytics.engine.exporter import Exporter
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models.yolo import classify, depth, detect, obb, pose, segment, semantic
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import ASSETS, DEFAULT_CFG, IS_RASPBERRYPI, WEIGHTS_DIR
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.torch_utils import unwrap_model


def _require_npu_device():
    """返回 NPU 测试设备；无 NPU 时跳过对应测试。"""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("NPU is required for this EMA test")
    torch.npu.set_device(0)
    return torch.device("npu:0")


def test_func(*args, **kwargs):
    """Test function used as a callback stub to verify callback registration."""
    print("callback test passed")


def test_export(monkeypatch, tmp_path):
    """Test model exporting functionality by adding a callback and verifying its execution."""
    monkeypatch.chdir(tmp_path)
    exporter = Exporter()
    exporter.add_callback("on_export_start", test_func)
    assert test_func in exporter.callbacks["on_export_start"], "on_export_start callback not registered"
    f = exporter(model=YOLO("yolo26n.yaml").model)
    YOLO(f)(SOURCE)  # exported model inference


def test_task_aligned_assigner_masks_invalid_nan_metrics_before_normalization():
    assigner = TaskAlignedAssigner(topk=1, num_classes=2)

    def get_pos_mask(*args, **kwargs):
        mask_pos = torch.tensor([[[1.0, 0.0, 0.0]]])
        align_metric = torch.tensor([[[0.5, float("nan"), float("nan")]]])
        overlaps = torch.tensor([[[0.7, float("nan"), float("nan")]]])
        return mask_pos, align_metric, overlaps

    assigner.get_pos_mask = get_pos_mask
    _, _, target_scores, fg_mask, _ = assigner(
        torch.zeros(1, 3, 2),
        torch.zeros(1, 3, 4),
        torch.zeros(3, 2),
        torch.zeros(1, 1, 1),
        torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
        torch.ones(1, 1, 1),
    )

    assert torch.isfinite(target_scores).all()
    assert torch.equal(fg_mask, torch.tensor([[True, False, False]]))


def test_task_aligned_assigner_cuda_oom_retries_on_cpu(monkeypatch):
    """确认非 NPU assigner 遇到 CUDA OOM 时保留 CPU 恢复路径。"""
    assigner = TaskAlignedAssigner(topk=1, num_classes=2)
    calls = 0

    def retry_on_cpu(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("CUDA out of memory")
        return args[:5]

    monkeypatch.setattr(assigner, "_forward", retry_on_cpu)
    result = assigner(
        torch.zeros(1, 3, 2),
        torch.zeros(1, 3, 4),
        torch.zeros(3, 2),
        torch.zeros(1, 1, 1),
        torch.tensor([[[0.0, 0.0, 1.0, 1.0]]]),
        torch.ones(1, 1, 1),
    )

    assert calls == 2
    assert all(t.device.type == "cpu" for t in result)


def test_task_aligned_assigner_npu_oom_fails_fast(monkeypatch):
    """确认 NPU assigner 遇到 OOM 时不把整批数据搬到 CPU 重试。"""
    device = _require_npu_device()
    assigner = TaskAlignedAssigner(topk=1, num_classes=2)
    calls = 0

    def fail_on_npu(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("NPU out of memory")

    monkeypatch.setattr(assigner, "_forward", fail_on_npu)
    with pytest.raises(RuntimeError, match="NPU out of memory"):
        assigner(
            torch.zeros(1, 3, 2, device=device),
            torch.zeros(1, 3, 4, device=device),
            torch.zeros(3, 2, device=device),
            torch.zeros(1, 1, 1, device=device),
            torch.tensor([[[0.0, 0.0, 1.0, 1.0]]], device=device),
            torch.ones(1, 1, 1, device=device),
        )

    assert calls == 1


def test_build_optimizer_ascend_fused_adamw_missing_fails(monkeypatch):
    """启用 Ascend fused optimizer 时，缺少 NpuFusedAdamW 必须直接失败。"""
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(lr0=0.001, momentum=0.9, warmup_bias_lr=0.0)
    trainer.data = {"nc": 80}
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))

    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_FUSED_OPTIMIZER", True)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(optim=SimpleNamespace()))

    with pytest.raises(RuntimeError, match="NpuFusedAdamW"):
        trainer.build_optimizer(model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-5)


def test_build_optimizer_ascend_fused_adamw_uses_explicit_class(monkeypatch):
    """Ascend AdamW 应显式调用 torch_npu.optim.NpuFusedAdamW。"""
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(lr0=0.001, momentum=0.9, warmup_bias_lr=0.0)
    trainer.data = {"nc": 80}
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.BatchNorm2d(4))

    class FakeNpuFusedAdamW:
        def __init__(self, params):
            self.param_groups = params

    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_FUSED_OPTIMIZER", True)
    monkeypatch.setitem(
        sys.modules, "torch_npu", SimpleNamespace(optim=SimpleNamespace(NpuFusedAdamW=FakeNpuFusedAdamW))
    )

    optimizer = trainer.build_optimizer(model, name="AdamW", lr=0.001, momentum=0.9, decay=1e-5)

    assert isinstance(optimizer, FakeNpuFusedAdamW)


def test_optimizer_step_ascend_fused_grad_clip_missing_fails(monkeypatch):
    """启用 fused grad clip 时，优化器缺少 clip_grad_norm_fused_ 必须直接失败。"""
    trainer = object.__new__(BaseTrainer)
    trainer.model = torch.nn.Linear(2, 2)
    trainer.optimizer = SimpleNamespace()
    trainer.ema = None

    class Scaler:
        def unscale_(self, optimizer):
            pass

    trainer.scaler = Scaler()
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_FUSED_GRAD_CLIP", True)

    with pytest.raises(RuntimeError, match="clip_grad_norm_fused_"):
        trainer.optimizer_step()


def test_save_metrics_includes_fitness_before_lr(tmp_path):
    """Test that training results CSV includes fitness before learning-rate columns."""
    trainer = object.__new__(BaseTrainer)
    trainer.csv = tmp_path / "results.csv"
    trainer.epoch = 0
    trainer.train_time_start = 0.0
    trainer.fitness = 0.75

    trainer.save_metrics({"train/loss": 1.0, "metrics/accuracy_top1": 0.5, "val/loss": 0.8, "lr/pg0": 0.01})

    header, row = [line.split(",") for line in trainer.csv.read_text(encoding="utf-8").splitlines()]
    assert header == ["epoch", "time", "train/loss", "metrics/accuracy_top1", "val/loss", "fitness", "lr/pg0"]
    assert row[header.index("fitness")] == "0.75"


def test_save_metrics_backfills_legacy_csv_fitness(tmp_path):
    """Test that appending to a legacy results CSV adds a fitness column with nan history."""
    trainer = object.__new__(BaseTrainer)
    trainer.csv = tmp_path / "results.csv"
    trainer.csv.write_text("epoch,time,train/loss,metrics/accuracy_top1,lr/pg0\n1,1,1,0.2,0.01\n", encoding="utf-8")
    trainer.epoch = 1
    trainer.train_time_start = 0.0
    trainer.fitness = 0.6

    trainer.save_metrics({"train/loss": 0.5, "metrics/accuracy_top1": 0.4, "lr/pg0": 0.02})

    rows = [line.split(",") for line in trainer.csv.read_text(encoding="utf-8").splitlines()]
    header = rows[0]
    fitness_index = header.index("fitness")
    assert header == ["epoch", "time", "train/loss", "metrics/accuracy_top1", "fitness", "lr/pg0"]
    assert rows[1][fitness_index] == "nan"
    assert rows[2][fitness_index] == "0.6"


def _minimal_plot_trainer(tmp_path):
    """Create a BaseTrainer shell with only the fields needed by plot_metrics."""
    trainer = object.__new__(BaseTrainer)
    trainer.csv = tmp_path / "results.csv"
    trainer.plots = {}
    trainer.on_plot = lambda *args, **kwargs: None
    return trainer


def test_plot_metrics_async_coalesces_pending_requests(tmp_path, monkeypatch):
    """Test threaded plot_metrics keeps a single worker and collapses repeated requests."""
    trainer = _minimal_plot_trainer(tmp_path)
    started, release = threading.Event(), threading.Event()
    calls = []

    def fake_plot_results(file, on_plot):
        calls.append(file)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)

    monkeypatch.setattr(trainer_module, "plot_results", fake_plot_results)

    thread = trainer.plot_metrics(threaded=True)
    assert started.wait(5)
    assert trainer.plot_metrics(threaded=True) is thread
    assert trainer.plot_metrics(threaded=True) is thread
    assert len(calls) == 1

    release.set()
    trainer._wait_for_plot_metrics()

    assert len(calls) == 2
    assert trainer._plot_thread is None


def test_plot_metrics_sync_waits_for_async_then_plots_latest(tmp_path, monkeypatch):
    """Test synchronous plot_metrics waits for active work and discards queued redraws."""
    trainer = _minimal_plot_trainer(tmp_path)
    started, release = threading.Event(), threading.Event()
    calls = []

    def fake_plot_results(file, on_plot):
        calls.append(file)
        if len(calls) == 1:
            started.set()
            assert release.wait(5)

    monkeypatch.setattr(trainer_module, "plot_results", fake_plot_results)

    trainer.plot_metrics(threaded=True)
    assert started.wait(5)
    trainer.plot_metrics(threaded=True)
    release.set()
    trainer.plot_metrics()

    assert len(calls) == 2
    assert trainer._plot_thread is None


def test_plot_metrics_async_warns_on_background_error(tmp_path, monkeypatch):
    """Test background plotting failures are logged without escaping the worker."""
    trainer = _minimal_plot_trainer(tmp_path)
    warnings = []

    def fake_plot_results(file, on_plot):
        raise RuntimeError("boom")

    monkeypatch.setattr(trainer_module, "plot_results", fake_plot_results)
    monkeypatch.setattr(trainer_module.LOGGER, "warning", lambda msg: warnings.append(msg))

    trainer.plot_metrics(threaded=True)
    trainer._wait_for_plot_metrics()

    assert trainer._plot_thread is None
    assert any("boom" in str(warning) for warning in warnings)


@pytest.mark.parametrize(
    "trainer_cls,validator_cls,predictor_cls,data,model,weights",
    [
        (
            detect.DetectionTrainer,
            detect.DetectionValidator,
            detect.DetectionPredictor,
            "coco8.yaml",
            "yolo26n.yaml",
            MODEL,
        ),
        (
            segment.SegmentationTrainer,
            segment.SegmentationValidator,
            segment.SegmentationPredictor,
            "coco8-seg.yaml",
            "yolo26n-seg.yaml",
            WEIGHTS_DIR / "yolo26n-seg.pt",
        ),
        (
            classify.ClassificationTrainer,
            classify.ClassificationValidator,
            classify.ClassificationPredictor,
            "imagenet10",
            "yolo26n-cls.yaml",
            None,
        ),
        (obb.OBBTrainer, obb.OBBValidator, obb.OBBPredictor, "dota8.yaml", "yolo26n-obb.yaml", None),
        (pose.PoseTrainer, pose.PoseValidator, pose.PosePredictor, "coco8-pose.yaml", "yolo26n-pose.yaml", None),
        (
            semantic.SemanticSegmentationTrainer,
            semantic.SemanticSegmentationValidator,
            semantic.SemanticSegmentationPredictor,
            "cityscapes8.yaml",
            "yolo26n-sem.yaml",
            None,
        ),
        (depth.DepthTrainer, depth.DepthValidator, depth.DepthPredictor, "depth8.yaml", "yolo26-depth.yaml", None),
    ],
)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_task(trainer_cls, validator_cls, predictor_cls, data, model, weights):
    """Test YOLO training, validation, and prediction for various tasks."""
    overrides = {
        "data": data,
        "model": model,
        "imgsz": 32,
        "epochs": 1,
        "save": False,
        "mask_ratio": 1,
        "overlap_mask": False,
    }

    # Trainer
    trainer = trainer_cls(overrides=overrides)
    trainer.add_callback("on_train_start", test_func)
    assert test_func in trainer.callbacks["on_train_start"], "on_train_start callback not registered"
    trainer.train()

    # Validator
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = data
    cfg.imgsz = 32
    val = validator_cls(args=cfg)
    val.add_callback("on_val_start", test_func)
    assert test_func in val.callbacks["on_val_start"], "on_val_start callback not registered"
    val(model=trainer.best)

    # Predictor
    pred = predictor_cls(overrides={"imgsz": [64, 64]})
    pred.add_callback("on_predict_start", test_func)
    assert test_func in pred.callbacks["on_predict_start"], "on_predict_start callback not registered"

    # Determine model path for prediction
    model_path = weights if weights else trainer.best
    if model == "yolo26n.yaml":  # only for detection
        # Confirm there is no issue with sys.argv being empty
        with mock.patch.object(sys, "argv", []):
            result = pred(source=ASSETS, model=model_path)
            assert len(result) > 0, f"Predictor returned no results for {model}"
    else:
        result = pred(source=ASSETS, model=model_path)
        assert len(result) > 0, f"Predictor returned no results for {model}"

    # Test resume functionality
    with pytest.raises(AssertionError):
        trainer_cls(overrides={**overrides, "resume": trainer.last}).train()


@pytest.mark.parametrize("task,weight,data", TASK_MODEL_DATA)
def test_resume_incomplete(task, weight, data, tmp_path):
    """Test training resumes from an incomplete checkpoint."""
    train_args = {
        "data": data,
        "epochs": 2,
        "save": True,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": task,
        "imgsz": 32,
        "exist_ok": True,
    }

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    def disable_final_eval(trainer):
        trainer.final_eval = lambda: None

    model = YOLO(weight)
    model.add_callback("on_train_start", disable_final_eval)
    model.add_callback("on_train_epoch_end", stop_after_first_epoch)
    model.train(**train_args)
    last_path = model.trainer.last
    _, ckpt = load_checkpoint(last_path)
    assert ckpt["epoch"] == 0, "checkpoint should be resumable"

    # Resume training using the checkpoint
    resume_model = YOLO(last_path)
    resume_model.train(resume=True, **train_args)
    assert resume_model.trainer.start_epoch == resume_model.trainer.epoch == 1, "resume test failed"


def test_distill_resume(tmp_path: Path):
    """Test knowledge distillation resumes from an incomplete checkpoint."""
    overrides = {
        "data": "coco8.yaml",
        "model": "yolo26n.yaml",
        "distill_model": WEIGHTS_DIR / "yolo26s.pt",
        "imgsz": 32,
        "multi_scale": 0.5,  # vary per-batch image size to exercise dynamic distillation score splitting
        "epochs": 2,
        "save": True,
        "plots": False,
        "workers": 0,
        "project": tmp_path,
        "name": "distill",
        "exist_ok": True,
    }

    # Train for one epoch then interrupt to produce a resumable checkpoint
    trainer = detect.DetectionTrainer(overrides=overrides)

    def stop_after_first_epoch(trainer):
        if trainer.epoch == 0:
            trainer.stop = True

    trainer.final_eval = lambda: None
    trainer.add_callback("on_train_epoch_end", stop_after_first_epoch)
    trainer.train()
    _, ckpt = load_checkpoint(trainer.last)
    assert ckpt["epoch"] == 0, "checkpoint should be resumable"
    assert isinstance(ckpt["ema"], DistillationModel), "distillation EMA wraps the student model"
    assert ckpt["ema"].teacher_model is None, "teacher should be stripped from the EMA/checkpoint"
    assert ckpt["ema"].projector is not None, "the distillation projector should be persisted in the EMA checkpoint"

    overrides["resume"] = trainer.last
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.final_eval = lambda: None
    trainer.train()
    model = unwrap_model(trainer.model)
    assert isinstance(model, DistillationModel), "resume should rebuild the DistillationModel"
    assert model.teacher_model is not None, "resume should rebuild the teacher from the distill_model path"
    assert trainer.start_epoch == trainer.epoch == 1, "resume test failed"


def test_distill_grayscale(tmp_path: Path):
    """Test knowledge distillation on a single-channel dataset (https://github.com/ultralytics/ultralytics/issues/25066)."""
    teacher = DetectionModel("yolo26n.yaml", ch=3, nc=80, verbose=False)
    teacher_path = tmp_path / "teacher.pt"
    torch.save({"model": teacher}, teacher_path)
    student = DetectionModel("yolo26n.yaml", ch=1, nc=80, verbose=False)
    student.args = SimpleNamespace(imgsz=32, dis=1.0)
    model = DistillationModel(teacher_model=teacher_path, student_model=student)
    assert isinstance(model, DistillationModel)
    assert model.teacher_model.yaml["channels"] == 1


@pytest.mark.parametrize(
    "ckpt",
    [
        {"model": OrderedDict([("a", torch.zeros(1))])},  # state_dict saved under the "model" key
        {"model": {"a": torch.zeros(1)}},  # plain-dict "model" value
        OrderedDict([("a", torch.zeros(1))]),  # bare state_dict, no "model" key
    ],
)
def test_load_checkpoint_state_dict_rejected(ckpt, tmp_path):
    """Test a state_dict checkpoint raises a clear TypeError instead of a cryptic AttributeError/KeyError."""
    weight = tmp_path / "bad.pt"
    torch.save(ckpt, weight)
    with pytest.raises(TypeError, match="supported Ultralytics checkpoint format"):
        load_checkpoint(weight)


def test_nan_recovery():
    """Test NaN loss detection and recovery during training."""
    nan_injected = [False]

    def inject_nan(trainer):
        """Inject NaN into loss during batch processing to test recovery mechanism."""
        if trainer.epoch == 1 and trainer.tloss is not None and not nan_injected[0]:
            trainer.tloss[next(iter(trainer.tloss))] *= float("nan")
            nan_injected[0] = True

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 3}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_batch_end", inject_nan)
    trainer.train()
    assert nan_injected[0], "NaN injection failed"


def test_checkpoint_fp16_overflow():
    """Test a finite model whose weights overflow fp16 is still checkpointed (clamped) instead of skipped."""

    def inflate_ema(trainer):
        """Push an EMA weight above the fp16 max (65504) so its fp16 snapshot would otherwise become Inf."""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = 1.0e5

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 2}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", inflate_ema)
    trainer.train()
    assert trainer.last.exists(), "checkpoint not saved for a finite model with fp16-overflowing weights"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )
    # Validation must leave the live EMA fp32 and unchanged; checkpoint serialization may clamp its fp16 copy.
    ema_param = next(iter(trainer.ema.ema.parameters()))
    assert ema_param.dtype == torch.float32 and torch.isfinite(ema_param).all() and ema_param.flatten()[0] == 1.0e5, (
        "validation corrupted the live EMA"
    )


def test_checkpoint_nonfinite_ema_resync():
    """Test a non-finite EMA on a finite model is resynced (not skipped) so the run still produces a checkpoint."""

    def poison_ema(trainer):
        """Make the live fp32 EMA genuinely non-finite while the model stays finite (sticky-NaN on a finite-loss run)."""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = float("inf")

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 2}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", poison_ema)
    trainer.train()
    assert trainer.last.exists(), "no checkpoint saved when the EMA went non-finite on a finite model"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )


def test_checkpoint_nonfinite_ema_and_model_sanitized():
    """Test a tensor non-finite in both EMA and model is sanitized (not skipped) so the run still produces a checkpoint."""

    def poison_ema_and_model(trainer):
        """Force the first parameter non-finite in both the live EMA and the model (finite-loss sticky-NaN)."""
        if trainer.ema is not None:
            next(iter(trainer.ema.ema.parameters())).data.flatten()[0] = float("inf")
            next(iter(unwrap_model(trainer.model).parameters())).data.flatten()[0] = float("nan")

    overrides = {"data": "coco8.yaml", "model": "yolo26n.yaml", "imgsz": 32, "epochs": 1}
    trainer = detect.DetectionTrainer(overrides=overrides)
    trainer.add_callback("on_train_epoch_end", poison_ema_and_model)
    trainer.train()
    assert trainer.last.exists(), "no checkpoint saved when a tensor went non-finite in both EMA and model"
    model, _ = load_checkpoint(trainer.last)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if isinstance(v, torch.Tensor)), (
        "saved checkpoint contains NaN/Inf"
    )


@pytest.mark.parametrize(
    "kwargs,uses_weights",
    [({}, True), ({"pretrained": True}, True), ({"pretrained": False}, False), ({"pretrained": MODEL}, True)],
)
@pytest.mark.skipif(IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_reuses_loaded_checkpoint_model(monkeypatch, kwargs, uses_weights):
    """Test training reuses loaded checkpoint config while respecting the pretrained argument."""
    model = YOLO("yolo26n.yaml")
    model.ckpt = {"checkpoint": True}
    model.ckpt_path = "/tmp/fake.pt"
    model.overrides["model"] = "ul://glenn-jocher/m2/exp-14"
    model.overrides["pretrained"] = False
    original_model = model.model
    captured = {}

    class FakeTrainer:
        def __init__(self, overrides=None, _callbacks=None):
            self.overrides = overrides
            self.callbacks = _callbacks
            self.model = None
            self.args = SimpleNamespace(save=False)
            self.validator = SimpleNamespace(metrics=None)
            self.best = MODEL.parent / "nonexistent-best.pt"
            self.last = MODEL
            captured["trainer"] = self

        def get_model(self, cfg=None, weights=None, verbose=True):
            captured["cfg"] = cfg
            captured["weights"] = weights
            return original_model

        def train(self):
            return None

    monkeypatch.setattr("ultralytics.engine.model.checks.check_pip_update_available", lambda: None)
    monkeypatch.setattr(model, "_smart_load", lambda key: FakeTrainer)
    monkeypatch.setattr(
        "ultralytics.engine.model.load_checkpoint",
        lambda path: (original_model, {"checkpoint": True}),
    )

    model.train(data="coco8.yaml", epochs=1, **kwargs)

    assert captured["trainer"].model is original_model, "Trainer model does not match original"
    assert captured["cfg"] == original_model.yaml, f"Config mismatch: {captured['cfg']} != {original_model.yaml}"
    assert captured["weights"] is (original_model if uses_weights else None), "Unexpected weights loaded"


def test_train_multi_custom_trainer_metrics_and_failure_keys(monkeypatch, tmp_path):
    """Test custom multi-dataset runs keep memory metrics and unique failure keys."""
    model = YOLO(MODEL)
    calls = 0

    class FakeTrainer:
        def __init__(self, overrides=None, _callbacks=None):
            pass

        def get_model(self, cfg=None, weights=None, verbose=True):
            return model.model

        def train(self):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("failed repeated dataset")
            self.validator = SimpleNamespace(metrics=SimpleNamespace(results_dict={"fitness": 1.0}))

    monkeypatch.setattr("ultralytics.engine.model.checks.check_pip_update_available", lambda: None)
    results = model.train(
        data=["coco8.yaml", "coco8.yaml"],
        project=tmp_path,
        plots=False,
        save=False,
        trainer=FakeTrainer,
    )

    assert model.trainer.trainer is FakeTrainer
    assert results == {"coco8": {"fitness": 1.0}, "coco8-2": None}


@pytest.mark.parametrize("pretrained,uses_weights", [(True, True), (False, False), (MODEL, True)])
def test_setup_model_respects_pretrained_arg_for_pt_models(monkeypatch, pretrained, uses_weights):
    """Test .pt models use checkpoint config while respecting the pretrained argument."""
    captured = {}
    checkpoint_model = SimpleNamespace(yaml={"nc": 80})
    trainer = object.__new__(BaseTrainer)
    trainer.model = "yolo26n.pt"
    trainer.args = SimpleNamespace(pretrained=pretrained)
    trainer.resume = False

    def fake_get_model(cfg=None, weights=None, verbose=True):
        captured["cfg"] = cfg
        captured["weights"] = weights
        return SimpleNamespace()

    trainer.get_model = fake_get_model
    monkeypatch.setattr(
        "ultralytics.engine.trainer.load_checkpoint", lambda path: (checkpoint_model, {"checkpoint": True})
    )

    trainer.setup_model()

    assert captured["cfg"] == checkpoint_model.yaml, "Checkpoint config was not used"
    assert captured["weights"] is (checkpoint_model if uses_weights else None), "Unexpected weights loaded"
