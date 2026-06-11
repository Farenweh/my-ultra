# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import json
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from types import ModuleType, SimpleNamespace
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
from ultralytics.models.yolo.detect import val as detect_val
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import DetectionModel, load_checkpoint
from ultralytics.utils import ASSETS, DEFAULT_CFG, IS_RASPBERRYPI, WEIGHTS_DIR
from ultralytics.utils.patches import torch_load
from ultralytics.utils.tal import TaskAlignedAssigner
from ultralytics.utils.torch_utils import ModelEMA, strip_optimizer, unwrap_model


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


def _install_fake_faster_coco_eval(monkeypatch):
    """安装最小 fake faster-coco-eval，避免单测依赖真实 COCO 评估。"""
    seen_iou_types = []
    fake_module = ModuleType("faster_coco_eval")

    class FakeCOCO:
        def __init__(self, annotation_file):
            self.annotation_file = annotation_file

        def loadRes(self, prediction_file):
            return {"prediction_file": prediction_file}

    class FakeCOCOeval:
        def __init__(self, anno, pred, iouType, lvis_style=False, print_function=print):
            self.anno = anno
            self.pred = pred
            self.iouType = iouType
            self.lvis_style = lvis_style
            self.print_function = print_function
            self.params = SimpleNamespace(imgIds=[])
            self.stats_as_dict = {
                "AP_50": 0.5,
                "AP_all": 0.4,
                "AP_small": 0.1,
                "AP_medium": 0.2,
                "AP_large": 0.3,
            }
            seen_iou_types.append(iouType)

        def evaluate(self):
            assert self.params.imgIds

        def accumulate(self):
            pass

        def summarize(self):
            self.print_function("fake summary", self.iouType)

    fake_module.COCO = FakeCOCO
    fake_module.COCOeval_faster = FakeCOCOeval
    monkeypatch.setitem(sys.modules, "faster_coco_eval", fake_module)
    monkeypatch.setattr(detect_val, "check_requirements", lambda *args, **kwargs: None)
    return seen_iou_types


def _coco_eval_label(tmp_path, *, segments=None, keypoints=None):
    """构造最小 YOLO dataset label。"""
    return {
        "im_file": str(tmp_path / "0001.jpg"),
        "shape": (100, 200),
        "cls": [[0]],
        "bboxes": [[0.5, 0.5, 0.2, 0.4]],
        "segments": segments or [],
        "keypoints": keypoints,
        "normalized": True,
        "bbox_format": "xywh",
    }


def _coco_eval_validator(tmp_path, labels, task="detect"):
    """构造只包含 coco_evaluate 所需属性的 DetectionValidator。"""
    validator = object.__new__(detect.DetectionValidator)
    validator.args = SimpleNamespace(save_json=True, task=task, single_cls=False)
    validator.is_lvis = False
    validator.is_coco = False
    validator.save_dir = tmp_path
    validator.jdict = [{"image_id": 1, "category_id": 1, "bbox": [80, 30, 40, 40], "score": 0.9}]
    validator.class_map = [1, 2]
    validator.names = {0: "class0", 1: "class1"}
    validator.nc = 2
    validator.dataloader = SimpleNamespace(
        dataset=SimpleNamespace(labels=labels, im_files=[label["im_file"] for label in labels])
    )
    if task == "pose":
        validator.kpt_shape = [2, 3]
    return validator


def _write_prediction_json(tmp_path):
    pred_json = tmp_path / "predictions.json"
    pred_json.write_text(
        '[{"image_id": 1, "category_id": 1, "bbox": [80, 30, 40, 40], "score": 0.9}]',
        encoding="utf-8",
    )
    return pred_json


def test_coco_eval_generates_annotations_for_yolo_dataset(tmp_path, monkeypatch):
    """普通 YOLO 数据集没有 annotation JSON 时，会在 save_dir 生成 COCO-style GT。"""
    _install_fake_faster_coco_eval(monkeypatch)
    validator = _coco_eval_validator(tmp_path, [_coco_eval_label(tmp_path)])

    stats = validator.coco_evaluate({}, _write_prediction_json(tmp_path), tmp_path / "missing_annotations.json")

    annotation_json = tmp_path / "coco_eval_annotations.json"
    data = json.loads(annotation_json.read_text(encoding="utf-8"))
    assert data["images"] == [{"id": 1, "file_name": "0001.jpg", "height": 100, "width": 200}]
    assert data["annotations"][0]["bbox"] == [80.0, 30.0, 40.0, 40.0]
    assert data["annotations"][0]["category_id"] == 1
    assert stats["metrics/mAP50(B)"] == 0.5
    assert stats["metrics/mAP50-95(B)"] == 0.4
    assert stats["fitness"] == pytest.approx(0.41)


def test_coco_eval_saves_summary_text(tmp_path, monkeypatch):
    """COCO-style eval summary 会同步保存到 save_dir/coco_eval.txt。"""
    _install_fake_faster_coco_eval(monkeypatch)
    validator = _coco_eval_validator(tmp_path, [_coco_eval_label(tmp_path)])

    validator.coco_evaluate({}, _write_prediction_json(tmp_path), tmp_path / "missing_annotations.json")

    summary = (tmp_path / "coco_eval.txt").read_text(encoding="utf-8")
    assert f"predictions: {tmp_path / 'predictions.json'}" in summary
    assert f"annotations: {tmp_path / 'coco_eval_annotations.json'}" in summary
    assert "iou_type: bbox" in summary
    assert "fake summary bbox" in summary


@pytest.mark.parametrize(
    "task,iou_types,suffix,label_kwargs,expected",
    [
        (
            "segment",
            ["bbox", "segm"],
            ["Box", "Mask"],
            {"segments": [[[0.4, 0.3], [0.6, 0.3], [0.6, 0.7], [0.4, 0.7]]]},
            ["bbox", "segm"],
        ),
        (
            "pose",
            ["bbox", "keypoints"],
            ["Box", "Pose"],
            {"keypoints": [[[0.45, 0.45, 2], [0.55, 0.55, 2]]]},
            ["bbox", "keypoints"],
        ),
    ],
)
def test_coco_eval_uses_task_iou_types_when_annotations_exist(
    tmp_path, monkeypatch, task, iou_types, suffix, label_kwargs, expected
):
    """segment/pose 标注完整时，会额外运行对应 COCO iou_type。"""
    seen_iou_types = _install_fake_faster_coco_eval(monkeypatch)
    validator = _coco_eval_validator(tmp_path, [_coco_eval_label(tmp_path, **label_kwargs)], task=task)

    validator.coco_evaluate(
        {},
        _write_prediction_json(tmp_path),
        tmp_path / "missing_annotations.json",
        iou_types,
        suffix,
    )

    assert seen_iou_types == expected


@pytest.mark.parametrize(
    "task,iou_types,suffix",
    [
        ("segment", ["bbox", "segm"], ["Box", "Mask"]),
        ("pose", ["bbox", "keypoints"], ["Box", "Pose"]),
    ],
)
def test_coco_eval_falls_back_to_bbox_when_task_annotations_missing(tmp_path, monkeypatch, task, iou_types, suffix):
    """segment/pose 标注缺失时，COCO-style eval 降级到 bbox 且不中断。"""
    seen_iou_types = _install_fake_faster_coco_eval(monkeypatch)
    validator = _coco_eval_validator(tmp_path, [_coco_eval_label(tmp_path)], task=task)

    validator.coco_evaluate(
        {},
        _write_prediction_json(tmp_path),
        tmp_path / "missing_annotations.json",
        iou_types,
        suffix,
    )

    assert seen_iou_types == ["bbox"]


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


def test_model_ema_averages_parameters_but_copies_buffers():
    """验证 EMA 只平均参数，BN running stats 等 buffers 直接跟随当前模型。"""
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1, bias=False), torch.nn.BatchNorm2d(1))
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[1].running_mean.fill_(0.0)
        model[1].running_var.fill_(1.0)
        model[1].num_batches_tracked.fill_(0)

    ema = ModelEMA(model)
    ema.decay = lambda _: 0.5

    with torch.no_grad():
        model[0].weight.fill_(3.0)
        model[1].running_mean.fill_(10.0)
        model[1].running_var.fill_(4.0)
        model[1].num_batches_tracked.fill_(7)

    ema.update(model)

    assert ema.ema[0].weight.item() == pytest.approx(2.0)
    assert ema.ema[1].running_mean.item() == pytest.approx(10.0)
    assert ema.ema[1].running_var.item() == pytest.approx(4.0)
    assert ema.ema[1].num_batches_tracked.item() == 7


def test_model_ema_foreach_fast_path_matches_loop_on_npu():
    """验证 NPU foreach 快路径与逐元素 EMA 更新等价。"""
    device = _require_npu_device()
    model = torch.nn.Sequential(torch.nn.Conv2d(2, 4, 1, bias=False), torch.nn.BatchNorm2d(4)).to(device)
    ema_loop = ModelEMA(model)
    ema_fast = ModelEMA(model)
    ema_loop._foreach_update_disabled = True
    ema_loop.decay = ema_fast.decay = lambda _: 0.5

    with torch.no_grad():
        model[0].weight.fill_(3.0)
        model[1].weight.fill_(5.0)
        model[1].bias.fill_(7.0)
        model[1].running_mean.fill_(11.0)
        model[1].running_var.fill_(13.0)
        model[1].num_batches_tracked.fill_(17)

    ema_loop.update(model)
    ema_fast.update(model)
    torch.npu.synchronize()

    assert not ema_fast._foreach_update_disabled
    assert ema_fast._foreach_update_cache is not None
    for (name, expected), (_, actual) in zip(ema_loop.ema.state_dict().items(), ema_fast.ema.state_dict().items()):
        if actual.is_floating_point():
            torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6, msg=f"EMA state mismatch for {name}")
        else:
            torch.testing.assert_close(actual, expected, msg=f"EMA state mismatch for {name}")


def test_model_ema_foreach_cache_reused_on_npu():
    """验证连续 update 时 foreach cache 不再每步重建。"""
    device = _require_npu_device()
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    ema = ModelEMA(model)
    ema.decay = lambda _: 0.5

    ema.update(model)
    torch.npu.synchronize()
    rebuilds = ema._foreach_update_cache_rebuilds

    with torch.no_grad():
        for value in (2.0, 3.0, 4.0):
            model.weight.fill_(value)
            ema.update(model)
    torch.npu.synchronize()

    assert ema._foreach_update_cache_rebuilds == rebuilds


def test_model_ema_foreach_cache_rebuilds_after_dtype_roundtrip_on_npu():
    """验证 EMA 模型 dtype/storage 往返后，foreach cache 会重建到 live storage。"""
    device = _require_npu_device()
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model)
    ema.decay = lambda _: 0.5

    ema.update(model)
    torch.npu.synchronize()
    old_cached_ptr = ema._foreach_update_cache["ema_params"][0].data_ptr()

    ema.ema.half()
    ema.ema.float()
    live_ptr_after_roundtrip = next(ema.ema.parameters()).data_ptr()
    with torch.no_grad():
        model.weight.fill_(3.0)

    ema.update(model)
    torch.npu.synchronize()

    live_param = next(ema.ema.parameters())
    assert ema._foreach_update_cache["ema_params"][0].data_ptr() == live_param.data_ptr()
    if live_ptr_after_roundtrip != old_cached_ptr:
        assert ema._foreach_update_cache["ema_params"][0].data_ptr() != old_cached_ptr
    torch.testing.assert_close(live_param.detach().cpu(), torch.tensor([[2.0]]))


def test_model_ema_foreach_cache_rebuilds_after_source_dtype_roundtrip_on_npu():
    """验证 source model dtype/storage 往返后，foreach cache 使用新的 live buffers。"""
    device = _require_npu_device()
    model = torch.nn.Sequential(torch.nn.Linear(1, 1, bias=False), torch.nn.BatchNorm1d(1)).to(device)
    with torch.no_grad():
        model[0].weight.fill_(1.0)
        model[1].running_mean.fill_(0.0)
    ema = ModelEMA(model)
    ema.decay = lambda _: 0.5

    ema.update(model)
    torch.npu.synchronize()
    rebuilds = ema._foreach_update_cache_rebuilds
    old_buffer_ptr = ema._foreach_update_cache["model_buffers"][0].data_ptr()

    model.half()
    model.float()
    with torch.no_grad():
        model[0].weight.fill_(3.0)
        model[1].running_mean.fill_(9.0)

    ema.update(model)
    torch.npu.synchronize()

    live_buffer = next(model.buffers())
    assert ema._foreach_update_cache_rebuilds == rebuilds + 1
    assert ema._foreach_update_cache["model_buffers"][0].data_ptr() == live_buffer.data_ptr()
    if live_buffer.data_ptr() != old_buffer_ptr:
        assert ema._foreach_update_cache["model_buffers"][0].data_ptr() != old_buffer_ptr
    torch.testing.assert_close(ema.ema[0].weight.detach().cpu(), torch.tensor([[2.0]]))
    torch.testing.assert_close(ema.ema[1].running_mean.detach().cpu(), torch.tensor([9.0]))


def test_model_ema_foreach_cache_uses_live_parameter_after_data_replacement_on_npu():
    """验证 source 参数 .data 替换后无需重建 cache 也能读取新参数值。"""
    device = _require_npu_device()
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model)
    ema.decay = lambda _: 0.5

    ema.update(model)
    torch.npu.synchronize()
    rebuilds = ema._foreach_update_cache_rebuilds
    model.weight.data = torch.full_like(model.weight, 3.0)

    ema.update(model)
    torch.npu.synchronize()

    assert ema._foreach_update_cache_rebuilds == rebuilds
    torch.testing.assert_close(next(ema.ema.parameters()).detach().cpu(), torch.tensor([[2.0]]))


def test_model_ema_foreach_cache_rebuilds_after_source_load_state_dict_on_npu():
    """验证 source model load_state_dict 替换 tensor 后会重建 foreach cache。"""
    device = _require_npu_device()
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model)
    ema.decay = lambda _: 0.5

    ema.update(model)
    torch.npu.synchronize()
    rebuilds = ema._foreach_update_cache_rebuilds
    state = {k: torch.full_like(v, 3.0) for k, v in model.state_dict().items()}

    model.load_state_dict(state, assign=True)
    ema.update(model)
    torch.npu.synchronize()

    assert ema._foreach_update_cache_rebuilds == rebuilds + 1
    torch.testing.assert_close(next(ema.ema.parameters()).detach().cpu(), torch.tensor([[2.0]]))


def test_model_ema_foreach_fallback_on_npu(monkeypatch):
    """验证 NPU foreach 不可用时会回退到逐元素 EMA 更新。"""
    device = _require_npu_device()
    model = torch.nn.Linear(1, 1, bias=False).to(device)
    with torch.no_grad():
        model.weight.fill_(1.0)
    ema = ModelEMA(model)
    ema.decay = lambda _: 0.5

    with torch.no_grad():
        model.weight.fill_(3.0)

    calls = {"lerp": 0}

    def fail_lerp(*args, **kwargs):
        calls["lerp"] += 1
        raise RuntimeError("forced foreach failure")

    monkeypatch.setattr(torch, "_foreach_lerp_", fail_lerp)
    ema.update(model)
    torch.npu.synchronize()

    assert calls["lerp"] == 1
    assert ema._foreach_update_disabled
    torch.testing.assert_close(next(ema.ema.parameters()).detach().cpu(), torch.tensor([[2.0]]))


def test_model_ema_decay_scales_with_effective_batch():
    """验证 EMA decay 按有效 batch/nbs 做样本数归一化。"""
    model = torch.nn.Linear(1, 1)
    ema = ModelEMA(model, decay=0.81, tau=1, batch_scale=4.0)

    assert ema.batch_scale == pytest.approx(4.0)
    assert ema.decay(100) == pytest.approx(0.81**4)


def test_trainer_ema_batch_scale_uses_accumulated_global_batch():
    """验证 Trainer 用每次 optimizer step 覆盖的全局图片数计算 EMA batch scale。"""
    trainer = object.__new__(BaseTrainer)
    trainer.batch_size = 1024
    trainer.accumulate = 1
    trainer.args = SimpleNamespace(nbs=64)
    assert trainer._ema_batch_scale() == pytest.approx(16.0)

    trainer.batch_size = 16
    trainer.accumulate = 4
    assert trainer._ema_batch_scale() == pytest.approx(1.0)


def test_update_ema_bn_stats_recomputes_moments_without_changing_parameters():
    """验证 EMA PreciseBN 使用 activation moments 重算 BN 统计，且不改变可学习参数。"""
    model = torch.nn.Sequential(torch.nn.BatchNorm2d(1))
    model.eval()
    with torch.no_grad():
        model[0].weight.fill_(2.0)
        model[0].bias.fill_(1.0)
        model[0].running_mean.fill_(100.0)
        model[0].running_var.fill_(100.0)

    trainer = object.__new__(BaseTrainer)
    trainer.world_size = 1
    trainer.device = torch.device("cpu")
    trainer.amp = False
    dataloader = [
        {"img": torch.tensor([[[[1.0, 3.0]]], [[[5.0, 7.0]]]])},
        {"img": torch.tensor([[[[2.0, 4.0]]], [[[6.0, 8.0]]]])},
    ]

    trainer._update_ema_bn_stats(model, dataloader, target_images=4)

    assert model[0].weight.item() == pytest.approx(2.0)
    assert model[0].bias.item() == pytest.approx(1.0)
    assert model[0].running_mean.item() == pytest.approx(4.5)
    assert model[0].running_var.item() == pytest.approx(5.25)
    assert model[0].num_batches_tracked.item() == 2
    assert model.training is False


def test_update_ema_bn_stats_uses_distributed_moment_reduction(monkeypatch):
    """验证 DDP 校准路径聚合 moments，而不是只使用 rank0 本地 BN buffer。"""
    calls = {"all_gather": 0, "all_reduce": 0}

    def fake_all_gather(outputs, tensor):
        calls["all_gather"] += 1
        for out in outputs:
            out.copy_(tensor)

    def fake_all_reduce(tensor, op=None):
        calls["all_reduce"] += 1

    monkeypatch.setattr(trainer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(trainer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(trainer_module.dist, "all_gather", fake_all_gather)
    monkeypatch.setattr(trainer_module.dist, "all_reduce", fake_all_reduce)

    model = torch.nn.Sequential(torch.nn.BatchNorm2d(1))
    trainer = object.__new__(BaseTrainer)
    trainer.world_size = 2
    trainer.device = torch.device("cpu")
    trainer.amp = False
    dataloader = [{"img": torch.ones(1, 1, 1, 2)}]

    trainer._update_ema_bn_stats(model, dataloader, target_images=2)

    assert calls["all_gather"] == 1
    assert calls["all_reduce"] == 1


def test_calibrate_ema_bn_caps_samples_to_8192():
    """验证 EMA BN 校准按图片数使用 min(8192, len(train_dataset))。"""
    recorded = []
    trainer = object.__new__(BaseTrainer)
    trainer.ema = SimpleNamespace(ema=torch.nn.BatchNorm2d(1))
    trainer.train_loader = SimpleNamespace(dataset=range(10000))
    trainer._build_ema_bn_calibration_loader = lambda: "loader"
    trainer._update_ema_bn_stats = lambda _model, _loader, target, _bn: recorded.append(target)

    trainer._calibrate_ema_bn()
    assert recorded == [8192]

    recorded.clear()
    trainer.train_loader = SimpleNamespace(dataset=range(17))
    trainer._calibrate_ema_bn()
    assert recorded == [17]


def test_ema_bn_calibration_loader_disables_pin_memory(monkeypatch):
    """验证短生命周期 EMA BN 校准 dataloader 不启动 pin_memory 后台线程。"""
    captured = {}

    def fake_build_dataloader(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(sampler=None)

    monkeypatch.setattr("ultralytics.data.build_dataloader", fake_build_dataloader)
    trainer = object.__new__(BaseTrainer)
    trainer.train_loader = SimpleNamespace(dataset=range(8), batch_size=4)
    trainer.batch_size = 4
    trainer.world_size = 1
    trainer.args = SimpleNamespace(workers=2)

    trainer._build_ema_bn_calibration_loader()

    assert captured["pin_memory"] is False


def test_prepare_ema_for_eval_or_save_calibrates_for_validation_or_checkpoint():
    """验证 validation 前和仅保存 checkpoint 前都会进入 EMA BN 校准。"""
    calls = {"clear": 0, "calibrate": 0}
    trainer = object.__new__(BaseTrainer)
    trainer.device = torch.device("cpu")
    trainer._clear_memory = lambda *args, **kwargs: calls.__setitem__("clear", calls["clear"] + 1)
    trainer._calibrate_ema_bn = lambda: calls.__setitem__("calibrate", calls["calibrate"] + 1)

    trainer._prepare_ema_for_eval_or_save(should_validate=False, should_save=False)
    assert calls == {"clear": 0, "calibrate": 0}

    trainer._prepare_ema_for_eval_or_save(should_validate=True, should_save=False)
    trainer._prepare_ema_for_eval_or_save(should_validate=False, should_save=True)
    assert calls == {"clear": 2, "calibrate": 2}


def test_save_model_checkpoint_saves_raw_model_and_ema(tmp_path):
    """验证训练中 checkpoint 同时保留原始 FP32 model 和 EMA 权重。"""
    raw_model = torch.nn.Linear(1, 1, bias=False)
    ema_model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        raw_model.weight.fill_(1.25)
        ema_model.weight.fill_(2.5)
    raw_model.args = {"task": "classify"}
    ema_model.args = {"task": "classify"}

    trainer = object.__new__(BaseTrainer)
    trainer.model = raw_model
    trainer.ema = SimpleNamespace(ema=ema_model, updates=7)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.scaler = SimpleNamespace(state_dict=lambda: {})
    trainer.args = SimpleNamespace(task="classify", imgsz=32)
    trainer.metrics = {}
    trainer.fitness = trainer.best_fitness = 0.1
    trainer.epoch = trainer.global_step = trainer.data_cycle = trainer._data_cycle_batch = 0
    trainer.wdir = tmp_path
    trainer.last = tmp_path / "last.pt"
    trainer.best = tmp_path / "best.pt"
    trainer.save_period = -1
    trainer.read_results_csv = lambda: {}

    assert trainer.save_model() is True

    ckpt = torch_load(trainer.last, map_location="cpu")
    assert ckpt["model"] is not None, "checkpoint 应包含原始 model 权重"
    assert ckpt["ema"] is not None, "checkpoint 应包含 EMA 权重"
    assert next(ckpt["model"].parameters()).dtype == torch.float32, "原始 model checkpoint 应保持 FP32"
    assert next(ckpt["ema"].parameters()).dtype == torch.float16, "EMA checkpoint 应保持现有 FP16 格式"
    assert next(ckpt["model"].parameters()).item() == pytest.approx(1.25)
    assert next(ckpt["ema"].parameters()).item() == pytest.approx(2.5)
    raw_param = next(raw_model.parameters())
    ema_param = next(ema_model.parameters())
    assert raw_param.dtype == torch.float32 and raw_param.item() == pytest.approx(1.25)
    assert ema_param.dtype == torch.float32 and ema_param.item() == pytest.approx(2.5)

    loaded_model, loaded_ckpt = load_checkpoint(trainer.last)
    loaded_param = next(loaded_model.parameters()).detach().cpu()
    ema_param = next(loaded_ckpt["ema"].float().parameters()).detach().cpu()
    assert torch.allclose(loaded_param, ema_param), "load_checkpoint 应继续优先使用 EMA 权重"


def test_strip_optimizer_saves_raw_sidecar(tmp_path):
    """验证 strip_optimizer 在 EMA 覆盖 model 前单独保存原始 model 权重。"""
    ckpt_path = tmp_path / "best.pt"
    raw_path = tmp_path / "best_raw.pt"
    raw_model = torch.nn.Linear(1, 1, bias=False)
    ema_model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        raw_model.weight.fill_(1.25)
        ema_model.weight.fill_(2.5)
    raw_model.criterion = torch.nn.MSELoss()
    ema_model.criterion = torch.nn.MSELoss()
    torch.save(
        {
            "epoch": 3,
            "model": raw_model,
            "ema": ema_model,
            "optimizer": {"state": {}},
            "best_fitness": 0.5,
            "updates": 7,
            "scaler": {"scale": 1.0},
            "train_args": {"imgsz": 32},
        },
        ckpt_path,
    )

    strip_optimizer(ckpt_path)

    assert raw_path.exists(), "raw sidecar checkpoint 未保存"
    stripped = torch_load(ckpt_path, map_location="cpu")
    raw = torch_load(raw_path, map_location="cpu")
    assert next(stripped["model"].parameters()).dtype == torch.float16
    assert next(raw["model"].parameters()).dtype == torch.float32
    assert next(stripped["model"].parameters()).item() == pytest.approx(2.5)
    assert next(raw["model"].parameters()).item() == pytest.approx(1.25)
    for ckpt in (stripped, raw):
        assert ckpt["epoch"] == -1
        assert all(ckpt[k] is None for k in ("optimizer", "best_fitness", "ema", "updates", "scaler"))
        assert getattr(ckpt["model"], "criterion", None) is None
        assert all(not p.requires_grad for p in ckpt["model"].parameters())


def test_strip_optimizer_raw_sidecar_uses_output_path(tmp_path):
    """验证传入 s 时 raw sidecar 基于输出路径命名。"""
    ckpt_path = tmp_path / "source.pt"
    output_path = tmp_path / "export.pt"
    raw_path = tmp_path / "export_raw.pt"
    raw_model = torch.nn.Linear(1, 1, bias=False)
    ema_model = torch.nn.Linear(1, 1, bias=False)
    torch.save(
        {
            "epoch": 3,
            "model": raw_model,
            "ema": ema_model,
            "optimizer": {"state": {}},
            "best_fitness": 0.5,
            "updates": 7,
            "scaler": {"scale": 1.0},
            "train_args": {"imgsz": 32},
        },
        ckpt_path,
    )

    strip_optimizer(ckpt_path, s=output_path)

    assert output_path.exists()
    assert raw_path.exists()
    assert not (tmp_path / "source_raw.pt").exists()


def test_strip_optimizer_skips_raw_sidecar_for_legacy_ema_checkpoint(tmp_path):
    """验证 model=None 的旧 checkpoint 不会生成 raw sidecar。"""
    ckpt_path = tmp_path / "last.pt"
    raw_path = tmp_path / "last_raw.pt"
    ema_model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        ema_model.weight.fill_(3.0)
    torch.save(
        {
            "epoch": 3,
            "model": None,
            "ema": ema_model,
            "optimizer": {"state": {}},
            "best_fitness": 0.5,
            "updates": 7,
            "scaler": {"scale": 1.0},
            "train_args": {"imgsz": 32},
        },
        ckpt_path,
    )

    strip_optimizer(ckpt_path)

    stripped = torch_load(ckpt_path, map_location="cpu")
    assert not raw_path.exists(), "旧 checkpoint 不应生成 raw sidecar"
    assert stripped["ema"] is None
    assert next(stripped["model"].parameters()).dtype == torch.float16
    assert next(stripped["model"].parameters()).item() == pytest.approx(3.0)


def test_strip_optimizer_unwraps_distillation_ema(tmp_path):
    """验证最终 checkpoint 将 DistillationModel EMA 解包为纯 student model。"""
    ckpt_path = tmp_path / "best.pt"
    student = torch.nn.Linear(1, 1, bias=False)
    distillation = DistillationModel.__new__(DistillationModel)
    torch.nn.Module.__init__(distillation)
    distillation.teacher_model = None
    distillation.student_model = student
    distillation.feats_idx = []
    distillation._teacher_feats = {}
    distillation._student_feats = {}
    distillation._teacher_hooks = []
    distillation._student_hooks = []
    torch.save({"model": None, "ema": distillation, "train_args": {}}, ckpt_path)

    strip_optimizer(ckpt_path)

    stripped = torch_load(ckpt_path, map_location="cpu")
    assert type(stripped["model"]) is torch.nn.Linear
    assert next(stripped["model"].parameters()).dtype == torch.float16
    assert stripped["ema"] is None


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
