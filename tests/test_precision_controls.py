# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import ultralytics
import ultralytics.engine.trainer as trainer_module
import ultralytics.nn.tasks as tasks_module
import ultralytics.utils.torch_utils as torch_utils
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils.checks import check_amp
from ultralytics.utils.torch_utils import resolve_amp_dtype
from tests import CFG


class DtypeModel(nn.Module):
    """Small native model used to verify weight and input precision without external checkpoints."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.names = {0: "item"}
        self.stride = torch.tensor([1])
        self.yaml = {"channels": 3}
        self.last_input_dtype = None

    def forward(self, x):
        self.last_input_dtype = x.dtype
        return x * self.weight


class InplaceModule(nn.Module):
    """记录模块自身原地操作策略的最小测试模块。"""

    def __init__(self, inplace):
        super().__init__()
        self.inplace = inplace


@pytest.mark.parametrize(
    ("module_inplace", "load_inplace", "expected"),
    [(False, True, False), (True, True, True), (False, False, False), (True, False, False)],
)
def test_load_checkpoint_treats_inplace_as_permission(monkeypatch, module_inplace, load_inplace, expected):
    """加载参数只能保留或收紧模块的原地策略，不能把模块显式的 False 强制改成 True。"""
    model = DtypeModel()
    model.task = "detect"
    model.inplace_probe = InplaceModule(module_inplace)
    checkpoint = {"model": model, "train_args": {}}
    monkeypatch.setattr(tasks_module, "torch_safe_load", lambda *_args, **_kwargs: (checkpoint, "model.pt"))

    loaded, _ = load_checkpoint("model.pt", inplace=load_inplace)

    assert loaded.inplace_probe.inplace is expected


@pytest.mark.parametrize(
    ("setting", "expected"),
    [
        (False, None),
        (True, torch.float16),
        ("fp16", torch.float16),
        ("FP16", torch.float16),
        ("BF16", torch.bfloat16),
        ("fp32", None),
    ],
)
def test_resolve_amp_dtype(setting, expected):
    """Public AMP values resolve independently from model quantization."""
    assert resolve_amp_dtype(setting) == expected


@pytest.mark.parametrize(
    ("enabled", "dtype", "expected"),
    [(False, torch.float16, False), (True, torch.float16, True), (True, torch.bfloat16, False)],
)
def test_grad_scaler_is_fp16_only(enabled, dtype, expected):
    """BF16 AMP must never enable GradScaler."""
    assert BaseTrainer._amp_uses_grad_scaler(enabled, dtype) is expected


@pytest.mark.parametrize(("dtype", "expected"), [(torch.float16, False), (torch.bfloat16, True)])
def test_amp_capability_check_uses_bf16_tolerance(monkeypatch, dtype, expected):
    """BF16 capability checks filter confidence and use the absorbed 2-pixel tolerance."""

    class FakePredictor:
        def __init__(self):
            self.calls = 0

        def __call__(self, *_args, **_kwargs):
            offset = 0.0 if self.calls == 0 else 1.5
            self.calls += 1
            boxes = torch.tensor([[offset, offset, offset, offset, 0.5, 0.0]])
            return [SimpleNamespace(boxes=SimpleNamespace(data=boxes))]

    model = nn.Linear(1, 1, device="meta")
    model.stride = torch.tensor([32])
    monkeypatch.setattr(ultralytics, "YOLO", lambda *_args, **_kwargs: FakePredictor())
    monkeypatch.setattr(torch_utils, "autocast", lambda *args, **kwargs: nullcontext())

    assert check_amp(model, dtype) is expected


@pytest.mark.parametrize(
    ("quantize", "expected_dtype"), [(None, torch.float32), (32, torch.float32), (16, torch.float16), ("bf16", torch.bfloat16)]
)
def test_training_validator_precision_does_not_mutate_ema(quantize, expected_dtype):
    """Manual validation uses a temporary low-precision EMA copy while AMP/FP32 uses the live FP32 EMA."""
    ema = DtypeModel().float()
    validator = object.__new__(BaseValidator)
    validator.args = SimpleNamespace(quantize=quantize)
    trainer = SimpleNamespace(
        amp=False,
        amp_enabled=False,
        amp_dtype=None,
        ema=SimpleNamespace(ema=ema),
        model=DtypeModel(),
        args=SimpleNamespace(compile=False),
    )

    validation_model, owns_model = validator._prepare_training_model(trainer)

    assert next(validation_model.parameters()).dtype == expected_dtype
    assert next(ema.parameters()).dtype == torch.float32
    assert owns_model is (quantize in {16, "bf16"})
    assert (validation_model is ema) is not owns_model
    assert validator.args.quantize == quantize


def test_training_validator_amp_keeps_fp32_model_and_input():
    """AMP validation must not reuse quantize as a hidden FP16 switch."""
    ema = DtypeModel().float()
    validator = object.__new__(BaseValidator)
    validator.args = SimpleNamespace(quantize=None)
    trainer = SimpleNamespace(
        amp=True,
        amp_enabled=True,
        amp_dtype=torch.bfloat16,
        ema=SimpleNamespace(ema=ema),
        model=DtypeModel(),
        args=SimpleNamespace(compile=False),
    )

    validation_model, owns_model = validator._prepare_training_model(trainer)

    assert validation_model is ema and not owns_model
    assert next(validation_model.parameters()).dtype == torch.float32
    assert validator.input_dtype == torch.float32
    assert validator.amp_enabled and validator.amp_dtype == torch.bfloat16
    assert validator.args.quantize is None


@pytest.mark.parametrize(
    ("enabled", "dtype", "expected_amp"),
    [(False, torch.float16, False), (True, torch.float16, "fp16"), (True, torch.bfloat16, "bf16")],
)
def test_final_validation_inherits_effective_training_amp(monkeypatch, tmp_path, enabled, dtype, expected_amp):
    """Final best.pt validation uses the same effective AMP state as per-epoch validation."""
    best = tmp_path / "best.pt"
    best.touch()
    captured = {}

    class FakeValidator:
        def __init__(self):
            self.args = SimpleNamespace(plots=False, compile=True, amp=False, quantize=None)

        def __call__(self, model=None):
            captured.update(model=model, amp=self.args.amp, quantize=self.args.quantize)
            return {"fitness": 1.0}

    trainer = object.__new__(BaseTrainer)
    trainer.best = best
    trainer.last = tmp_path / "missing-last.pt"
    trainer.validator = FakeValidator()
    trainer.args = SimpleNamespace(plots=False)
    trainer.amp_enabled = enabled
    trainer.amp_dtype = dtype
    trainer.epoch = 1
    trainer.run_callbacks = lambda *_args: None
    monkeypatch.setattr(trainer_module, "strip_optimizer", lambda *_args, **_kwargs: {})

    trainer.final_eval()

    assert captured == {"model": best, "amp": expected_amp, "quantize": None}


@pytest.mark.parametrize(
    ("fp16", "bf16", "expected_dtype"),
    [(False, False, torch.float32), (True, False, torch.float16), (False, True, torch.bfloat16)],
)
def test_native_backend_uses_one_weight_and_input_dtype(fp16, bf16, expected_dtype):
    """Native backend manual precision casts both the whole model and floating input."""
    model = DtypeModel()
    backend = AutoBackend(model, device=torch.device("cpu"), fp16=fp16, bf16=bf16, fuse=False, verbose=False)
    output = backend(torch.ones(1, 3, 2, 2, dtype=torch.float32))

    assert backend.dtype == expected_dtype
    assert next(backend.model.parameters()).dtype == expected_dtype
    assert backend.model.last_input_dtype == expected_dtype
    assert output.dtype == expected_dtype


def test_bf16_backend_rejects_exported_formats_before_loading():
    """Manual BF16 is limited to native PyTorch runtime models."""
    with pytest.raises(ValueError, match="only native PyTorch"):
        AutoBackend("model.onnx", device=torch.device("cpu"), bf16=True, verbose=False)


def test_autobatch_receives_training_amp_dtype(monkeypatch):
    """AutoBatch profiles under the same autocast dtype as the training forward path."""
    import importlib

    autobatch_module = importlib.import_module("ultralytics.utils.autobatch")
    captured = {}

    def fake_autocast(**kwargs):
        captured.update(kwargs)
        return nullcontext()

    monkeypatch.setattr(autobatch_module, "autocast", fake_autocast)
    monkeypatch.setattr(autobatch_module, "autobatch", lambda *_args, **_kwargs: 7)

    result = autobatch_module.check_train_batch_size(DtypeModel(), amp=True, amp_dtype=torch.bfloat16)

    assert result == 7
    assert captured == {"enabled": True, "device": "cpu", "dtype": torch.bfloat16}


def test_repeated_predict_reconfigures_amp_and_quantize():
    """Changing precision on a reused predictor rebuilds its backend instead of retaining stale dtype state."""

    class FakePredictor:
        def __init__(self, overrides, _callbacks):
            self.args = get_cfg(overrides=overrides)
            self.model = None
            self.setup_calls = []

        def setup_model(self, model, verbose=False):
            self.setup_calls.append((self.args.amp, self.args.quantize))
            self.model = SimpleNamespace(dynamic=True)

        def __call__(self, source=None, stream=False):
            return []

    model = YOLO(CFG)
    model.predict(torch.zeros(1, 3, 32, 32), predictor=FakePredictor, amp=False, quantize=16)
    model.predict(torch.zeros(1, 3, 32, 32), amp="bf16", quantize=None)

    assert model.predictor.setup_calls == [(False, 16), ("bf16", None)]
