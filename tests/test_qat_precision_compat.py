"""上游 INT8 QAT 与 NPU 运行时精度扩展的兼容检查。"""

import pytest
import torch
from torch import nn

import ultralytics.engine.trainer as trainer_module
from ultralytics.cfg import get_cfg
from ultralytics.engine.predictor import BasePredictor


class TensorQuantizer(nn.Identity):
    """模拟已存在的 QAT 标记，测试不安装或运行 ModelOpt。"""


class PrecisionProbe(nn.Module):
    def __init__(self, qat):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.quantizer = TensorQuantizer() if qat else nn.Identity()
        self.names = {0: "目标"}
        self.stride = torch.tensor([32])
        self.yaml = {"channels": 3}

    def forward(self, x):
        return self.quantizer(x * self.weight)


@pytest.mark.parametrize("mode", ("train", "predict", "val"))
@pytest.mark.parametrize("amp", (False, True, "bf16"))
@pytest.mark.parametrize("quantize", (8, "int8"))
def test_qat_is_compatible_with_amp(mode, amp, quantize):
    cfg = get_cfg(overrides={"mode": mode, "amp": amp, "quantize": quantize})
    assert cfg.quantize == 8
    assert cfg.amp == amp


@pytest.mark.parametrize("quantize", (16, "bf16", 32))
def test_manual_whole_model_precision_still_requires_amp_disabled(quantize):
    with pytest.raises(ValueError, match="mutually exclusive"):
        get_cfg(overrides={"mode": "train", "amp": True, "quantize": quantize})


@pytest.mark.parametrize("qat", (False, True))
def test_int8_native_inference_requires_existing_qat(qat, tmp_path):
    predictor = BasePredictor(
        overrides={
            "mode": "predict",
            "device": "cpu",
            "amp": True,
            "quantize": 8,
            "project": str(tmp_path),
            "name": "qat",
        }
    )
    model = PrecisionProbe(qat)
    if qat:
        predictor.setup_model(model, verbose=False)
        assert predictor.model.dtype == torch.float32
    else:
        with pytest.raises(ValueError, match="native PyTorch runtime precision"):
            predictor.setup_model(model, verbose=False)


@pytest.mark.parametrize("resume", (False, True))
@pytest.mark.parametrize("amp", (False, True, "bf16"))
def test_training_prepares_or_restores_qat_before_compile(monkeypatch, resume, amp):
    """训练入口保留上游 QAT 准备及恢复，并允许 NPU 扩展的 AMP 配置。"""
    trainer = object.__new__(trainer_module.BaseTrainer)
    trainer.args = get_cfg(overrides={"mode": "train", "quantize": 8, "amp": amp, "imgsz": 32})
    trainer.model = PrecisionProbe(False)
    trainer.device = torch.device("cpu")
    trainer.batch_size = trainer.world_size = 1
    trainer.resume = resume
    trainer.data = {"train": "合成数据"}
    state = {"校准状态": 1}
    trainer.setup_model = lambda: {"modelopt": state} if resume else {}
    trainer.set_model_attributes = lambda: None
    loader = object()
    trainer.get_dataloader = lambda *args, **kwargs: loader
    trainer.preprocess_batch = lambda batch: batch
    calls = []

    def prepare(model, data, preprocess):
        assert data is loader
        calls.append("准备")
        model.quantizer = TensorQuantizer()
        return model

    def restore(model, saved):
        assert saved == state
        calls.append("恢复")
        model.quantizer = TensorQuantizer()

    def compile_model(model, **kwargs):
        assert isinstance(model.quantizer, TensorQuantizer)
        assert trainer.args.amp == amp
        raise RuntimeError("QAT 已先于编译配置完成")

    monkeypatch.setattr(trainer_module, "prepare_qat", prepare)
    monkeypatch.setattr(trainer_module, "restore_qat", restore)
    monkeypatch.setattr(trainer_module, "attempt_compile", compile_model)
    with pytest.raises(RuntimeError, match="QAT 已先于编译配置完成"):
        trainer._setup_train()
    assert calls == ["恢复" if resume else "准备"]
