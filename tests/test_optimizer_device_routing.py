import sys
from types import SimpleNamespace

import pytest
import torch

from ultralytics.engine import trainer as trainer_module
from ultralytics.engine.trainer import BaseTrainer


@pytest.mark.parametrize("device_type", ("cpu", "cuda"))
@pytest.mark.parametrize("name", ("SGD", "AdamW"))
@pytest.mark.parametrize("setting", (None, False, True))
def test_non_npu_training_ignores_ascend_fused_options(monkeypatch, device_type, name, setting):
    """Ascend 主机上的非 NPU 训练应选择标准优化器并完成参数更新。"""
    trainer = object.__new__(BaseTrainer)
    trainer.device = torch.device(device_type)
    trainer.args = SimpleNamespace(lr0=0.01, momentum=0.9, warmup_bias_lr=0.0)
    trainer.data = {"nc": 2}
    trainer.model = torch.nn.Linear(2, 1)
    trainer.ema = None
    trainer.scaler = torch.amp.GradScaler("cpu", enabled=False)

    def forbidden_fused_optimizer(**kwargs):
        raise AssertionError("非 NPU 训练不应构造 NPU 融合优化器")

    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_FUSED_OPTIMIZER", setting, raising=False)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_FUSED_GRAD_CLIP", setting, raising=False)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(optim=SimpleNamespace(**{f"NpuFused{name}": forbidden_fused_optimizer})),
    )
    trainer.optimizer = trainer.build_optimizer(trainer.model, name=name, lr=0.01)
    assert type(trainer.optimizer) is getattr(torch.optim, name)

    # CUDA 仅模拟设备路由；CPU 张量即可验证标准优化器和梯度裁剪的实际更新。
    before = trainer.model.weight.detach().clone()
    trainer.model(torch.ones(2, 2)).square().sum().backward()
    trainer.optimizer_step()
    assert not torch.equal(before, trainer.model.weight)
    assert torch.isfinite(trainer.model.weight).all()
