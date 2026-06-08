from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import ultralytics.engine.trainer as trainer_module
from ultralytics.engine.trainer import BaseTrainer


class RTDETRDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


def _capture_ddp_kwargs(monkeypatch, model, compile):
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(
        amp=False,
        channels_last=False,
        compile=compile,
        distill_model=None,
        freeze=None,
        imgsz=32,
    )
    trainer.batch_size = 1
    trainer.device = torch.device("cpu")
    trainer.model = model
    trainer.setup_model = lambda: None
    trainer.set_model_attributes = lambda: None
    trainer.world_size = 2

    captured = {}

    def distributed_data_parallel(model, **kwargs):
        captured.update(kwargs)
        return model

    def stop_after_ddp():
        raise RuntimeError("DDP configured")

    monkeypatch.setattr(trainer_module, "RANK", -1)
    monkeypatch.setattr(trainer_module, "IS_ASCEND", False)
    monkeypatch.setattr(trainer_module, "PROFILE", "")
    monkeypatch.setattr(trainer_module, "attempt_compile", lambda model, **kwargs: model)
    monkeypatch.setattr(nn.parallel, "DistributedDataParallel", distributed_data_parallel)
    trainer._build_train_pipeline = stop_after_ddp

    with pytest.raises(RuntimeError, match="DDP configured"):
        trainer._setup_train()
    return captured


@pytest.mark.parametrize(("nccl_available", "expected_backend"), [(True, "nccl"), (False, "gloo")])
def test_setup_ddp_selects_cuda_backend_without_hccl_probe(monkeypatch, nccl_available, expected_backend):
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(device="0,1")
    trainer.world_size = 2
    captured = {}
    fake_dist = SimpleNamespace(
        init_process_group=lambda **kwargs: captured.update(kwargs),
        is_nccl_available=lambda: nccl_available,
    )

    monkeypatch.setattr(trainer_module, "dist", fake_dist)
    monkeypatch.setattr(trainer_module, "IS_ASCEND", False)
    monkeypatch.setattr(trainer_module, "LOCAL_RANK", 0)
    monkeypatch.setattr(trainer_module, "RANK", 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)

    trainer._setup_ddp()

    assert captured["backend"] == expected_backend


def test_setup_ddp_selects_hccl_on_ascend_without_cuda_backend_probe(monkeypatch):
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(device="npu:0,1")
    trainer.world_size = 2
    captured = {}
    fake_dist = SimpleNamespace(init_process_group=lambda **kwargs: captured.update(kwargs))
    fake_accelerator = SimpleNamespace(set_device=lambda index: None)

    monkeypatch.setattr(trainer_module, "dist", fake_dist)
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "LOCAL_RANK", 0)
    monkeypatch.setattr(trainer_module, "RANK", 0)
    monkeypatch.setattr(trainer_module, "get_torch_device_backend", lambda device: fake_accelerator)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(set_device=lambda index: None), raising=False)

    trainer._setup_ddp()

    assert captured["backend"] == "hccl"


def test_resolve_ddp_find_unused_parameters_defaults_to_false_for_rtdetr():
    assert BaseTrainer._resolve_ddp_find_unused_parameters(RTDETRDetectionModel()) is False


def test_resolve_ddp_find_unused_parameters_defaults_to_true_for_generic_model():
    assert BaseTrainer._resolve_ddp_find_unused_parameters(nn.Linear(4, 4)) is True


def test_resolve_ddp_find_unused_parameters_env_override(monkeypatch):
    monkeypatch.setenv("ULTRALYTICS_DDP_FIND_UNUSED_PARAMETERS", "1")
    assert BaseTrainer._resolve_ddp_find_unused_parameters(RTDETRDetectionModel()) is True

    monkeypatch.setenv("ULTRALYTICS_DDP_FIND_UNUSED_PARAMETERS", "0")
    assert BaseTrainer._resolve_ddp_find_unused_parameters(nn.Linear(4, 4)) is False


@pytest.mark.parametrize(
    ("model", "compile", "expected_static_graph", "expected_find_unused"),
    [
        (nn.Linear(4, 4), False, False, True),
        (RTDETRDetectionModel(), False, False, False),
        (nn.Linear(4, 4), True, True, False),
    ],
)
def test_setup_train_passes_ddp_graph_kwargs(monkeypatch, model, compile, expected_static_graph, expected_find_unused):
    kwargs = _capture_ddp_kwargs(monkeypatch, model, compile)

    assert kwargs["static_graph"] is expected_static_graph
    assert kwargs["find_unused_parameters"] is expected_find_unused


def test_resolve_ddp_gradient_as_bucket_view_defaults_to_true(monkeypatch):
    monkeypatch.setattr(trainer_module, "IS_ASCEND", False)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_FUSED_OPTIMIZER", True)

    assert BaseTrainer._resolve_ddp_gradient_as_bucket_view() is True


def test_resolve_ddp_gradient_as_bucket_view_stays_true_without_fused_optimizer(monkeypatch):
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_FUSED_OPTIMIZER", False)

    assert BaseTrainer._resolve_ddp_gradient_as_bucket_view() is True


def test_resolve_ddp_gradient_as_bucket_view_disabled_for_ascend_fused_optimizer(monkeypatch):
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_FUSED_OPTIMIZER", True)

    assert BaseTrainer._resolve_ddp_gradient_as_bucket_view() is False
