from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ultralytics.engine.trainer as trainer_module
from ultralytics.engine.trainer import BaseTrainer


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
