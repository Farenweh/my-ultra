from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import ultralytics.engine.trainer as trainer_module
import ultralytics.engine.validator as validator_module
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator


@pytest.mark.parametrize(("nccl_available", "expected_backend"), [(True, "nccl"), (False, "gloo")])
def test_setup_ddp_selects_cuda_backend_without_hccl_probe(monkeypatch, nccl_available, expected_backend):
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(device="0,1")
    trainer.device = torch.device("cuda", 0)
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


@pytest.mark.parametrize("device", ("0,1", "npu:0,1"))
def test_setup_ddp_selects_hccl_on_ascend_without_cuda_backend_probe(monkeypatch, device):
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(device=device)
    trainer.device = torch.device("npu", 0)
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


def test_validator_ddp_unprefixed_device_uses_resolved_backend(monkeypatch):
    validator = object.__new__(BaseValidator)
    validator.args = SimpleNamespace(
        augment=False,
        model="model.pt",
        end2end=None,
        max_det=300,
        agnostic_nms=False,
        data="coco8.yaml",
        device="0,1",
        dnn=False,
        quantize=None,
    )
    selected_device = torch.device("npu", 0)
    captured = {}

    def stop_after_backend(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("backend configured")

    monkeypatch.setattr(validator_module, "RANK", 0)
    monkeypatch.setattr(validator_module, "LOCAL_RANK", 0)
    monkeypatch.setattr(validator_module.callbacks, "add_integration_callbacks", lambda validator: None)
    monkeypatch.setattr(validator_module, "convert_ndjson_to_yolo_if_needed", lambda data: data)
    monkeypatch.setattr(validator_module, "torch_distributed_zero_first", lambda rank: nullcontext())
    monkeypatch.setattr(validator_module, "select_device", lambda device, verbose: selected_device)
    monkeypatch.setattr(
        validator_module,
        "get_torch_device_backend",
        lambda device: SimpleNamespace(current_device=lambda: 1),
    )
    monkeypatch.setattr(validator_module, "AutoBackend", stop_after_backend)

    with pytest.raises(RuntimeError, match="backend configured"):
        validator()

    assert captured["device"] == torch.device("npu", 1)


@pytest.mark.parametrize(
    ("local_batches", "local_loss", "remote_batches", "remote_loss", "expected"),
    (
        (2, (2.0, 4.0), 2, (2.0, 4.0), (1.0, 2.0)),
        (2, (2.0, 4.0), 3, (3.0, 6.0), (1.0, 2.0)),
        (0, (0.0, 0.0), 5, (5.0, 10.0), (1.0, 2.0)),
        (2, (1.38622, 1.134), 18, (20.79374, 17.01), (1.108998, 0.9072)),
    ),
)
def test_validator_reduces_loss_by_global_batch_count(
    monkeypatch, local_batches, local_loss, remote_batches, remote_loss, expected
):
    validator = object.__new__(BaseValidator)
    validator.device = torch.device("cpu")
    validator.dataloader = [None] * local_batches
    validator.loss = {
        "box_loss": torch.tensor(local_loss[0]),
        "cls_loss": torch.tensor(local_loss[1]),
    }
    reduced = iter((*remote_loss, float(remote_batches)))
    ops = []

    def reduce(value, dst, op):
        ops.append((dst, op))
        value.add_(next(reduced))

    monkeypatch.setattr(validator_module, "RANK", 0)
    monkeypatch.setattr(validator_module.dist, "reduce", reduce)

    loss = validator._reduce_training_loss(SimpleNamespace(world_size=4))

    assert torch.isclose(loss["box_loss"], torch.tensor(expected[0]))
    assert torch.isclose(loss["cls_loss"], torch.tensor(expected[1]))
    assert ops == [(0, validator_module.dist.ReduceOp.SUM)] * 3


def test_validator_single_rank_loss_preserves_local_batch_mean(monkeypatch):
    validator = object.__new__(BaseValidator)
    validator.device = torch.device("cpu")
    validator.dataloader = [None, None]
    validator.loss = {"box_loss": torch.tensor(3.0)}
    monkeypatch.setattr(validator_module, "RANK", 0)
    monkeypatch.setattr(
        validator_module.dist,
        "reduce",
        lambda *_args, **_kwargs: pytest.fail("single-rank validation must not reduce"),
    )

    loss = validator._reduce_training_loss(SimpleNamespace(world_size=1))

    assert torch.equal(loss["box_loss"], torch.tensor(1.5))


def test_validator_nonzero_rank_reduces_before_returning(monkeypatch):
    validator = object.__new__(BaseValidator)
    validator.device = torch.device("cpu")
    validator.dataloader = [None]
    validator.loss = {"box_loss": torch.tensor(1.0)}
    calls = []
    monkeypatch.setattr(validator_module, "RANK", 1)
    monkeypatch.setattr(validator_module.dist, "reduce", lambda value, dst, op: calls.append((value, dst, op)))

    assert validator._reduce_training_loss(SimpleNamespace(world_size=2)) is None
    assert len(calls) == 2
    assert all(dst == 0 and op == validator_module.dist.ReduceOp.SUM for _, dst, op in calls)
