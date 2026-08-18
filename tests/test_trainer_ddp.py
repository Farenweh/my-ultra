from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import ultralytics.engine.trainer as trainer_module
import ultralytics.engine.validator as validator_module
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.engine.validator import BaseValidator


class RTDETRDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class DDPBufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bn_float_buffers", torch.zeros(249344 // 4, dtype=torch.float32))
        self.register_buffer("bn_count_buffers", torch.zeros(1336 // 8, dtype=torch.int64))


def _padding_buffers(model):
    return {
        name: buffer for name, buffer in model.named_buffers() if name.startswith(BaseTrainer._DDP_BUFFER_ALIGN_PREFIX)
    }


def _buffer_bytes_by_dtype(model):
    bytes_by_dtype = {}
    for buffer in model.buffers():
        bytes_by_dtype[buffer.dtype] = bytes_by_dtype.get(buffer.dtype, 0) + buffer.numel() * buffer.element_size()
    return bytes_by_dtype


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
    assert captured["device_id"] == torch.device("cuda", 0)


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
    assert captured["device_id"] == torch.device("npu", 0)


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
    monkeypatch.setattr(trainer_module, "USE_ASCEND_FUSED_OPTIMIZER", True, raising=False)

    assert BaseTrainer._resolve_ddp_gradient_as_bucket_view() is True


def test_resolve_ddp_gradient_as_bucket_view_stays_true_without_fused_optimizer(monkeypatch):
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_FUSED_OPTIMIZER", False, raising=False)

    assert BaseTrainer._resolve_ddp_gradient_as_bucket_view() is True


@pytest.mark.parametrize("setting", [None, True])
def test_resolve_ddp_gradient_as_bucket_view_disabled_for_possible_ascend_fused_optimizer(monkeypatch, setting):
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_FUSED_OPTIMIZER", setting, raising=False)

    assert BaseTrainer._resolve_ddp_gradient_as_bucket_view() is False


@pytest.mark.parametrize("setting", [None, True])
def test_align_ddp_broadcast_buffers_pads_each_dtype_for_per_rank_alignment(monkeypatch, setting):
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_DDP_BUFFER_ALIGN", setting, raising=False)
    model = DDPBufferModel()

    padding = BaseTrainer._align_ddp_broadcast_buffers(model, world_size=2)

    bytes_by_dtype = _buffer_bytes_by_dtype(model)
    assert bytes_by_dtype[torch.float32] % 1024 == 0
    assert bytes_by_dtype[torch.int64] % 1024 == 0
    assert sorted(padding.values()) == [512, 712]
    assert sorted(buffer.numel() * buffer.element_size() for buffer in _padding_buffers(model).values()) == [
        512,
        712,
    ]


def test_align_ddp_broadcast_buffers_padding_is_not_persistent(monkeypatch):
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_DDP_BUFFER_ALIGN", True, raising=False)
    model = DDPBufferModel()

    BaseTrainer._align_ddp_broadcast_buffers(model, world_size=2)

    assert _padding_buffers(model)
    assert not any(name.startswith(BaseTrainer._DDP_BUFFER_ALIGN_PREFIX) for name in model.state_dict())


def test_align_ddp_broadcast_buffers_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_DDP_BUFFER_ALIGN", False, raising=False)
    model = DDPBufferModel()

    padding = BaseTrainer._align_ddp_broadcast_buffers(model, world_size=2)

    assert padding == {}
    assert _padding_buffers(model) == {}


def test_align_ddp_broadcast_buffers_skips_outside_ascend(monkeypatch):
    monkeypatch.setattr(trainer_module, "IS_ASCEND", False)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_DDP_BUFFER_ALIGN", True, raising=False)
    model = DDPBufferModel()

    padding = BaseTrainer._align_ddp_broadcast_buffers(model, world_size=2)

    assert padding == {}
    assert _padding_buffers(model) == {}


def test_align_ddp_broadcast_buffers_skips_single_process(monkeypatch):
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_DDP_BUFFER_ALIGN", True, raising=False)
    model = DDPBufferModel()

    padding = BaseTrainer._align_ddp_broadcast_buffers(model, world_size=1)

    assert padding == {}
    assert _padding_buffers(model) == {}


def test_align_ddp_broadcast_buffers_is_idempotent(monkeypatch):
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module, "USE_ASCEND_DDP_BUFFER_ALIGN", True, raising=False)
    model = DDPBufferModel()

    BaseTrainer._align_ddp_broadcast_buffers(model, world_size=2)
    first_padding = _padding_buffers(model)
    first_total_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())
    BaseTrainer._align_ddp_broadcast_buffers(model, world_size=2)
    second_padding = _padding_buffers(model)
    second_total_bytes = sum(buffer.numel() * buffer.element_size() for buffer in model.buffers())

    assert sorted(buffer.numel() * buffer.element_size() for buffer in first_padding.values()) == [512, 712]
    assert sorted(buffer.numel() * buffer.element_size() for buffer in second_padding.values()) == [512, 712]
    assert first_total_bytes == second_total_bytes
