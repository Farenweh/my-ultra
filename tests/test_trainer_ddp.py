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
from ultralytics.models.rtdetr.train import RTDETRTrainer


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


def _capture_ddp_kwargs(monkeypatch, model, compile, trainer_class=BaseTrainer):
    trainer = object.__new__(trainer_class)
    trainer.resume = False
    trainer.args = SimpleNamespace(
        amp=False,
        quantize=None,
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
    lifecycle = []

    def build_train_pipeline():
        lifecycle.append("训练流程已准备")

    def distributed_data_parallel(model, **kwargs):
        assert lifecycle == ["训练流程已准备"], "DDP 必须在训练流程确定参数物理布局后构造"
        lifecycle.append("DDP 已构造")
        captured.update(kwargs)
        return model

    def stop_after_ddp():
        lifecycle.append("验证器入口")
        raise RuntimeError("DDP 已配置")

    monkeypatch.setattr(trainer_module, "RANK", -1)
    monkeypatch.setattr(trainer_module, "IS_ASCEND", False)
    monkeypatch.setattr(trainer_module, "PROFILE", "")
    monkeypatch.setattr(trainer_module, "attempt_compile", lambda model, **kwargs: model)
    monkeypatch.setattr(nn.parallel, "DistributedDataParallel", distributed_data_parallel)
    trainer._build_train_pipeline = build_train_pipeline
    trainer.get_validator = stop_after_ddp

    with pytest.raises(RuntimeError, match="DDP 已配置"):
        trainer._setup_train()
    assert lifecycle == ["训练流程已准备", "DDP 已构造", "验证器入口"]
    return captured


@pytest.mark.parametrize("ddp", (False, True))
@pytest.mark.parametrize("batch", (0.5, 1, 3, 2.0, True))
def test_train_rejects_invalid_distributed_batch_before_launch(monkeypatch, ddp, batch):
    """父进程和外部 torchrun worker 都应拒绝不能均分的全局 batch。"""
    trainer = object.__new__(BaseTrainer)
    trainer.world_size = 2
    trainer.batch_size = batch
    trainer.ddp = ddp
    cleaned = []
    trainer.callbacks = {"teardown": [lambda _: cleaned.append("回调")]}
    trainer._teardown_train_resources = lambda: cleaned.append("资源")
    trainer._do_train = lambda: pytest.fail("不应开始训练")
    monkeypatch.setattr(trainer_module, "unset_deterministic", lambda: None)

    with pytest.raises(ValueError, match="WORLD_SIZE=2"):
        trainer.train()
    assert cleaned == ([] if ddp else ["回调", "资源"])


@pytest.mark.parametrize("batch", (2, 4, 8))
def test_distributed_batch_accepts_exact_multiples(batch):
    trainer = object.__new__(BaseTrainer)
    trainer.world_size = 2
    trainer.batch_size = batch
    trainer._validate_distributed_batch_size()


def test_build_train_pipeline_rechecks_distributed_batch():
    """重建 dataloader 时也应防止向下取整丢失样本。"""
    trainer = object.__new__(BaseTrainer)
    trainer.world_size = 3
    trainer.batch_size = 4

    with pytest.raises(ValueError, match="WORLD_SIZE=3"):
        trainer._build_train_pipeline()


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("cuda:0", "cuda:0"),
        ("cuda:0,1", "cuda:0,1"),
        (torch.device("cuda", 1), "cuda:1"),
        (torch.device("cuda"), "cuda"),
    ],
)
def test_trainer_preserves_explicit_cuda_backend(monkeypatch, requested, expected):
    """训练器传给设备选择器的请求应保留显式 CUDA 后端。"""
    selected = []

    def capture(device):
        selected.append(str(device))
        raise RuntimeError("已捕获设备请求")

    monkeypatch.setattr(trainer_module, "is_k8s_training_enabled", lambda: False)
    monkeypatch.setattr(BaseTrainer, "check_resume", lambda self, overrides: None)
    monkeypatch.setattr(trainer_module, "select_device", capture)

    with pytest.raises(RuntimeError, match="已捕获设备请求"):
        BaseTrainer(overrides={"device": requested})

    assert selected == [expected]


@pytest.mark.parametrize(("nccl_available", "expected_backend"), [(True, "nccl"), (False, "gloo")])
@pytest.mark.parametrize("is_ascend", (False, True))
def test_setup_ddp_selects_cuda_backend_without_hccl_probe(monkeypatch, nccl_available, expected_backend, is_ascend):
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
    monkeypatch.setattr(trainer_module, "IS_ASCEND", is_ascend)
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
        nms=None,
        max_det=300,
        agnostic_nms=False,
        data="coco8.yaml",
        task="detect",
        split="val",
        fraction=1.0,
        device="0,1",
        dnn=False,
        quantize=None,
        channels_last=None,
    )
    selected_device = torch.device("npu", 0)
    captured = {}

    def stop_after_backend(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("backend configured")

    monkeypatch.setattr(validator_module, "RANK", 0)
    monkeypatch.setattr(validator_module, "LOCAL_RANK", 0)
    monkeypatch.setattr(validator_module.callbacks, "add_integration_callbacks", lambda validator: None)
    monkeypatch.setattr(validator_module, "convert_ndjson_to_yolo_if_needed", lambda data, fraction, split: data)
    monkeypatch.setattr(validator_module, "check_det_dataset", lambda data, split: {split: "images"})
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


def test_setup_train_rtdetr_compile_uses_dynamic_ddp_graph(monkeypatch):
    """实际 DDP 构造入口应采用 RT-DETR trainer 的动态训练图配置。"""
    kwargs = _capture_ddp_kwargs(monkeypatch, RTDETRDetectionModel(), True, RTDETRTrainer)
    assert kwargs["static_graph"] is False
    assert kwargs["find_unused_parameters"] is False


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
