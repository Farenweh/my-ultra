from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import ultralytics.engine.trainer as trainer_module
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import checks


def test_parse_empty_vram_cache_bool(monkeypatch):
    monkeypatch.delenv("EMPTY_VRAM_CACHE", raising=False)
    assert checks._parse_env_bool("EMPTY_VRAM_CACHE", False) is False

    monkeypatch.setenv("EMPTY_VRAM_CACHE", "")
    assert checks._parse_env_bool("EMPTY_VRAM_CACHE", False) is False

    monkeypatch.setenv("EMPTY_VRAM_CACHE", "False")
    assert checks._parse_env_bool("EMPTY_VRAM_CACHE", False) is False

    monkeypatch.setenv("EMPTY_VRAM_CACHE", "True")
    assert checks._parse_env_bool("EMPTY_VRAM_CACHE", False) is True


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 0.6),
        ("", 0.6),
        ("True", 0.6),
        ("False", False),
        ("0", False),
        ("0.75", 0.75),
        ("1", 1.0),
    ],
)
def test_parse_vram_target(raw, expected):
    assert checks._parse_vram_target(raw) == expected


@pytest.mark.parametrize("raw", ["bad", "-0.1", "1.1"])
def test_parse_vram_target_rejects_invalid_values(raw):
    with pytest.raises(ValueError, match="VRAM_TARGET"):
        checks._parse_vram_target(raw)


def test_vram_target_rejects_empty_cache_conflict():
    with pytest.raises(ValueError, match="VRAM_TARGET 已启用"):
        checks._validate_vram_target_config(0.8, True)


def _make_vram_trainer(device_type: str = "cuda"):
    trainer = object.__new__(BaseTrainer)
    trainer.device = torch.device(device_type, 0)
    trainer._vram_target_reserve = None
    trainer._vram_target_applied = False
    return trainer


def test_vram_target_waits_until_third_epoch(monkeypatch):
    trainer = _make_vram_trainer()
    calls = []

    monkeypatch.setattr(trainer_module, "VRAM_TARGET", 0.8)
    monkeypatch.setattr(trainer_module.torch, "empty", lambda *args, **kwargs: calls.append(args))

    trainer._maybe_apply_vram_target(epoch=0)
    trainer._maybe_apply_vram_target(epoch=1)

    assert calls == []
    assert trainer._vram_target_applied is False


def test_vram_target_fills_cuda_reserved_gap_once(monkeypatch):
    trainer = _make_vram_trainer()
    infos = []
    allocated = []
    reserved_bytes = {"value": 200}

    monkeypatch.setattr(trainer_module, "VRAM_TARGET", 0.8)
    monkeypatch.setattr(trainer_module.LOGGER, "info", lambda message: infos.append(message))
    monkeypatch.setattr(trainer_module.torch.cuda, "memory_reserved", lambda device=None: reserved_bytes["value"])
    monkeypatch.setattr(
        trainer_module.torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(total_memory=1000)
    )
    monkeypatch.setattr(trainer_module.torch.cuda, "synchronize", lambda device=None: None)

    def fake_empty(shape, *, dtype, device):
        allocated.append((shape, dtype, device))
        reserved_bytes["value"] += shape[0]
        return {"shape": shape, "dtype": dtype, "device": device}

    monkeypatch.setattr(trainer_module.torch, "empty", fake_empty)

    trainer._maybe_apply_vram_target(epoch=2)
    trainer._maybe_apply_vram_target(epoch=3)

    assert allocated == [((600,), torch.uint8, torch.device("cuda", 0))]
    assert trainer._vram_target_reserve == {
        "shape": (600,),
        "dtype": torch.uint8,
        "device": torch.device("cuda", 0),
    }
    assert trainer._vram_target_applied is True
    assert "填充前VRAM占用大小" in infos[0]
    assert "百分比=20.00%" in infos[0]
    assert "预计准备填充的tensor shape=(600,)" in infos[0]
    assert "预计填充完后的VRAM占用大小" in infos[0]
    assert "当前VRAM占用大小" in infos[0]
    assert "百分比=80.00%" in infos[0]


def test_vram_target_skips_when_already_at_target(monkeypatch):
    trainer = _make_vram_trainer()
    infos = []

    monkeypatch.setattr(trainer_module, "VRAM_TARGET", 0.8)
    monkeypatch.setattr(trainer_module.LOGGER, "info", lambda message: infos.append(message))
    monkeypatch.setattr(trainer_module.torch.cuda, "memory_reserved", lambda device=None: 900)
    monkeypatch.setattr(
        trainer_module.torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(total_memory=1000)
    )
    monkeypatch.setattr(trainer_module.torch.cuda, "synchronize", lambda device=None: None)
    monkeypatch.setattr(trainer_module.torch, "empty", lambda *args, **kwargs: pytest.fail("不应分配保留张量"))

    trainer._maybe_apply_vram_target(epoch=2)

    assert trainer._vram_target_reserve is None
    assert trainer._vram_target_applied is True
    assert "预计准备填充的tensor shape=(0,)" in infos[0]
    assert "百分比=90.00%" in infos[0]


def test_vram_target_oom_warns_and_disables_retry(monkeypatch):
    trainer = _make_vram_trainer()
    warnings = []
    calls = []

    monkeypatch.setattr(trainer_module, "VRAM_TARGET", 0.8)
    monkeypatch.setattr(trainer_module.LOGGER, "warning", lambda message: warnings.append(message))
    monkeypatch.setattr(trainer_module.torch.cuda, "memory_reserved", lambda device=None: 200)
    monkeypatch.setattr(
        trainer_module.torch.cuda, "get_device_properties", lambda device=None: SimpleNamespace(total_memory=1000)
    )
    monkeypatch.setattr(trainer_module.torch.cuda, "synchronize", lambda device=None: None)

    def fake_empty(*args, **kwargs):
        calls.append(args)
        raise torch.cuda.OutOfMemoryError("out of memory")

    monkeypatch.setattr(trainer_module.torch, "empty", fake_empty)

    trainer._maybe_apply_vram_target(epoch=2)
    trainer._maybe_apply_vram_target(epoch=3)

    assert len(calls) == 1
    assert trainer._vram_target_applied is True
    assert "VRAM_TARGET显存填充失败" in warnings[0]


def test_vram_target_uses_npu_reserved_metric(monkeypatch):
    trainer = _make_vram_trainer("npu")
    allocated = []
    fake_npu = SimpleNamespace(
        memory_reserved=lambda: 250,
        get_device_properties=lambda device=None: SimpleNamespace(total_memory=1000),
        synchronize=lambda device=None: None,
    )

    monkeypatch.setattr(trainer_module, "VRAM_TARGET", 0.5)
    monkeypatch.setattr(trainer_module.LOGGER, "info", lambda message: None)
    monkeypatch.setattr(trainer_module.torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(
        trainer_module.torch,
        "empty",
        lambda shape, *, dtype, device: allocated.append((shape, dtype, device)) or object(),
    )

    trainer._maybe_apply_vram_target(epoch=2)

    assert allocated == [((250,), torch.uint8, torch.device("npu", 0))]
