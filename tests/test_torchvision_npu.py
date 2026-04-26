from __future__ import annotations

import builtins
import logging
import types

import ultralytics.utils.torch_utils as torch_utils


def test_enable_torchvision_npu_skips_without_available_npu(monkeypatch):
    calls = []
    original_import = builtins.__import__

    monkeypatch.setattr(torch_utils, "_TORCHVISION_NPU_AVAILABLE", None)
    if hasattr(torch_utils.torch, "npu"):
        monkeypatch.setattr(torch_utils.torch.npu, "is_available", lambda: False, raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "torchvision_npu":
            calls.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert torch_utils.enable_torchvision_npu() is False
    assert calls == []


def test_enable_torchvision_npu_warns_when_import_fails(monkeypatch):
    warnings = []
    original_import = builtins.__import__

    monkeypatch.setattr(torch_utils, "_TORCHVISION_NPU_AVAILABLE", None)
    monkeypatch.setattr(torch_utils.torch, "npu", types.SimpleNamespace(is_available=lambda: True), raising=False)
    monkeypatch.setattr(torch_utils.LOGGER, "warning", lambda message: warnings.append(message))

    def fake_import(name, *args, **kwargs):
        if name == "torchvision_npu":
            raise ImportError("missing torchvision_npu")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert torch_utils.enable_torchvision_npu() is False
    assert len(warnings) == 1
    assert "torchvision_npu 加载失败" in warnings[0]


def test_enable_torchvision_npu_imports_only_once(monkeypatch):
    calls = []
    fake_module = types.ModuleType("torchvision_npu")
    original_import = builtins.__import__

    monkeypatch.setattr(torch_utils, "_TORCHVISION_NPU_AVAILABLE", None)
    monkeypatch.setattr(torch_utils.torch, "npu", types.SimpleNamespace(is_available=lambda: True), raising=False)

    def fake_import(name, *args, **kwargs):
        if name == "torchvision_npu":
            calls.append(name)
            return fake_module
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    assert torch_utils.enable_torchvision_npu() is True
    assert torch_utils.enable_torchvision_npu() is True
    assert calls == ["torchvision_npu"]


def test_enable_torchvision_npu_preserves_root_logging_and_hides_env_info(monkeypatch, capsys):
    calls = []
    fake_module = types.ModuleType("torchvision_npu")
    original_import = builtins.__import__
    root_logger = logging.getLogger()
    env_logger = logging.getLogger("torch_npu.env")
    original_root_level = root_logger.level
    original_root_handlers = list(root_logger.handlers)
    original_env_level = env_logger.level

    monkeypatch.delenv("TORCH_NPU_LOGS", raising=False)
    monkeypatch.setattr(torch_utils, "_TORCHVISION_NPU_AVAILABLE", None)
    monkeypatch.setattr(torch_utils.torch, "npu", types.SimpleNamespace(is_available=lambda: True), raising=False)

    for handler in original_root_handlers:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.ERROR)
    env_logger.setLevel(logging.NOTSET)

    def fake_import(name, *args, **kwargs):
        if name == "torchvision_npu":
            calls.append(name)
            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s | %(message)s")
            logging.getLogger("torch_npu.env").info("get env WORLD_SIZE = 8")
            return fake_module
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        assert torch_utils.enable_torchvision_npu() is True
        captured = capsys.readouterr()

        assert calls == ["torchvision_npu"]
        assert root_logger.level == logging.ERROR
        assert root_logger.handlers == []
        assert "get env WORLD_SIZE" not in captured.out
        assert "get env WORLD_SIZE" not in captured.err
    finally:
        root_logger.setLevel(original_root_level)
        for handler in list(root_logger.handlers):
            root_logger.removeHandler(handler)
        for handler in original_root_handlers:
            root_logger.addHandler(handler)
        env_logger.setLevel(original_env_level)


def test_enable_torchvision_npu_respects_torch_npu_logs_env(monkeypatch):
    fake_module = types.ModuleType("torchvision_npu")
    original_import = builtins.__import__
    env_logger = logging.getLogger("torch_npu.env")
    original_env_level = env_logger.level

    monkeypatch.setenv("TORCH_NPU_LOGS", "env")
    monkeypatch.setattr(torch_utils, "_TORCHVISION_NPU_AVAILABLE", None)
    monkeypatch.setattr(torch_utils.torch, "npu", types.SimpleNamespace(is_available=lambda: True), raising=False)
    env_logger.setLevel(logging.INFO)

    def fake_import(name, *args, **kwargs):
        if name == "torchvision_npu":
            return fake_module
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        assert torch_utils.enable_torchvision_npu() is True
        assert env_logger.level == logging.INFO
    finally:
        env_logger.setLevel(original_env_level)
