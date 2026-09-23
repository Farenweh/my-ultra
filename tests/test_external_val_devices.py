"""通过公共验证分派入口验证外部 torchrun 与 K8s 的设备参数。"""
from types import SimpleNamespace

import pytest
import torch

from ultralytics.engine import val_runtime


@pytest.fixture
def external_context(tmp_path, monkeypatch):
    for key, value in {
        "LOCAL_RANK": "0", "RANK": "0", "WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2",
        "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "12345",
        "ULTRALYTICS_DISTRIBUTED_VAL_WORKER": "0", "ULTRALYTICS_DISTRIBUTED_VAL_CONFIG": "",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(val_runtime, "IS_ASCEND", True)
    monkeypatch.setattr(val_runtime, "_visible_devices", lambda: ("npu", [0, 1]))
    monkeypatch.setattr(val_runtime, "_run_worker", lambda owner, args, direct: val_runtime._load_worker_config())
    return {"task": "detect", "mode": "val", "batch": 4, "project": str(tmp_path), "name": "val"}


@pytest.mark.parametrize("device", ([0, 1], (0, 1), "0,1", "npu:0,1", None, "", "none"))
def test_external_validation_accepts_supported_device_forms(external_context, device):
    result = val_runtime.run_or_launch_distributed_validation(SimpleNamespace(), {**external_context, "device": device}, lambda args: args)
    assert result["device_ids"] == [0, 1]
    assert result["device_argument"] == "npu:0,1"
    assert result["global_batch"] == 4 and result["local_batch"] == 2


@pytest.mark.parametrize("device", ("cuda:0,1", torch.device("cuda")))
def test_external_validation_preserves_explicit_cuda(external_context, monkeypatch, device):
    """显式 CUDA 请求在昇腾主机上也应保留 CUDA 后端。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    result = val_runtime.run_or_launch_distributed_validation(
        SimpleNamespace(), {**external_context, "device": device}, lambda args: args
    )

    assert result["device_type"] == "cuda"
    assert result["device_ids"] == [0, 1]
    assert result["device_argument"] == "cuda:0,1"


def test_explicit_cuda_validation_rejects_missing_cuda(external_context, monkeypatch):
    """只有 NPU 可用时，显式 CUDA 验证请求应直接报错。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="没有可用的CUDA"):
        val_runtime.run_or_launch_distributed_validation(
            SimpleNamespace(), {**external_context, "device": "cuda:0,1"}, lambda args: args
        )


@pytest.mark.parametrize(("device", "message"), (([0], "LOCAL_WORLD_SIZE"), ([0, 0], "互不重复"),
                                                ([-2, 1], "非负整数"), ("invalid", "无法解析")))
def test_external_validation_keeps_device_errors(external_context, device, message):
    with pytest.raises(ValueError, match=message):
        val_runtime.run_or_launch_distributed_validation(SimpleNamespace(), {**external_context, "device": device}, lambda args: args)


@pytest.mark.parametrize("device", ([0, 1], (0, 1), 0, "npu:0,1"))
def test_k8s_rejects_explicit_devices_with_expected_error(external_context, monkeypatch, device):
    monkeypatch.setenv("LOCAL_RANK", "-1")
    monkeypatch.setattr(val_runtime, "is_k8s_distributed_parent", lambda: True)
    with pytest.raises(ValueError, match="K8S多节点验证不应手动设置device"):
        val_runtime.run_or_launch_distributed_validation(SimpleNamespace(), {**external_context, "device": device}, lambda args: args)
