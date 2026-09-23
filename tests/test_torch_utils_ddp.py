from __future__ import annotations

import itertools
import os
import subprocess
import sys
import types

import pytest

import ultralytics.utils.torch_utils as torch_utils


def test_restore_thop_state_removes_only_new_artifacts():
    """THOP profiling cleanup must preserve user hooks while removing partial profiling state."""
    module = torch_utils.torch.nn.ReLU()
    user_handle = module.register_forward_hook(lambda *_: None)
    state = torch_utils._snapshot_thop_state((module,))
    thop_handle = module.register_forward_hook(lambda *_: None)
    module.register_buffer("total_ops", torch_utils.torch.zeros(1))
    module.register_buffer("total_params", torch_utils.torch.zeros(1))

    torch_utils._restore_thop_state(state)

    assert user_handle.id in module._forward_hooks
    assert thop_handle.id not in module._forward_hooks
    assert "total_ops" not in module._buffers
    assert "total_params" not in module._buffers
    user_handle.remove()


def test_select_cpu_does_not_hide_npu_from_later_work(monkeypatch):
    """CPU-only work in a warm Ascend process must not poison later NPU initialization."""
    visible = "0,1"
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(torch_utils, "IS_ASCEND", True)

    assert str(torch_utils.select_device("cpu", verbose=False)) == "cpu"
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == visible


def test_select_device_npu_list_uses_logical_device_indices(monkeypatch):
    selected = []
    fake_npu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda index: "Ascend910B",
        set_device=selected.append,
    )

    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch_utils, "IS_ASCEND", True)
    monkeypatch.setattr(torch_utils.torch, "npu", fake_npu, raising=False)
    device = torch_utils.select_device([1], verbose=False)

    assert "ASCEND_RT_VISIBLE_DEVICES" not in os.environ
    assert str(device) == "npu:1"
    assert selected == [1]


@pytest.mark.parametrize("device_request", ([0, 1], (0, 1), "0, 1", "0,1"))
def test_select_device_unprefixed_list_routes_to_npu(monkeypatch, device_request):
    fake_npu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda index: f"Ascend910B-{index}",
        set_device=lambda index: None,
    )

    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(torch_utils, "IS_ASCEND", True)
    monkeypatch.setattr(torch_utils.torch, "npu", fake_npu, raising=False)
    device = torch_utils.select_device(device_request, verbose=False)

    assert torch_utils.parse_device(device_request) == "0,1"
    assert "ASCEND_RT_VISIBLE_DEVICES" not in os.environ
    assert str(device) == "npu:0"


def test_select_device_preserves_existing_npu_visibility(monkeypatch):
    """已设置物理卡可见范围时，逻辑卡号不得覆盖该范围。"""
    fake_npu = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda index: f"Ascend910B-{index}",
        set_device=lambda index: None,
    )
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "6,7")
    monkeypatch.setattr(torch_utils, "IS_ASCEND", True)
    monkeypatch.setattr(torch_utils.torch, "npu", fake_npu, raising=False)

    assert str(torch_utils.select_device("1", verbose=False)) == "npu:1"
    assert os.environ["ASCEND_RT_VISIBLE_DEVICES"] == "6,7"


def test_select_device_explicit_cuda_on_ascend_host(monkeypatch):
    """显式 CUDA 请求应始终使用 CUDA 后端。"""
    selected = []
    monkeypatch.setattr(torch_utils, "IS_ASCEND", True)
    monkeypatch.setattr(torch_utils.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch_utils.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch_utils.torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(torch_utils, "get_gpu_info", lambda index, device_type=None: "测试设备")

    assert str(torch_utils.select_device("cuda:0", verbose=False)) == "cuda:0"
    assert selected == [0]


def test_select_device_explicit_cuda_rejects_missing_cuda_on_ascend_host(monkeypatch):
    """有 NPU 但无 CUDA 时，显式 CUDA 请求不能静默切换到 NPU。"""
    monkeypatch.setattr(torch_utils, "IS_ASCEND", True)
    monkeypatch.setattr(torch_utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch_utils.torch.cuda, "device_count", lambda: 0)

    with pytest.raises(ValueError, match="Invalid CUDA"):
        torch_utils.select_device("cuda:0", verbose=False)


@pytest.mark.skipif(
    not hasattr(torch_utils.torch, "npu") or not torch_utils.torch.npu.is_available(),
    reason="NPU is not available",
)
def test_nn_modules_import_does_not_lock_visible_devices():
    script = """
import os
import torch
import ultralytics.nn.modules

os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
x = torch.ones(1, device="npu:0")
print(x.cpu().item())
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "1.0" in result.stdout


class FakeStore:
    def __init__(self):
        self.values = {}
        self.waits = []

    def set(self, key, value):
        self.values[key] = value

    def wait(self, keys):
        self.waits.append(tuple(keys))

    def check(self, keys):
        return all(key in self.values for key in keys)

    def get(self, key):
        return str(self.values[key]).encode()


def setup_dist(monkeypatch, backend="hccl", rank=0, world_size=8):
    monkeypatch.setattr(torch_utils.dist, "is_available", lambda: True)
    monkeypatch.setattr(torch_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(torch_utils.dist, "get_backend", lambda: backend)
    monkeypatch.setattr(torch_utils.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(torch_utils.dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(torch_utils.torch.cuda, "current_device", lambda: rank)
    monkeypatch.setattr(torch_utils, "_ZERO_FIRST_COUNTER", itertools.count())


def test_zero_first_hccl_global_waits_on_store(monkeypatch):
    setup_dist(monkeypatch, rank=3)
    store = FakeStore()
    store.set("ultralytics/zero_first/0/0/done", "1")
    barriers = []
    monkeypatch.setattr(torch_utils, "_get_distributed_store", lambda: store)
    monkeypatch.setattr(torch_utils.dist, "barrier", lambda *args, **kwargs: barriers.append((args, kwargs)))

    seen = []
    with torch_utils.torch_distributed_zero_first(3, global_rank=True):
        seen.append("body")

    assert seen == ["body"]
    assert store.waits == [("ultralytics/zero_first/0/0/done",)]
    assert barriers == []


def test_zero_first_hccl_leader_sets_store_done(monkeypatch):
    setup_dist(monkeypatch, rank=0)
    store = FakeStore()
    barriers = []
    monkeypatch.setattr(torch_utils, "_get_distributed_store", lambda: store)
    monkeypatch.setattr(torch_utils.dist, "barrier", lambda *args, **kwargs: barriers.append((args, kwargs)))

    with torch_utils.torch_distributed_zero_first(0, global_rank=True):
        store.set("inside", "1")

    assert store.values["inside"] == "1"
    assert store.values["ultralytics/zero_first/0/0/done"] == "1"
    assert barriers == []


def test_zero_first_hccl_local_scope_waits_for_each_node_leader(monkeypatch):
    setup_dist(monkeypatch, rank=9, world_size=16)
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    store = FakeStore()
    store.set("ultralytics/zero_first/0/0/done", "1")
    store.set("ultralytics/zero_first/0/8/done", "1")
    monkeypatch.setattr(torch_utils, "_get_distributed_store", lambda: store)

    with torch_utils.torch_distributed_zero_first(1):
        pass

    assert store.waits == [("ultralytics/zero_first/0/0/done", "ultralytics/zero_first/0/8/done")]


def test_zero_first_nccl_keeps_device_id_barrier(monkeypatch):
    setup_dist(monkeypatch, backend="nccl", rank=2)
    barriers = []
    monkeypatch.setattr(torch_utils.dist, "barrier", lambda *args, **kwargs: barriers.append((args, kwargs)))

    with torch_utils.torch_distributed_zero_first(2):
        pass

    assert barriers == [((), {"device_ids": [2]})]


def test_zero_first_gloo_keeps_plain_barrier(monkeypatch):
    setup_dist(monkeypatch, backend="gloo", rank=2)
    barriers = []
    monkeypatch.setattr(torch_utils.dist, "barrier", lambda *args, **kwargs: barriers.append((args, kwargs)))

    with torch_utils.torch_distributed_zero_first(2):
        pass

    assert barriers == [((), {})]
