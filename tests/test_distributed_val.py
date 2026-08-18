# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from ultralytics.engine import val_runtime


class _PromptModule(torch.nn.Module):
    """用于验证内存模型非参数状态也进入临时快照。"""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.names = {0: "自定义类别"}
        self.register_buffer("txt_feats", torch.tensor([[1.0, 2.0]], dtype=torch.float16))


def _gloo_scheduler_worker(rank, world_size, process_group_port, store_port, output_dir):
    """在真实Gloo进程中领取动态batch并落盘本rank结果。"""
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{process_group_port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    store = torch.distributed.TCPStore(
        "127.0.0.1",
        store_port,
        world_size,
        rank == 0,
        timedelta(seconds=30),
        True,
    )
    scheduler = val_runtime.DynamicBatchScheduler(store, total_samples=23, batch_size=3, namespace="gloo-test")
    scheduler.initialize(rank)
    claims = []
    while claimed := scheduler.claim():
        claims.append(claimed)
    Path(output_dir, f"rank{rank}.json").write_text(json.dumps(claims), encoding="utf-8")
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


class _AtomicStore:
    """用于验证原子batch领取的内存store。"""

    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = int(value)

    def add(self, key, value):
        self.values[key] = self.values.get(key, 0) + int(value)
        return self.values[key]


def test_global_batch_is_strict_and_returns_local_batch():
    assert val_runtime._validate_global_batch(128, 8) == (128, 16)
    with pytest.raises(ValueError, match="batch >= world_size"):
        val_runtime._validate_global_batch(4, 8)
    with pytest.raises(ValueError, match="整除world_size"):
        val_runtime._validate_global_batch(10, 8)
    with pytest.raises(TypeError, match="必须是整数"):
        val_runtime._validate_global_batch(8.0, 8)


def test_dynamic_scheduler_claims_every_sample_once_with_real_tail():
    store = _AtomicStore()
    scheduler = val_runtime.DynamicBatchScheduler(store, total_samples=10, batch_size=4, namespace="test")
    store.set(scheduler.key, 0)
    claims = []
    while claimed := scheduler.claim():
        claims.append(claimed)
    assert claims == [(0, [0, 1, 2, 3]), (1, [4, 5, 6, 7]), (2, [8, 9])]
    assert [index for _, indices in claims for index in indices] == list(range(10))


def test_dynamic_scheduler_reserves_one_initial_batch_per_rank(monkeypatch):
    store = _AtomicStore()
    monkeypatch.setattr(val_runtime.dist, "barrier", lambda: None)
    rank0 = val_runtime.DynamicBatchScheduler(store, total_samples=10, batch_size=2, namespace="reserved")
    rank1 = val_runtime.DynamicBatchScheduler(store, total_samples=10, batch_size=2, namespace="reserved")

    rank0.initialize(rank=0, world_size=2)
    rank1.initialize(rank=1, world_size=2)

    assert rank0.claim() == (0, [0, 1])
    assert rank1.claim() == (1, [2, 3])
    assert rank0.claim() == (2, [4, 5])


@pytest.mark.parametrize("world_size", [2, 4])
def test_tcpstore_scheduler_is_atomic_across_real_gloo_processes(tmp_path, world_size):
    process_group_port = val_runtime.find_free_network_port()
    store_port = val_runtime.find_free_network_port()
    while store_port == process_group_port:
        store_port = val_runtime.find_free_network_port()
    torch.multiprocessing.spawn(
        _gloo_scheduler_worker,
        args=(world_size, process_group_port, store_port, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )
    claims = []
    for rank in range(world_size):
        claims.extend(json.loads((tmp_path / f"rank{rank}.json").read_text(encoding="utf-8")))
    batch_ids = [batch_id for batch_id, _ in claims]
    indices = [index for _, batch_indices in claims for index in batch_indices]
    assert sorted(batch_ids) == list(range(8))
    assert sorted(indices) == list(range(23))
    assert len(indices) == len(set(indices))
    assert sorted(len(batch_indices) for _, batch_indices in claims) == [2, 3, 3, 3, 3, 3, 3, 3]


def test_distributed_sampler_records_claim_order():
    store = _AtomicStore()
    scheduler = val_runtime.DynamicBatchScheduler(store, total_samples=5, batch_size=2, namespace="test")
    store.set(scheduler.key, 0)
    context = SimpleNamespace(claimed_batches=[], claimed_indices=[])
    sampler = val_runtime.DynamicBatchSampler(scheduler, context)
    assert list(sampler) == [[0, 1], [2, 3], [4]]
    assert context.claimed_batches == [0, 1, 2]
    assert context.claimed_indices == [0, 1, 2, 3, 4]


def test_context_publishes_dataset_total_for_parent_progress(monkeypatch, tmp_path):
    store = _AtomicStore()
    monkeypatch.setattr(val_runtime.dist, "barrier", lambda: None)
    context = val_runtime.DistributedValContext(
        rank=0,
        local_rank=0,
        world_size=2,
        device=torch.device("cpu"),
        global_batch=8,
        local_batch=4,
        save_dir=tmp_path,
        metrics_path=tmp_path / "metrics.pt",
        progress_path=tmp_path / ".dist" / "progress.rank0.json",
        store=store,
        namespace="progress-test",
        started_at=0.0,
    )

    context.make_scheduler(23)

    metadata = json.loads((tmp_path / ".dist" / "dataset.json").read_text(encoding="utf-8"))
    assert metadata == {"total_samples": 23}


def test_single_device_path_does_not_launch(monkeypatch):
    owner = SimpleNamespace(model=torch.nn.Linear(2, 2), callbacks=defaultdict(list), metrics=None)
    monkeypatch.setattr(val_runtime, "is_k8s_distributed_parent", lambda: False)
    result = val_runtime.run_or_launch_distributed_validation(
        owner,
        {"device": "cpu", "batch": 1},
        lambda args: ("direct", args["batch"]),
    )
    assert result == ("direct", 1)


def test_multi_device_path_calls_launcher(monkeypatch):
    expected = object()
    owner = SimpleNamespace(model=torch.nn.Linear(2, 2), callbacks=defaultdict(list), metrics=None)
    monkeypatch.setattr(val_runtime, "is_k8s_distributed_parent", lambda: False)
    monkeypatch.setattr(val_runtime, "_device_request", lambda device: ("npu", [0, 1]))
    monkeypatch.setattr(val_runtime, "_launch", lambda *args: expected)
    result = val_runtime.run_or_launch_distributed_validation(
        owner,
        {"device": "0,1", "batch": 2},
        lambda args: None,
    )
    assert result is expected


def test_k8s_parent_uses_all_visible_devices_and_rejects_explicit_device(monkeypatch):
    expected = object()
    owner = SimpleNamespace(model=torch.nn.Linear(2, 2), callbacks=defaultdict(list), metrics=None)
    monkeypatch.setattr(val_runtime, "is_k8s_distributed_parent", lambda: True)
    monkeypatch.setattr(val_runtime, "_visible_devices", lambda: ("npu", [0, 1, 2, 3]))
    monkeypatch.setattr(val_runtime, "_launch", lambda *args: expected)
    result = val_runtime.run_or_launch_distributed_validation(
        owner,
        {"device": None, "batch": 8},
        lambda args: None,
    )
    assert result is expected

    with pytest.raises(ValueError, match="不应手动设置device"):
        val_runtime.run_or_launch_distributed_validation(
            owner,
            {"device": "0,1,2,3", "batch": 8},
            lambda args: None,
        )


def test_callbacks_reject_lambda_and_local_function():
    with pytest.raises(TypeError, match="无法在torchrun worker中导入"):
        val_runtime._validate_callbacks({"on_val_start": [lambda validator: None]})


def test_snapshot_preserves_current_fp32_state_and_moves_parent_to_cpu(tmp_path):
    model = torch.nn.Linear(2, 2).half()
    owner = SimpleNamespace(model=model)
    expected = {key: value.float().clone() for key, value in model.state_dict().items()}
    original_device = val_runtime._save_model_snapshot(owner, tmp_path / "snapshot.pt")
    checkpoint = torch.load(tmp_path / "snapshot.pt", map_location="cpu", weights_only=False)
    assert original_device.type == "cpu"
    assert all(value.dtype == torch.float32 for value in checkpoint["model"].state_dict().values())
    for key, value in checkpoint["model"].state_dict().items():
        torch.testing.assert_close(value, expected[key])


def test_snapshot_preserves_names_and_prompt_state(tmp_path):
    owner = SimpleNamespace(model=_PromptModule())
    val_runtime._save_model_snapshot(owner, tmp_path / "snapshot.pt")
    checkpoint = torch.load(tmp_path / "snapshot.pt", map_location="cpu", weights_only=False)
    snapshot = checkpoint["model"]
    assert snapshot.names == {0: "自定义类别"}
    assert snapshot.txt_feats.dtype == torch.float32
    torch.testing.assert_close(snapshot.txt_feats, torch.tensor([[1.0, 2.0]]))


def test_k8s_parent_coordination_uses_dedicated_tcpstore(monkeypatch):
    calls = []
    expected = object()

    def fake_store(*args):
        calls.append(args)
        return expected

    monkeypatch.setenv("ULTRALYTICS_VAL_PARENT_STORE_PORT", "23456")
    monkeypatch.setattr(val_runtime.dist, "TCPStore", fake_store)
    k8s = SimpleNamespace(master_addr="10.0.0.1", master_port=12345, nnodes=3, node_rank=1)
    assert val_runtime.create_k8s_parent_store(k8s) is expected
    assert calls[0][:4] == ("10.0.0.1", 23456, 3, False)
    assert calls[0][-1] is True


def test_oom_message_contains_global_and_local_batch():
    context = SimpleNamespace(
        global_batch=128,
        local_batch=16,
        rank=3,
        device=torch.device("cpu"),
    )
    error = val_runtime._distributed_oom_error(context, RuntimeError("out of memory"))
    assert "global_batch=128" in str(error)
    assert "local_batch=16" in str(error)
    assert "rank=3" in str(error)


def test_parent_reads_ranked_worker_failures(tmp_path):
    dist_dir = tmp_path / ".dist"
    dist_dir.mkdir()
    (dist_dir / "failure.rank1.json").write_text(
        json.dumps({"rank": 1, "type": "RuntimeError", "message": "global_batch=16 OOM"}),
        encoding="utf-8",
    )
    (dist_dir / "failure.rank0.json").write_text("invalid", encoding="utf-8")
    assert val_runtime._read_worker_failures(tmp_path) == ["rank=1 RuntimeError: global_batch=16 OOM"]


def test_dynamic_tensor_broadcast_allocates_receiver(monkeypatch):
    context = SimpleNamespace(rank=1, device=torch.device("cpu"))
    monkeypatch.setattr(val_runtime, "_ACTIVE_CONTEXT", context)

    def broadcast_metadata(metadata, src, device):
        assert src == 0
        assert device.type == "cpu"
        metadata[0] = ((2, 3), torch.float32)

    monkeypatch.setattr(val_runtime.dist, "broadcast_object_list", broadcast_metadata)
    monkeypatch.setattr(val_runtime.dist, "broadcast", lambda tensor, src: tensor.fill_(7))
    result = val_runtime.broadcast_val_tensor(None)
    assert result.shape == (2, 3)
    assert torch.equal(result, torch.full((2, 3), 7.0))


@pytest.mark.parametrize("rank,expected_factory_calls", [(0, 1), (1, 0)])
def test_yoloe_prompt_embedding_is_built_only_on_global_rank_zero(monkeypatch, rank, expected_factory_calls):
    from ultralytics.models.yolo.yoloe import val as yoloe_val

    validator = object.__new__(yoloe_val.YOLOEDetectValidator)
    calls = []
    monkeypatch.setattr(yoloe_val, "RANK", rank)
    monkeypatch.setattr(val_runtime, "get_distributed_val_context", lambda: object())
    monkeypatch.setattr(
        val_runtime,
        "broadcast_val_tensor",
        lambda value: torch.tensor([rank], dtype=torch.float32) if value is None else value,
    )

    result = validator._shared_prompt_embedding(lambda: calls.append("factory") or torch.tensor([7.0]))

    assert len(calls) == expected_factory_calls
    assert result.shape == (1,)
