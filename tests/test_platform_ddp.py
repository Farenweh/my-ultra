# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ultralytics.engine.trainer as trainer_module
import ultralytics.models.yolo.detect.val as detect_val
import ultralytics.utils.dist as dist_utils


def test_normalize_k8s_launch_config(monkeypatch):
    """Test K8s pod-level rendezvous envs are normalized for torchrun."""
    monkeypatch.setenv("K8S_TRAINING", "1")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "master.service")
    monkeypatch.setenv("MASTER_PORT", "23456")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "5")

    config = dist_utils.normalize_k8s_launch_config(2)

    assert config is not None
    assert config.master_addr == "master.service"
    assert config.master_port == 23456
    assert config.node_rank == 1
    assert config.nnodes == 4


def test_k8s_training_requires_explicit_flag(monkeypatch):
    """Test rendezvous envs alone do not trigger K8s launch mode without K8S_TRAINING=1."""
    monkeypatch.delenv("K8S_TRAINING", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "master.service")
    monkeypatch.setenv("MASTER_PORT", "23456")
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "5")

    assert dist_utils.is_k8s_training_enabled() is False
    assert dist_utils.is_k8s_distributed_parent() is False
    assert dist_utils.normalize_k8s_launch_config(2) is None


@pytest.mark.parametrize(
    ("rank", "world_size", "master_port", "message"),
    [
        ("0", "1", "23456", "WORLD_SIZE=1"),
        ("4", "4", "23456", "RANK=4"),
        ("0", "5", "bad-port", "MASTER_PORT='bad-port'"),
    ],
)
def test_normalize_k8s_launch_config_rejects_invalid_env(monkeypatch, rank, world_size, master_port, message):
    """Test malformed K8s rendezvous envs fail early with clear messages."""
    monkeypatch.setenv("K8S_TRAINING", "1")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "master.service")
    monkeypatch.setenv("MASTER_PORT", master_port)
    monkeypatch.setenv("RANK", rank)
    monkeypatch.setenv("WORLD_SIZE", world_size)

    with pytest.raises(ValueError, match=message):
        dist_utils.normalize_k8s_launch_config(2)


def test_generate_k8s_ddp_command(monkeypatch, tmp_path):
    """Test K8s DDP launch uses pod envs to construct a multi-node torchrun command."""
    monkeypatch.setenv("K8S_TRAINING", "1")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "master.service")
    monkeypatch.setenv("MASTER_PORT", "23456")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "5")

    runner = tmp_path / "runner.py"
    runner.write_text("print('runner')", encoding="utf-8")
    monkeypatch.setattr(dist_utils, "generate_ddp_file", lambda trainer: runner)

    trainer = SimpleNamespace(resume=False, save_dir=tmp_path / "runs" / "exp", local_world_size=2, world_size=8)
    cmd, file = dist_utils.generate_k8s_ddp_command(trainer)

    assert file == str(runner)
    assert "--nnodes" in cmd and cmd[cmd.index("--nnodes") + 1] == "4"
    assert "--node_rank" in cmd and cmd[cmd.index("--node_rank") + 1] == "1"
    assert "--master_addr" in cmd and cmd[cmd.index("--master_addr") + 1] == "master.service"
    assert "--master_port" in cmd and cmd[cmd.index("--master_port") + 1] == "23456"
    assert "--nproc_per_node" in cmd and cmd[cmd.index("--nproc_per_node") + 1] == "2"


def test_generate_k8s_ddp_command_keeps_existing_save_dir(monkeypatch, tmp_path):
    """Test K8s launch no longer deletes the shared save_dir before creating the runner file."""
    monkeypatch.setenv("K8S_TRAINING", "1")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("MASTER_ADDR", "master.service")
    monkeypatch.setenv("MASTER_PORT", "23456")
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "5")

    save_dir = tmp_path / "runs" / "exp"
    save_dir.mkdir(parents=True)
    sentinel = save_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    trainer = SimpleNamespace(
        resume=False,
        save_dir=save_dir,
        local_world_size=2,
        world_size=8,
        args=SimpleNamespace(model="yolo11n.yaml", augmentations=None),
        hub_session=None,
    )
    _, file = dist_utils.generate_k8s_ddp_command(trainer)

    assert sentinel.exists()
    assert Path(file).exists()


def test_generate_ddp_file_serializes_resolved_save_dir(tmp_path):
    """Test generated DDP runner scripts pass the resolved parent save_dir directly to child trainers."""
    trainer = SimpleNamespace(
        args=SimpleNamespace(model="yolo11n.yaml", augmentations=None),
        save_dir=tmp_path / "runs" / "exp",
        hub_session=None,
    )

    runner = dist_utils.generate_ddp_file(trainer, dist_dir=tmp_path / ".dist")
    content = runner.read_text(encoding="utf-8")

    assert str(trainer.save_dir.resolve()) in content


@pytest.mark.parametrize("device", [0, "0,1", [0, 1], "-1", "cpu"])
def test_k8s_parent_rejects_user_device(monkeypatch, device):
    """Test K8s parent processes reject explicit device settings with a Chinese error."""
    monkeypatch.setattr(
        trainer_module,
        "get_cfg",
        lambda cfg, overrides: SimpleNamespace(device=device, seed=0, deterministic=False),
    )
    monkeypatch.setattr(trainer_module, "is_k8s_training_enabled", lambda: True)
    monkeypatch.setattr(trainer_module, "is_k8s_distributed_parent", lambda: True)

    with pytest.raises(ValueError, match="在任务提交分布式状态下不应手动设置此值"):
        trainer_module.BaseTrainer(cfg={}, overrides={})


def test_k8s_parent_auto_resolves_visible_devices(monkeypatch):
    """Test K8s parent processes auto-expand visible local accelerators when device is unset."""
    captured = {}

    def fake_select_device(device):
        captured["device"] = device
        raise RuntimeError("stop after select_device")

    monkeypatch.setattr(
        trainer_module,
        "get_cfg",
        lambda cfg, overrides: SimpleNamespace(device="", seed=0, deterministic=False),
    )
    monkeypatch.setattr(trainer_module, "is_k8s_training_enabled", lambda: True)
    monkeypatch.setattr(trainer_module, "is_k8s_distributed_parent", lambda: True)
    monkeypatch.setattr(trainer_module.BaseTrainer, "check_resume", lambda self, overrides: setattr(self, "resume", False))
    monkeypatch.setattr(trainer_module, "select_device", fake_select_device)
    monkeypatch.setattr(trainer_module, "IS_ASCEND", True)
    monkeypatch.setattr(trainer_module.torch.npu, "is_available", lambda: True)
    monkeypatch.setattr(trainer_module.torch.npu, "device_count", lambda: 2)

    with pytest.raises(RuntimeError, match="stop after select_device"):
        trainer_module.BaseTrainer(cfg={}, overrides={})

    assert captured["device"] == "0,1"


def test_detection_validator_reduces_confusion_matrix_as_float32(monkeypatch):
    """Test DDP confusion matrix reduction uses a collective-compatible dtype and preserves summed values."""
    source = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    validator = object.__new__(detect_val.DetectionValidator)
    validator.args = SimpleNamespace(plots=True)
    validator.device = torch.device("cpu")
    validator.metrics = SimpleNamespace(stats={"tp": []}, box=None, clear_stats=lambda: None)
    validator.jdict = []
    validator.dataloader = SimpleNamespace(dataset=[0, 1])
    validator.confusion_matrix = SimpleNamespace(matrix=source.copy())
    validator._gather_image_metrics = lambda metric: None
    reduced = {}

    def gather_object(value, output, dst):
        if output is not None:
            output[:] = [value] * len(output)

    def reduce(matrix, dst, op):
        reduced["dtype"] = matrix.dtype
        reduced["dst"] = dst
        reduced["op"] = op
        matrix.mul_(2)

    monkeypatch.setattr(detect_val, "RANK", 0)
    monkeypatch.setattr(detect_val.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(detect_val.dist, "gather_object", gather_object)
    monkeypatch.setattr(detect_val.dist, "reduce", reduce)

    detect_val.DetectionValidator.gather_stats(validator)

    assert reduced == {"dtype": torch.float32, "dst": 0, "op": detect_val.dist.ReduceOp.SUM}
    assert validator.confusion_matrix.matrix.dtype == np.float32
    np.testing.assert_array_equal(validator.confusion_matrix.matrix, source * 2)
