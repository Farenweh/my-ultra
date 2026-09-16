"""训练结束后的公共模型、检查点和分布式结果交接回归。"""
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import ultralytics.engine.trainer as trainer_module
from tests.rtdetr_helpers import run_gloo_workers, tiny_rtdetr
from ultralytics import RTDETR, YOLO
from ultralytics.cfg import get_cfg
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_file


@pytest.mark.parametrize("kind", ("yolo", "rtdetr"))
@pytest.mark.parametrize("pretrained", (False, True))
@pytest.mark.parametrize("checkpoint", (None, "last", "best"))
@pytest.mark.parametrize("save", (False, True))
def test_public_training_receives_current_result(tmp_path, monkeypatch, kind, pretrained, checkpoint, save):
    """从 YAML 或权重启动均接收本轮结果，并使已有 predictor 失效。"""
    if kind == "yolo":
        owner = YOLO("yolo11n.yaml")
    else:
        cfg = tmp_path / "rtdetr-tiny.yaml"
        YAML.save(cfg, tiny_rtdetr().yaml)
        owner = RTDETR(cfg)
    if pretrained:
        weight = tmp_path / "initial.pt"
        owner.save(weight)
        owner = type(owner)(weight)
    original = owner.model
    owner.predictor = object()

    class ResultTrainer(BaseTrainer):
        def __init__(self, overrides, _callbacks):
            self.args = get_cfg(overrides=overrides)
            self.model = None
            self.ema = None
            self.best, self.last = tmp_path / "best.pt", tmp_path / "last.pt"
            self.best.write_bytes("旧文件不得被读取".encode())
            self.last.write_bytes("旧文件不得被读取".encode())
            self._saved_checkpoints = set()
            self.validator = SimpleNamespace(metrics={"验证指标": 0.5})

        def get_model(self, weights=None, cfg=None):
            result = deepcopy(weights) if weights is not None else deepcopy(original)
            self.seed_weight = next(weights.parameters()).detach().clone() if weights is not None else None
            result.args = vars(self.args).copy()
            return result

        def train(self):
            if self.model is None:
                self.model = self.get_model()
            with torch.no_grad():
                next(self.model.parameters()).fill_(0.25)
            self.ema = SimpleNamespace(ema=deepcopy(self.model))
            with torch.no_grad():
                next(self.ema.ema.parameters()).fill_(0.75)
            if checkpoint:
                path = getattr(self, checkpoint)
                torch.save({"model": self.ema.ema, "train_args": vars(self.args)}, path)
                self._saved_checkpoints.add(path)

    monkeypatch.setattr("ultralytics.engine.model.checks.check_pip_update_available", lambda: None)
    assert owner.train(trainer=ResultTrainer, data="offline.yaml", save=save) == {"验证指标": 0.5}
    torch.testing.assert_close(next(owner.model.parameters()), torch.full_like(next(owner.model.parameters()), 0.75))
    assert owner.predictor is None
    assert owner.ckpt
    if checkpoint is None:
        assert owner.ckpt_path is None
        assert owner.ckpt.get("epoch", -1) == -1 and owner.ckpt.get("optimizer") is None
    else:
        assert Path(owner.ckpt_path) == getattr(owner.trainer, checkpoint)
    output = tmp_path / "trained.pt"
    owner.save(output)
    loaded, _ = load_checkpoint(output)
    torch.testing.assert_close(next(loaded.parameters()), next(owner.model.parameters()))
    # 第二次微调必须使用刚取得的权重。
    owner.train(trainer=ResultTrainer, data="offline.yaml", save=save)
    torch.testing.assert_close(owner.trainer.seed_weight, torch.full_like(owner.trainer.seed_weight, 0.75))


def _trainer(tmp_path):
    trainer = object.__new__(BaseTrainer)
    trainer.model = torch.nn.Linear(2, 1)
    trainer.model.args = {"task": "classify"}
    trainer.ema = None
    trainer.args = SimpleNamespace(task="classify", model="source.yaml")
    trainer.metrics = {"指标": 1.0}
    trainer.validator = None
    trainer.best, trainer.last = tmp_path / "best.pt", tmp_path / "last.pt"
    trainer._saved_checkpoints = set()
    return trainer


def test_result_rejects_missing_model_and_stale_files(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.best.write_bytes("旧权重".encode())
    trainer.last.write_bytes("旧权重".encode())
    trainer.model = "source.yaml"
    with pytest.raises(RuntimeError, match="训练结果"):
        trainer._get_training_result()


def test_final_eval_ignores_old_files(tmp_path, monkeypatch):
    trainer = _trainer(tmp_path)
    trainer.best.write_bytes("旧权重".encode())
    trainer.last.write_bytes("旧权重".encode())
    monkeypatch.setattr(trainer_module, "RANK", -1)
    monkeypatch.setattr(trainer_module, "strip_optimizer", lambda *a, **k: pytest.fail("不得处理旧权重"))
    trainer.final_eval()
    assert trainer.best.read_bytes() == "旧权重".encode()


def test_ddp_runner_publishes_and_cleans_training_result(tmp_path):
    trainer = _trainer(tmp_path)
    trainer.save_dir = tmp_path
    trainer.callbacks = {}
    runner = generate_ddp_file(trainer)
    content = runner.read_text()
    assert "_save_training_result" in content
    result_path = runner.with_suffix(".result.pt")
    trainer._save_training_result(result_path)
    received = trainer._load_training_result(result_path)
    torch.testing.assert_close(received["model"].weight, trainer.model.weight)
    ddp_cleanup(trainer, runner)
    assert not result_path.exists()


def _result_worker(rank, world_size, rendezvous, folder):
    torch.set_num_threads(1)
    trainer_module.RANK = rank
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=40))
    try:
        trainer = _trainer(Path(folder))
        model = torch.nn.parallel.DistributedDataParallel(trainer.model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.ones(2, 2) * (rank + 1)).sum().backward()
        optimizer.step()
        trainer.model = model
        trainer._save_training_result(Path(folder) / "result.pt")
        dist.barrier()
        result = trainer._load_training_result(Path(folder) / "result.pt")
        torch.testing.assert_close(result["model"].weight, model.module.weight)
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_result_handoff(tmp_path):
    run_gloo_workers(_result_worker, tmp_path, str(tmp_path))


def test_ddp_parent_receives_result_before_cleanup(tmp_path, monkeypatch):
    """覆盖父训练入口的读取顺序，防止清理先于交接。"""
    parent = _trainer(tmp_path)
    parent.ddp = True
    parent.k8s_launch_config = None
    parent.local_world_size = parent.world_size = 2
    parent.save_dir = tmp_path
    parent.callbacks = {}
    parent.args.rect = False
    parent.args.batch = 2
    child = _trainer(tmp_path)
    with torch.no_grad():
        child.model.weight.fill_(2.0)
    paths = []

    def run(command, check):
        path = Path(command[-1]).with_suffix(".result.pt")
        paths.append(path)
        child._save_training_result(path)

    monkeypatch.setattr(trainer_module, "RANK", -1)
    monkeypatch.setattr(trainer_module.subprocess, "run", run)
    parent.train()
    result = parent._get_training_result()
    torch.testing.assert_close(result["model"].weight, child.model.weight)
    assert paths and not any(p.exists() for p in paths)


def test_memory_result_preserves_qat_without_mutating_source(tmp_path, monkeypatch):
    """模拟动态 QAT 类型，确认交接可序列化且只剥离副本。"""
    class DynamicQATLinear(torch.nn.Linear):
        pass

    trainer = _trainer(tmp_path)
    trainer.model = DynamicQATLinear(2, 1)
    state = {"量化范围": 2}

    def strip(model):
        assert model is not trainer.model
        model.__class__ = torch.nn.Linear

    monkeypatch.setattr(trainer_module, "qat_state", lambda model: state)
    monkeypatch.setattr(trainer_module, "strip_qat", strip)
    monkeypatch.setattr(trainer_module, "restore_qat", lambda model, saved: setattr(model, "restored", saved))
    path = tmp_path / "result.pt"
    trainer._save_training_result(path)
    result = trainer._load_training_result(path)
    assert result["model"].restored == state
    assert isinstance(trainer.model, DynamicQATLinear)
