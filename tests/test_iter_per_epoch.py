from __future__ import annotations

from types import SimpleNamespace

import pytest
from torch.utils.data import Dataset

import ultralytics.engine.trainer as trainer_module
from ultralytics.cfg import get_cfg
from ultralytics.data.build import InfiniteDataLoader
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.callbacks import comet as comet_callbacks


class RangeDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, index):
        return index


def _make_cycle_trainer(train_loader):
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.train_loader = train_loader
    trainer._train_cycle_size = len(train_loader)
    trainer._restore_train_cycle_state()
    return trainer


def test_get_cfg_accepts_positive_iters_per_epoch():
    cfg = get_cfg(DEFAULT_CFG, overrides={"iters_per_epoch": 3})
    assert cfg.iters_per_epoch == 3


def test_get_cfg_rejects_non_positive_iters_per_epoch():
    with pytest.raises(ValueError, match="iters_per_epoch"):
        get_cfg(DEFAULT_CFG, overrides={"iters_per_epoch": 0})


def test_get_cfg_defaults_val_batch_factor_to_none():
    cfg = get_cfg(DEFAULT_CFG)
    assert cfg.val_batch_factor is None


@pytest.mark.parametrize("val_batch_factor", [1, 2, 4])
def test_get_cfg_accepts_positive_int_val_batch_factor(val_batch_factor):
    cfg = get_cfg(DEFAULT_CFG, overrides={"val_batch_factor": val_batch_factor})
    assert cfg.val_batch_factor == val_batch_factor


@pytest.mark.parametrize(
    ("val_batch_factor", "error_type"),
    [(0, ValueError), (-1, ValueError), (1.5, TypeError), ("2", TypeError), (True, TypeError)],
)
def test_get_cfg_rejects_invalid_val_batch_factor_with_chinese_error(val_batch_factor, error_type):
    with pytest.raises(error_type, match="无效|正整数"):
        get_cfg(DEFAULT_CFG, overrides={"val_batch_factor": val_batch_factor})


@pytest.mark.parametrize(
    ("task", "val_batch_factor", "expected_batch_size"),
    [
        ("detect", None, 8),
        ("obb", None, 4),
        ("semantic", None, 4),
        ("detect", 3, 12),
    ],
)
def test_resolve_val_batch_size_respects_factor_or_task_defaults(task, val_batch_factor, expected_batch_size):
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(task=task, val_batch_factor=val_batch_factor)

    assert trainer._resolve_val_batch_size(train_batch_size=4) == expected_batch_size


def test_resolve_val_batch_size_warns_when_obb_factor_increases_batch(monkeypatch):
    warnings = []
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(task="obb", val_batch_factor=2)
    monkeypatch.setattr(trainer_module.LOGGER, "warning", warnings.append)

    assert trainer._resolve_val_batch_size(train_batch_size=4) == 8
    assert len(warnings) == 1
    assert "可能增加验证阶段显存占用" in warnings[0]


def test_next_train_batch_continues_across_logical_epochs():
    loader = InfiniteDataLoader(RangeDataset(), batch_size=2, shuffle=False, num_workers=0)
    trainer = _make_cycle_trainer(loader)

    batches = []
    for _ in range(3):
        epoch_batches = []
        for _ in range(2):
            epoch_batches.extend(trainer._next_train_batch().tolist())
        batches.append(epoch_batches)

    assert batches == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 0, 1]]


def test_train_cycle_reseeds_only_when_cycle_rolls_over(monkeypatch):
    class DummySampler:
        def __init__(self):
            self.calls = []

        def set_epoch(self, epoch):
            self.calls.append(epoch)

    class DummyLoader:
        def __init__(self):
            self.sampler = DummySampler()

        def __len__(self):
            return 3

        def __iter__(self):
            epoch = self.sampler.calls[-1] if self.sampler.calls else -1
            return iter([(epoch, 0), (epoch, 1), (epoch, 2)])

    trainer = _make_cycle_trainer(DummyLoader())
    monkeypatch.setattr(trainer_module, "RANK", 0)

    assert trainer._next_train_batch() == (0, 0)
    assert trainer.train_loader.sampler.calls == [0]
    assert trainer._next_train_batch() == (0, 1)
    assert trainer.train_loader.sampler.calls == [0]
    assert trainer._next_train_batch() == (0, 2)
    assert trainer.train_loader.sampler.calls == [0]
    assert trainer._next_train_batch() == (1, 0)
    assert trainer.train_loader.sampler.calls == [0, 1]


def test_comet_metadata_uses_global_step():
    trainer = SimpleNamespace(
        epoch=1,
        global_step=8,
        iters_per_epoch=4,
        train_loader=[0, 1, 2],
        batch_size=2,
        epochs=3,
        args=SimpleNamespace(save=False, save_period=1),
    )

    metadata = comet_callbacks._fetch_trainer_metadata(trainer)

    assert metadata["curr_epoch"] == 2
    assert metadata["curr_step"] == 8
