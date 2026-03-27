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
