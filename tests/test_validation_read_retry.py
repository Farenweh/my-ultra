"""验证读取失败只能重试原图，不能改变评估样本集合。"""
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tests.detection_data_helpers import detection_data
from tests.rtdetr_helpers import run_gloo_workers
from ultralytics.cfg import get_cfg
from ultralytics.data.base import BaseDataset
from ultralytics.data.build import build_yolo_dataset
from ultralytics.models.rtdetr.val import build_rtdetr_dataset


def _retry_dataset(tmp_path, augment, count=2, retries=2):
    dataset = object.__new__(BaseDataset)
    dataset.augment = augment
    dataset.prefix = "读取回归: "
    dataset.labels = [{} for _ in range(count)]
    dataset.im_files = [str(tmp_path / f"{i}.png") for i in range(count)]
    dataset.data_retries = retries
    dataset._data_error_report = None
    dataset.transforms = lambda value: value
    return dataset


@pytest.mark.parametrize("count", (1, 2))
def test_validation_retries_same_image_and_recovers(tmp_path, count):
    dataset = _retry_dataset(tmp_path, False, count)
    attempts = []

    def get(index):
        attempts.append(index)
        if len(attempts) < 3:
            raise OSError("暂时读取失败")
        return {"index": index}

    dataset.get_image_and_label = get
    dataset._replacement_index = lambda index: pytest.fail("验证不得替换图片")
    assert dataset[0] == {"index": 0}
    assert attempts == [0, 0, 0]


@pytest.mark.parametrize("retries", (0, 2))
def test_validation_exhaustion_reports_original_image(tmp_path, retries):
    dataset = _retry_dataset(tmp_path, False, retries=retries)
    attempts = []

    def fail(index):
        attempts.append(index)
        raise FileNotFoundError(dataset.im_files[index])

    dataset.get_image_and_label = fail
    dataset._replacement_index = lambda index: pytest.fail("验证不得替换图片")
    with pytest.raises(RuntimeError, match=f"已重试{retries}次") as raised:
        dataset[0]
    assert str(tmp_path / "0.png") in str(raised.value)
    assert attempts == [0] * (retries + 1)
    assert dataset._data_error_report.exists()


def test_training_still_replaces_bad_image(tmp_path):
    dataset = _retry_dataset(tmp_path, True)
    dataset._replacement_index = lambda index: 1

    def get(index):
        if index == 0:
            raise OSError("坏训练图片")
        return {"index": index}

    dataset.get_image_and_label = get
    assert dataset[0] == {"index": 1}


@pytest.mark.parametrize("builder", (build_yolo_dataset, build_rtdetr_dataset))
def test_real_validation_loader_rejects_missing_image(tmp_path, builder):
    data = detection_data(tmp_path, count=2)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 32, "data_retries": 1})
    dataset = builder(cfg, data["val"], 1, data, "val")
    Path(dataset.im_files[0]).unlink()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)
    with pytest.raises(RuntimeError, match="0.png"):
        list(loader)


def _failed_validation_worker(rank, world_size, rendezvous, dataset):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size,
                            timeout=timedelta(seconds=15))
    try:
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=[rank], collate_fn=dataset.collate_fn)
        list(loader)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_rank_validation_failure_exits(tmp_path):
    data = detection_data(tmp_path, count=2)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 32, "data_retries": 1})
    dataset = build_yolo_dataset(cfg, data["val"], 1, data, "val")
    Path(dataset.im_files[0]).unlink()
    with pytest.raises(torch.multiprocessing.ProcessRaisedException):
        run_gloo_workers(_failed_validation_worker, tmp_path, dataset)
