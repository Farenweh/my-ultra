"""RT-DETR 训练及验证应使用用户选择的数据校验、缓存和重试策略。"""
from pathlib import Path

import pytest

from tests.detection_data_helpers import detection_data
from ultralytics.cfg import get_cfg
from ultralytics.models.rtdetr.val import build_rtdetr_dataset


@pytest.mark.parametrize("mode", ("train", "val"))
@pytest.mark.parametrize("annotation_format", ("txt", "coco"))
def test_rtdetr_forwards_data_options(tmp_path, mode, annotation_format):
    data = detection_data(tmp_path, annotation_format)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 32, "data_verify": "full", "data_retries": 0,
                            "metadata_cache": "shared"})
    dataset = build_rtdetr_dataset(cfg, data[mode], 2, data, mode)
    assert (dataset.data_verify, dataset.data_retries, dataset.metadata_cache) == ("full", 0, "shared")
    Path(dataset.im_files[0]).unlink()
    with pytest.raises(RuntimeError, match="已重试0次"):
        dataset[0]


def test_rtdetr_full_check_refreshes_labels_and_local_stage(tmp_path):
    data = detection_data(tmp_path)
    local = tmp_path / "local"
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 32, "metadata_cache": str(local)})
    original = build_rtdetr_dataset(cfg, data["val"], 2, data, "val")
    assert original.labels.cache_dir.parent == local
    (tmp_path / "labels/train/0.txt").write_text("1 0.5 0.5 0.25 0.25\n")
    cfg.data_verify = "full"
    refreshed = build_rtdetr_dataset(cfg, data["val"], 2, data, "val")
    assert refreshed.labels[0]["cls"].item() == 1
    assert refreshed.labels.cache_dir != original.labels.cache_dir
