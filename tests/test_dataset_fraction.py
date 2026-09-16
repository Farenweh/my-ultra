"""检测数据各缓存路径保持比例与整数样本数语义。"""
import pytest

from tests.detection_data_helpers import detection_data
from ultralytics.cfg import get_cfg
from ultralytics.data import dataset as dataset_module
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import get_split_fraction
from ultralytics.models.rtdetr.val import build_rtdetr_dataset


@pytest.mark.parametrize("builder", (build_yolo_dataset, build_rtdetr_dataset))
@pytest.mark.parametrize("annotation_format", ("txt", "coco"))
@pytest.mark.parametrize(("fraction", "expected"), ((2, 2), (100, 5), (0.01, 1), (0.4, 2), (1.0, 5)))
def test_fraction_cold_hot_and_full(tmp_path, builder, annotation_format, fraction, expected):
    data = detection_data(tmp_path, annotation_format)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 32, "fraction": fraction})
    for verify in ("fast", "fast", "full"):
        cfg.data_verify = verify
        dataset = builder(cfg, data["train"], 2, data, "train")
        assert len(dataset) == len(dataset.im_files) == expected
        assert dataset[0]["img"].numel() > 0


@pytest.mark.parametrize("annotation_format", ("txt", "coco"))
@pytest.mark.parametrize(("fraction", "expected"), ((2, 2), (0.01, 1)))
def test_fraction_without_writable_cache(tmp_path, monkeypatch, annotation_format, fraction, expected):
    data = detection_data(tmp_path, annotation_format)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 32, "fraction": fraction})

    def denied(*args, **kwargs):
        raise PermissionError("模拟缓存不可写")

    monkeypatch.setattr(dataset_module, "write_metadata_store", denied)
    dataset = build_yolo_dataset(cfg, data["train"], 2, data, "train")
    assert isinstance(dataset.labels, list)
    assert len(dataset) == len(dataset.im_files) == expected


def test_fraction_per_split_and_test_zero(tmp_path):
    data = detection_data(tmp_path)
    cfg = get_cfg(overrides={"task": "detect", "imgsz": 32, "fraction": [2, 0.01, 0]})
    assert len(build_yolo_dataset(cfg, data["train"], 2, data, "train")) == 2
    assert len(build_yolo_dataset(cfg, data["val"], 2, data, "val")) == 1
    assert get_split_fraction(cfg.fraction, "test") == 0.0
    for split in ("train", "val"):
        with pytest.raises(ValueError, match="at least one"):
            get_split_fraction([0, 0, 0], split)
