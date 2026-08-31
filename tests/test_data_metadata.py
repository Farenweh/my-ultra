from __future__ import annotations

import builtins
import gc
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ultralytics.cfg import get_cfg
from ultralytics.data import YOLODataset, build_dataloader, build_yolo_dataset, verify_dataset
from ultralytics.data import base as data_base
from ultralytics.data import dataset as data_dataset
from ultralytics.data import metadata
from ultralytics.data import utils as data_utils
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG


def _label(image: Path, cls: int = 1) -> dict:
    """构造最小检测标签。"""
    return {
        "im_file": str(image),
        "shape": (20, 30),
        "cls": np.array([[cls]], dtype=np.float32),
        "bboxes": np.array([[0.5, 0.5, 0.2, 0.3]], dtype=np.float32),
        "segments": [],
        "keypoints": None,
        "normalized": True,
        "bbox_format": "xywh",
    }


def test_image_inventory_hot_path_does_not_walk(tmp_path, monkeypatch):
    """目录签名未变化时应直接读取图片清单。"""
    root = tmp_path / "images" / "train"
    root.mkdir(parents=True)
    Image.new("RGB", (30, 20)).save(root / "b.jpg")
    Image.new("RGB", (30, 20)).save(root / "a.jpg")

    files, content_id = metadata.load_or_create_image_inventory(root)
    monkeypatch.setattr(metadata.os, "walk", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("walk")))
    cached_files, cached_content_id = metadata.load_or_create_image_inventory(root)

    assert files == cached_files
    assert content_id == cached_content_id
    assert [Path(path).name for path in files] == ["a.jpg", "b.jpg"]


def test_image_inventory_invalidation_and_corruption_recovery(tmp_path, monkeypatch):
    """目录变化或清单损坏后应重新枚举并原子恢复。"""
    root = tmp_path / "images" / "train"
    root.mkdir(parents=True)
    Image.new("RGB", (30, 20)).save(root / "a.jpg")
    metadata.load_or_create_image_inventory(root)
    Image.new("RGB", (30, 20)).save(root / "b.jpg")
    original_walk = metadata.os.walk
    walks = 0

    def counted_walk(*args, **kwargs):
        nonlocal walks
        walks += 1
        return original_walk(*args, **kwargs)

    monkeypatch.setattr(metadata.os, "walk", counted_walk)
    files, _ = metadata.load_or_create_image_inventory(root)
    assert walks == 1
    assert [Path(path).name for path in files] == ["a.jpg", "b.jpg"]

    metadata.image_inventory_path(root).write_bytes(b"corrupt")
    files, _ = metadata.load_or_create_image_inventory(root)
    assert walks == 2
    assert len(files) == 2


def test_compact_metadata_round_trip_filter_reorder_and_summary(tmp_path):
    """紧凑缓存应保持标签字段，并支持过滤和重排。"""
    images = [tmp_path / f"{index}.jpg" for index in range(2)]
    labels = [_label(images[0], 0), _label(images[1], 1)]
    labels[0]["texts"] = [["zero"]]
    labels[1]["texts"] = [["one"]]
    signature = {"content_id": "fixture", "schema": metadata.METADATA_CACHE_VERSION}
    store_dir = metadata.write_metadata_store(
        tmp_path / "metadata", labels, num_samples=2, source_signature=signature
    )

    store = metadata.MMapLabelSequence(store_dir)
    filtered = store.with_filter([1], False)
    reordered = store.reordered(np.array([1, 0]))

    assert store[0]["im_file"] == str(images[0])
    assert store[0]["shape"] == (20, 30)
    assert filtered[0]["cls"].shape == (0, 1)
    assert filtered[1]["cls"].tolist() == [[1.0]]
    assert Path(reordered[0]["im_file"]).name == "1.jpg"
    assert store.summary["class_counts"] == [1, 1]
    assert store.summary["max_num_obj"] == 1
    assert store.summary["text_counts"] == {"zero": 1, "one": 1}


def test_rect_retry_keeps_replacement_in_same_shape_batch(monkeypatch):
    """rect模式替代样本必须保持相同batch形状。"""
    dataset = object.__new__(data_base.BaseDataset)
    dataset.rect = True
    dataset.batch = np.array([0, 0, 1, 1])
    dataset.labels = [{}, {}, {}, {}]
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("RANK", "0")

    assert dataset._replacement_index(0) == 1
    assert dataset._replacement_index(2) == 3


def test_yolo_legacy_cache_migrates_before_directory_scan(tmp_path, monkeypatch):
    """已有传统标签缓存时不得重新枚举图片目录。"""
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    image = image_dir / "one.jpg"
    Image.new("RGB", (30, 20)).save(image)
    legacy_path = label_dir / "train.cache"
    with open(legacy_path, "wb") as file:
        np.save(
            file,
            {
                "labels": [_label(image, 0)],
                "hash": "legacy",
                "results": (1, 0, 0, 0, 1),
                "msgs": [],
                "version": "1.0.4",
            },
        )
    monkeypatch.setattr(
        data_base,
        "load_or_create_image_inventory",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("不应扫描图片目录")),
    )
    cfg = get_cfg(DEFAULT_CFG, overrides={"imgsz": 32, "metadata_cache": "shared"})

    dataset = YOLODataset(
        img_path=str(image_dir),
        imgsz=32,
        batch_size=1,
        augment=False,
        hyp=cfg,
        data={"names": {0: "item"}, "channels": 3},
        metadata_cache="shared",
    )

    assert len(dataset) == 1
    assert isinstance(dataset.labels, metadata.MMapLabelSequence)
    assert dataset.labels[0]["cls"].tolist() == [[0.0]]


def test_yolo_text_image_list_hot_start_does_not_reread_list(tmp_path, monkeypatch):
    """文本图片清单首次构建紧凑缓存后，热启动不应重新读取大清单。"""
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image = image_dir / "one.jpg"
    Image.new("RGB", (30, 20)).save(image)
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    image_list = tmp_path / "train.txt"
    image_list.write_text(f"{image}\n", encoding="utf-8")
    cfg = get_cfg(DEFAULT_CFG, overrides={"imgsz": 32, "metadata_cache": "shared"})
    kwargs = {
        "img_path": str(image_list),
        "imgsz": 32,
        "batch_size": 1,
        "augment": False,
        "hyp": cfg,
        "data": {"names": {0: "item"}, "channels": 3},
        "metadata_cache": "shared",
    }
    assert len(YOLODataset(**kwargs)) == 1
    original_open = builtins.open

    def guarded_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)) and Path(file) == image_list:
            raise AssertionError("热启动不应重新读取图片清单")
        return original_open(file, *args, **kwargs)

    original_stat = Path.stat

    def guarded_stat(path: Path, *args, **kwargs):
        if path in {image, label_dir / "one.txt"}:
            raise AssertionError("热启动不应逐样本stat")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    assert len(YOLODataset(**kwargs)) == 1


def test_compact_cache_corruption_is_quarantined_and_rebuilt(tmp_path):
    """紧凑缓存截断后应保留损坏副本，并从传统缓存自动重建。"""
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image = image_dir / "one.jpg"
    Image.new("RGB", (30, 20)).save(image)
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    cfg = get_cfg(DEFAULT_CFG, overrides={"imgsz": 32, "metadata_cache": "shared"})
    kwargs = {
        "img_path": str(image_dir),
        "imgsz": 32,
        "batch_size": 1,
        "augment": False,
        "hyp": cfg,
        "data": {"names": {0: "item"}, "channels": 3},
        "metadata_cache": "shared",
    }
    first = YOLODataset(**kwargs)
    store_dir = first.labels.cache_dir
    (store_dir / "records.bin").write_bytes(b"corrupt")

    rebuilt = YOLODataset(**kwargs)

    assert rebuilt.labels[0]["cls"].tolist() == [[0.0]]
    assert list(store_dir.parent.glob(f"{store_dir.name}.corrupt-*"))


def test_legacy_cache_errors_restore_gc_and_failed_save_preserves_old_cache(tmp_path, monkeypatch):
    """传统缓存损坏不应关闭GC，原子发布失败也不得删除旧缓存。"""
    corrupt = tmp_path / "corrupt.cache"
    corrupt.write_bytes(b"corrupt")
    with pytest.raises(Exception):
        data_utils.load_dataset_cache_file(corrupt)
    assert gc.isenabled()

    existing = tmp_path / "labels.cache"
    existing.write_bytes(b"old-cache")
    monkeypatch.setattr(data_utils.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("发布失败")))
    data_utils.save_dataset_cache_file("test: ", existing, {"labels": []}, "1.0.4")
    assert existing.read_bytes() == b"old-cache"


def test_full_verify_refreshes_compact_cache_after_in_place_label_change(tmp_path):
    """完整校验应刷新目录指纹无法感知的标签原地修改。"""
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (30, 20)).save(image_dir / "one.jpg")
    label_file = label_dir / "one.txt"
    label_file.write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    cfg = get_cfg(DEFAULT_CFG, overrides={"imgsz": 32, "metadata_cache": "shared"})

    def build(data_verify: str) -> YOLODataset:
        return YOLODataset(
            img_path=str(image_dir),
            imgsz=32,
            batch_size=1,
            augment=False,
            hyp=cfg,
            data={"names": {0: "zero", 1: "one"}, "channels": 3},
            metadata_cache="shared",
            data_verify=data_verify,
        )

    assert build("fast").labels[0]["cls"].tolist() == [[0.0]]
    label_file.write_text("1 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    assert build("full").labels[0]["cls"].tolist() == [[1.0]]
    assert build("fast").labels[0]["cls"].tolist() == [[1.0]]


def test_cache_false_does_not_probe_npy(tmp_path, monkeypatch):
    """未启用磁盘图片缓存时不得检查同名NPY。"""
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image = image_dir / "one.jpg"
    Image.new("RGB", (30, 20)).save(image)
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    cfg = get_cfg(DEFAULT_CFG, overrides={"imgsz": 32, "metadata_cache": "shared"})
    dataset = YOLODataset(
        img_path=str(image_dir),
        imgsz=32,
        batch_size=1,
        augment=False,
        hyp=cfg,
        data={"names": {0: "item"}, "channels": 3},
        metadata_cache="shared",
    )
    original_exists = Path.exists

    def guarded_exists(path: Path) -> bool:
        if path.suffix == ".npy":
            raise AssertionError("cache=False不应查询NPY")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    assert dataset.load_image(0)[1] == (20, 30)


def test_verify_dataset_api_and_dataloader_prefetch_regression(tmp_path):
    """验证API可用，且DataLoader预取量保持为4。"""
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    Image.new("RGB", (30, 20)).save(image_dir / "one.jpg")
    (label_dir / "one.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        f"path: {tmp_path}\ntrain: images/train\nval: images/train\nnames:\n  0: item\n", encoding="utf-8"
    )

    result = verify_dataset(yaml_path, mode="fast", splits="train", metadata_cache="shared")
    loader = build_dataloader(range(16), batch=2, workers=1)
    try:
        assert result["train"]["images"] == 1
        assert loader.prefetch_factor == 4
        assert loader.iterator is not None  # InfiniteDataLoader继续持有并复用同一worker迭代器
    finally:
        loader.close()


def test_validation_workers_still_double(monkeypatch):
    """检测训练器的验证worker计算方式应保持为训练配置的两倍。"""
    from ultralytics.models.yolo.detect import train as detect_train

    trainer = object.__new__(detect_train.DetectionTrainer)
    trainer.args = SimpleNamespace(workers=3, compile=False)
    trainer.device = "cpu"
    dataset = SimpleNamespace(rect=False)
    trainer.build_dataset = lambda *_args, **_kwargs: dataset
    captured = {}

    def fake_build_dataloader(built_dataset, **kwargs):
        captured.update(kwargs)
        return built_dataset

    monkeypatch.setattr(detect_train, "build_dataloader", fake_build_dataloader)
    result = trainer.get_dataloader("unused", batch_size=2, rank=-1, mode="val")

    assert result is dataset
    assert captured["workers"] == 6


def test_node_local_stage_and_bad_sample_retry_report(tmp_path, monkeypatch):
    """节点缓存应可复用，坏样本应有限重试并输出报告。"""
    source = tmp_path / "shared" / "content"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text('{"version":"1.0.0"}', encoding="utf-8")
    (source / "records.bin").write_bytes(b"records")
    local_root = tmp_path / "local"
    staged = metadata.stage_metadata_cache(source, str(local_root))
    assert staged == local_root / "content"
    assert (staged / "records.bin").read_bytes() == b"records"

    dataset = object.__new__(data_base.BaseDataset)
    dataset.data_retries = 1
    dataset.labels = [{}, {}]
    dataset.im_files = ["missing.jpg", "valid.jpg"]
    dataset.prefix = "test: "
    dataset._data_error_report = None
    dataset.transforms = lambda value: value
    monkeypatch.setattr(dataset, "_replacement_index", lambda _index: 1)

    def get_label(index):
        if index == 0:
            raise FileNotFoundError("missing.jpg")
        return {"index": index}

    monkeypatch.setattr(dataset, "get_image_and_label", get_label)
    assert dataset[0] == {"index": 1}
    assert dataset._data_error_report.is_file()
    assert "missing.jpg" in dataset._data_error_report.read_text(encoding="utf-8")


def test_node_local_stage_is_concurrent_safe_and_falls_back_when_full(tmp_path, monkeypatch):
    """并发暂存应只形成一个完整目录，空间不足时应使用共享缓存。"""
    source = tmp_path / "shared" / "content"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text('{"version":"1.0.0"}', encoding="utf-8")
    (source / "records.bin").write_bytes(b"records")
    local_root = tmp_path / "local"
    with ThreadPoolExecutor(max_workers=4) as pool:
        staged = list(pool.map(lambda _: metadata.stage_metadata_cache(source, str(local_root)), range(4)))
    assert staged == [local_root / "content"] * 4
    assert (local_root / "content" / "records.bin").read_bytes() == b"records"

    full_root = tmp_path / "full"
    monkeypatch.setattr(metadata.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    assert metadata.stage_metadata_cache(source, str(full_root)) == source


def test_coco_fast_init_does_not_touch_image_files(tmp_path, monkeypatch):
    """COCO fast初始化只信任JSON中的逻辑路径和尺寸。"""
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image = image_dir / "one.jpg"
    Image.new("RGB", (30, 20)).save(image)
    annotation = tmp_path / "instances.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "one.jpg", "width": 30, "height": 20}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]}],
                "categories": [{"id": 1, "name": "item"}],
            }
        ),
        encoding="utf-8",
    )
    yaml_path = tmp_path / "data.yaml"
    yaml_path.write_text(
        f"path: {tmp_path}\ntrain: images\nval: images\nannotations:\n  train: instances.json\n  val: instances.json\n",
        encoding="utf-8",
    )
    data = check_det_dataset(yaml_path)
    cfg = get_cfg(DEFAULT_CFG, overrides={"task": "detect", "imgsz": 32, "metadata_cache": "shared"})
    original_exists = Path.exists

    def guarded_exists(path: Path) -> bool:
        if path == image:
            raise AssertionError("fast初始化不应检查图片是否存在")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)
    monkeypatch.setattr(data_dataset, "check_image", lambda *_args: pytest.fail("fast初始化不应打开图片"))
    dataset = build_yolo_dataset(cfg, data["train"], batch=1, data=data, mode="train")

    assert len(dataset) == 1


def test_coco_category_cache_avoids_reloading_json(tmp_path, monkeypatch):
    """COCO类别缓存命中后不应再次解析完整JSON。"""
    annotation = tmp_path / "instances.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1}],
                "annotations": [{"image_id": 1, "category_id": 2}],
                "categories": [{"id": 2, "name": "item"}],
            }
        ),
        encoding="utf-8",
    )
    assert data_utils._load_coco_json_categories(annotation) == [(2, "item")]
    monkeypatch.setattr(data_utils.json, "load", lambda *_args, **_kwargs: pytest.fail("不应重新解析JSON"))
    assert data_utils._load_coco_json_categories(annotation) == [(2, "item")]
