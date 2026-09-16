"""共享元数据重建后的节点缓存代次与并发读取。"""
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from PIL import Image

from ultralytics.cfg import get_cfg
from ultralytics.data import YOLODataset
from ultralytics.data import metadata


@pytest.mark.parametrize("policy", ("explicit", "auto"))
@pytest.mark.parametrize("first_verify", ("fast", "full"))
def test_full_verification_refreshes_staged_labels(tmp_path, monkeypatch, policy, first_verify):
    images, labels = tmp_path / "images/train", tmp_path / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    Image.new("RGB", (32, 32)).save(images / "one.png")
    label = labels / "one.txt"
    label.write_text("0 0.5 0.5 0.2 0.3\n")
    local = tmp_path / "local"
    monkeypatch.setenv("ULTRALYTICS_DATA_CACHE_DIR", str(local))
    monkeypatch.setattr(metadata, "is_remote_filesystem", lambda path: True)
    cache_policy = "auto" if policy == "auto" else str(local)
    cfg = get_cfg(overrides={"imgsz": 32})

    def build(verify):
        return YOLODataset(img_path=str(images), imgsz=32, batch_size=1, augment=False, hyp=cfg,
                           data={"names": {0: "甲", 1: "乙"}, "channels": 3},
                           metadata_cache=cache_policy, data_verify=verify)

    old = build(first_verify)
    label.write_text("1 0.5 0.5 0.2 0.3\n")
    new = build("full")
    assert new.labels[0]["cls"].item() == 1
    assert old.labels[0]["cls"].item() == 0
    assert old.labels.cache_dir != new.labels.cache_dir
    assert old.labels.manifest["generation"] != new.labels.manifest["generation"]
    assert build("fast").labels.cache_dir == new.labels.cache_dir
    with ThreadPoolExecutor(max_workers=4) as pool:
        refreshed = list(pool.map(lambda _: build("full"), range(4)))
    assert all(d.labels[0]["cls"].item() == 1 for d in refreshed)
    assert old.labels[0]["cls"].item() == 0


def test_legacy_stage_identity_tracks_fixed_file_metadata(tmp_path):
    """没有 generation 的旧缓存不复用内容已变化的本地副本。"""
    source = tmp_path / "shared" / "content"
    source.mkdir(parents=True)
    (source / "manifest.json").write_text(json.dumps({"version": metadata.METADATA_CACHE_VERSION}))
    records = source / "records.bin"
    records.write_bytes(b"old")
    first = metadata.stage_metadata_cache(source, str(tmp_path / "local"))
    records.write_bytes(b"updated records")
    second = metadata.stage_metadata_cache(source, str(tmp_path / "local"))
    assert first != second
    assert (first / "records.bin").read_bytes() == b"old"
    assert (second / "records.bin").read_bytes() == b"updated records"


def test_staging_uses_source_identity_and_atomic_generation(tmp_path):
    """不同源路径的同名目录不能互相污染；并发读者共享完整副本。"""
    sources = [tmp_path / name / "same" for name in ("a", "b")]
    for source, value in zip(sources, (b"a", b"b")):
        source.mkdir(parents=True)
        (source / "manifest.json").write_text(json.dumps({"generation": "same-id"}))
        (source / "records.bin").write_bytes(value)
    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(pool.map(lambda i: metadata.stage_metadata_cache(sources[i % 2], str(tmp_path / "local")), range(16)))
    assert len(set(paths)) == 2
    assert all((p / "records.bin").read_bytes() == (b"a" if i % 2 == 0 else b"b") for i, p in enumerate(paths))


def test_stage_permission_error_falls_back_to_shared(tmp_path, monkeypatch):
    source = tmp_path / "shared"
    source.mkdir()
    (source / "manifest.json").write_text('{"generation":"one"}')
    (source / "records.bin").write_bytes(b"records")

    def denied(*args, **kwargs):
        raise PermissionError("模拟无写权限")

    monkeypatch.setattr(metadata.shutil, "copytree", denied)
    assert metadata.stage_metadata_cache(source, str(tmp_path / "local")) == source
