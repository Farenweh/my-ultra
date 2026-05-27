from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from ultralytics.nn.modules import Detect, OBB, Pose, Segment, YOLOEDetect
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel, load_checkpoint


def tiny_detect_cfg(*, legacy: bool) -> dict:
    """构造不依赖外部 YAML 文件的轻量 Detect 模型配置。"""
    cfg = {
        "nc": 3,
        "scale": "n",
        "backbone": [[-1, 1, "Conv", [16, 3, 2]], [-1, 1, "C3k2", [32, False]]],
        "head": [[[1], 1, "Detect", ["nc"]]],
    }
    if legacy:
        cfg["legacy_yolo_head"] = True
    return cfg


def assert_legacy_topology(head: Detect, expected: bool) -> None:
    """断言实例元数据与已构造的 cv3 拓扑一致。"""
    assert "legacy" in head.__dict__
    assert head.legacy is expected
    if expected:
        assert isinstance(head.cv3[0][0], Conv)
    else:
        assert isinstance(head.cv3[0][0], nn.Sequential)


def checkpoint_payload(model: DetectionModel) -> dict:
    """生成 load_checkpoint 支持的最小完整 checkpoint。"""
    return {"model": model, "train_args": {}}


def test_parse_model_keeps_legacy_state_per_instance():
    legacy_model = DetectionModel(tiny_detect_cfg(legacy=True), verbose=False, summary=False)
    legacy_head = legacy_model.model[-1]
    modern_model = DetectionModel(tiny_detect_cfg(legacy=False), verbose=False, summary=False)
    second_legacy_model = DetectionModel(tiny_detect_cfg(legacy=True), verbose=False, summary=False)

    assert_legacy_topology(legacy_head, True)
    assert_legacy_topology(modern_model.model[-1], False)
    assert_legacy_topology(second_legacy_model.model[-1], True)
    assert Detect.legacy is False
    assert sum(p.numel() for p in legacy_head.cv3.parameters()) > sum(
        p.numel() for p in modern_model.model[-1].cv3.parameters()
    )


@pytest.mark.parametrize(
    ("head_cls", "extra"),
    (
        (Detect, {}),
        (Segment, {}),
        (Pose, {"kpt_shape": (2, 3)}),
        (OBB, {}),
        (YOLOEDetect, {"embed": 64, "with_bn": True}),
    ),
)
def test_detect_subclasses_receive_instance_legacy(head_cls, extra):
    legacy_head = head_cls(nc=3, ch=(32, 64, 128), legacy=True, **extra)
    modern_head = head_cls(nc=3, ch=(32, 64, 128), legacy=False, **extra)

    assert_legacy_topology(legacy_head, True)
    assert_legacy_topology(modern_head, False)


def test_load_checkpoint_restores_missing_legacy_from_topology(tmp_path):
    model = DetectionModel(tiny_detect_cfg(legacy=True), verbose=False, summary=False).eval()
    head = model.model[-1]
    image = torch.rand(1, 3, 64, 64)
    expected = model(image)[0]
    del head.__dict__["legacy"]  # 模拟由旧类属性方案保存的 checkpoint
    weight = tmp_path / "legacy-old-format.pt"
    torch.save(checkpoint_payload(model), weight)

    loaded, _ = load_checkpoint(weight)

    assert_legacy_topology(loaded.model[-1], True)
    torch.testing.assert_close(loaded(image)[0], expected)


def test_load_checkpoint_preserves_new_instance_legacy(tmp_path):
    model = DetectionModel(tiny_detect_cfg(legacy=True), verbose=False, summary=False)
    weight = tmp_path / "legacy-new-format.pt"
    torch.save(checkpoint_payload(model), weight)

    loaded, _ = load_checkpoint(weight)

    assert_legacy_topology(loaded.model[-1], True)


def test_load_checkpoint_repairs_metadata_from_actual_topology(tmp_path):
    model = DetectionModel(tiny_detect_cfg(legacy=True), verbose=False, summary=False)
    model.model[-1].legacy = False
    weight = tmp_path / "legacy-mismatched-metadata.pt"
    torch.save(checkpoint_payload(model), weight)

    loaded, _ = load_checkpoint(weight)

    assert_legacy_topology(loaded.model[-1], True)


def test_embedded_cfg_rebuilds_without_external_yaml(tmp_path):
    cfg = tiny_detect_cfg(legacy=True)
    model = DetectionModel(deepcopy(cfg), verbose=False, summary=False)
    weight = tmp_path / "self-contained.pt"
    torch.save(checkpoint_payload(model), weight)

    loaded, _ = load_checkpoint(weight)
    rebuilt = DetectionModel(deepcopy(loaded.yaml), nc=2, verbose=False, summary=False)

    assert rebuilt.yaml["nc"] == 2
    assert_legacy_topology(rebuilt.model[-1], True)
    assert rebuilt.model[-1].cv3[0][-1].out_channels == 2


def test_class_head_transfer_keeps_cls_remap_behavior():
    source = DetectionModel(tiny_detect_cfg(legacy=True), nc=3, verbose=False, summary=False)
    source.names = {0: "cat", 1: "dog", 2: "car"}
    for branch in source.model[-1].cv3:
        branch[-1].bias.data.copy_(torch.tensor([10.0, 20.0, 30.0]))

    no_remap = DetectionModel(tiny_detect_cfg(legacy=True), nc=2, verbose=False, summary=False)
    initial_biases = [branch[-1].bias.detach().clone() for branch in no_remap.model[-1].cv3]
    no_remap.load(source, verbose=False)
    assert all(torch.equal(branch[-1].bias, initial) for branch, initial in zip(no_remap.model[-1].cv3, initial_biases))

    remapped = DetectionModel(tiny_detect_cfg(legacy=True), nc=2, verbose=False, summary=False)
    remapped.names = {0: "dog", 1: "cat"}
    remapped.load(source, verbose=False)
    assert all(branch[-1].bias.tolist() == [20.0, 10.0] for branch in remapped.model[-1].cv3)

    same_classes = DetectionModel(tiny_detect_cfg(legacy=True), nc=3, verbose=False, summary=False)
    same_classes.names = source.names
    same_classes.load(source, verbose=False)
    assert all(branch[-1].bias.tolist() == [10.0, 20.0, 30.0] for branch in same_classes.model[-1].cv3)
