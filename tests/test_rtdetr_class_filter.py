"""指定类别的 RT-DETR 验证应与推理一致，并在类别过滤后截断。"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.models.rtdetr.val import RTDETRValidator


@pytest.mark.parametrize(("classes", "conf", "limit", "expected"), (
    (None, 0.1, 2, [1, 0]), ([0], 0.1, 1, [0]), ([0, 1], 0.1, 2, [1, 0]),
    ([9], 0.1, 1, []), ([], 0.1, 1, []), ([0], 0.8, 1, []),
))
@pytest.mark.parametrize("empty", (False, True))
def test_class_filter_matches_prediction(tmp_path, classes, conf, limit, expected, empty):
    validator = RTDETRValidator(save_dir=tmp_path, args={"imgsz": 64, "conf": conf, "classes": classes, "max_det": limit})
    raw = torch.tensor([[[.5, .5, .5, .5, .9, 1], [.5, .5, .5, .5, .8, 0]]])
    if empty:
        raw = raw[:, :0]
    original = raw.clone()
    actual = validator.postprocess(raw)[0]
    predictor = object.__new__(RTDETRPredictor)
    predictor.args = validator.args
    predictor.model = SimpleNamespace(names={0: "甲", 1: "乙"})
    predictor.batch = (["17.jpg"],)
    reference = predictor.postprocess(raw, torch.zeros(1, 3, 64, 64), [np.zeros((64, 64, 3), np.uint8)])[0]
    assert actual["cls"].tolist() == ([] if empty else expected)
    torch.testing.assert_close(actual["bboxes"], reference.boxes.xyxy)
    torch.testing.assert_close(actual["conf"], reference.boxes.conf)
    torch.testing.assert_close(raw, original)


def test_class_filter_preserves_metrics_and_exports(tmp_path):
    validator = RTDETRValidator(dataloader=torch.utils.data.DataLoader([0]), save_dir=tmp_path,
                               args={"imgsz": 64, "classes": [0], "max_det": 1,
                                     "save_json": True, "save_txt": True, "plots": False})
    validator.device = torch.device("cpu")
    validator.data = {"val": "自定义数据集", "nc": 2}
    validator.init_metrics(SimpleNamespace(names={0: "甲", 1: "乙"}))
    raw = torch.tensor([[[.5, .5, .5, .5, .9, 1], [.5, .5, .5, .5, .8, 0]]])
    predictions = validator.postprocess(raw)
    batch = {"img": torch.zeros(1, 3, 64, 64), "cls": torch.zeros(1, 1),
             "bboxes": torch.tensor([[.5, .5, .5, .5]]), "batch_idx": torch.zeros(1),
             "ori_shape": [(64, 64)], "ratio_pad": [((1., 1.), (0, 0))], "im_file": ["17.jpg"]}
    validator.update_metrics(predictions, batch)
    assert validator.metrics.stats["tp"][0].all()
    assert [prediction["category_id"] for prediction in validator.jdict] == [1]
    assert [float(x) for x in (tmp_path / "labels/17.txt").read_text().split()] == pytest.approx([0, .5, .5, .5, .5])
