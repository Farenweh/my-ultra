from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ultralytics.models.rtdetr.predict import RTDETRPredictor
from ultralytics.models.rtdetr.val import RTDETRValidator


@pytest.mark.parametrize("shape", ((320, 1280), (1280, 320), (320, 320), (640, 640)))
@pytest.mark.parametrize("save_txt,save_json", ((True, False), (False, True), (True, True)))
@pytest.mark.parametrize("empty", (False, True))
def test_rtdetr_validation_export_coordinates(tmp_path, shape, save_txt, save_json, empty):
    """TXT 和 JSON 共用原图坐标，横图、竖图和空预测均不修改指标输入。"""
    validator = RTDETRValidator(
        dataloader=torch.utils.data.DataLoader([0]),
        save_dir=tmp_path,
        args={"task": "detect", "imgsz": 640, "save_txt": save_txt, "save_json": save_json, "plots": False},
    )
    validator.data = {"val": "自定义数据集", "nc": 1}
    validator.device = torch.device("cpu")
    validator.init_metrics(SimpleNamespace(names={0: "目标"}))
    h, w = shape
    batch = {
        "img": torch.zeros(1, 3, 640, 640),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        "batch_idx": torch.zeros(1),
        "ori_shape": [shape],
        "ratio_pad": [((640 / h, 640 / w), (0, 0))],
        "im_file": ["17.jpg"],
    }
    prediction = {
        "bboxes": torch.empty(0, 4) if empty else torch.tensor([[160.0, 160.0, 480.0, 480.0]]),
        "conf": torch.empty(0) if empty else torch.tensor([0.8]),
        "cls": torch.empty(0) if empty else torch.tensor([0.0]),
    }
    original = {key: value.clone() for key, value in prediction.items()}
    validator.update_metrics([prediction], batch)
    torch.testing.assert_close(prediction, original)
    txt_file = tmp_path / "labels" / "17.txt"
    if save_txt and not empty:
        assert [float(x) for x in txt_file.read_text().split()] == pytest.approx([0, 0.5, 0.5, 0.5, 0.5])
    else:
        assert not txt_file.exists()
    if save_json and not empty:
        assert len(validator.jdict) == 1
        assert validator.jdict[0]["bbox"] == pytest.approx([w / 4, h / 4, w / 2, h / 2])
    else:
        assert validator.jdict == []
    scaled = validator.scale_preds(prediction, validator._prepare_batch(0, batch))
    predictor = object.__new__(RTDETRPredictor)
    predictor.args = SimpleNamespace(conf=0.1, classes=None, max_det=300)
    predictor.model = SimpleNamespace(names={0: "目标"})
    predictor.batch = (["17.jpg"],)
    raw = torch.empty(1, 0, 6) if empty else torch.tensor([[[0.5, 0.5, 0.5, 0.5, 0.8, 0.0]]])
    result = predictor.postprocess(raw, batch["img"], [np.zeros((h, w, 3), dtype=np.uint8)])[0]
    torch.testing.assert_close(scaled["bboxes"], result.boxes.xyxy)
