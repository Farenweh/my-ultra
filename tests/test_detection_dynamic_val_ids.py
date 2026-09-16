from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from ultralytics.engine import val_runtime
from ultralytics.models.rtdetr.val import RTDETRValidator
from ultralytics.models.yolo.detect import val as detection_val
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops


class _Scheduler:
    """以不同领取顺序重现全局动态采样，保留真实 batch sampler。"""

    def __init__(self, batches):
        self.batches = iter(enumerate(batches))
        self.total_batches = len(batches)

    def claim(self):
        return next(self.batches, None)


def _evaluate_partition(validator_class, tmp_path, monkeypatch, size, batches=None, coco=False):
    context = None
    if batches is None:
        loader = torch.utils.data.DataLoader(list(range(size)), batch_size=2)
    else:
        context = SimpleNamespace(claimed_batches=[], claimed_indices=[])
        sampler = val_runtime.DynamicBatchSampler(_Scheduler(batches), context)
        loader = torch.utils.data.DataLoader(list(range(size)), batch_sampler=sampler)
    monkeypatch.setattr(val_runtime, "_ACTIVE_CONTEXT", context)
    validator = validator_class(
        dataloader=loader, save_dir=tmp_path, args={"task": "detect", "imgsz": 64, "plots": False}
    )
    validator.device = torch.device("cpu")
    validator.training = False
    validator.data = {"val": "自定义数据集", "nc": 1}
    if coco:
        validator.data.update(annotation_formats={"val": "coco_json"}, coco_category_ids=[7])
    validator.init_metrics(SimpleNamespace(names={0: "目标"}))
    if context is not None and not coco:
        assert validator.eval_ids is context.claimed_indices
        assert validator.eval_ids == []  # 初始化不得提前消费动态 sampler。
    for indices in loader:
        indices = indices.tolist()
        count = len(indices)
        boxes = torch.tensor([[0.2 + i * 0.1, 0.5, 0.1, 0.2] for i in indices])
        batch = {
            "img": torch.zeros(count, 3, 64, 64),
            "cls": torch.zeros(count, 1),
            "bboxes": boxes,
            "batch_idx": torch.arange(count),
            "ori_shape": [(64, 64)] * count,
            "ratio_pad": [((1.0, 1.0), (0, 0))] * count,
            "im_file": [f"{i}.jpg" for i in indices],
            "image_id": [100 + i for i in indices],
        }
        predictions = []
        for i, box in zip(indices, boxes):
            predictions.append(
                {
                    "bboxes": torch.empty(0, 4) if i == 1 else ops.xywh2xyxy(box[None]) * 64,
                    "conf": torch.empty(0) if i == 1 else torch.tensor([0.9]),
                    "cls": torch.empty(0) if i == 1 else torch.tensor([0.0]),
                }
            )
        validator.update_metrics(predictions, batch)
    return validator, context


@pytest.mark.parametrize("validator_class", (DetectionValidator, RTDETRValidator))
@pytest.mark.parametrize(
    "partitions", (((0, 1), (2, 3)), ((2, 3), (0, 1, 4)), ((0, 1, 4), (2, 3)))
)
def test_dynamic_val_global_ids_and_coco_metrics(validator_class, partitions, tmp_path, monkeypatch):
    """跨 rank 汇总保留所有图片和标注，尾批及空预测不改变 COCO 指标。"""
    pytest.importorskip("faster_coco_eval")
    size = sum(map(len, partitions))
    validators, contexts = [], []
    for rank, indices in enumerate(partitions):
        batches = [list(indices[i : i + 2]) for i in range(0, len(indices), 2)]
        validator, context = _evaluate_partition(
            validator_class, tmp_path / f"rank{rank}", monkeypatch, size, batches
        )
        validators.append(validator)
        contexts.append(context)
    payloads = iter(
        [
            [(c.claimed_indices, deepcopy(v.metrics.stats)) for c, v in zip(contexts, validators)],
            [(deepcopy(v.jdict), deepcopy(v.gdict), v.pred_counts) for v in validators],
        ]
    )

    def gather_object(value, output, dst):
        output[:] = next(payloads)

    merged = validators[0]
    monkeypatch.setattr(val_runtime, "_ACTIVE_CONTEXT", contexts[0])
    monkeypatch.setattr(detection_val, "RANK", 0)
    monkeypatch.setattr(detection_val.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(detection_val.dist, "gather_object", gather_object)
    monkeypatch.setattr(merged, "_gather_image_metrics", lambda metric: None)
    merged.gather_stats()
    assert sorted(image["id"] for image in merged.gdict["images"]) == list(range(size))
    assert len({annotation["id"] for annotation in merged.gdict["annotations"]}) == size
    assert len(merged.metrics.stats["target_cls"]) == size
    merged_stats = merged.eval_json({})
    reference, _ = _evaluate_partition(validator_class, tmp_path / "single", monkeypatch, size)
    reference_stats = reference.eval_json({})
    assert merged_stats["metrics/mAP50-95(B)"] > 0
    assert merged_stats == pytest.approx(reference_stats)


@pytest.mark.parametrize("validator_class", (DetectionValidator, RTDETRValidator))
def test_dynamic_val_keeps_native_coco_ids(validator_class, tmp_path, monkeypatch):
    """原生 COCO 标注使用其自带 ID，不参与自定义数据集的编号逻辑。"""
    validator, _ = _evaluate_partition(validator_class, tmp_path, monkeypatch, 4, [[2, 3]], coco=True)
    assert validator.eval_ids is None
    assert validator.gdict is None
    assert {p["image_id"] for p in validator.jdict} == {102, 103}
    assert {p["category_id"] for p in validator.jdict} == {7}
