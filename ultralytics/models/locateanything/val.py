# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""LocateAnything专用的MS COCO单卡与分布式验证器。"""

from __future__ import annotations

import json
import os
import random
import re
import signal
import subprocess
import tempfile
from datetime import timedelta
from queue import Full, Queue
from threading import Thread
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.runtime import CallbackHost, initialize_distributed_runtime
from ultralytics.utils import LOGGER, TQDM, SimpleClass
from ultralytics.utils.dist import (
    build_torchrun_command,
    find_free_network_port,
    is_k8s_distributed_parent,
    normalize_k8s_launch_config,
)
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import get_torch_device_backend

from .val_preprocess import (
    LEGACY_PROTOCOL,
    LEGACY_PROTOCOL_ID,
    PAPER_PROTOCOL,
    PAPER_PROTOCOL_ID,
    PAPER_SHORT_SIDE,
    LocateAnythingValPreprocessor,
)

DEFAULT_DEVICES = "0"
_LABEL_TRAILING_PUNCTUATION = " \t\r\n.,;:!?，。；：！？"


class LocateMetrics(SimpleClass):
    """与Ultralytics其他validator一致的LocateAnything验证指标对象。"""

    def __init__(self, payload: dict[str, Any], save_dir: str | Path) -> None:
        self.save_dir = Path(save_dir)
        self.config = payload["config"]
        self.counts = payload["counts"]
        self.official = payload["official_locate_metrics"]
        self.coco_ap = payload["nonstandard_constant_score_coco_ap"]
        self.speed = payload["speed"]
        self.per_class = {threshold: values["per_class"] for threshold, values in self.official["f1"].items()}

    @classmethod
    def from_file(cls, path: str | Path) -> "LocateMetrics":
        """从rank 0写出的metrics.json恢复指标对象。"""
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), path.parent)

    @property
    def keys(self) -> list[str]:
        """返回results_dict的稳定指标名称。"""
        keys = [
            "metrics/precision50(B)",
            "metrics/recall50(B)",
            "metrics/F1-50(B)",
            "metrics/F1-95(B)",
        ]
        keys.append("metrics/F1-mean(B)" if "mean" in self.official["f1"] else "metrics/mean-IoU(B)")
        if "mean" in self.official["f1"]:
            keys.append("metrics/mean-GT-IoU(B)")
        return keys + ["metrics/nonstandard-mAP50(B)", "metrics/nonstandard-mAP50-95(B)", "fitness"]

    @property
    def fitness(self) -> float:
        """以论文定义的Mean F1作为无confidence验证适应度。"""
        f1 = self.official["f1"]
        if "mean" in f1:
            return float(f1["mean"]["macro"]["f1"])
        return 0.5 * (float(f1["0.50"]["micro"]["f1"]) + float(f1["0.95"]["micro"]["f1"]))

    @property
    def results_dict(self) -> dict[str, float]:
        """返回与DetMetrics用法一致的扁平指标字典。"""
        f1 = self.official["f1"]
        ap = self.coco_ap
        if "mean" in f1:
            values = [
                float(f1["0.50"]["macro"]["precision"]),
                float(f1["0.50"]["macro"]["recall"]),
                float(f1["0.50"]["macro"]["f1"]),
                float(f1["0.95"]["macro"]["f1"]),
                float(f1["mean"]["macro"]["f1"]),
                float(self.official["mean_gt_iou"]),
            ]
        else:
            values = [
                float(f1["0.50"]["micro"]["precision"]),
                float(f1["0.50"]["micro"]["recall"]),
                float(f1["0.50"]["micro"]["f1"]),
                float(f1["0.95"]["micro"]["f1"]),
                float(self.official["mean_gt_iou"]),
            ]
        values.extend((float(ap.get("AP50", 0.0)), float(ap.get("AP50_95", 0.0)), self.fitness))
        return dict(zip(self.keys, values))

    def summary(self, decimals: int = 5) -> list[dict[str, Any]]:
        """返回每个COCO类别的论文口径验证摘要。"""
        f1_50, f1_95 = self.per_class["0.50"], self.per_class["0.95"]
        f1_mean = self.per_class.get("mean")
        return [
            {
                "Class": name,
                "Precision50": round(float(values["precision"]), decimals),
                "Recall50": round(float(values["recall"]), decimals),
                "F1-50": round(float(values["f1"]), decimals),
                "F1-95": round(float(f1_95[name]["f1"]), decimals),
                **({"F1-mean": round(float(f1_mean[name]["f1"]), decimals)} if f1_mean else {}),
                "TP50": int(values["tp"]),
                "FP50": int(values["fp"]),
                "FN50": int(values["fn"]),
            }
            for name, values in f1_50.items()
        ]


def parse_devices(device_spec: str) -> list[int]:
    """解析并验证一个或多个互不重复的NPU编号。"""
    try:
        devices = [int(item.strip()) for item in device_spec.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"非法NPU列表：{device_spec!r}") from error
    if not devices or len(devices) != len(set(devices)):
        raise ValueError(f"devices必须包含至少一个互不重复的NPU编号，得到{devices}")
    if min(devices) < 0:
        raise ValueError(f"NPU编号不能为负数，得到{devices}")
    return devices


def shard_images(images: list[dict[str, Any]], rank: int, world_size: int) -> list[dict[str, Any]]:
    """按步长切分图片，避免DistributedSampler补齐造成重复。"""
    if not 0 <= rank < world_size:
        raise ValueError(f"rank={rank}不在[0,{world_size})范围内")
    return images[rank::world_size]


def batch_images(images: list[dict[str, Any]], batch: int) -> list[list[dict[str, Any]]]:
    """严格按用户batch分组，最后一组仅包含真实剩余样本。"""
    if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
        raise ValueError(f"batch必须是大于等于1的整数，得到{batch!r}")
    return [images[offset : offset + batch] for offset in range(0, len(images), batch)]


class _DynamicImageQueue:
    """基于共享输出目录的单节点跨rank原子任务队列。"""

    def __init__(self, output_dir: str | Path) -> None:
        dist_dir = Path(output_dir) / ".dist"
        self.tasks_path = dist_dir / "dynamic_tasks.json"
        self.state_path = dist_dir / "dynamic_state.txt"
        if not self.tasks_path.is_file() or not self.state_path.is_file():
            raise FileNotFoundError(f"动态任务队列尚未初始化：{dist_dir}")
        self.image_ids = [int(value) for value in json.loads(self.tasks_path.read_text(encoding="utf-8"))]

    @property
    def total(self) -> int:
        """返回本轮尚需处理的总图片数。"""
        return len(self.image_ids)

    def claim(self, count: int) -> list[int]:
        """原子领取至多count个连续image id，不引入HCCL同步。"""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"动态任务领取数必须是正整数，得到{count!r}")
        import fcntl

        with self.state_path.open("r+", encoding="utf-8") as state_file:
            fcntl.flock(state_file.fileno(), fcntl.LOCK_EX)
            try:
                value = state_file.read().strip()
                offset = int(value or 0)
                end = min(offset + count, len(self.image_ids))
                claimed = self.image_ids[offset:end]
                state_file.seek(0)
                state_file.write(str(end))
                state_file.truncate()
                state_file.flush()
                os.fsync(state_file.fileno())
            finally:
                fcntl.flock(state_file.fileno(), fcntl.LOCK_UN)
        return claimed


class _TCPDynamicImageQueue:
    """使用TCPStore在单节点或多节点worker间原子领取图片。"""

    def __init__(
        self,
        store: dist.Store,
        image_ids: list[int],
        namespace: str,
        rank: int,
        world_size: int,
        initial_count: int,
    ) -> None:
        self.store = store
        self.image_ids = image_ids
        self.key = f"{namespace}/next_image"
        initial_start = rank * initial_count
        initial_end = min(initial_start + initial_count, len(image_ids))
        self.initial_image_ids = image_ids[initial_start:initial_end]
        if rank == 0:
            self.store.set(self.key, str(min(world_size * initial_count, len(image_ids))))
        dist.barrier()

    @property
    def total(self) -> int:
        """返回需要处理的图片总数。"""
        return len(self.image_ids)

    def claim(self, count: int) -> list[int]:
        """原子领取至多count个连续image id。"""
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError(f"动态任务领取数必须是正整数，得到{count!r}")
        claimed = self.initial_image_ids[:count]
        del self.initial_image_ids[:count]
        remaining = count - len(claimed)
        if not remaining:
            return claimed
        end = int(self.store.add(self.key, remaining))
        start = end - remaining
        claimed.extend(self.image_ids[start : min(end, len(self.image_ids))])
        return claimed


def _remaining_dynamic_image_ids(
    output_dir: str | Path,
    images: list[dict[str, Any]],
    world_size: int,
    *,
    resume: bool,
) -> list[int]:
    """根据所有rank已落盘记录返回待处理image id。"""
    completed: set[int] = set()
    if resume:
        output_dir = Path(output_dir)
        for rank in range(world_size):
            completed.update(
                image_id
                for image_id, record in read_jsonl_records(output_dir / f"predictions.rank{rank}.jsonl").items()
                if not record.get("error")
            )
    return [int(image["id"]) for image in images if int(image["id"]) not in completed]


def _initialize_dynamic_queue(
    output_dir: str | Path,
    images: list[dict[str, Any]],
    world_size: int,
    *,
    resume: bool,
) -> list[int]:
    """根据全部rank已成功落盘的记录重建动态队列。"""
    output_dir = Path(output_dir)
    image_ids = _remaining_dynamic_image_ids(output_dir, images, world_size, resume=resume)
    dist_dir = output_dir / ".dist"
    dist_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = dist_dir / "dynamic_tasks.json"
    temporary_path = tasks_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(image_ids), encoding="utf-8")
    temporary_path.replace(tasks_path)
    (dist_dir / "dynamic_state.txt").write_text("0", encoding="utf-8")
    return image_ids


def normalize_label(label: str) -> str:
    """仅规范空白、大小写和末尾标点，不进行模糊类别猜测。"""
    return re.sub(r"\s+", " ", str(label)).strip().rstrip(_LABEL_TRAILING_PUNCTUATION).casefold()


def xyxy_to_xywh(xyxy: list[float] | tuple[float, ...]) -> list[float]:
    """将绝对像素xyxy转换为COCO xywh。"""
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    return [x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0)]


def coco_xywh_to_xyxy(xywh: list[float] | tuple[float, ...]) -> list[float]:
    """将COCO xywh转换为绝对像素xyxy。"""
    x, y, width, height = (float(value) for value in xywh)
    return [x, y, x + max(width, 0.0), y + max(height, 0.0)]


def bbox_iou(first: list[float], second: list[float]) -> float:
    """计算两个xyxy框的IoU。"""
    intersection_width = max(min(first[2], second[2]) - max(first[0], second[0]), 0.0)
    intersection_height = max(min(first[3], second[3]) - max(first[1], second[1]), 0.0)
    intersection = intersection_width * intersection_height
    first_area = max(first[2] - first[0], 0.0) * max(first[3] - first[1], 0.0)
    second_area = max(second[2] - second[0], 0.0) * max(second[3] - second[1], 0.0)
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def bbox_ioa(first: list[float], second: list[float]) -> float:
    """计算first框被second crowd框覆盖的比例。"""
    intersection_width = max(min(first[2], second[2]) - max(first[0], second[0]), 0.0)
    intersection_height = max(min(first[3], second[3]) - max(first[1], second[1]), 0.0)
    intersection = intersection_width * intersection_height
    first_area = max(first[2] - first[0], 0.0) * max(first[3] - first[1], 0.0)
    return intersection / first_area if first_area > 0 else 0.0


def _resolve_path(root: Path, value: str | Path) -> Path:
    """相对数据集根目录解析路径。"""
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def load_coco_validation(
    data: str | Path,
    *,
    allow_download: bool = False,
    max_images: int = 0,
) -> dict[str, Any]:
    """读取COCO val2017图片、标注和类别，并禁止隐式下载。"""
    dataset = check_det_dataset(str(data), autodownload=allow_download)
    root = Path(dataset["path"])
    annotation_value = (dataset.get("annotations") or {}).get("val")
    annotation_path = Path(annotation_value) if annotation_value else root / "annotations" / "instances_val2017.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(f"未找到COCO val2017标注：{annotation_path}。默认不会自动下载数据。")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    required = {"images", "annotations", "categories"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError(f"COCO标注缺少字段：{sorted(required - set(payload or {}))}")

    categories = sorted(
        ({"id": int(item["id"]), "name": str(item["name"])} for item in payload["categories"]),
        key=lambda item: item["id"],
    )
    normalized_names = [normalize_label(item["name"]) for item in categories]
    if len(categories) != 80 or len(set(normalized_names)) != len(categories):
        raise ValueError(f"预期MS COCO 80类，标注中得到{len(categories)}类")

    val_source = dataset["val"]
    if isinstance(val_source, (list, tuple)):
        sources = [Path(item) for item in val_source]
    else:
        sources = [Path(val_source)]
    image_by_name: dict[str, Path] = {}
    for source in sources:
        if source.is_file() and source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    image_path = _resolve_path(root, line.strip())
                    image_by_name[image_path.name] = image_path
        elif source.is_dir():
            image_by_name.update({path.name: path for path in source.iterdir() if path.is_file()})

    images = []
    for item in sorted(payload["images"], key=lambda value: int(value["id"])):
        file_name = str(item["file_name"])
        image_path = image_by_name.get(Path(file_name).name, root / "images" / "val2017" / file_name)
        if not image_path.is_file():
            raise FileNotFoundError(f"COCO图片不存在：{image_path}")
        images.append(
            {
                "id": int(item["id"]),
                "file_name": file_name,
                "path": str(image_path),
                "height": int(item["height"]),
                "width": int(item["width"]),
            }
        )
    if len(images) != 5000:
        raise ValueError(f"预期COCO val2017包含5000张图片，得到{len(images)}张")
    if max_images:
        images = images[:max_images]
    selected_image_ids = {int(image["id"]) for image in images}
    evaluation_image_ids = [int(item["id"]) for item in payload["images"] if int(item["id"]) in selected_image_ids]
    return {
        "annotation_path": str(annotation_path),
        "images": images,
        "evaluation_image_ids": evaluation_image_ids,
        "annotations": payload["annotations"],
        "categories": categories,
    }


def result_to_record(
    result: Any,
    image: dict[str, Any],
    category_by_name: dict[str, dict[str, Any]],
    preprocessor: LocateAnythingValPreprocessor | None = None,
) -> dict[str, Any]:
    """将一个LocateAnythingResult转换为可恢复的rank JSONL记录。"""
    predictions = []
    unknown_labels = []
    for box in result.boxes:
        normalized = normalize_label(box.label)
        category = category_by_name.get(normalized)
        if category is None:
            unknown_labels.append(str(box.label))
            continue
        xyxy = [float(value) for value in box.xyxy]
        if preprocessor is not None:
            xyxy = preprocessor.box_to_original(xyxy, image)
        predictions.append(
            {
                "image_id": int(image["id"]),
                "category_id": int(category["id"]),
                "category_name": str(category["name"]),
                "bbox": xyxy_to_xywh(xyxy),
                "xyxy": xyxy,
            }
        )
    stats = result.stats if isinstance(getattr(result, "stats", None), dict) else {}
    return {
        "image_id": int(image["id"]),
        "file_name": str(image["file_name"]),
        "raw_output": str(result.raw_output),
        "parse_warnings": list(result.parse_warnings),
        "unknown_labels": unknown_labels,
        "predictions": predictions,
        "speed": {key: float(value) for key, value in result.speed.items()},
        "output_tokens": int(stats.get("output_tokens", 0)),
        "batch_id": int(stats.get("batch_id", -1)),
        "batch_size": int(stats.get("batch_size", 1)),
        "batch_generation_seconds": float(
            stats.get("batch_generation_seconds", float(result.speed.get("inference", 0.0)) / 1000)
        ),
        "batch_output_tokens": int(stats.get("batch_output_tokens", stats.get("output_tokens", 0))),
        "generation_stats": stats,
        "validation_preprocess": dict(image.get("validation_preprocess") or {}),
        "error": None,
    }


def read_jsonl_records(path: str | Path) -> dict[int, dict[str, Any]]:
    """读取rank JSONL，忽略崩溃留下的尾部坏行，并让后写记录覆盖旧记录。"""
    records: dict[int, dict[str, Any]] = {}
    path = Path(path)
    if not path.is_file():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            records[int(record["image_id"])] = record
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            LOGGER.warning(f"忽略{path}第{line_number}行的不完整记录")
    return records


def validate_resume_protocol(output_dir: str | Path, world_size: int, protocol: str) -> None:
    """防止resume把论文预处理结果与旧协议分片混合。"""
    if protocol == LEGACY_PROTOCOL:
        return
    expected = PAPER_PROTOCOL_ID
    for rank in range(world_size):
        path = Path(output_dir) / f"predictions.rank{rank}.jsonl"
        for image_id, record in read_jsonl_records(path).items():
            if record.get("error"):
                continue
            actual = (record.get("validation_preprocess") or {}).get("protocol_id")
            if actual != expected:
                raise RuntimeError(
                    f"resume分片{path}中image_id={image_id}的验证协议为{actual or '旧版/未记录'}，"
                    f"当前要求{expected}。请使用新output_dir重新验证，不要混用旧分片。"
                )


def merge_prediction_shards(output_dir: str | Path, world_size: int) -> list[dict[str, Any]]:
    """合并所有rank分片并按image_id去重排序。"""
    merged: dict[int, dict[str, Any]] = {}
    for rank in range(world_size):
        for image_id, record in read_jsonl_records(Path(output_dir) / f"predictions.rank{rank}.jsonl").items():
            previous = merged.get(image_id)
            if previous is None or previous.get("error") or not record.get("error"):
                merged[image_id] = record
    return [merged[image_id] for image_id in sorted(merged)]


def build_constant_score_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造faster-coco-eval所需的非标准常数分数预测。"""
    predictions = []
    for record in records:
        for prediction in record.get("predictions", []):
            predictions.append(
                {
                    "image_id": int(prediction["image_id"]),
                    "category_id": int(prediction["category_id"]),
                    "bbox": [float(value) for value in prediction["bbox"]],
                    "score": 1.0,
                }
            )
    return predictions


def _greedy_matches(predictions: list[list[float]], targets: list[list[float]], threshold: float) -> list[float]:
    """按IoU降序执行确定性一对一匹配。"""
    candidates = sorted(
        (
            (bbox_iou(prediction, target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predictions)
            for target_index, target in enumerate(targets)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_predictions, used_targets, matches = set(), set(), []
    for iou, prediction_index, target_index in candidates:
        if iou < threshold:
            break
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
        matches.append(iou)
    return matches


def _greedy_match_indices(
    predictions: list[list[float]], targets: list[list[float]], threshold: float
) -> tuple[set[int], set[int], list[float]]:
    """返回一对一匹配使用的预测、GT索引及IoU。"""
    candidates = sorted(
        (
            (bbox_iou(prediction, target), prediction_index, target_index)
            for prediction_index, prediction in enumerate(predictions)
            for target_index, target in enumerate(targets)
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    used_predictions, used_targets, ious = set(), set(), []
    for iou, prediction_index, target_index in candidates:
        if iou < threshold:
            break
        if prediction_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(prediction_index)
        used_targets.add(target_index)
        ious.append(iou)
    return used_predictions, used_targets, ious


def _ratios(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    """由TP/FP/FN计算precision、recall和F1。"""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def _f1_from_precision_recall(precision: float, recall: float) -> float:
    """按论文评估脚本对聚合后的precision和recall取调和平均。"""
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _safe_positive_mean(values: list[float]) -> float:
    """复现官方FastEval包装层忽略非正数的safe_mean。"""
    positive = [float(value) for value in values if value > 0]
    return float(np.mean(positive)) if positive else 0.0


def _paper_match_counts(
    predictions: list[list[float]],
    targets: list[list[float]],
    crowds: list[list[float]],
    threshold: float,
    *,
    max_detections: int = 100,
) -> tuple[dict[str, int], list[float]]:
    """按COCO/FastEval的生成顺序执行普通GT一对一与crowd忽略匹配。"""
    counts, matched_ious, _ = _paper_match_sequence(
        predictions,
        targets,
        crowds,
        threshold,
        max_detections=max_detections,
    )
    return counts, matched_ious


def _paper_match_sequence(
    predictions: list[list[float]],
    targets: list[list[float]],
    crowds: list[list[float]],
    threshold: float,
    *,
    max_detections: int = 100,
) -> tuple[dict[str, int], list[float], list[int | None]]:
    """返回FastEval需要的生成顺序TP/FP序列；None表示crowd忽略项。"""
    used_targets: set[int] = set()
    matched_ious: list[float] = []
    outcomes: list[int | None] = []
    true_positives = false_positives = 0
    for prediction in predictions[:max_detections]:
        best: tuple[bool, int] | None = None
        best_overlap = float(threshold)
        for target_index, target in enumerate(targets):
            if target_index in used_targets:
                continue
            overlap = bbox_iou(prediction, target)
            if overlap < best_overlap:
                continue
            best_overlap = overlap
            best = (False, target_index)
        if best is None:
            for crowd_index, crowd in enumerate(crowds):
                overlap = bbox_ioa(prediction, crowd)
                if overlap < best_overlap:
                    continue
                best_overlap = overlap
                best = (True, crowd_index)
        if best is None:
            false_positives += 1
            outcomes.append(0)
        elif not best[0]:
            used_targets.add(best[1])
            matched_ious.append(best_overlap)
            true_positives += 1
            outcomes.append(1)
        else:
            outcomes.append(None)
    return (
        {"tp": true_positives, "fp": false_positives, "fn": len(targets) - len(used_targets)},
        matched_ious,
        outcomes,
    )


def _fasteval_precision(outcomes: list[int | None], total_gt: int) -> float:
    """复现官方FastEval从COCO插值PR曲线取出的单类precision。"""
    if not outcomes or total_gt <= 0:
        return 0.0
    true_positives = false_positives = 0
    recalls: list[float] = []
    precisions: list[float] = []
    for outcome in outcomes:
        true_positives += outcome == 1
        false_positives += outcome == 0
        recalls.append(true_positives / total_gt)
        precisions.append(
            true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
        )
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    sampled = []
    for recall_threshold in np.arange(0.0, 1.01, 0.01):
        precision = 0.0
        for index, recall in enumerate(recalls):
            if recall > recall_threshold:
                precision = precisions[index]
                break
        sampled.append(precision)
    positive = [value for value in sampled if value > 0]
    return positive[-1] if positive else 0.0


def compute_locate_metrics(
    records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    image_ids: set[int],
    image_order: list[int] | None = None,
) -> dict[str, Any]:
    """按LocateAnything论文的COCO/FastEval口径计算P/R/F1。"""
    category_name = {int(item["id"]): str(item["name"]) for item in categories}
    positive_categories: dict[int, set[int]] = defaultdict(set)
    targets: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    crowds: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in image_ids:
            continue
        category_id = int(annotation["category_id"])
        positive_categories[image_id].add(category_id)
        key = (image_id, category_id)
        destination = crowds if annotation.get("iscrowd", 0) else targets
        destination[key].append(coco_xywh_to_xyxy(annotation["bbox"]))

    predictions: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    dropped_predictions = 0
    for record in records:
        image_id = int(record["image_id"])
        for prediction in record.get("predictions", []):
            category_id = int(prediction["category_id"])
            if image_id in positive_categories and category_id not in positive_categories[image_id]:
                dropped_predictions += 1
                continue
            predictions[(image_id, category_id)].append([float(value) for value in prediction["xyxy"]])

    ordered_image_ids = [int(image_id) for image_id in (image_order or sorted(image_ids)) if image_id in image_ids]
    populated_keys = set(targets) | set(crowds) | set(predictions)
    category_image_ids = {
        category_id: [image_id for image_id in ordered_image_ids if (image_id, category_id) in populated_keys]
        for category_id in category_name
    }
    thresholds: dict[str, Any] = {}
    threshold_values = [round(float(value), 2) for value in np.arange(0.5, 0.951, 0.05)]
    for threshold in threshold_values:
        per_category = {category_id: {"tp": 0, "fp": 0, "fn": 0} for category_id in category_name}
        per_category_outcomes: dict[int, list[int | None]] = defaultdict(list)
        for category_id in category_name:
            for image_id in category_image_ids[category_id]:
                matched, _, outcomes = _paper_match_sequence(
                    predictions[(image_id, category_id)],
                    targets[(image_id, category_id)],
                    crowds[(image_id, category_id)],
                    threshold,
                )
                counts = per_category[category_id]
                for name in ("tp", "fp", "fn"):
                    counts[name] += matched[name]
                per_category_outcomes[category_id].extend(outcomes)

        per_class = {}
        for category_id, counts in per_category.items():
            values = _ratios(**counts)
            values["count_precision"] = values["precision"]
            values["precision"] = _fasteval_precision(
                per_category_outcomes[category_id],
                int(counts["tp"] + counts["fn"]),
            )
            values["f1"] = _f1_from_precision_recall(float(values["precision"]), float(values["recall"]))
            per_class[category_name[category_id]] = values
        totals = {key: sum(int(metrics[key]) for metrics in per_class.values()) for key in ("tp", "fp", "fn")}
        macro_precision = _safe_positive_mean([float(metrics["precision"]) for metrics in per_class.values()])
        macro_recall = _safe_positive_mean([float(metrics["recall"]) for metrics in per_class.values()])
        macro = {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1": _f1_from_precision_recall(macro_precision, macro_recall),
        }
        thresholds[f"{threshold:.2f}"] = {"micro": _ratios(**totals), "macro": macro, "per_class": per_class}

    per_class_mean = {}
    for class_name in category_name.values():
        precision = float(
            np.mean([thresholds[f"{value:.2f}"]["per_class"][class_name]["precision"] for value in threshold_values])
        )
        recall = float(
            np.mean([thresholds[f"{value:.2f}"]["per_class"][class_name]["recall"] for value in threshold_values])
        )
        per_class_mean[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": _f1_from_precision_recall(precision, recall),
        }
    mean_precision = _safe_positive_mean([metrics["precision"] for metrics in per_class_mean.values()])
    mean_recall = _safe_positive_mean([metrics["recall"] for metrics in per_class_mean.values()])
    thresholds["mean"] = {
        "macro": {
            "precision": mean_precision,
            "recall": mean_recall,
            "f1": _f1_from_precision_recall(mean_precision, mean_recall),
        },
        "per_class": per_class_mean,
    }

    total_gt, matched_iou_sum = 0, 0.0
    for key, target_boxes in targets.items():
        prediction_boxes = predictions[key]
        total_gt += len(target_boxes)
        matched_iou_sum += sum(_greedy_matches(prediction_boxes, target_boxes, threshold=0.0))
    return {
        "f1": thresholds,
        "mean_gt_iou": matched_iou_sum / total_gt if total_gt else 0.0,
        "evaluated_non_crowd_gt": total_gt,
        "protocol": "coco_fasteval_paper",
        "iou_thresholds": threshold_values,
        "aggregation": "per_category_precision_recall_safe_mean_then_harmonic_f1",
        "positive_only": True,
        "max_detections_per_image_category": 100,
        "positive_only_dropped_predictions": dropped_predictions,
    }


def compute_legacy_locate_metrics(
    records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    image_ids: set[int],
) -> dict[str, Any]:
    """保留原验证器的IoU降序匹配、micro/macro F1和mean GT IoU。"""
    category_name = {int(item["id"]): str(item["name"]) for item in categories}
    targets: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    crowds: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        if image_id not in image_ids:
            continue
        key = (image_id, int(annotation["category_id"]))
        (crowds if annotation.get("iscrowd", 0) else targets)[key].append(coco_xywh_to_xyxy(annotation["bbox"]))
    predictions: dict[tuple[int, int], list[list[float]]] = defaultdict(list)
    for record in records:
        for prediction in record.get("predictions", []):
            predictions[(int(record["image_id"]), int(prediction["category_id"]))].append(
                [float(value) for value in prediction["xyxy"]]
            )

    keys = set(targets) | set(crowds) | set(predictions)
    thresholds: dict[str, Any] = {}
    for threshold in (0.5, 0.95):
        per_category = {category_id: {"tp": 0, "fp": 0, "fn": 0} for category_id in category_name}
        for image_id, category_id in keys:
            prediction_boxes = predictions[(image_id, category_id)]
            target_boxes = targets[(image_id, category_id)]
            crowd_boxes = crowds[(image_id, category_id)]
            used_predictions, used_targets, _ = _greedy_match_indices(prediction_boxes, target_boxes, threshold)
            ignored_predictions = {
                index
                for index, prediction in enumerate(prediction_boxes)
                if index not in used_predictions
                and any(bbox_ioa(prediction, crowd) >= threshold for crowd in crowd_boxes)
            }
            counts = per_category[category_id]
            counts["tp"] += len(used_targets)
            counts["fp"] += len(prediction_boxes) - len(used_predictions) - len(ignored_predictions)
            counts["fn"] += len(target_boxes) - len(used_targets)
        per_class = {category_name[category_id]: _ratios(**counts) for category_id, counts in per_category.items()}
        totals = {name: sum(int(metrics[name]) for metrics in per_class.values()) for name in ("tp", "fp", "fn")}
        macro = {
            name: float(np.mean([float(metrics[name]) for metrics in per_class.values()]))
            for name in ("precision", "recall", "f1")
        }
        thresholds[f"{threshold:.2f}"] = {"micro": _ratios(**totals), "macro": macro, "per_class": per_class}

    total_gt = sum(len(boxes) for boxes in targets.values())
    matched_iou_sum = sum(
        sum(_greedy_matches(predictions[key], target_boxes, threshold=0.0)) for key, target_boxes in targets.items()
    )
    return {
        "f1": thresholds,
        "mean_gt_iou": matched_iou_sum / total_gt if total_gt else 0.0,
        "evaluated_non_crowd_gt": total_gt,
        "protocol": "legacy_class_aware_iou_greedy",
    }


def run_nonstandard_coco_ap(
    annotation_path: str | Path,
    prediction_path: str | Path,
    image_ids: list[int],
) -> dict[str, Any]:
    """以固定score=1.0运行COCO AP，并显式标记其非标准性质。"""
    warning = "LocateAnything不输出confidence；本节为所有框固定score=1.0，COCO AP不具备标准排序意义。"
    predictions = json.loads(Path(prediction_path).read_text(encoding="utf-8"))
    if not predictions:
        return {"status": "empty_predictions", "score_policy": "constant_1.0", "warning": warning}
    try:
        from faster_coco_eval import COCO, COCOeval_faster

        ground_truth = COCO(str(annotation_path))
        prediction_set = ground_truth.loadRes(str(prediction_path))
        evaluator = COCOeval_faster(ground_truth, prediction_set, iouType="bbox", print_function=LOGGER.info)
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        values = evaluator.stats_as_dict
        return {
            "status": "ok_nonstandard_constant_score",
            "score_policy": "constant_1.0",
            "warning": warning,
            "AP50_95": float(values["AP_all"]),
            "AP50": float(values["AP_50"]),
            "AP75": float(values["AP_75"]),
            "AP_small": float(values["AP_small"]),
            "AP_medium": float(values["AP_medium"]),
            "AP_large": float(values["AP_large"]),
        }
    except Exception as error:
        return {
            "status": "evaluation_failed",
            "score_policy": "constant_1.0",
            "warning": warning,
            "error": f"{type(error).__name__}: {error}",
        }


def _seed_for_image(seed: int, image_id: int) -> None:
    """按image_id设置可复现的采样随机状态。"""
    value = int(seed + image_id)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.manual_seed_all(value)


def _error_record(image: dict[str, Any], error: Exception) -> dict[str, Any]:
    """将单图推理错误保存为可汇总记录。"""
    return {
        "image_id": int(image["id"]),
        "file_name": str(image["file_name"]),
        "raw_output": "",
        "parse_warnings": [],
        "unknown_labels": [],
        "predictions": [],
        "speed": {},
        "output_tokens": 0,
        "batch_id": -1,
        "batch_size": 1,
        "batch_generation_seconds": 0.0,
        "batch_output_tokens": 0,
        "generation_stats": {},
        "validation_preprocess": dict(image.get("validation_preprocess") or {}),
        "error": f"{type(error).__name__}: {error}",
    }


def _write_record(file, record: dict[str, Any]) -> None:
    """追加并立即flush一条JSONL记录。"""
    file.write(json.dumps(record, ensure_ascii=False) + "\n")
    file.flush()


class _RecordSink:
    """可选后台JSONL写入器；正常退出时保证所有记录均已flush。"""

    _END = object()

    def __init__(self, path: Path, mode: str, *, asynchronous: bool) -> None:
        self.path = path
        self.mode = mode
        self.asynchronous = asynchronous
        self.file = None
        self.queue: Queue | None = Queue(maxsize=256) if asynchronous else None
        self.error: BaseException | None = None
        self.thread: Thread | None = None

    def __enter__(self):
        self.file = self.path.open(self.mode, encoding="utf-8")
        if self.asynchronous:
            self.thread = Thread(target=self._run, name="locate-jsonl-writer", daemon=True)
            self.thread.start()
        return self

    def _run(self) -> None:
        while True:
            record = self.queue.get()
            try:
                if record is self._END:
                    return
                if self.error is None:
                    try:
                        _write_record(self.file, record)
                    except BaseException as error:
                        self.error = error
            finally:
                self.queue.task_done()

    def write(self, record: dict[str, Any]) -> None:
        if self.error is not None:
            raise RuntimeError("LocateAnything后台JSONL写入失败") from self.error
        if not self.asynchronous:
            _write_record(self.file, record)
            return
        while True:
            try:
                self.queue.put(record, timeout=0.1)
                return
            except Full:
                if self.error is not None:
                    raise RuntimeError("LocateAnything后台JSONL写入失败") from self.error

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.asynchronous and self.thread is not None:
            self.queue.put(self._END)
            self.queue.join()
            self.thread.join()
        self.file.close()
        if self.error is not None and exc_value is None:
            raise RuntimeError("LocateAnything后台JSONL写入失败") from self.error


def _resolve_worker_output(args: Any) -> Path:
    """返回父进程已解析并写入分布式配置的输出目录。"""
    path = Path(args.output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _validate_npu_count(devices: list[int]) -> None:
    """验证当前进程可见的NPU足以覆盖设备列表。"""
    try:
        import torch_npu  # noqa: F401
    except ImportError as error:
        raise RuntimeError("LocateAnything NPU验证需要安装torch_npu") from error
    if not torch.npu.is_available():
        raise RuntimeError("torch_npu未检测到可用Ascend NPU")
    count = torch.npu.device_count()
    if max(devices) >= count:
        raise RuntimeError(f"请求NPU {devices}，但当前仅检测到{count}张卡")


def _run_inference(
    args: Any,
    model: Any,
    rank: int,
    world_size: int,
    device: torch.device,
    output_dir: Path,
    coco: dict[str, Any],
) -> tuple[dict[str, float | int], Any]:
    """当前rank使用本地NPU处理固定分片或共享动态队列。"""
    categories = coco["categories"]
    category_by_name = {normalize_label(item["name"]): item for item in categories}
    preprocessor = LocateAnythingValPreprocessor(
        coco["annotations"],
        categories,
        protocol=getattr(args, "protocol", PAPER_PROTOCOL),
    )
    image_by_id = {int(image["id"]): image for image in coco["images"]}
    dynamic_scheduling = bool(getattr(args, "dynamic_scheduling", False))
    continuous_batching = bool(getattr(args, "continuous_batching", False))
    dynamic_queue = getattr(args, "dynamic_queue", None)
    if dynamic_scheduling and dynamic_queue is None:
        dynamic_queue = _DynamicImageQueue(output_dir)
    shard_path = output_dir / f"predictions.rank{rank}.jsonl"
    prior = read_jsonl_records(shard_path) if args.resume else {}
    completed = {image_id for image_id, record in prior.items() if not record.get("error")}
    if dynamic_queue is None:
        assigned = shard_images(coco["images"], rank, world_size)
        remaining = [image for image in assigned if int(image["id"]) not in completed]
        task_count = len(remaining)
    else:
        remaining = []
        task_count = dynamic_queue.total
    mode = "a" if args.resume else "w"

    if task_count and model is None:
        from .model import LocateAnything

        model = LocateAnything(
            args.model,
            revision=args.revision,
            device=device,
            dtype="bfloat16",
            local_files_only=not args.allow_download,
            npu_fast_path=args.npu_fast_path,
        )
    if model is not None:
        from .npu_fast import configure_npu_kernel_fusions

        configure_npu_kernel_fusions(
            model.model,
            fused_qkv=getattr(args, "fused_qkv", False),
            fused_add_rms_norm=getattr(args, "fused_add_rms_norm", True),
            fused_mlp=getattr(args, "fused_mlp", False),
        )

    if hasattr(torch, "npu"):
        torch.npu.reset_peak_memory_stats(device)
    ready_path = output_dir / ".dist" / f"ready.rank{rank}"
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text(str(os.getpid()), encoding="utf-8")
    if dynamic_scheduling and dist.is_initialized():
        # 只在模型加载完成后同步一次，防止先完成加载的rank预先领取多轮任务。
        dist.barrier()

    start = time.perf_counter()
    processed = 0
    processed_boxes = 0
    output_tokens = 0
    generation_seconds = 0.0
    progress = TQDM(
        total=task_count,
        desc=f"rank {rank}验证",
        disable=rank != 0 or dynamic_scheduling or bool(getattr(args, "parent_progress", False)),
    )

    def claim_images(count: int) -> list[dict[str, Any]]:
        if dynamic_queue is None:
            raise RuntimeError("固定分片路径不能领取动态任务")
        return [image_by_id[image_id] for image_id in dynamic_queue.claim(count)]

    with _RecordSink(shard_path, mode, asynchronous=continuous_batching) as record_sink:
        if args.batch == 1:
            if dynamic_queue is None:
                image_groups = ([image] for image in remaining)
            else:

                def dynamic_single_groups():
                    while images := claim_images(1):
                        yield images

                image_groups = dynamic_single_groups()
            for image_group in image_groups:
                image = image_group[0]
                image_id = int(image["id"])
                _seed_for_image(args.seed, image_id)
                context = image
                try:
                    source, question, context = preprocessor.prepare(image)
                    result = model.predict(
                        source,
                        task="raw",
                        prompt=question,
                        generation_mode=args.generation_mode,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=1.1,
                    )[0]
                    record = result_to_record(result, context, category_by_name, preprocessor)
                    record["batch_id"] = image_id
                    record["batch_size"] = 1
                    record["batch_output_tokens"] = record["output_tokens"]
                    record["generation_stats"].update(
                        {
                            "batch_id": image_id,
                            "batch_size": 1,
                            "batch_generation_seconds": record["batch_generation_seconds"],
                            "batch_output_tokens": record["output_tokens"],
                        }
                    )
                except Exception as error:
                    LOGGER.error(f"rank={rank} image_id={image_id}验证失败：{error}")
                    record = _error_record(context, error)
                    if hasattr(torch, "npu"):
                        torch.npu.empty_cache()
                record_sink.write(record)
                processed += 1
                processed_boxes += len(record["predictions"])
                output_tokens += int(record["output_tokens"])
                generation_seconds += float(record["batch_generation_seconds"])
                progress.update(1)
                if generation_seconds:
                    progress.set_postfix(**{"tok/s": f"{output_tokens / generation_seconds:.1f}"})
        else:

            def store_result(image: dict[str, Any], result: Any) -> None:
                nonlocal processed, processed_boxes, output_tokens
                try:
                    record = result_to_record(result, image, category_by_name, preprocessor)
                except Exception as error:
                    LOGGER.error(f"rank={rank} image_id={image['id']}结果解析失败：{error}")
                    record = _error_record(image, error)
                    stats = result.stats
                    record.update(
                        batch_id=int(stats.get("batch_id", image["id"])),
                        batch_size=int(stats.get("batch_size", args.batch)),
                        continuous_window_size=int(stats.get("continuous_window_size", 0)),
                        batch_generation_seconds=float(stats.get("batch_generation_seconds", 0.0)),
                        batch_output_tokens=int(stats.get("batch_output_tokens", 0)),
                        generation_stats=dict(stats),
                    )
                record_sink.write(record)
                processed += 1
                processed_boxes += len(record["predictions"])
                output_tokens += int(record["output_tokens"])
                progress.update(1)

            if continuous_batching:
                local_offset = 0

                def provide_sources(count: int) -> list[tuple[Any, int, Any, str]]:
                    nonlocal local_offset
                    if dynamic_queue is not None:
                        images = claim_images(count)
                    else:
                        images = remaining[local_offset : local_offset + count]
                        local_offset += len(images)
                    prepared = [preprocessor.prepare(image) for image in images]
                    return [
                        (source, int(args.seed + int(context["id"])), context, question)
                        for source, question, context in prepared
                    ]

                refill_batch = int(getattr(args, "refill_batch", 0)) or max(1, args.batch // 16)
                try:
                    stream_stats = model._predict_continuous(
                        provide_sources,
                        store_result,
                        None,
                        default_label="target",
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        repetition_penalty=1.1,
                        scheduler=args.scheduler,
                        slot_capacity=args.batch,
                        refill_batch_size=refill_batch,
                        max_provider_inputs=task_count,
                        preprocess_request_size=refill_batch if dynamic_queue is not None else None,
                        paged_kv_cache=getattr(args, "paged_kv_cache", False),
                        max_duplicate_boxes=getattr(args, "max_duplicate_boxes", 0),
                        shape_bucketing=getattr(args, "shape_bucketing", False),
                        kv_bucket_size=getattr(args, "kv_bucket_size", 128),
                        npu_graph=getattr(args, "npu_graph", False),
                        visual_batching=getattr(args, "visual_batching", False),
                        direct_paged_decode=getattr(args, "direct_paged_decode", True),
                        device_repetition_cache=getattr(args, "device_repetition_cache", True),
                        qsample_reservoir=getattr(args, "qsample_reservoir", False),
                        overlap_prefill=getattr(args, "overlap_prefill", True),
                        candidate_top_p=getattr(args, "candidate_top_p", True),
                    )
                except Exception as error:
                    raise _batch_inference_error(error, args.batch, args.batch, device) from error
                generation_seconds = float(stream_stats["generation_seconds"])
                if int(stream_stats["processed"]) != processed:
                    raise RuntimeError(
                        f"continuous batching完成数不一致：runtime={stream_stats['processed']} records={processed}"
                    )
                if generation_seconds:
                    progress.set_postfix(**{"tok/s": f"{output_tokens / generation_seconds:.1f}"})
            else:
                window_size = args.batch * int(getattr(args, "continuous_window", 1))
                if dynamic_queue is None:
                    grouped_images = iter(batch_images(remaining, window_size))
                else:

                    def dynamic_groups():
                        while images := claim_images(window_size):
                            yield images

                    grouped_images = dynamic_groups()
                for images in grouped_images:
                    batch_id = int(images[0]["id"])
                    prepared = [preprocessor.prepare(image) for image in images]
                    sources = [item[0] for item in prepared]
                    questions = [item[1] for item in prepared]
                    contexts = [item[2] for item in prepared]
                    seeds = [int(args.seed + int(image["id"])) for image in contexts]
                    try:
                        results = model._predict_batch(
                            sources,
                            questions,
                            default_label="target",
                            max_new_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            repetition_penalty=1.1,
                            scheduler=args.scheduler,
                            seeds=seeds,
                            batch_id=batch_id,
                            slot_capacity=args.batch,
                            static_kv_cache=getattr(args, "static_kv_cache", False),
                            paged_kv_cache=getattr(args, "paged_kv_cache", False),
                            max_duplicate_boxes=getattr(args, "max_duplicate_boxes", 0),
                            shape_bucketing=getattr(args, "shape_bucketing", False),
                            kv_bucket_size=getattr(args, "kv_bucket_size", 128),
                            npu_graph=getattr(args, "npu_graph", False),
                            visual_batching=getattr(args, "visual_batching", False),
                            direct_paged_decode=getattr(args, "direct_paged_decode", True),
                            device_repetition_cache=getattr(args, "device_repetition_cache", True),
                            qsample_reservoir=getattr(args, "qsample_reservoir", False),
                            overlap_prefill=getattr(args, "overlap_prefill", True),
                            candidate_top_p=getattr(args, "candidate_top_p", True),
                        )
                    except Exception as error:
                        raise _batch_inference_error(error, args.batch, min(args.batch, len(images)), device) from error
                    for image, result in zip(contexts, results):
                        store_result(image, result)
                    batch_seconds = float(results[0].stats["batch_generation_seconds"])
                    generation_seconds += batch_seconds
                    if generation_seconds:
                        progress.set_postfix(**{"tok/s": f"{output_tokens / generation_seconds:.1f}"})
    progress.close()
    if hasattr(torch, "npu"):
        torch.npu.synchronize(device)
        peak_memory = int(torch.npu.max_memory_allocated(device))
        total_memory = int(torch.npu.get_device_properties(device).total_memory)
    else:
        peak_memory = total_memory = 0
    return (
        {
            "wall_seconds": time.perf_counter() - start,
            "processed": processed,
            "boxes": processed_boxes,
            "output_tokens": output_tokens,
            "generation_seconds": generation_seconds,
            "peak_memory_bytes": peak_memory,
            "total_memory_bytes": total_memory,
        },
        model,
    )


def _batch_inference_error(
    error: Exception, requested_batch: int, effective_batch: int, device: torch.device
) -> RuntimeError:
    """构造不隐式降级的批量失败信息，OOM时附带当前显存状态。"""
    message = str(error)
    is_oom = "out of memory" in message.lower() or "oom" in message.lower()
    memory = ""
    if hasattr(torch, "npu"):
        allocated = int(torch.npu.memory_allocated(device))
        reserved = int(torch.npu.memory_reserved(device))
        free, total = (int(value) for value in torch.npu.mem_get_info(device))
        gib = 1024**3
        memory = (
            f"：allocated={allocated / gib:.2f}GiB, reserved={reserved / gib:.2f}GiB, "
            f"free={free / gib:.2f}GiB, total={total / gib:.2f}GiB"
        )
    kind = "NPU OOM" if is_oom else "批量推理失败"
    return RuntimeError(
        f"{kind}，请求batch={requested_batch}，当前尾批effective_batch={effective_batch}{memory}。"
        "本次验证不会自动降低batch。"
        f"原始错误：{type(error).__name__}: {error}"
    )


def _aggregate_speed(
    records: list[dict[str, Any]],
    global_wall_seconds: float,
    processed: int,
    processed_boxes: int,
    processed_tokens: int = 0,
    global_generation_seconds: float = 0.0,
    peak_memory_bytes: int = 0,
    total_memory_bytes: int = 0,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """汇总逐图阶段耗时及本轮全局吞吐。"""
    stage_values: dict[str, list[float]] = defaultdict(list)
    total_output_tokens = 0
    for record in records:
        for stage, value in record.get("speed", {}).items():
            stage_values[stage].append(float(value))
        tokens = record.get("output_tokens")
        if tokens is None and tokenizer is not None:
            encoded = tokenizer(str(record.get("raw_output", "")), add_special_tokens=False)
            token_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            tokens = len(token_ids[0] if token_ids and isinstance(token_ids[0], (list, tuple)) else token_ids)
        total_output_tokens += int(tokens or 0)
    return {
        "global_wall_seconds": global_wall_seconds,
        "processed_this_run": processed,
        "boxes_processed_this_run": processed_boxes,
        "output_tokens": total_output_tokens,
        "output_tokens_this_run": processed_tokens,
        "tokens_per_second": processed_tokens / global_generation_seconds if global_generation_seconds else 0.0,
        "average_tokens_per_image": total_output_tokens / len(records) if records else 0.0,
        "global_generation_seconds": global_generation_seconds,
        "images_per_second": processed / global_wall_seconds if global_wall_seconds else 0.0,
        "boxes_per_second": processed_boxes / global_wall_seconds if global_wall_seconds else 0.0,
        "peak_memory_bytes": peak_memory_bytes,
        "total_memory_bytes": total_memory_bytes,
        "peak_memory_percent": 100 * peak_memory_bytes / total_memory_bytes if total_memory_bytes else 0.0,
        "average_milliseconds": {
            stage: float(np.mean(values)) if values else 0.0 for stage, values in sorted(stage_values.items())
        },
    }


def _summary_text(metrics: dict[str, Any]) -> str:
    """生成便于终端和文件阅读的中文摘要。"""
    f1_50 = metrics["official_locate_metrics"]["f1"]["0.50"]
    f1_95 = metrics["official_locate_metrics"]["f1"]["0.95"]
    ap = metrics["nonstandard_constant_score_coco_ap"]
    paper = "mean" in metrics["official_locate_metrics"]["f1"]
    lines = ["LocateAnything MS COCO 2017 val验证结果"]
    if paper:
        f1_mean = metrics["official_locate_metrics"]["f1"]["mean"]["macro"]
        lines.extend(
            (
                "协议：LocateAnything论文COCO协议（短边840、GT正类prompt、FastEval式匹配）",
                f"实际global batch={metrics['config']['global_batch']}，"
                f"每rank local batch={metrics['config']['local_batch']}；论文每设备batch=1，"
                f"local batch是否一致={metrics['config']['paper_batch_matches']}",
                f"图片：{metrics['counts']['images']}，预测框：{metrics['counts']['boxes']}",
                f"F1@0.50={f1_50['macro']['f1']:.6f} "
                f"(P={f1_50['macro']['precision']:.6f}, R={f1_50['macro']['recall']:.6f})",
                f"F1@0.95={f1_95['macro']['f1']:.6f} "
                f"(P={f1_95['macro']['precision']:.6f}, R={f1_95['macro']['recall']:.6f})",
                f"Mean F1={f1_mean['f1']:.6f} (P={f1_mean['precision']:.6f}, R={f1_mean['recall']:.6f})",
            )
        )
    else:
        lines.extend(
            (
                "协议：legacy（原图、80类prompt、IoU降序匹配）",
                f"图片：{metrics['counts']['images']}，预测框：{metrics['counts']['boxes']}",
                f"F1@0.50 micro={f1_50['micro']['f1']:.6f}, macro={f1_50['macro']['f1']:.6f}",
                f"F1@0.95 micro={f1_95['micro']['f1']:.6f}, macro={f1_95['macro']['f1']:.6f}",
            )
        )
    lines.extend(
        (
            f"mean GT IoU={metrics['official_locate_metrics']['mean_gt_iou']:.6f}（诊断项，非论文Mean F1）",
            f"吞吐：{metrics['speed']['images_per_second']:.4f} images/s, "
            f"{metrics['speed']['boxes_per_second']:.4f} boxes/s, "
            f"{metrics['speed']['tokens_per_second']:.2f} tokens/s",
            "",
            "警告：LocateAnything不输出confidence；以下COCO AP使用固定score=1.0，不是标准COCO AP。",
            f"COCO AP状态：{ap['status']}",
        )
    )
    if metrics["counts"].get("repetition_stopped_images"):
        lines.insert(2, f"退化重复终止图片：{metrics['counts']['repetition_stopped_images']}")
    if ap.get("AP50_95") is not None:
        lines.append(f"AP50:95={ap['AP50_95']:.6f}, AP50={ap['AP50']:.6f}, AP75={ap['AP75']:.6f}")
    if "average_npu_utilization_percent" in metrics["speed"]:
        lines.append(f"平均NPU利用率={metrics['speed']['average_npu_utilization_percent']:.2f}%")
    if ap.get("error"):
        lines.append(f"COCO AP错误：{ap['error']}")
    return "\n".join(lines) + "\n"


def _prepare_run_config_file(args: Any, output_dir: Path) -> None:
    """写入或验证resume所需的协议、batch和节点布局元数据。"""
    path = output_dir / ".dist" / "run_config.json"
    expected = {
        "protocol": args.protocol,
        "global_batch": int(args.global_batch),
        "local_batch": int(args.batch),
        "world_size": int(args.global_batch // args.batch),
        "local_world_size": int(getattr(args, "local_world_size", 1)),
        "nnodes": int(getattr(args, "nnodes", 1)),
    }
    if args.resume:
        if not path.is_file():
            raise RuntimeError(
                f"resume目录缺少新版分布式布局元数据：{path}。请使用新output_dir重新验证，不要混用旧分片。"
            )
        actual = json.loads(path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(f"resume分布式布局不一致：已有{actual}，当前{expected}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_results(
    args: Any,
    output_dir: Path,
    coco: dict[str, Any],
    world_size: int,
    global_wall_seconds: float,
    processed: int,
    processed_boxes: int,
    processed_tokens: int = 0,
    global_generation_seconds: float = 0.0,
    peak_memory_bytes: int = 0,
    total_memory_bytes: int = 0,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """由rank 0合并分片、计算指标并写入最终产物。"""
    records = merge_prediction_shards(output_dir, world_size)
    expected_ids = {int(image["id"]) for image in coco["images"]}
    actual_ids = {int(record["image_id"]) for record in records}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise RuntimeError(f"验证记录未完整覆盖图片：missing={missing[:20]} unexpected={unexpected[:20]}")

    predictions = build_constant_score_predictions(records)
    prediction_path = output_dir / "predictions.json"
    prediction_path.write_text(json.dumps(predictions, ensure_ascii=False), encoding="utf-8")
    protocol = getattr(args, "protocol", PAPER_PROTOCOL)
    metric_function = compute_locate_metrics if protocol == PAPER_PROTOCOL else compute_legacy_locate_metrics
    metric_kwargs = {"image_order": coco.get("evaluation_image_ids")} if protocol == PAPER_PROTOCOL else {}
    official = metric_function(records, coco["annotations"], coco["categories"], expected_ids, **metric_kwargs)
    coco_ap = run_nonstandard_coco_ap(coco["annotation_path"], prediction_path, sorted(expected_ids))
    metrics = {
        "config": {
            "model": args.model,
            "data": args.data,
            "devices": args.devices,
            "generation_mode": args.generation_mode,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "max_images": args.max_images,
            "batch": getattr(args, "global_batch", getattr(args, "batch", 1)),
            "global_batch": getattr(args, "global_batch", getattr(args, "batch", 1)),
            "local_batch": getattr(args, "batch", 1),
            "protocol": protocol,
            "protocol_id": PAPER_PROTOCOL_ID if protocol == PAPER_PROTOCOL else LEGACY_PROTOCOL_ID,
            "short_side": PAPER_SHORT_SIDE if protocol == PAPER_PROTOCOL else None,
            "resize_interpolation": "bilinear" if protocol == PAPER_PROTOCOL else None,
            "prompt_categories": "image_gt_positive" if protocol == PAPER_PROTOCOL else "all_80",
            "paper_reference_batch": 1,
            "paper_batch_matches": getattr(args, "batch", 1) == 1,
            "scheduler": getattr(args, "scheduler", "pipeline"),
            "continuous_window": getattr(args, "continuous_window", 1),
            "continuous_batching": getattr(args, "continuous_batching", False),
            "dynamic_scheduling": getattr(args, "dynamic_scheduling", False),
            "refill_batch": getattr(args, "refill_batch", 0),
            "effective_refill_batch": (getattr(args, "refill_batch", 0) or max(1, getattr(args, "batch", 1) // 16)),
            "static_kv_cache": getattr(args, "static_kv_cache", False),
            "paged_kv_cache": getattr(args, "paged_kv_cache", False),
            "max_duplicate_boxes": getattr(args, "max_duplicate_boxes", 0),
            "shape_bucketing": getattr(args, "shape_bucketing", False),
            "kv_bucket_size": getattr(args, "kv_bucket_size", 128),
            "npu_graph": getattr(args, "npu_graph", False),
            "visual_batching": getattr(args, "visual_batching", False),
            "direct_paged_decode": getattr(args, "direct_paged_decode", True),
            "device_repetition_cache": getattr(args, "device_repetition_cache", True),
            "qsample_reservoir": getattr(args, "qsample_reservoir", False),
            "overlap_prefill": getattr(args, "overlap_prefill", True),
            "candidate_top_p": getattr(args, "candidate_top_p", True),
            "cpu_affinity": getattr(args, "cpu_affinity", False),
            "npu_fast_path": getattr(args, "npu_fast_path", "auto"),
            "fused_qkv": getattr(args, "fused_qkv", False),
            "fused_add_rms_norm": getattr(args, "fused_add_rms_norm", True),
            "fused_mlp": getattr(args, "fused_mlp", False),
            "world_size": world_size,
            "local_world_size": getattr(args, "local_world_size", world_size),
            "nnodes": getattr(args, "nnodes", 1),
        },
        "counts": {
            "images": len(records),
            "boxes": len(predictions),
            "unknown_labels": sum(len(record.get("unknown_labels", [])) for record in records),
            "parse_warning_images": sum(bool(record.get("parse_warnings")) for record in records),
            "empty_prediction_images": sum(not record.get("predictions") for record in records),
            "inference_errors": sum(bool(record.get("error")) for record in records),
            "repetition_stopped_images": sum(
                bool(record.get("generation_stats", {}).get("stopped_repetition")) for record in records
            ),
        },
        "official_locate_metrics": official,
        "nonstandard_constant_score_coco_ap": coco_ap,
        "speed": _aggregate_speed(
            records,
            global_wall_seconds,
            processed,
            processed_boxes,
            processed_tokens,
            global_generation_seconds,
            peak_memory_bytes,
            total_memory_bytes,
            tokenizer,
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = _summary_text(metrics)
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    LOGGER.info("\n" + summary.rstrip())
    return metrics


def _run_distributed_worker(args: Any, model: Any = None) -> LocateMetrics | None:
    """执行当前torchrun rank的验证并在rank 0返回指标对象。"""
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    _configure_worker_progress(rank)
    devices = parse_devices(args.devices)
    if getattr(args, "cpu_affinity", True):
        _configure_worker_cpu_runtime(devices[local_rank])
    device, _, _ = initialize_distributed_runtime(
        device_type="npu",
        device_spec="npu:" + ",".join(str(value) for value in devices),
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        dist_module=dist,
        accelerator_resolver=get_torch_device_backend,
        is_ascend=True,
    )
    try:
        output_dir = _resolve_worker_output(args)
        if rank == 0:
            _prepare_run_config_file(args, output_dir)
        dist.barrier()
        coco = load_coco_validation(
            args.data,
            allow_download=args.allow_download,
            max_images=args.max_images,
        )
        if args.resume:
            validate_resume_protocol(
                output_dir,
                world_size,
                getattr(args, "protocol", PAPER_PROTOCOL),
            )
        if getattr(args, "dynamic_scheduling", False):
            if getattr(args, "store_port", 0):
                dynamic_store = dist.TCPStore(
                    args.store_host,
                    int(args.store_port),
                    world_size,
                    rank == 0,
                    timedelta(seconds=1800),
                    True,
                )
                image_ids = _remaining_dynamic_image_ids(
                    output_dir,
                    coco["images"],
                    world_size,
                    resume=bool(args.resume),
                )
                args.dynamic_queue = _TCPDynamicImageQueue(
                    dynamic_store,
                    image_ids,
                    getattr(args, "launch_id", "locateanything-val"),
                    rank,
                    world_size,
                    int(args.batch),
                )
            else:
                if rank == 0:
                    _initialize_dynamic_queue(
                        output_dir,
                        coco["images"],
                        world_size,
                        resume=bool(args.resume),
                    )
                dist.barrier()
        runtime, model = _run_inference(args, model, rank, world_size, device, output_dir, coco)
        # HCCL不支持FP64 all_reduce，秒级墙钟时间使用FP32已足够。
        wall_tensor = torch.tensor(runtime["wall_seconds"], dtype=torch.float32, device=device)
        generation_tensor = torch.tensor(runtime["generation_seconds"], dtype=torch.float32, device=device)
        processed_tensor = torch.tensor(runtime["processed"], dtype=torch.long, device=device)
        boxes_tensor = torch.tensor(runtime["boxes"], dtype=torch.long, device=device)
        tokens_tensor = torch.tensor(runtime["output_tokens"], dtype=torch.long, device=device)
        peak_memory_tensor = torch.tensor(runtime["peak_memory_bytes"], dtype=torch.long, device=device)
        total_memory_tensor = torch.tensor(runtime["total_memory_bytes"], dtype=torch.long, device=device)
        dist.all_reduce(wall_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(generation_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(processed_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(boxes_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(tokens_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(peak_memory_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(total_memory_tensor, op=dist.ReduceOp.MAX)
        dist.barrier()
        metrics = None
        if rank == 0:
            payload = finalize_results(
                args,
                output_dir,
                coco,
                world_size,
                float(wall_tensor.cpu()),
                int(processed_tensor.cpu()),
                int(boxes_tensor.cpu()),
                int(tokens_tensor.cpu()),
                float(generation_tensor.cpu()),
                int(peak_memory_tensor.cpu()),
                int(total_memory_tensor.cpu()),
                getattr(model, "tokenizer", None),
            )
            metrics = LocateMetrics(payload, output_dir)
        dist.barrier()
        return metrics
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_single_worker(args: Any, model: Any) -> LocateMetrics:
    """不创建process group，直接在当前NPU上执行单卡验证。"""
    output_dir = _resolve_worker_output(args)
    coco = load_coco_validation(
        args.data,
        allow_download=args.allow_download,
        max_images=args.max_images,
    )
    if args.resume:
        validate_resume_protocol(output_dir, 1, getattr(args, "protocol", PAPER_PROTOCOL))
    if getattr(args, "dynamic_scheduling", False):
        _initialize_dynamic_queue(output_dir, coco["images"], 1, resume=bool(args.resume))
    runtime, model = _run_inference(args, model, 0, 1, model.device, output_dir, coco)
    payload = finalize_results(
        args,
        output_dir,
        coco,
        1,
        float(runtime["wall_seconds"]),
        int(runtime["processed"]),
        int(runtime["boxes"]),
        int(runtime["output_tokens"]),
        float(runtime["generation_seconds"]),
        int(runtime["peak_memory_bytes"]),
        int(runtime["total_memory_bytes"]),
        getattr(model, "tokenizer", None),
    )
    return LocateMetrics(payload, output_dir)


def _configure_worker_progress(rank: int) -> None:
    """仅让rank 0显示Transformers权重加载进度。"""
    if rank != 0:
        from transformers.utils import logging as transformers_logging

        transformers_logging.disable_progress_bar()


def _parse_cpu_list(value: str) -> set[int]:
    """解析Linux cpulist，例如``0-3,8-11``。"""
    cpus: set[int] = set()
    for part in value.split(","):
        bounds = [int(item) for item in part.strip().split("-", 1)]
        cpus.update(range(bounds[0], bounds[-1] + 1))
    return cpus


def _configure_worker_cpu_runtime(device_id: int) -> set[int]:
    """限制PyTorch Host线程，并按npu-smi拓扑绑定当前worker。"""
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "topo"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        match = re.search(
            rf"^NPU{device_id}\s+.*?([0-9]+(?:-[0-9]+)?(?:,[0-9]+(?:-[0-9]+)?)*)\s*$", result.stdout, re.M
        )
        if not match:
            return set(os.sched_getaffinity(0))
        cpus = _parse_cpu_list(match.group(1))
        os.sched_setaffinity(0, cpus)
        return cpus
    except (OSError, ValueError, subprocess.SubprocessError):
        return set(os.sched_getaffinity(0))


class _DistributedProgress:
    """增量读取所有rank JSONL，不引入额外HCCL同步点。"""

    def __init__(self, output_dir: Path, world_size: int, total: int) -> None:
        self.paths = [output_dir / f"predictions.rank{rank}.jsonl" for rank in range(world_size)]
        self.positions = [path.stat().st_size if path.is_file() else 0 for path in self.paths]
        self.buffers = [b""] * world_size
        self.completed: set[int] = set()
        for path in self.paths:
            self.completed.update(
                image_id for image_id, record in read_jsonl_records(path).items() if not record.get("error")
            )
        self.initial_completed = len(self.completed)
        self.batch_stats: dict[tuple[int, int], tuple[int, float]] = {}
        self.stream_stats: dict[int, tuple[int, float]] = {}
        self.progress = TQDM(
            total=total,
            initial=self.initial_completed,
            desc=f"LocateAnything {world_size}卡验证",
            unit="image",
            unit_scale=False,
        )

    @property
    def tokens_per_second(self) -> float:
        rank_seconds: dict[int, float] = defaultdict(float)
        rank_tokens: dict[int, int] = defaultdict(int)
        for rank, (stream_tokens, stream_seconds) in self.stream_stats.items():
            rank_tokens[rank] = stream_tokens
            rank_seconds[rank] = stream_seconds
        for (rank, _), (batch_tokens, batch_seconds) in self.batch_stats.items():
            if rank in self.stream_stats:
                continue
            rank_tokens[rank] += batch_tokens
            rank_seconds[rank] += batch_seconds
        global_seconds = max(rank_seconds.values(), default=0.0)
        return sum(rank_tokens.values()) / global_seconds if global_seconds else 0.0

    def poll(self) -> None:
        """读取从上次offset起新增的完整JSONL行。"""
        before = len(self.completed)
        for rank, path in enumerate(self.paths):
            if not path.is_file():
                continue
            size = path.stat().st_size
            if size < self.positions[rank]:
                self.positions[rank] = 0
                self.buffers[rank] = b""
            with path.open("rb") as file:
                file.seek(self.positions[rank])
                chunk = file.read()
                self.positions[rank] = file.tell()
            if not chunk:
                continue
            lines = (self.buffers[rank] + chunk).split(b"\n")
            self.buffers[rank] = lines.pop()
            for line in lines:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    image_id = int(record["image_id"])
                except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
                    continue
                self.completed.add(image_id)
                generation_stats = record.get("generation_stats", {})
                if "scheduler_generation_seconds" in generation_stats:
                    self.stream_stats[rank] = (
                        int(generation_stats.get("scheduler_output_tokens", 0)),
                        float(generation_stats["scheduler_generation_seconds"]),
                    )
                    continue
                batch_id = int(record.get("batch_id", image_id))
                self.batch_stats[(rank, batch_id)] = (
                    int(record.get("batch_output_tokens", record.get("output_tokens", 0))),
                    float(record.get("batch_generation_seconds", 0.0)),
                )
        increment = len(self.completed) - before
        if increment:
            self.progress.update(increment)
        self.progress.set_postfix(**{"global tok/s": f"{self.tokens_per_second:.1f}"})

    def close(self) -> None:
        self.poll()
        self.progress.close()


def _read_npu_utilization(device_ids: list[int]) -> list[float]:
    """读取一次各卡NPU utilization；采样失败不影响验证。"""
    values = []
    for device_id in device_ids:
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "usages", "-i", str(device_id)],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            match = re.search(r"NPU Utilization\(%\)\s*:\s*(\d+(?:\.\d+)?)", result.stdout)
            if match:
                values.append(float(match.group(1)))
        except (OSError, subprocess.SubprocessError):
            return []
    return values


def _distributed_worker_env(npu_graph: bool) -> dict[str, str]:
    """构造torchrun worker环境；Graph才局部切换TorchNPU task queue模式。"""
    worker_env = os.environ.copy()
    if npu_graph:
        worker_env["TASK_QUEUE_ENABLE"] = "1"
    return worker_env


class LocateAnythingValidator(CallbackHost):
    """LocateAnything原生单卡与分布式MS COCO validator。"""

    def __init__(
        self,
        *,
        model: Any,
        data: str | Path = "coco.yaml",
        device: str | None = None,
        output_dir: str | Path | None = None,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 8192,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch: int = 1,
        scheduler: str = "pipeline",
        protocol: str = PAPER_PROTOCOL,
        continuous_window: int = 1,
        continuous_batching: bool | None = None,
        dynamic_scheduling: bool | None = None,
        refill_batch: int | None = None,
        static_kv_cache: bool = False,
        paged_kv_cache: bool | None = None,
        max_duplicate_boxes: int = 0,
        shape_bucketing: bool = False,
        kv_bucket_size: int = 128,
        npu_graph: bool = False,
        visual_batching: bool | None = None,
        direct_paged_decode: bool = True,
        device_repetition_cache: bool = True,
        qsample_reservoir: bool = False,
        overlap_prefill: bool = True,
        candidate_top_p: bool = True,
        fused_qkv: bool = False,
        fused_add_rms_norm: bool = True,
        fused_mlp: bool = False,
        cpu_affinity: bool = True,
        seed: int = 0,
        max_images: int = 0,
        resume: bool = False,
        allow_download: bool = False,
        callbacks_: dict | None = None,
    ) -> None:
        local_rank = int(os.getenv("LOCAL_RANK", "-1"))
        if is_k8s_distributed_parent():
            if device not in {None, "", "none"}:
                raise ValueError("K8S多节点LocateAnything验证请使用device=None自动选择可见NPU")
            _validate_npu_count([0])
            devices = list(range(torch.npu.device_count()))
            k8s = normalize_k8s_launch_config(len(devices))
            requested_world_size = len(devices) * k8s.nnodes
        else:
            if local_rank >= 0 and device in {None, "", "none"}:
                devices = list(range(int(os.getenv("LOCAL_WORLD_SIZE", "1"))))
            else:
                devices = parse_devices(str(device or DEFAULT_DEVICES))
            k8s = None
            requested_world_size = int(os.getenv("WORLD_SIZE", "1")) if local_rank >= 0 else len(devices)
        protocol = str(protocol).strip().lower()
        if protocol not in {PAPER_PROTOCOL, LEGACY_PROTOCOL}:
            raise ValueError(f"protocol必须是'paper'或'legacy'，得到{protocol!r}")
        if generation_mode not in {"fast", "slow", "hybrid"}:
            raise ValueError("generation_mode必须是'fast'、'slow'或'hybrid'")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens必须大于0")
        if temperature < 0:
            raise ValueError("temperature不能为负数")
        if not 0 < top_p <= 1:
            raise ValueError("top_p必须位于(0,1]")
        if max_images < 0:
            raise ValueError("max_images不能为负数")
        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError(f"batch必须是大于等于1的整数，得到{batch!r}")
        if requested_world_size > 1 and (batch < requested_world_size or batch % requested_world_size):
            raise ValueError(
                "LocateAnything分布式验证使用全局batch："
                f"要求batch >= world_size且能整除world_size，得到batch={batch}, "
                f"world_size={requested_world_size}"
            )
        local_batch = batch // requested_world_size
        use_batched_runtime = local_batch > 1
        if continuous_batching is None:
            continuous_batching = use_batched_runtime
        if dynamic_scheduling is None:
            dynamic_scheduling = requested_world_size > 1
        if refill_batch is None:
            refill_batch = min(8, local_batch) if use_batched_runtime else 0
        if paged_kv_cache is None:
            paged_kv_cache = use_batched_runtime
        if visual_batching is None:
            visual_batching = use_batched_runtime
        if isinstance(continuous_window, bool) or not isinstance(continuous_window, int) or continuous_window < 1:
            raise ValueError(f"continuous_window必须是大于等于1的整数，得到{continuous_window!r}")
        if not isinstance(continuous_batching, bool):
            raise ValueError(f"continuous_batching必须是bool，得到{continuous_batching!r}")
        if not isinstance(dynamic_scheduling, bool):
            raise ValueError(f"dynamic_scheduling必须是bool，得到{dynamic_scheduling!r}")
        if isinstance(refill_batch, bool) or not isinstance(refill_batch, int) or not 0 <= refill_batch <= local_batch:
            raise ValueError(f"refill_batch按每rank解释，必须位于[0,{local_batch}]，得到{refill_batch!r}")
        if not isinstance(static_kv_cache, bool):
            raise ValueError(f"static_kv_cache必须是bool，得到{static_kv_cache!r}")
        if not isinstance(paged_kv_cache, bool):
            raise ValueError(f"paged_kv_cache必须是bool，得到{paged_kv_cache!r}")
        if isinstance(max_duplicate_boxes, bool) or not isinstance(max_duplicate_boxes, int) or max_duplicate_boxes < 0:
            raise ValueError(f"max_duplicate_boxes必须是大于等于0的整数，得到{max_duplicate_boxes!r}")
        if not isinstance(shape_bucketing, bool):
            raise ValueError(f"shape_bucketing必须是bool，得到{shape_bucketing!r}")
        if isinstance(kv_bucket_size, bool) or not isinstance(kv_bucket_size, int) or kv_bucket_size < 1:
            raise ValueError(f"kv_bucket_size必须是正整数，得到{kv_bucket_size!r}")
        if not isinstance(npu_graph, bool):
            raise ValueError(f"npu_graph必须是bool，得到{npu_graph!r}")
        if not isinstance(visual_batching, bool):
            raise ValueError(f"visual_batching必须是bool，得到{visual_batching!r}")
        if not isinstance(direct_paged_decode, bool):
            raise ValueError(f"direct_paged_decode必须是bool，得到{direct_paged_decode!r}")
        if not isinstance(device_repetition_cache, bool):
            raise ValueError(f"device_repetition_cache必须是bool，得到{device_repetition_cache!r}")
        if not isinstance(qsample_reservoir, bool):
            raise ValueError(f"qsample_reservoir必须是bool，得到{qsample_reservoir!r}")
        if not isinstance(overlap_prefill, bool):
            raise ValueError(f"overlap_prefill必须是bool，得到{overlap_prefill!r}")
        if not isinstance(candidate_top_p, bool):
            raise ValueError(f"candidate_top_p必须是bool，得到{candidate_top_p!r}")
        if not isinstance(fused_qkv, bool):
            raise ValueError(f"fused_qkv必须是bool，得到{fused_qkv!r}")
        if not isinstance(fused_add_rms_norm, bool):
            raise ValueError(f"fused_add_rms_norm必须是bool，得到{fused_add_rms_norm!r}")
        if not isinstance(fused_mlp, bool):
            raise ValueError(f"fused_mlp必须是bool，得到{fused_mlp!r}")
        if npu_graph and not shape_bucketing:
            raise ValueError("npu_graph要求同时启用shape_bucketing")
        if static_kv_cache and paged_kv_cache:
            raise ValueError("static_kv_cache与paged_kv_cache不能同时启用")
        if not isinstance(cpu_affinity, bool):
            raise ValueError(f"cpu_affinity必须是bool，得到{cpu_affinity!r}")
        from .batch import normalize_scheduler

        scheduler = normalize_scheduler(scheduler)
        if local_batch > 1 and generation_mode != "hybrid":
            raise ValueError("local_batch>1时generation_mode必须为'hybrid'")
        if continuous_batching and local_batch == 1:
            raise ValueError("continuous_batching要求每rank local_batch>1")
        if continuous_batching and generation_mode != "hybrid":
            raise ValueError("continuous_batching要求generation_mode='hybrid'")
        if continuous_batching and continuous_window != 1:
            raise ValueError("continuous_batching与旧式continuous_window不能同时启用")
        if continuous_batching and static_kv_cache:
            raise ValueError("continuous_batching暂不支持static_kv_cache，请使用paged_kv_cache")

        self.owner = model
        self.device_ids = devices
        self.local_world_size = len(devices)
        self.world_size = requested_world_size
        self.k8s_launch_config = k8s
        self.requested_output = Path(output_dir or "runs/locateanything/val")
        manual_launch = local_rank >= 0
        local_world_size = int(os.getenv("LOCAL_WORLD_SIZE", str(len(devices)))) if manual_launch else len(devices)
        nnodes = requested_world_size // local_world_size
        store_host = os.getenv("MASTER_ADDR", "127.0.0.1") if manual_launch else ""
        store_port = 0
        launch_id = ""
        if manual_launch:
            master_port = int(os.getenv("MASTER_PORT", "29500"))
            default_store_port = master_port + 1 if master_port < 65535 else master_port - 1
            store_port = int(os.getenv("ULTRALYTICS_VAL_STORE_PORT", "0")) or default_store_port
            if not 1 <= store_port <= 65535 or store_port == master_port:
                raise ValueError(
                    f"ULTRALYTICS_VAL_STORE_PORT={store_port}非法或与torchrun MASTER_PORT={master_port}冲突"
                )
            launch_id = f"locate-external-{store_host}-{master_port}"
        self.args = SimpleNamespace(
            model=model.model_name,
            revision=model.revision,
            data=str(data),
            devices=",".join(str(value) for value in devices),
            output_dir="",
            generation_mode=generation_mode,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            global_batch=batch,
            batch=local_batch,
            scheduler=scheduler,
            protocol=protocol,
            continuous_window=continuous_window,
            continuous_batching=continuous_batching,
            dynamic_scheduling=dynamic_scheduling,
            refill_batch=refill_batch,
            static_kv_cache=static_kv_cache,
            paged_kv_cache=paged_kv_cache,
            max_duplicate_boxes=max_duplicate_boxes,
            shape_bucketing=shape_bucketing,
            kv_bucket_size=kv_bucket_size,
            npu_graph=npu_graph,
            visual_batching=visual_batching,
            direct_paged_decode=direct_paged_decode,
            device_repetition_cache=device_repetition_cache,
            qsample_reservoir=qsample_reservoir,
            overlap_prefill=overlap_prefill,
            candidate_top_p=candidate_top_p,
            fused_qkv=fused_qkv,
            fused_add_rms_norm=fused_add_rms_norm,
            fused_mlp=fused_mlp,
            cpu_affinity=cpu_affinity,
            npu_fast_path=getattr(model, "npu_fast_path", "auto"),
            seed=seed,
            max_images=max_images,
            resume=bool(resume),
            allow_download=bool(allow_download),
            parent_progress=False,
            local_world_size=local_world_size,
            nnodes=nnodes,
            store_host=store_host,
            store_port=store_port,
            launch_id=launch_id,
        )
        self.setup_callbacks(callbacks_)
        self.save_dir: Path | None = None
        self.metrics: LocateMetrics | None = None

    def __call__(self) -> LocateMetrics:
        """执行单卡直跑或启动分布式验证，返回metrics对象。"""
        self.run_callbacks("on_val_start")
        if int(os.getenv("LOCAL_RANK", "-1")) >= 0:
            self.args.output_dir = str(self.requested_output.resolve())
            metrics = _run_distributed_worker(self.args, self.owner)
            if metrics is None:
                return metrics
            self.metrics = metrics
        elif self.world_size == 1:
            output_dir = self._resolve_output()
            self._prepare_run_config(output_dir)
            self.args.output_dir = str(output_dir)
            self.args.parent_progress = False
            self.metrics = _run_single_worker(self.args, self.owner)
        else:
            self.metrics = self._launch_distributed()
        self.save_dir = self.metrics.save_dir
        self.run_callbacks("on_val_end")
        return self.metrics

    def _resolve_output(self) -> Path:
        """解析新验证目录或显式resume目录。"""
        if self.args.resume:
            if not self.requested_output.is_dir():
                raise FileNotFoundError(f"resume要求已有输出目录：{self.requested_output}")
            return self.requested_output.resolve()
        return increment_path(self.requested_output, mkdir=True).resolve()

    def _serializable_config(self) -> dict[str, Any]:
        """返回临时torchrun worker所需的JSON配置。"""
        return vars(self.args).copy()

    def _prepare_run_config(self, output_dir: Path) -> None:
        """写入或验证resume所需的分布式布局元数据。"""
        _prepare_run_config_file(self.args, output_dir)

    def _launch_distributed(self) -> LocateMetrics:
        """生成轻量worker，通过torchrun启动单节点或多节点HCCL rank。"""
        _validate_npu_count(self.device_ids)
        k8s = self.k8s_launch_config
        node_rank = k8s.node_rank if k8s else 0
        nnodes = k8s.nnodes if k8s else 1
        if k8s:
            from ultralytics.engine.val_runtime import create_k8s_parent_store

            parent_store = create_k8s_parent_store(k8s)
        else:
            parent_store = None
        if node_rank == 0:
            output_dir = self._resolve_output()
            self._prepare_run_config(output_dir)
            self.args.output_dir = str(output_dir)
            self.args.parent_progress = True
            self.args.local_world_size = self.local_world_size
            self.args.nnodes = nnodes
            self.args.store_host = k8s.master_addr if k8s else "127.0.0.1"
            self.args.store_port = int(os.getenv("ULTRALYTICS_VAL_STORE_PORT", "0")) or find_free_network_port()
            parent_store_port = (
                int(os.getenv("ULTRALYTICS_VAL_PARENT_STORE_PORT", str(k8s.master_port + 1))) if k8s else None
            )
            if k8s and self.args.store_port in {k8s.master_port, parent_store_port}:
                raise ValueError(
                    f"ULTRALYTICS_VAL_STORE_PORT={self.args.store_port}与torchrun或父进程协调端口冲突，请指定其他端口"
                )
            self.args.launch_id = f"locate-val-{self.args.store_port}-{time.time_ns()}"
        else:
            config_path = Path(parent_store.get("config_path").decode())
            output_dir = Path(json.loads(config_path.read_text(encoding="utf-8"))["output_dir"])

        dist_dir = output_dir / ".dist"
        dist_dir.mkdir(parents=True, exist_ok=True)
        if node_rank == 0:
            config_path = dist_dir / "config.json"
            config_path.write_text(
                json.dumps(self._serializable_config(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            if parent_store is not None:
                parent_store.set("config_path", str(config_path))
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=f"{id(self)}.py",
            prefix="_locate_val_",
            dir=dist_dir,
            delete=False,
            encoding="utf-8",
        ) as file:
            file.write(
                "from ultralytics.models.locateanything.val import distributed_val_from_config\n"
                f"distributed_val_from_config({str(config_path)!r})\n"
            )
            runner = Path(file.name)
        torchrun_port = k8s.master_port if k8s else find_free_network_port()
        while not k8s and torchrun_port == self.args.store_port:
            torchrun_port = find_free_network_port()
        command = build_torchrun_command(
            runner=runner,
            nproc_per_node=self.local_world_size,
            master_port=torchrun_port,
            nnodes=nnodes,
            node_rank=node_rank,
            master_addr=k8s.master_addr if k8s else None,
        )
        if node_rank == 0:
            LOGGER.info("LocateAnything分布式验证启动命令：" + " ".join(command))

        original_device = self.owner.device
        self.owner.model.to("cpu")
        if hasattr(torch, "npu"):
            torch.npu.empty_cache()
        for marker in dist_dir.glob("ready.rank*"):
            marker.unlink(missing_ok=True)
        total_images = self.args.max_images or 5000
        progress = _DistributedProgress(output_dir, self.world_size, total_images) if node_rank == 0 else None
        utilization_samples: list[float] = []
        worker_env = _distributed_worker_env(self.args.npu_graph)
        try:
            process = subprocess.Popen(command, env=worker_env)
            next_utilization_sample = 0.0
            try:
                while process.poll() is None:
                    if progress is not None:
                        progress.poll()
                    now = time.monotonic()
                    ready = node_rank == 0 and any(dist_dir.glob("ready.rank*"))
                    if ready and now >= next_utilization_sample:
                        utilization_samples.extend(_read_npu_utilization(self.device_ids))
                        next_utilization_sample = now + 5.0
                    time.sleep(1.0)
            except KeyboardInterrupt:
                LOGGER.warning("收到中断信号，正在优雅停止LocateAnything分布式验证并保留JSONL分片…")
                process.send_signal(signal.SIGINT)
                process.wait()
                raise
            finally:
                if progress is not None:
                    progress.close()
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, command)
        finally:
            runner.unlink(missing_ok=True)
            self.owner.model.to(original_device)
        metrics_path = output_dir / "metrics.json"
        if not metrics_path.is_file():
            raise RuntimeError(f"LocateAnything验证未生成指标文件：{metrics_path}")
        if utilization_samples and node_rank == 0:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            payload["speed"]["average_npu_utilization_percent"] = float(np.mean(utilization_samples))
            payload["speed"]["npu_utilization_samples"] = len(utilization_samples)
            metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            (output_dir / "summary.txt").write_text(_summary_text(payload), encoding="utf-8")
        if node_rank == 0:
            LOGGER.info(f"LocateAnything验证完成，结果保存在{output_dir}")
        if parent_store is not None:
            parent_store.set(f"parent_done/{node_rank}", "1")
            if node_rank == 0:
                parent_store.wait([f"parent_done/{rank}" for rank in range(nnodes)])
        return LocateMetrics.from_file(metrics_path)


def distributed_val_from_config(config_path: str | Path) -> LocateMetrics | None:
    """torchrun临时worker入口。"""
    args = SimpleNamespace(**json.loads(Path(config_path).read_text(encoding="utf-8")))
    return _run_distributed_worker(args)


__all__ = "LocateAnythingValPreprocessor", "LocateAnythingValidator", "LocateMetrics", "distributed_val_from_config"
