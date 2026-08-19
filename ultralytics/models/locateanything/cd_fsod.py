# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""LocateAnything在六个CD-FSOD目标域上的纯zero-shot验证。"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ultralytics.utils import LOGGER, SimpleClass

from .val_preprocess import CLOSED_SET_PROTOCOL, PAPER_PROTOCOL, PAPER_PROTOCOL_ID, PAPER_SHORT_SIDE

CD_FSOD_PROTOCOL_ID = "locateanything-paper-cdfsod-v1"
CD_FSOD_CLOSED_SET_PROTOCOL_ID = "locateanything-closed-set-cdfsod-v1"
CD_FSOD_DATASET_ORDER = ("ArTaxOr", "DIOR", "FISH", "NEU-DET", "UODD", "clipart1k")
CD_FSOD_DISPLAY_NAMES = {
    "ArTaxOr": "ArTaxOr",
    "DIOR": "DIOR",
    "FISH": "FISH (DeepFish)",
    "NEU-DET": "NEU-DET",
    "UODD": "UODD",
    "clipart1k": "clipart1k",
}
_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "cfg" / "datasets" / "cd-fsod"
DEFAULT_CD_FSOD_DATA = tuple(_CONFIG_ROOT / f"{name}-1shot.yaml" for name in CD_FSOD_DATASET_ORDER)


def naturalize_category_name(name: str) -> str:
    """将标注中的连接符转换为更适合语言模型的自然英文空格。"""
    return re.sub(r"\s+", " ", re.sub(r"[_-]+", " ", str(name))).strip()


def _normalize_label(label: str) -> str:
    from .val import normalize_label

    return normalize_label(label)


def build_category_alias_map(categories: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """同时接受原始类别名和自然化prompt别名，并拒绝歧义映射。"""
    aliases: dict[str, dict[str, Any]] = {}
    for category in categories:
        for name in (str(category["name"]), str(category.get("prompt_name", category["name"]))):
            key = _normalize_label(name)
            previous = aliases.get(key)
            if previous is not None and int(previous["id"]) != int(category["id"]):
                raise ValueError(f"类别别名冲突：{name!r}同时对应category id {previous['id']}和{category['id']}")
            aliases[key] = category
    return aliases


def _resolve_data_files(data: Any = None) -> list[Path]:
    values = DEFAULT_CD_FSOD_DATA if data is None else data
    if isinstance(values, (str, Path)):
        path = Path(values)
        if path.is_dir():
            values = [path / f"{name}-1shot.yaml" for name in CD_FSOD_DATASET_ORDER]
        else:
            values = [path]
    paths = [Path(value) for value in values]
    if not paths:
        raise ValueError("CD-FSOD验证至少需要一个数据集配置")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"CD-FSOD数据配置不存在：{missing}")
    return paths


def _dataset_name(path: Path, root: str | Path) -> str:
    candidates = (Path(root).name, path.stem.rsplit("-1shot", 1)[0])
    lookup = {name.casefold(): name for name in CD_FSOD_DATASET_ORDER}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    raise ValueError(f"无法从配置{path}或数据根目录{root}识别CD-FSOD数据集")


def load_cd_fsod_validation(
    data: Any = None,
    *,
    allow_download: bool = False,
    max_images_per_dataset: int = 0,
) -> dict[str, Any]:
    """加载CD-FSOD测试集，并构造全局唯一的内部image/category id。"""
    from .val import load_detection_validation

    if isinstance(max_images_per_dataset, bool) or not isinstance(max_images_per_dataset, int):
        raise ValueError("max_images_per_dataset必须是大于等于0的整数")
    if max_images_per_dataset < 0:
        raise ValueError("max_images_per_dataset不能为负数")

    loaded = []
    seen_names: set[str] = set()
    for config_path in _resolve_data_files(data):
        detection = load_detection_validation(
            config_path,
            allow_download=allow_download,
            max_images=max_images_per_dataset,
            drop_empty_categories=True,
            validation_only=True,
        )
        name = _dataset_name(config_path, detection["dataset_root"])
        if name in seen_names:
            raise ValueError(f"CD-FSOD数据集重复：{name}")
        seen_names.add(name)
        loaded.append((CD_FSOD_DATASET_ORDER.index(name), name, config_path, detection))
    loaded.sort(key=lambda item: item[0])

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    datasets: list[dict[str, Any]] = []
    category_by_name_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    category_ids_by_dataset: dict[str, list[int]] = {}
    category_aliases: dict[int, str] = {}
    next_image_id = next_category_id = 1

    manifest_hasher = hashlib.sha256()
    for _, name, config_path, detection in loaded:
        annotation_path = Path(detection["annotation_path"])
        annotation_digest = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
        manifest_hasher.update(name.encode())
        manifest_hasher.update(annotation_digest.encode())

        source_to_global_category: dict[int, int] = {}
        dataset_categories = []
        for source_category in detection["categories"]:
            source_id = int(source_category["id"])
            category = {
                "id": next_category_id,
                "name": str(source_category["name"]),
                "prompt_name": naturalize_category_name(source_category["name"]),
                "dataset_id": name,
                "source_category_id": source_id,
            }
            categories.append(category)
            dataset_categories.append(category)
            source_to_global_category[source_id] = next_category_id
            category_aliases[next_category_id] = category["prompt_name"]
            next_category_id += 1
        category_by_name_by_dataset[name] = build_category_alias_map(dataset_categories)
        category_ids_by_dataset[name] = [int(category["id"]) for category in dataset_categories]

        source_to_global_image: dict[Any, int] = {}
        dataset_images = []
        for source_image in detection["images"]:
            source_id = source_image["id"]
            image = {
                **source_image,
                "id": next_image_id,
                "dataset_id": name,
                "dataset_display_name": CD_FSOD_DISPLAY_NAMES[name],
                "source_image_id": source_id,
            }
            images.append(image)
            dataset_images.append(image)
            source_to_global_image[source_id] = next_image_id
            next_image_id += 1

        selected_source_ids = set(source_to_global_image)
        dataset_annotations = []
        for source_annotation in detection["annotations"]:
            source_image_id = source_annotation["image_id"]
            source_category_id = int(source_annotation["category_id"])
            if source_image_id not in selected_source_ids or source_category_id not in source_to_global_category:
                continue
            annotation = {
                **source_annotation,
                "image_id": source_to_global_image[source_image_id],
                "category_id": source_to_global_category[source_category_id],
                "source_image_id": source_image_id,
                "source_category_id": source_category_id,
                "dataset_id": name,
            }
            annotations.append(annotation)
            dataset_annotations.append(annotation)

        datasets.append(
            {
                "id": name,
                "display_name": CD_FSOD_DISPLAY_NAMES[name],
                "data": str(config_path),
                "annotation_path": str(annotation_path),
                "annotation_sha256": annotation_digest,
                "full_image_count": int(detection["full_image_count"]),
                "images": dataset_images,
                "annotations": dataset_annotations,
                "categories": dataset_categories,
                "evaluation_image_ids": [int(image["id"]) for image in dataset_images],
            }
        )

    return {
        "benchmark": "cd_fsod_zero_shot",
        "images": images,
        "annotations": annotations,
        "categories": categories,
        "evaluation_image_ids": [int(image["id"]) for image in images],
        "datasets": datasets,
        "category_aliases": category_aliases,
        "category_by_name_by_dataset": category_by_name_by_dataset,
        "category_ids_by_dataset": category_ids_by_dataset,
        "manifest_sha256": manifest_hasher.hexdigest(),
        "protocol_id": CD_FSOD_PROTOCOL_ID,
    }


class CDFsodMetrics(SimpleClass):
    """六个CD-FSOD目标域的LocateAnything zero-shot汇总指标。"""

    def __init__(self, payload: dict[str, Any], save_dir: str | Path) -> None:
        from .val import LocateMetrics

        self.save_dir = Path(save_dir)
        self.config = payload["config"]
        self.counts = payload["counts"]
        self.speed = payload["speed"]
        self.per_dataset = {
            name: LocateMetrics(dataset_payload, self.save_dir / name)
            for name, dataset_payload in payload["datasets"].items()
        }
        self.aggregate = payload["aggregate"]

    @classmethod
    def from_file(cls, path: str | Path) -> "CDFsodMetrics":
        path = Path(path)
        return cls(json.loads(path.read_text(encoding="utf-8")), path.parent)

    @property
    def fitness(self) -> float:
        return float(self.aggregate["F1_mean"])

    @property
    def results_dict(self) -> dict[str, float]:
        if self.config.get("protocol") == CLOSED_SET_PROTOCOL:
            return {
                "metrics/closed-set-F1-50(mean-datasets)": float(self.aggregate["F1_50"]),
                "metrics/closed-set-F1-95(mean-datasets)": float(self.aggregate["F1_95"]),
                "metrics/closed-set-F1-mean(mean-datasets)": float(self.aggregate["F1_mean"]),
                "metrics/paper-style-F1-50(mean-datasets)": float(self.aggregate["paper_F1_50"]),
                "metrics/paper-style-F1-95(mean-datasets)": float(self.aggregate["paper_F1_95"]),
                "metrics/paper-style-F1-mean(mean-datasets)": float(self.aggregate["paper_F1_mean"]),
                "metrics/nonstandard-mAP50(mean-datasets)": float(self.aggregate["nonstandard_AP50"]),
                "metrics/nonstandard-mAP50-95(mean-datasets)": float(self.aggregate["nonstandard_AP50_95"]),
                "fitness": self.fitness,
            }
        return {
            "metrics/F1-50(mean-datasets)": float(self.aggregate["F1_50"]),
            "metrics/F1-95(mean-datasets)": float(self.aggregate["F1_95"]),
            "metrics/F1-mean(mean-datasets)": float(self.aggregate["F1_mean"]),
            "metrics/nonstandard-mAP50(mean-datasets)": float(self.aggregate["nonstandard_AP50"]),
            "metrics/nonstandard-mAP50-95(mean-datasets)": float(self.aggregate["nonstandard_AP50_95"]),
            "fitness": self.fitness,
        }

    def summary(self, decimals: int = 5) -> list[dict[str, Any]]:
        rows = []
        for name, metrics in self.per_dataset.items():
            row = {
                "Dataset": CD_FSOD_DISPLAY_NAMES.get(name, name),
                "Images": metrics.counts["images"],
                "F1-50": round(float(metrics.official["f1"]["0.50"]["macro"]["f1"]), decimals),
                "F1-95": round(float(metrics.official["f1"]["0.95"]["macro"]["f1"]), decimals),
                "F1-mean": round(float(metrics.official["f1"]["mean"]["macro"]["f1"]), decimals),
                "nonstandard-AP50": round(float(metrics.coco_ap.get("AP50", 0.0)), decimals),
            }
            if metrics.auxiliary_paper:
                row.update(
                    {
                        "paper-style-F1-50": round(
                            float(metrics.auxiliary_paper["f1"]["0.50"]["macro"]["f1"]), decimals
                        ),
                        "paper-style-F1-95": round(
                            float(metrics.auxiliary_paper["f1"]["0.95"]["macro"]["f1"]), decimals
                        ),
                        "paper-style-F1-mean": round(
                            float(metrics.auxiliary_paper["f1"]["mean"]["macro"]["f1"]), decimals
                        ),
                    }
                )
            rows.append(row)
        return rows


def build_cd_fsod_validator(
    *,
    model: Any,
    data: Any = None,
    device: str | None = None,
    output_dir: str | Path = "runs/locateanything/cd_fsod",
    batch: int = 512,
    max_images_per_dataset: int = 0,
    callbacks_: dict | None = None,
    **kwargs: Any,
):
    """构造带CD-FSOD元数据的共享LocateAnything validator。"""
    from .val import LocateAnythingValidator

    protocol = str(kwargs.pop("protocol", "paper")).strip().lower()
    if protocol not in {PAPER_PROTOCOL, CLOSED_SET_PROTOCOL}:
        raise ValueError("CD-FSOD zero-shot仅支持protocol='paper'或'closed_set'")
    if "max_images" in kwargs:
        raise TypeError("请使用max_images_per_dataset，不要为CD-FSOD传入max_images")
    data_files = _resolve_data_files(data)
    validator = LocateAnythingValidator(
        model=model,
        data=data_files,
        device=device,
        output_dir=output_dir,
        batch=batch,
        protocol=protocol,
        max_images=max_images_per_dataset,
        callbacks_=callbacks_,
        **kwargs,
    )
    suite = load_cd_fsod_validation(
        data_files,
        allow_download=bool(getattr(validator.args, "allow_download", False)),
        max_images_per_dataset=max_images_per_dataset,
    )
    validator.args.benchmark = "cd_fsod"
    validator.args.dataset_manifest_sha256 = suite["manifest_sha256"]
    validator.args.total_images = len(suite["images"])
    return validator


class LocateAnythingCDFsodValidator:
    """复用LocateAnything验证运行时的一次启动CD-FSOD suite。"""

    def __init__(self, **kwargs: Any) -> None:
        self._validator = build_cd_fsod_validator(**kwargs)

    def __call__(self) -> CDFsodMetrics:
        return self._validator()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._validator, name)


def _dataset_summary_text(name: str, metrics: dict[str, Any]) -> str:
    f1 = metrics["official_locate_metrics"]["f1"]
    ap = metrics["nonstandard_constant_score_coco_ap"]
    closed_set = metrics["config"]["protocol"] == CLOSED_SET_PROTOCOL
    lines = [
        f"LocateAnything CD-FSOD zero-shot：{CD_FSOD_DISPLAY_NAMES.get(name, name)}",
        (
            "协议：严格closed-set（短边840、数据集全类别prompt、错误类别计FP、零分参与宏平均）"
            if closed_set
            else "协议：短边840、PIL Bilinear、逐图GT正类别prompt、FastEval式生成顺序匹配"
        ),
        "类别prompt：下划线/连字符自然化；未使用任何1/5/10-shot训练样本",
        f"图片：{metrics['counts']['images']}，GT：{metrics['counts']['ground_truths']}，"
        f"预测框：{metrics['counts']['boxes']}",
        f"F1@0.50={f1['0.50']['macro']['f1']:.6f}",
        f"F1@0.95={f1['0.95']['macro']['f1']:.6f}",
        f"Mean F1={f1['mean']['macro']['f1']:.6f}",
    ]
    if closed_set:
        auxiliary = metrics["auxiliary_paper_metrics"]["f1"]
        lines.append(
            f"辅助paper-style：F1@0.50={auxiliary['0.50']['macro']['f1']:.6f}，"
            f"F1@0.95={auxiliary['0.95']['macro']['f1']:.6f}，"
            f"Mean F1={auxiliary['mean']['macro']['f1']:.6f}"
        )
    lines.extend(
        (
            f"mean GT IoU={metrics['official_locate_metrics']['mean_gt_iou']:.6f}",
            "警告：LocateAnything不输出confidence；以下AP固定score=1.0，仅为非标准诊断。",
            f"AP状态：{ap['status']}，AP50={ap.get('AP50', 0.0):.6f}，AP50:95={ap.get('AP50_95', 0.0):.6f}",
        )
    )
    return "\n".join(lines) + "\n"


def suite_summary_text(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    speed = payload["speed"]
    dataset_count = len(payload["datasets"])
    dataset_count_text = "六" if dataset_count == 6 else str(dataset_count)
    lines = [
        f"LocateAnything CD-FSOD {dataset_count_text}数据集zero-shot验证结果",
        (
            "未使用1/5/10-shot训练标注；主指标为严格closed-set计数F1。"
            if payload["config"].get("protocol") == CLOSED_SET_PROTOCOL
            else "未使用1/5/10-shot训练标注；主指标为LocateAnything论文式F1。"
        ),
        (
            "逐图prompt使用数据集全部类别；主指标严格计入错误类别预测和零分类别。"
            if payload["config"].get("protocol") == CLOSED_SET_PROTOCOL
            else "逐图prompt使用GT正类别，因此这是oracle positive-category协议，不是开放类别发现。"
        ),
        "",
    ]
    for name, metrics in payload["datasets"].items():
        f1 = metrics["official_locate_metrics"]["f1"]
        ap = metrics["nonstandard_constant_score_coco_ap"]
        lines.append(
            f"{CD_FSOD_DISPLAY_NAMES.get(name, name)}: images={metrics['counts']['images']}, "
            f"F1@0.50={f1['0.50']['macro']['f1']:.6f}, F1@0.95={f1['0.95']['macro']['f1']:.6f}, "
            f"Mean F1={f1['mean']['macro']['f1']:.6f}, nonstandard AP50={ap.get('AP50', 0.0):.6f}"
        )
        if metrics.get("auxiliary_paper_metrics"):
            auxiliary = metrics["auxiliary_paper_metrics"]["f1"]
            lines.append(
                f"  辅助paper-style: F1@0.50={auxiliary['0.50']['macro']['f1']:.6f}, "
                f"F1@0.95={auxiliary['0.95']['macro']['f1']:.6f}, "
                f"Mean F1={auxiliary['mean']['macro']['f1']:.6f}"
            )
    lines.extend(
        (
            "",
            f"{dataset_count_text}数据集等权平均：F1@0.50={aggregate['F1_50']:.6f}, "
            f"F1@0.95={aggregate['F1_95']:.6f}, Mean F1={aggregate['F1_mean']:.6f}",
            f"总吞吐：{speed['images_per_second']:.4f} images/s, "
            f"{speed['boxes_per_second']:.4f} boxes/s, {speed['tokens_per_second']:.2f} tokens/s",
            "固定score=1.0的AP不是标准CD-FSOD AP，不参与fitness。",
        )
    )
    if payload["config"].get("protocol") == CLOSED_SET_PROTOCOL:
        lines.insert(
            -2,
            f"辅助paper-style等权平均：F1@0.50={aggregate['paper_F1_50']:.6f}, "
            f"F1@0.95={aggregate['paper_F1_95']:.6f}, Mean F1={aggregate['paper_F1_mean']:.6f}",
        )
    if "average_npu_utilization_percent" in speed:
        lines.append(f"平均NPU利用率={speed['average_npu_utilization_percent']:.2f}%")
    return "\n".join(lines) + "\n"


def _write_coco_eval_remap(
    dataset: dict[str, Any],
    dataset_dir: Path,
    source_predictions: list[dict[str, Any]],
) -> tuple[Path, Path, list[int]]:
    """将可能为字符串的源image id稳定映射为faster-coco-eval支持的整数。"""
    source_payload = json.loads(Path(dataset["annotation_path"]).read_text(encoding="utf-8"))
    source_ids = [image["source_image_id"] for image in dataset["images"]]
    image_id_map = {source_id: index for index, source_id in enumerate(source_ids, 1)}
    selected_ids = set(source_ids)
    active_category_ids = {int(category["source_category_id"]) for category in dataset["categories"]}
    evaluation_payload = {
        key: value for key, value in source_payload.items() if key not in {"images", "annotations", "categories"}
    }
    evaluation_payload["images"] = [
        {**image, "id": image_id_map[image["id"]]} for image in source_payload["images"] if image["id"] in selected_ids
    ]
    evaluation_payload["annotations"] = [
        {**annotation, "image_id": image_id_map[annotation["image_id"]]}
        for annotation in source_payload["annotations"]
        if annotation["image_id"] in selected_ids and int(annotation["category_id"]) in active_category_ids
    ]
    evaluation_payload["categories"] = [
        category for category in source_payload["categories"] if int(category["id"]) in active_category_ids
    ]
    evaluation_predictions = [
        {**prediction, "image_id": image_id_map[prediction["image_id"]]} for prediction in source_predictions
    ]
    annotation_path = dataset_dir / "evaluation_annotations.json"
    prediction_path = dataset_dir / "predictions.eval.json"
    annotation_path.write_text(json.dumps(evaluation_payload, ensure_ascii=False), encoding="utf-8")
    prediction_path.write_text(json.dumps(evaluation_predictions, ensure_ascii=False), encoding="utf-8")
    (dataset_dir / "image_id_map.json").write_text(
        json.dumps(
            [
                {"evaluation_image_id": image_id_map[source_id], "source_image_id": source_id}
                for source_id in source_ids
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return annotation_path, prediction_path, list(range(1, len(source_ids) + 1))


def finalize_cd_fsod_results(
    args: Any,
    output_dir: Path,
    suite: dict[str, Any],
    world_size: int,
    global_wall_seconds: float,
    processed: int,
    processed_boxes: int,
    processed_tokens: int,
    global_generation_seconds: float,
    peak_memory_bytes: int,
    total_memory_bytes: int,
    tokenizer: Any = None,
) -> dict[str, Any]:
    """按数据集拆分统一rank分片，生成六份指标和一份等权汇总。"""
    from .val import (
        _aggregate_speed,
        build_constant_score_predictions,
        compute_closed_set_metrics,
        compute_locate_metrics,
        merge_prediction_shards,
        run_nonstandard_coco_ap,
    )

    records = merge_prediction_shards(output_dir, world_size)
    expected_ids = {int(image["id"]) for image in suite["images"]}
    actual_ids = {int(record["image_id"]) for record in records}
    if actual_ids != expected_ids:
        raise RuntimeError(
            f"CD-FSOD记录未完整覆盖图片：missing={sorted(expected_ids - actual_ids)[:20]} "
            f"unexpected={sorted(actual_ids - expected_ids)[:20]}"
        )

    suite_speed = _aggregate_speed(
        records,
        global_wall_seconds,
        processed,
        processed_boxes,
        processed_tokens,
        global_generation_seconds,
        peak_memory_bytes,
        total_memory_bytes,
        tokenizer,
    )
    category_by_id = {int(item["id"]): item for item in suite["categories"]}
    dataset_payloads: dict[str, dict[str, Any]] = {}
    rows = []
    for dataset in suite["datasets"]:
        name = dataset["id"]
        dataset_dir = output_dir / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_records = [record for record in records if record.get("dataset_id") == name]
        dataset_ids = set(dataset["evaluation_image_ids"])
        closed_set = args.protocol == CLOSED_SET_PROTOCOL
        metric_function = compute_closed_set_metrics if closed_set else compute_locate_metrics
        official = metric_function(
            dataset_records,
            dataset["annotations"],
            dataset["categories"],
            dataset_ids,
            image_order=dataset["evaluation_image_ids"],
        )
        auxiliary_paper = None
        if closed_set:
            auxiliary_paper = compute_locate_metrics(
                dataset_records,
                dataset["annotations"],
                dataset["categories"],
                dataset_ids,
                image_order=dataset["evaluation_image_ids"],
            )
            auxiliary_paper["protocol"] = "paper_style_on_closed_set_predictions"
        global_predictions = build_constant_score_predictions(dataset_records)
        source_predictions = []
        record_by_image_id = {int(record["image_id"]): record for record in dataset_records}
        for prediction in global_predictions:
            category = category_by_id[int(prediction["category_id"])]
            record = record_by_image_id[int(prediction["image_id"])]
            source_predictions.append(
                {
                    **prediction,
                    "image_id": record["source_image_id"],
                    "category_id": int(category["source_category_id"]),
                }
            )
        prediction_path = dataset_dir / "predictions.json"
        prediction_path.write_text(json.dumps(source_predictions, ensure_ascii=False), encoding="utf-8")
        eval_annotation_path, eval_prediction_path, eval_image_ids = _write_coco_eval_remap(
            dataset, dataset_dir, source_predictions
        )
        coco_ap = run_nonstandard_coco_ap(eval_annotation_path, eval_prediction_path, eval_image_ids)
        dataset_tokens = sum(int(record.get("output_tokens", 0)) for record in dataset_records)
        dataset_speed = {
            **suite_speed,
            "scope": "dataset_contribution_over_shared_suite_runtime",
            "processed_this_run": len(dataset_records),
            "boxes_processed_this_run": len(source_predictions),
            "output_tokens": dataset_tokens,
            "output_tokens_this_run": dataset_tokens,
            "images_per_second": len(dataset_records) / global_wall_seconds if global_wall_seconds else 0.0,
            "boxes_per_second": len(source_predictions) / global_wall_seconds if global_wall_seconds else 0.0,
            "tokens_per_second": dataset_tokens / global_generation_seconds if global_generation_seconds else 0.0,
            "average_tokens_per_image": dataset_tokens / len(dataset_records) if dataset_records else 0.0,
        }
        metrics = {
            "config": {
                "model": args.model,
                "data": dataset["data"],
                "dataset": name,
                "benchmark": "cd_fsod_zero_shot",
                "support_shots_used": 0,
                "devices": args.devices,
                "global_batch": args.global_batch,
                "local_batch": args.batch,
                "generation_mode": args.generation_mode,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "protocol": args.protocol,
                "protocol_id": (CD_FSOD_CLOSED_SET_PROTOCOL_ID if closed_set else CD_FSOD_PROTOCOL_ID),
                "reference_coco_protocol_id": PAPER_PROTOCOL_ID,
                "short_side": PAPER_SHORT_SIDE,
                "resize_interpolation": "bilinear",
                "prompt_categories": ("all_dataset_categories" if closed_set else "image_gt_positive_oracle"),
                "class_name_policy": "naturalize_underscore_hyphen",
                "world_size": world_size,
            },
            "counts": {
                "images": len(dataset_records),
                "ground_truths": int(official["evaluated_non_crowd_gt"]),
                "boxes": len(source_predictions),
                "unknown_labels": sum(len(record.get("unknown_labels", [])) for record in dataset_records),
                "parse_warning_images": sum(bool(record.get("parse_warnings")) for record in dataset_records),
                "empty_prediction_images": sum(not record.get("predictions") for record in dataset_records),
                "inference_errors": sum(bool(record.get("error")) for record in dataset_records),
                "repetition_stopped_images": sum(
                    bool(record.get("generation_stats", {}).get("stopped_repetition")) for record in dataset_records
                ),
            },
            "official_locate_metrics": official,
            **({"auxiliary_paper_metrics": auxiliary_paper} if auxiliary_paper is not None else {}),
            "nonstandard_constant_score_coco_ap": coco_ap,
            "speed": dataset_speed,
        }
        dataset_payloads[name] = metrics
        (dataset_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        (dataset_dir / "summary.txt").write_text(_dataset_summary_text(name, metrics), encoding="utf-8")
        rows.append(
            {
                "dataset": name,
                "display_name": dataset["display_name"],
                "images": len(dataset_records),
                "ground_truths": metrics["counts"]["ground_truths"],
                "boxes": len(source_predictions),
                "F1_50": official["f1"]["0.50"]["macro"]["f1"],
                "F1_95": official["f1"]["0.95"]["macro"]["f1"],
                "F1_mean": official["f1"]["mean"]["macro"]["f1"],
                "mean_gt_iou": official["mean_gt_iou"],
                "nonstandard_AP50": coco_ap.get("AP50", 0.0),
                "nonstandard_AP50_95": coco_ap.get("AP50_95", 0.0),
                **(
                    {
                        "paper_F1_50": auxiliary_paper["f1"]["0.50"]["macro"]["f1"],
                        "paper_F1_95": auxiliary_paper["f1"]["0.95"]["macro"]["f1"],
                        "paper_F1_mean": auxiliary_paper["f1"]["mean"]["macro"]["f1"],
                    }
                    if auxiliary_paper is not None
                    else {}
                ),
            }
        )

    aggregate = {
        key: float(np.mean([float(row[key]) for row in rows])) if rows else 0.0
        for key in ("F1_50", "F1_95", "F1_mean", "nonstandard_AP50", "nonstandard_AP50_95")
    }
    if args.protocol == CLOSED_SET_PROTOCOL:
        aggregate.update(
            {
                key: float(np.mean([float(row[key]) for row in rows])) if rows else 0.0
                for key in ("paper_F1_50", "paper_F1_95", "paper_F1_mean")
            }
        )
    payload = {
        "config": {
            "model": args.model,
            "revision": args.revision,
            "benchmark": "cd_fsod_zero_shot",
            "datasets": [dataset["id"] for dataset in suite["datasets"]],
            "support_shots_used": 0,
            "protocol": args.protocol,
            "global_batch": args.global_batch,
            "local_batch": args.batch,
            "world_size": world_size,
            "protocol_id": (
                CD_FSOD_CLOSED_SET_PROTOCOL_ID if args.protocol == CLOSED_SET_PROTOCOL else CD_FSOD_PROTOCOL_ID
            ),
            "manifest_sha256": suite["manifest_sha256"],
            "prompt_categories": (
                "all_dataset_categories" if args.protocol == CLOSED_SET_PROTOCOL else "image_gt_positive_oracle"
            ),
            "class_name_policy": "naturalize_underscore_hyphen",
        },
        "counts": {
            "datasets": len(dataset_payloads),
            "images": len(records),
            "boxes": sum(row["boxes"] for row in rows),
            "ground_truths": sum(row["ground_truths"] for row in rows),
        },
        "aggregate": aggregate,
        "datasets": dataset_payloads,
        "speed": suite_speed,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["dataset"])
        writer.writeheader()
        writer.writerows(rows)
    summary = suite_summary_text(payload)
    (output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    LOGGER.info("\n" + summary.rstrip())
    return payload


__all__ = (
    "CD_FSOD_CLOSED_SET_PROTOCOL_ID",
    "CD_FSOD_PROTOCOL_ID",
    "CDFsodMetrics",
    "DEFAULT_CD_FSOD_DATA",
    "LocateAnythingCDFsodValidator",
    "build_category_alias_map",
    "build_cd_fsod_validator",
    "finalize_cd_fsod_results",
    "load_cd_fsod_validation",
    "naturalize_category_name",
    "suite_summary_text",
)
