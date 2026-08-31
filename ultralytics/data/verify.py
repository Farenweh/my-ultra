# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""数据集完整性校验API。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG


def verify_dataset(
    data: str | Path,
    *,
    mode: str = "full",
    splits: str | list[str] | tuple[str, ...] | None = None,
    metadata_cache: str = "auto",
) -> dict[str, dict[str, Any]]:
    """校验检测数据集并返回各划分的样本摘要。

    Args:
        data: 数据集YAML路径或已注册的数据集名称。
        mode: ``fast``校验O(1)源指纹，``full``扫描图片和标签对。
        splits: 单个划分、划分序列；为None时校验所有已配置划分。
        metadata_cache: 传递给数据集的元数据缓存策略。

    Returns:
        数据集划分名称到校验摘要的映射。
    """
    mode = str(mode).lower()
    if mode not in {"fast", "full"}:
        raise ValueError("mode必须是'fast'或'full'")
    dataset_info = check_det_dataset(str(data))
    if splits is None:
        selected = [split for split in ("train", "val", "test", "minival") if dataset_info.get(split)]
    elif isinstance(splits, str):
        selected = [item.strip() for item in splits.split(",") if item.strip()]
    else:
        selected = list(splits)
    if not selected:
        raise ValueError("数据集没有可校验的已配置划分")

    cfg = get_cfg(
        DEFAULT_CFG,
        overrides={
            "task": "detect",
            "imgsz": 32,
            "cache": False,
            "fraction": 1.0,
            "data_verify": mode,
            "metadata_cache": metadata_cache,
        },
    )
    summaries: dict[str, dict[str, Any]] = {}
    for split in selected:
        if not dataset_info.get(split):
            raise FileNotFoundError(f"数据集未配置'{split}'划分")
        dataset = build_yolo_dataset(
            cfg,
            dataset_info[split],
            batch=1,
            data=dataset_info,
            mode="train" if split == "train" else split,
        )
        summaries[split] = {
            "mode": mode,
            "images": len(dataset),
            "metadata_cache": str(getattr(getattr(dataset, "labels", None), "cache_dir", "memory")),
        }
    return summaries
