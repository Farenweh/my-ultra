"""检测数据回归使用的离线图片与 TXT/COCO JSON 标注。"""
import json
from pathlib import Path

from PIL import Image

from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import YAML


def detection_data(root: Path, annotation_format="txt", count=5):
    images, labels = root / "images/train", root / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(count):
        Image.new("RGB", (48, 32), (i * 20, 60, 80)).save(images / f"{i}.png")
        (labels / f"{i}.txt").write_text(f"{i % 2} 0.5 0.5 0.25 0.25\n")
    cfg = {"path": str(root), "train": "images/train", "val": "images/train"}
    if annotation_format == "coco":
        annotation = {
            "images": [{"id": i, "file_name": f"{i}.png", "width": 48, "height": 32} for i in range(count)],
            "categories": [{"id": 1, "name": "甲"}, {"id": 2, "name": "乙"}],
            "annotations": [{"id": i+1, "image_id": i, "category_id": i % 2 + 1,
                             "bbox": [18, 12, 12, 8], "area": 96, "iscrowd": 0} for i in range(count)],
        }
        (root / "instances.json").write_text(json.dumps(annotation))
        cfg["annotations"] = {"train": "instances.json", "val": "instances.json"}
    else:
        cfg["names"] = {0: "甲", 1: "乙"}
    path = root / "data.yaml"
    YAML.save(path, cfg)
    return check_det_dataset(path)
