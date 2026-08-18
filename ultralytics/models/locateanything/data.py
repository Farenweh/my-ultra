# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""LocateAnything官方JSONL与YOLO检测数据适配。"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import DEFAULT_CFG


class LazyJsonl:
    """按字节偏移读取大型JSONL文件。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"LocateAnything标注文件不存在：{self.path}")
        self.offsets: list[int] = []
        offset = 0
        with self.path.open("rb") as file:
            while line := file.readline():
                if line.strip():
                    self.offsets.append(offset)
                offset = file.tell()

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as file:
            file.seek(self.offsets[index])
            try:
                return json.loads(file.readline().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"{self.path}第{index + 1}条非空记录不是合法JSON：{error}") from error


@dataclass(frozen=True)
class _RecipeEntry:
    loader: LazyJsonl
    index: int
    root: Path
    data_augment: bool


class ConversationDataset(Dataset):
    """官方ShareGPT JSONL recipe数据集。"""

    def __init__(self, data: str | Path, *, seed: int = 0) -> None:
        path = Path(data)
        self.seed = seed
        if path.suffix.lower() == ".jsonl":
            recipe = {path.stem: {"annotation": str(path), "root": "", "repeat_time": 1.0}}
            recipe_dir = path.parent
        else:
            with path.open(encoding="utf-8") as file:
                recipe = json.load(file)
            recipe_dir = path.parent
        if not isinstance(recipe, dict) or not recipe:
            raise ValueError("LocateAnything recipe必须是非空JSON对象")

        self.entries: list[_RecipeEntry] = []
        for name, spec in recipe.items():
            if not isinstance(spec, dict) or "annotation" not in spec:
                raise ValueError(f"recipe数据集{name!r}缺少annotation")
            annotations = spec["annotation"] if isinstance(spec["annotation"], list) else [spec["annotation"]]
            root = _resolve_path(recipe_dir, spec.get("root", ""), allow_empty=True)
            repeat_time = float(spec.get("repeat_time", 1.0))
            if repeat_time <= 0:
                raise ValueError(f"recipe数据集{name!r}的repeat_time必须大于0")
            for annotation in annotations:
                loader = LazyJsonl(_resolve_path(recipe_dir, annotation))
                indices = list(range(len(loader)))
                whole = int(repeat_time)
                fraction = repeat_time - whole
                selected = indices * whole
                if fraction:
                    count = int(round(len(indices) * fraction))
                    selected.extend(
                        random.Random(f"{seed}:{name}:{annotation}").sample(indices, min(count, len(indices)))
                    )
                if repeat_time < 1:
                    count = int(round(len(indices) * repeat_time))
                    selected = random.Random(f"{seed}:{name}:{annotation}").sample(indices, min(count, len(indices)))
                self.entries.extend(
                    _RecipeEntry(loader, index, root, bool(spec.get("data_augment", False))) for index in selected
                )
        if not self.entries:
            raise ValueError("LocateAnything recipe没有可用样本")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[index]
        sample = entry.loader[entry.index]
        if "video" in sample or "video_list" in sample:
            raise ValueError(f"首版LocateAnything训练不支持视频：{entry.loader.path}记录{entry.index + 1}")
        conversations = sample.get("conversations")
        if not isinstance(conversations, list) or not conversations:
            raise ValueError(f"{entry.loader.path}记录{entry.index + 1}缺少非空conversations")
        for turn in conversations:
            if (
                not isinstance(turn, dict)
                or turn.get("from") not in {"human", "gpt"}
                or not isinstance(turn.get("value"), str)
            ):
                raise ValueError(f"{entry.loader.path}记录{entry.index + 1}含非法conversation turn")
        media = sample.get("image_list")
        if media is None and sample.get("image") is not None:
            media = [sample["image"]]
        media = media or []
        if not isinstance(media, list) or not all(isinstance(item, (str, Path)) for item in media):
            raise ValueError(f"{entry.loader.path}记录{entry.index + 1}的image/image_list格式非法")
        images = [str(_resolve_path(entry.root, item)) for item in media]
        return {
            "conversations": conversations,
            "images": images,
            "data_augment": entry.data_augment,
            "source": f"{entry.loader.path}:{entry.index + 1}",
        }


class YOLOConversationDataset(Dataset):
    """将Ultralytics检测数据在线转换为定位对话。"""

    def __init__(
        self,
        data: str | Path,
        *,
        seed: int = 0,
        negative_ratio: float = 1.0,
        max_negative_classes: int = 32,
    ) -> None:
        if negative_ratio < 0:
            raise ValueError("negative_ratio不能为负数")
        dataset_info = check_det_dataset(str(data))
        self.names = {int(k): str(v).split("/", 1)[0].strip() for k, v in dataset_info["names"].items()}
        self.seed = seed
        self.negative_ratio = negative_ratio
        self.max_negative_classes = max_negative_classes
        self.dataset = YOLODataset(
            img_path=dataset_info["train"],
            data=dataset_info,
            task="detect",
            augment=False,
            hyp=DEFAULT_CFG,
            batch_size=1,
        )

    def __len__(self) -> int:
        return len(self.dataset.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        label = self.dataset.labels[index]
        return format_yolo_conversation(
            label["im_file"],
            label["cls"],
            label["bboxes"],
            self.names,
            seed=self.seed + index,
            negative_ratio=self.negative_ratio,
            max_negative_classes=self.max_negative_classes,
        )


def format_yolo_conversation(
    image: str | Path,
    classes: np.ndarray,
    bboxes: np.ndarray,
    names: dict[int, str],
    *,
    seed: int = 0,
    negative_ratio: float = 1.0,
    max_negative_classes: int = 32,
) -> dict[str, Any]:
    """把单张YOLO标签转换为确定性的LocateAnything训练样本。"""
    cls_ids = [int(x) for x in np.asarray(classes).reshape(-1)]
    boxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
    if len(cls_ids) != len(boxes):
        raise ValueError("YOLO类别数与定位框数量不一致")
    unknown = sorted(set(cls_ids) - set(names))
    if unknown:
        raise ValueError(f"YOLO标签引用未定义类别：{unknown}")

    positive_ids = sorted(set(cls_ids))
    negative_pool = sorted(set(names) - set(positive_ids))
    if positive_ids:
        negative_count = min(
            math.ceil(len(positive_ids) * negative_ratio),
            len(positive_ids),
            max_negative_classes,
            len(negative_pool),
        )
    else:
        negative_count = min(1, len(negative_pool))
    negatives = random.Random(seed).sample(negative_pool, negative_count) if negative_count else []
    query_ids = sorted([*positive_ids, *negatives])
    if not query_ids and names:
        query_ids = [min(names)]
    query = "</c>".join(names[i] for i in query_ids)
    prompt = f"Locate all the instances that matches the following description: {query}."

    response_parts = []
    ordered = sorted(range(len(cls_ids)), key=lambda i: (cls_ids[i], float(boxes[i, 0]), float(boxes[i, 1]), i))
    for i in ordered:
        cx, cy, width, height = boxes[i]
        coords = np.clip(
            [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
            0.0,
            1.0,
        )
        tokens = "".join(f"<{int(round(float(value) * 1000))}>" for value in coords)
        response_parts.append(f"<ref>{names[cls_ids[i]]}</ref><box>{tokens}</box>")
    response = "".join(response_parts) if response_parts else "<box>none</box>"
    return {
        "conversations": [
            {"from": "human", "value": f"<image-1>{prompt}"},
            {"from": "gpt", "value": response},
        ],
        "images": [str(image)],
        "data_augment": False,
        "source": str(image),
    }


def build_locate_dataset(data: str | Path, **kwargs: Any) -> Dataset:
    """根据扩展名创建官方recipe或YOLO检测适配数据集。"""
    suffix = Path(data).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return YOLOConversationDataset(data, **kwargs)
    if suffix in {".json", ".jsonl"}:
        allowed = {"seed": kwargs.get("seed", 0)}
        return ConversationDataset(data, **allowed)
    raise ValueError("data必须是YOLO .yaml/.yml、官方recipe .json或标注 .jsonl")


def encode_training_sample(
    sample: dict[str, Any],
    processor: Any,
    *,
    max_seq_length: int = 4096,
    block_size: int = 6,
) -> dict[str, torch.Tensor]:
    """把一条规范对话编码成含PBD block监督的模型输入。"""
    if block_size != 6:
        raise ValueError("当前LocateAnything权重要求block_size=6")
    messages = _conversation_messages(sample)
    message_text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    sample_seed = int.from_bytes(sha256(str(sample.get("source", message_text)).encode()).digest()[:8], "big")
    image_inputs, video_inputs = processor.process_vision_info(messages)
    if image_inputs and sample.get("data_augment", False):
        rng = random.Random(sample_seed)
        image_inputs = [_resize_augmentation(image, rng) for image in image_inputs]
    inputs = processor(
        text=[message_text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
    )
    input_ids = inputs["input_ids"][0]
    labels = build_pbd_labels(
        input_ids,
        processor.tokenizer,
        block_size=block_size,
        max_length=max_seq_length,
        seed=sample_seed,
    )

    if "pixel_values" in inputs:
        pixel_values = inputs["pixel_values"]
        image_grid_hws = torch.as_tensor(inputs["image_grid_hws"], dtype=torch.int32)
        image_flags = torch.tensor([len(image_grid_hws)], dtype=torch.long)
        _validate_image_alignment(input_ids, pixel_values, image_grid_hws, processor)
    else:
        pixel_values = torch.zeros((4, 3, 14, 14), dtype=torch.float32)
        image_grid_hws = torch.tensor([[2, 2]], dtype=torch.int32)
        image_flags = torch.tensor([0], dtype=torch.long)
    return {
        **labels,
        "pixel_values": pixel_values,
        "image_grid_hws": image_grid_hws,
        "image_flags": image_flags,
    }


def build_pbd_labels(
    input_ids: torch.Tensor,
    tokenizer: Any,
    *,
    block_size: int = 6,
    max_length: int = 4096,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """生成普通next-token标签和追加的box-aligned MTP block。"""
    ignore_id = -100
    ids = input_ids.detach().cpu().long()
    targets = torch.full_like(ids, ignore_id)
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_ids = tokenizer.encode("assistant", add_special_tokens=False)
    if len(assistant_ids) != 1:
        raise ValueError("LocateAnything tokenizer中的'assistant'必须是单token")
    assistant_id = assistant_ids[0]

    starts = set((torch.where(ids == im_start_id)[0] + 1).tolist())
    response_ranges: list[tuple[int, int, int, int]] = []
    for assistant_index in torch.where(ids == assistant_id)[0].tolist():
        if assistant_index not in starts:
            continue
        content_start = assistant_index + 2
        ends = torch.where(ids[content_start:] == im_end_id)[0]
        if not len(ends):
            continue
        content_end = content_start + int(ends[0])
        targets[content_start : content_end + 1] = ids[content_start : content_end + 1]
        response_ranges.append((content_start - 1, content_end + 1, content_start, content_end))
    if not response_ranges:
        raise ValueError("训练样本未找到assistant监督区间")

    mask_id = tokenizer.convert_tokens_to_ids("<text_mask>")
    null_id = tokenizer.convert_tokens_to_ids("<null>")
    box_end_id = tokenizer.convert_tokens_to_ids("</box>")
    ref_end_id = tokenizer.convert_tokens_to_ids("</ref>")
    mask_blocks: list[torch.Tensor] = []
    target_blocks: list[torch.Tensor] = []
    position_blocks: list[torch.Tensor] = []
    available_blocks = max((max_length - len(ids) - 1) // block_size, 0)

    has_structured_output = bool(torch.any(ids == box_end_id) or torch.any(ids == ref_end_id))
    if has_structured_output:
        for start, end, _, _ in response_ranges:
            current = start
            while current < end and len(mask_blocks) < available_blocks:
                if ids[current] == im_end_id:
                    break
                candidates = ids[current + 1 : min(current + 1 + block_size, end + 1)]
                if not len(candidates):
                    break
                valid_len = len(candidates)
                eos = torch.where(candidates == im_end_id)[0]
                if len(eos):
                    first_eos = int(eos[0])
                    valid_len = 1 if first_eos == 0 else first_eos
                if valid_len > 1 or (len(eos) and int(eos[0]) != 0):
                    for boundary in (ref_end_id, box_end_id):
                        boundary_positions = torch.where(candidates[:valid_len] == boundary)[0]
                        if len(boundary_positions):
                            valid_len = min(valid_len, int(boundary_positions[0]) + 1)
                _append_pbd_block(
                    ids,
                    current,
                    candidates[:valid_len],
                    block_size,
                    mask_id,
                    null_id,
                    mask_blocks,
                    target_blocks,
                    position_blocks,
                )
                current += valid_len
    else:
        rng = random.Random(seed)
        for _, _, label_start, label_end in response_ranges:
            valid_positions = [index for index in range(label_start, label_end + 1) if targets[index] != ignore_id]
            num_blocks = min(len(valid_positions) // block_size, available_blocks - len(mask_blocks))
            if num_blocks <= 0:
                continue
            remaining = len(valid_positions) - num_blocks * block_size
            offset = rng.randint(0, remaining) if remaining else 0
            for block_index in range(num_blocks):
                block_positions = valid_positions[
                    offset + block_index * block_size : offset + (block_index + 1) * block_size
                ]
                anchor = max(block_positions[0] - 1, 0)
                candidates = ids[block_positions]
                target_block = torch.full((block_size,), ignore_id, dtype=torch.long)
                target_block[: len(candidates)] = candidates
                mask_block = torch.full((block_size,), mask_id, dtype=torch.long)
                mask_block[0] = ids[anchor]
                mask_blocks.append(mask_block)
                target_blocks.append(target_block)
                position_blocks.append(torch.arange(anchor, anchor + block_size, dtype=torch.long))

    positions = torch.arange(len(ids), dtype=torch.long)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if mask_blocks:
        mask_tensor = torch.cat(mask_blocks)
        target_tensor = torch.cat(target_blocks)
        position_tensor = torch.cat(position_blocks)
        ids = torch.cat((ids, mask_tensor, torch.tensor([pad_id], dtype=torch.long)))
        targets = torch.cat((targets, torch.tensor([ignore_id], dtype=torch.long), target_tensor))
        positions = torch.cat((positions, position_tensor, torch.tensor([int(position_tensor[-1]) + 1])))
    return {
        "input_ids": ids,
        "labels": targets,
        "position_ids": positions,
        "attention_mask": ids.ne(pad_id),
    }


def _append_pbd_block(
    ids: torch.Tensor,
    anchor: int,
    candidates: torch.Tensor,
    block_size: int,
    mask_id: int,
    null_id: int,
    mask_blocks: list[torch.Tensor],
    target_blocks: list[torch.Tensor],
    position_blocks: list[torch.Tensor],
) -> None:
    """追加一个与官方box/ref边界对齐的PBD监督块。"""
    mask_block = torch.full((block_size,), mask_id, dtype=torch.long)
    mask_block[0] = ids[anchor]
    target_block = torch.full((block_size,), null_id, dtype=torch.long)
    target_block[: len(candidates)] = candidates
    mask_blocks.append(mask_block)
    target_blocks.append(target_block)
    position_blocks.append(torch.arange(anchor, anchor + block_size, dtype=torch.long))


class LocateAnythingCollator:
    """编码并填充LocateAnything训练batch。"""

    def __init__(self, processor: Any, *, max_seq_length: int = 4096, block_size: int = 6) -> None:
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.block_size = block_size

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [
            encode_training_sample(
                sample,
                self.processor,
                max_seq_length=self.max_seq_length,
                block_size=self.block_size,
            )
            for sample in samples
        ]
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.processor.tokenizer.eos_token_id
        max_len = max(len(item["input_ids"]) for item in encoded)

        def pad(key: str, value: int) -> torch.Tensor:
            rows = []
            for item in encoded:
                tensor = item[key]
                rows.append(torch.nn.functional.pad(tensor, (0, max_len - len(tensor)), value=value))
            return torch.stack(rows)

        return {
            "input_ids": pad("input_ids", pad_id),
            "labels": pad("labels", -100),
            "position_ids": pad("position_ids", 0),
            "attention_mask": pad("attention_mask", 0).bool(),
            "pixel_values": torch.cat([item["pixel_values"] for item in encoded]),
            "image_grid_hws": torch.cat([item["image_grid_hws"] for item in encoded]),
            "image_flags": torch.cat([item["image_flags"] for item in encoded]),
        }


def _conversation_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """将ShareGPT turn和图片路径转换为Processor消息。"""
    messages = []
    images = list(sample.get("images") or [])
    image_items = [{"type": "image", "image": image} for image in images]
    attached = False
    for turn in sample["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        if role == "user" and image_items and not attached:
            content: Any = [*image_items, {"type": "text", "text": turn["value"]}]
            attached = True
        else:
            content = turn["value"]
        messages.append({"role": role, "content": content})
    return messages


def _resize_augmentation(image: Image.Image, rng: random.Random) -> Image.Image:
    """按官方recipe语义以50%概率改变长边，保持宽高比。"""
    if rng.random() < 0.5:
        return image
    width, height = image.size
    target = rng.randint(640, 2560)
    scale = target / max(width, height)
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)


def _validate_image_alignment(
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor,
    image_grid_hws: torch.Tensor,
    processor: Any,
) -> None:
    """在进入模型前验证图片patch数和占位token数严格一致。"""
    merge_height, merge_width = processor.image_processor.merge_kernel_size
    expected_tokens = sum(int(height) * int(width) // (merge_height * merge_width) for height, width in image_grid_hws)
    actual_tokens = int((input_ids == processor.image_token_id).sum())
    if actual_tokens != expected_tokens:
        raise ValueError(f"图片占位token数量不一致：actual={actual_tokens}, expected={expected_tokens}")
    expected_patches = sum(int(height) * int(width) for height, width in image_grid_hws)
    if len(pixel_values) != expected_patches:
        raise ValueError(f"图片patch数量不一致：actual={len(pixel_values)}, expected={expected_patches}")


def _resolve_path(base: Path, value: str | Path, *, allow_empty: bool = False) -> Path:
    """相对recipe目录解析路径。"""
    if value in {"", None} and allow_empty:
        return base
    path = Path(value)
    return path if path.is_absolute() else base / path


__all__ = (
    "ConversationDataset",
    "LazyJsonl",
    "YOLOConversationDataset",
    "build_locate_dataset",
    "build_pbd_labels",
    "encode_training_sample",
    "format_yolo_conversation",
    "LocateAnythingCollator",
)
