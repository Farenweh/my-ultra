# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import math
import os
import random
import tempfile
import time
from collections.abc import Sequence
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset, get_worker_info

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, check_file_speeds, get_split_fraction
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, RANK, TQDM
from ultralytics.utils.patches import imread

from .metadata import MMapLabelSequence, NpyPathSequence, load_or_create_image_inventory


class BaseDataset(Dataset):
    """Base dataset class for loading and processing image data.

    This class provides core functionality for loading images, caching, and preparing data for training and inference in
    object detection tasks.

    Attributes:
        img_path (str | list[str]): Path to the folder containing images.
        imgsz (int): Target image size for resizing.
        augment (bool): Whether to apply data augmentation.
        single_cls (bool): Whether to treat all objects as a single class.
        prefix (str): Prefix to print in log messages.
        fraction (float | int): Dataset ratio or image count to use.
        channels (int): Number of channels in the images (1 for grayscale, 3 for color). Color images loaded with OpenCV
            are in BGR channel order.
        cv2_flag (int): OpenCV flag for reading images.
        im_files (list[str]): List of image file paths.
        labels (list[dict]): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        rect (bool): Whether to use rectangular training.
        batch_size (int): Size of batches.
        stride (int): Stride used in the model.
        pad (float): Padding value.
        buffer (list): Buffer for mosaic images.
        max_buffer_length (int): Maximum buffer size.
        ims (list): List of loaded images.
        im_hw0 (list): List of original image dimensions (h, w).
        im_hw (list): List of resized image dimensions (h, w).
        npy_files (list[Path]): List of numpy file paths.
        cache (str | None): Cache setting ('ram', 'disk', or None for no caching).
        transforms (callable): Image transformation function.
        batch_shapes (np.ndarray): Batch shapes for rectangular training.
        batch (np.ndarray): Batch index of each image.

    Methods:
        get_img_files: Read image files from the specified path.
        update_labels: Update labels to include only specified classes.
        load_image: Load an image from the dataset.
        cache_images: Cache images to memory or disk.
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
        check_cache_disk: Check image caching requirements vs available disk space.
        check_cache_ram: Check image caching requirements vs available memory.
        set_rectangle: Sort images by aspect ratio and set batch shapes for rectangular training.
        get_image_and_label: Get and return label information from the dataset.
        update_labels_info: Custom label format method to be implemented by subclasses.
        build_transforms: Build transformation pipeline to be implemented by subclasses.
        get_labels: Get labels method to be implemented by subclasses.
    """

    class _ImageCache:
        """Store images in one contiguous array to preserve copy-on-write sharing between workers."""

        def __init__(self, images: list[np.ndarray]):
            """Pack images and their layouts into contiguous NumPy arrays."""
            self.shapes = np.array([im.shape for im in images])
            self.dtypes = np.array([im.dtype.str for im in images])
            self.offsets = np.concatenate(([0], np.cumsum([im.nbytes for im in images])))
            self.buffer = np.empty(self.offsets[-1], dtype=np.uint8)
            for i, im in enumerate(images):
                self.buffer[self.offsets[i] : self.offsets[i + 1]] = im.reshape(-1).view(np.uint8)
                images[i] = None

        def __getitem__(self, i: int) -> np.ndarray:
            """Return an image view by index."""
            i = range(len(self.shapes))[i]
            return self.buffer[self.offsets[i] : self.offsets[i + 1]].view(self.dtypes[i]).reshape(self.shapes[i])

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
        metadata_cache: str = "auto",
        data_verify: str = "fast",
        data_retries: int = 3,
    ):
        """Initialize BaseDataset with given configuration and options.

        Args:
            img_path (str | list[str]): Path to the folder containing images or list of image paths.
            imgsz (int): Image size for resizing.
            cache (bool | str): Cache images to RAM or disk during training.
            augment (bool): If True, data augmentation is applied.
            hyp (dict[str, Any]): Hyperparameters to apply data augmentation.
            prefix (str): Prefix to print in log messages.
            rect (bool): If True, rectangular training is used.
            batch_size (int): Size of batches.
            stride (int): Stride used in the model.
            pad (float): Padding value.
            single_cls (bool): If True, single class training is used.
            classes (list[int], optional): List of included classes.
            fraction (float | int): Dataset ratio or image count to use.
            channels (int): Number of channels in the images (1 for grayscale, 3 for color). Color images loaded with
                OpenCV are in BGR channel order.
            metadata_cache (str): 元数据缓存策略，可选'auto'、'shared'或显式本地目录。
            data_verify (str): 数据集校验策略，可选'fast'或'full'。
            data_retries (int): 图片读取失败后尝试的替代样本数。
        """
        super().__init__()
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = get_split_fraction(fraction, "train")
        self.channels = channels
        self.metadata_cache = str(metadata_cache)
        self.data_verify = str(data_verify).lower()
        if self.data_verify not in {"fast", "full"}:
            raise ValueError("data_verify必须是'fast'或'full'")
        self.data_retries = max(int(data_retries), 0)
        self._data_error_report: Path | None = None
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels: Sequence[dict[str, Any]] = self.get_labels()
        if isinstance(self.labels, MMapLabelSequence):
            self.im_files = self.labels.im_files
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = NpyPathSequence(self.im_files)
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Read image files from the specified path.

        Args:
            img_path (str | list[str]): Path or list of paths to image directories or files.

        Returns:
            (list[str]): List of image file paths.

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
        """
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    inventory, _ = load_or_create_image_inventory(p, full_verify=self.data_verify == "full")
                    f += inventory
                elif p.is_file():  # file
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in t]  # local to global
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        count = self.fraction if isinstance(self.fraction, int) else round(len(im_files) * self.fraction)
        im_files = im_files[:count] if count < len(im_files) else im_files
        check_file_speeds(im_files, prefix=self.prefix)  # check image read speeds
        return im_files

    def update_labels(self, include_class: list[int] | None) -> None:
        """Update labels to include only specified classes.

        Args:
            include_class (list[int], optional): List of classes to include. If None, all classes are included.
        """
        if isinstance(self.labels, MMapLabelSequence):
            self.labels = self.labels.with_filter(include_class, self.single_cls)
            self.im_files = self.labels.im_files
            return
        if include_class is None and not self.single_cls:
            return
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i].get("keypoints")
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:] = 0

    def load_image(
        self, i: int, rect_mode: bool = True, resize_short: bool = False
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular resizing (long side to imgsz).
            resize_short (bool): Whether to resize the shorter side to imgsz while maintaining aspect ratio. Overrides
                rect_mode when True.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if self.cache == "disk" and fn.exists():  # 仅在启用磁盘图片缓存时读取NPY
                try:
                    im = np.load(fn)
                    npy_channels = im.shape[-1] if im.ndim >= 3 else 1
                    if npy_channels != self.channels:
                        LOGGER.warning(
                            f"{self.prefix}Removing stale *.npy image file {fn} with {npy_channels} channels, expected {self.channels}"
                        )
                        Path(fn).unlink(missing_ok=True)
                        im = imread(f, flags=self.cv2_flag)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                if resize_short:  # resize short side to imgsz while maintaining aspect ratio
                    r = self.imgsz / min(h0, w0)  # ratio
                    if r != 1:  # if sizes are not equal
                        w, h = (math.ceil(w0 * r), self.imgsz) if h0 < w0 else (self.imgsz, math.ceil(h0 * r))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
                else:
                    r = self.imgsz / max(h0, w0)  # ratio
                    if r != 1:  # if sizes are not equal
                        w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

            # Add to buffer if training with augmentations
            if self.augment and self.cache != "ram":
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self) -> None:
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()
        if self.cache == "ram":
            self.ims = self._ImageCache(self.ims)

    def cache_images_to_disk(self, i: int) -> None:
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            try:
                np.save(f.as_posix(), imread(self.im_files[i], flags=self.cv2_flag), allow_pickle=False)
            except Exception as e:
                f.unlink(missing_ok=True)
                LOGGER.warning(f"{self.prefix}WARNING ⚠️ Failed to cache image {f}: {e}")

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough disk space for caching images.

        Args:
            safety_margin (float): Safety margin factor for disk space calculation.

        Returns:
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}Skipping caching images to disk, directory not writable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, _used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 1.0) -> bool:
        """Check if there's enough RAM for caching images.

        Args:
            safety_margin (float): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            b += self.load_image(random.randrange(self.ni))[0].nbytes
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    def set_rectangle(self) -> None:
        """Sort images by aspect ratio and set batch shapes for rectangular training."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = (
            np.asarray(self.labels.shapes_array)
            if isinstance(self.labels, MMapLabelSequence)
            else np.array([x["shape"] for x in self.labels])
        )  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        if isinstance(self.labels, MMapLabelSequence):
            self.labels = self.labels.reordered(irect)
            self.im_files = self.labels.im_files
        else:
            self.im_files = [self.im_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return transformed label information for given index."""
        current = index
        for attempt in range(self.data_retries + 1):
            try:
                return self.transforms(self.get_image_and_label(current))
            except (FileNotFoundError, OSError) as error:
                report = self._record_data_error(current, error, attempt)
                if attempt >= self.data_retries or len(self) <= 1:
                    raise RuntimeError(
                        f"读取数据样本失败，已重试{attempt}次；错误报告：{report}"
                    ) from error
                current = self._replacement_index(current)
        raise RuntimeError("数据样本重试流程异常结束")

    def _replacement_index(self, failed_index: int) -> int:
        """从当前rank对应的近似分布式步长中选择替代样本。"""
        world_size = max(int(os.getenv("WORLD_SIZE", "1")), 1)
        rank = max(int(os.getenv("RANK", str(max(RANK, 0)))), 0) % world_size
        if self.rect and hasattr(self, "batch"):
            same_shape = np.flatnonzero(self.batch == self.batch[failed_index]).tolist()
            rank_candidates = [index for index in same_shape if index % world_size == rank]
            candidates = rank_candidates if len(rank_candidates) > 1 else same_shape
        else:
            candidates = range(rank, len(self), world_size)
        if len(candidates) <= 1:
            return failed_index if self.rect else (failed_index + 1) % len(self)
        replacement = candidates[random.randrange(len(candidates))]
        return replacement if replacement != failed_index else candidates[(candidates.index(replacement) + 1) % len(candidates)]

    def _record_data_error(self, index: int, error: Exception, attempt: int) -> Path:
        """为读取失败样本追加一条worker本地JSONL记录。"""
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        if self._data_error_report is None:
            report_root = Path(tempfile.gettempdir()) / "ultralytics-data-errors"
            report_root.mkdir(parents=True, exist_ok=True)
            self._data_error_report = report_root / f"rank{max(RANK, 0)}-worker{worker_id}-pid{os.getpid()}.jsonl"
        record = {
            "time": time.time(),
            "rank": max(RANK, 0),
            "worker": worker_id,
            "index": int(index),
            "im_file": str(self.im_files[index]) if 0 <= index < len(self.im_files) else "",
            "attempt": int(attempt),
            "error": repr(error),
        }
        with open(self._data_error_report, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        action = (
            f"正在重试（{attempt + 1}/{self.data_retries}）"
            if attempt < self.data_retries
            else "重试次数已耗尽"
        )
        LOGGER.warning(f"{self.prefix}读取样本失败，{action}：{record['im_file']}，{error}")
        return self._data_error_report

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (dict[str, Any]): Label dictionary with image and metadata.
        """
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self) -> int:
        """Return the length of the labels list for the dataset."""
        return len(self.labels)

    def get_class_counts(self, num_classes: int) -> np.ndarray | None:
        """紧凑元数据可用时返回缓存的类别计数。"""
        if not isinstance(self.labels, MMapLabelSequence) or not self.labels.summary_valid:
            return None
        counts = np.asarray(self.labels.summary.get("class_counts", ()), dtype=np.float32)
        counts = np.pad(counts, (0, max(0, num_classes - len(counts))))[:num_classes]
        if self.labels.include_class is not None:
            selected = np.zeros(num_classes, dtype=bool)
            selected[np.asarray(self.labels.include_class, dtype=int).ravel()] = True
            counts[~selected] = 0
        if self.labels.single_cls:
            counts = np.pad(np.array([counts.sum()], dtype=np.float32), (0, max(0, num_classes - 1)))[:num_classes]
        return counts

    def get_plot_labels(self) -> tuple[np.ndarray, np.ndarray] | None:
        """返回紧凑元数据中有上限的绘图抽样。"""
        if not isinstance(self.labels, MMapLabelSequence):
            return None
        root = self.labels.cache_dir
        boxes, cls = np.load(root / "plot_bboxes.npy"), np.load(root / "plot_cls.npy")
        if self.labels.include_class is not None:
            selected = (cls == self.labels.include_class).any(1)
            boxes, cls = boxes[selected], cls[selected]
        if self.labels.single_cls:
            cls[:] = 0
        return boxes, cls

    def get_max_num_obj(self) -> int | None:
        """返回缓存的单张图片最大目标数。"""
        if (
            not isinstance(self.labels, MMapLabelSequence)
            or not self.labels.summary_valid
            or self.labels.include_class is not None
        ):
            return None
        return int(self.labels.summary.get("max_num_obj", 0))

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Customize your label format here."""
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """Users can customize augmentations here.

        Examples:
            >>> if self.augment:
            ...     # Training transforms
            ...     return Compose([])
            >>> else:
            ...    # Val transforms
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> Sequence[dict[str, Any]]:
        """Users can customize their own format here.

        Examples:
            Ensure output is a dictionary with the following keys:
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # format: (height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # xy
            ...     keypoints=keypoints,  # xy
            ...     normalized=True,  # or False
            ...     bbox_format="xyxy",  # or xywh, ltwh
            ... )
        """
        raise NotImplementedError
