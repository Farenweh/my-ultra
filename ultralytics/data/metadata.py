# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""紧凑、可内存映射的数据集元数据缓存。"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from filelock import FileLock

from ultralytics.data.utils import IMG_FORMATS, img2label_paths
from ultralytics.utils import LOGGER, is_dir_writeable

METADATA_CACHE_VERSION = "1.0.0"
IMAGE_INVENTORY_VERSION = "1.1.0"
REMOTE_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "fuse.s3fs",
        "fuse.sshfs",
        "gcsfuse",
        "glusterfs",
        "lustre",
        "nfs",
        "nfs4",
        "panfs",
        "smb3",
        "smbfs",
    }
)


def _stat_signature(path: str | Path) -> tuple[Any, ...]:
    """返回单个路径的低开销身份信息。"""
    path = Path(path)
    try:
        stat = path.stat()
        return (
            str(path.resolve()),
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(stat.st_ctime_ns),
        )
    except OSError:
        return (str(path.absolute()), "missing")


def directory_signature(path: str | Path) -> tuple[int, int, int, int, int]:
    """返回与图片清单1.1.0兼容的目录签名。"""
    stat = Path(path).stat()
    return int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns)


def _json_digest(value: Any) -> str:
    """稳定散列JSON兼容对象。"""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def dataset_source_signature(
    img_path: str | Path | Sequence[str | Path],
    *,
    annotation_file: str | Path | None = None,
    task: str = "detect",
    channels: int = 3,
) -> dict[str, Any]:
    """构造不随样本数增长的缓存源签名。"""
    if isinstance(img_path, (list, tuple)):
        source_count = len(img_path)
        paths = list(img_path) if source_count <= 16 else [img_path[0], img_path[-1]]
    else:
        source_count = 1
        paths = [img_path]
    image_sources = [_stat_signature(path) for path in paths]
    label_sources = []
    if annotation_file is None:
        for source in paths:
            source = Path(source)
            if source.is_dir() or not source.suffix:
                label_root = Path(img2label_paths([str(source / "placeholder.jpg")])[0]).parent
                label_sources.append(_stat_signature(label_root))
            elif source.is_file():
                label_sources.append(_stat_signature(source))
    signature = {
        "schema": METADATA_CACHE_VERSION,
        "task": task,
        "channels": int(channels),
        "image_sources": image_sources,
        "label_sources": label_sources,
        "annotation": _stat_signature(annotation_file) if annotation_file else None,
    }
    if source_count > len(paths):
        signature["image_source_count"] = source_count
    signature["content_id"] = _json_digest(signature)
    return signature


def deterministic_label_cache_path(img_path: str | Path | Sequence[str | Path]) -> Path | None:
    """在扫描图片前推导传统YOLO标签缓存路径。"""
    if isinstance(img_path, (list, tuple)):
        return None if not img_path else deterministic_label_cache_path(img_path[0])
    source = Path(img_path)
    if source.is_file():
        if source.suffix.lower().lstrip(".") in IMG_FORMATS:
            label_file = Path(img2label_paths([str(source)])[0])
            return label_file.parent.with_suffix(".cache")
        return source.with_suffix(".cache")
    placeholder = img2label_paths([str(source / "placeholder.jpg")])[0]
    return Path(placeholder).parent.with_suffix(".cache")


def image_inventory_path(path: str | Path) -> Path:
    """返回目录对应的图片清单缓存路径。"""
    return Path(path).with_suffix(".images.cache")


def _load_numpy_dict(path: Path) -> dict[str, Any]:
    """读取NumPy对象字典，并确保文件句柄及时关闭。"""
    return np.load(str(path), allow_pickle=True).item()


def _inventory_is_valid(cache: dict[str, Any], root: Path, formats: tuple[str, ...]) -> bool:
    """仅检查目录级签名，不访问单个图片。"""
    if (
        cache.get("version") != IMAGE_INVENTORY_VERSION
        or cache.get("root") != str(root.resolve())
        or tuple(cache.get("formats", ())) != formats
        or not isinstance(cache.get("files"), list)
    ):
        return False
    directories = cache.get("directories")
    if not isinstance(directories, dict) or not directories:
        return False
    try:
        return all(directory_signature(root / rel) == tuple(signature) for rel, signature in directories.items())
    except OSError:
        return False


def load_or_create_image_inventory(root: str | Path, *, full_verify: bool = False) -> tuple[list[str], str]:
    """读取或构建目录图片清单，返回绝对路径列表和内容ID。"""
    root = Path(root).resolve()
    cache_path = image_inventory_path(root)
    formats = tuple(sorted(IMG_FORMATS))

    def load_cached() -> tuple[list[str], str] | None:
        """读取有效的1.1.0图片清单。"""
        try:
            cache = _load_numpy_dict(cache_path)
            if _inventory_is_valid(cache, root, formats):
                return [str(root / relative) for relative in cache["files"]], str(cache["content_id"])
        except (FileNotFoundError, ValueError, AttributeError, EOFError, pickle.UnpicklingError):
            pass
        return None

    if not full_verify and (cached := load_cached()) is not None:
        return cached

    # 可写目录中冷启动只允许一个进程递归枚举；只读数据集保持原有扫描回退行为。
    writeable = is_dir_writeable(cache_path.parent)
    lock = FileLock(str(cache_path) + ".lock") if writeable else contextlib.nullcontext()
    with lock:
        if not full_verify and (cached := load_cached()) is not None:
            return cached
        relative_files: list[str] = []
        directories: dict[str, tuple[int, int, int, int, int]] = {}
        for current_root, dirnames, filenames in os.walk(root):
            dirnames.sort()
            filenames.sort()
            current = Path(current_root)
            relative_dir = current.relative_to(root).as_posix()
            directories[relative_dir] = directory_signature(current)
            relative_files.extend(
                (current / name).relative_to(root).as_posix()
                for name in filenames
                if name.rpartition(".")[-1].lower() in IMG_FORMATS
            )
        relative_files.sort()
        if not relative_files:
            return [], ""
        content_id = _json_digest({"root": str(root), "directories": directories, "files": relative_files})
        cache = {
            "version": IMAGE_INVENTORY_VERSION,
            "root": str(root),
            "formats": formats,
            "files": relative_files,
            "directories": directories,
            "content_id": content_id,
        }
        if not writeable:
            return [str(root / relative) for relative in relative_files], content_id
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.tmp")
        try:
            with open(temp_path, "wb") as file:
                np.save(file, cache)
            os.replace(temp_path, cache_path)
        finally:
            temp_path.unlink(missing_ok=True)
        return [str(root / relative) for relative in relative_files], content_id


class MMapLabelSequence(Sequence[dict[str, Any]]):
    """以连续字节和偏移表保存标签，worker只反序列化当前样本。"""

    def __init__(
        self,
        cache_dir: str | Path,
        *,
        order: np.ndarray | None = None,
        include_class: Sequence[int] | None = None,
        single_cls: bool = False,
        summary_valid: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.order = order
        self.include_class = None if include_class is None else np.asarray(include_class).reshape(1, -1)
        self.single_cls = bool(single_cls)
        self.summary_valid = bool(summary_valid)
        self._open()

    def _open(self) -> None:
        self.manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("version") != METADATA_CACHE_VERSION:
            raise ValueError(f"不支持的元数据缓存版本：{self.manifest.get('version')}")
        num_samples = int(self.manifest["num_samples"])
        self._path_offsets = np.load(self.cache_dir / "path_offsets.npy", mmap_mode="r")
        self._record_offsets = np.load(self.cache_dir / "record_offsets.npy", mmap_mode="r")
        self._shapes = np.load(self.cache_dir / "shapes.npy", mmap_mode="r")
        if (
            self._path_offsets.shape != (num_samples + 1,)
            or self._record_offsets.shape != (num_samples + 1,)
            or self._shapes.shape != (num_samples, 2)
        ):
            raise ValueError(f"元数据缓存数组形状不一致：{self.cache_dir}")
        if int(self._path_offsets[-1]) != (self.cache_dir / "paths.bin").stat().st_size:
            raise ValueError(f"元数据路径字节长度不一致：{self.cache_dir}")
        if int(self._record_offsets[-1]) != (self.cache_dir / "records.bin").stat().st_size:
            raise ValueError(f"元数据标签字节长度不一致：{self.cache_dir}")
        self._paths = np.memmap(self.cache_dir / "paths.bin", dtype=np.uint8, mode="r")
        self._records = np.memmap(self.cache_dir / "records.bin", dtype=np.uint8, mode="r")

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        for key in ("manifest", "_path_offsets", "_record_offsets", "_shapes", "_paths", "_records"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._open()

    def _physical_index(self, index: int) -> int:
        index = range(len(self))[index]
        return int(self.order[index]) if self.order is not None else index

    def __len__(self) -> int:
        return len(self.order) if self.order is not None else int(self.manifest["num_samples"])

    def path(self, index: int) -> str:
        physical = self._physical_index(index)
        start, end = int(self._path_offsets[physical]), int(self._path_offsets[physical + 1])
        return bytes(self._paths[start:end]).decode("utf-8")

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        physical = self._physical_index(index)
        start, end = int(self._record_offsets[physical]), int(self._record_offsets[physical + 1])
        # 缓存仅由本地代码生成，不接受网络载荷。
        label = pickle.loads(memoryview(self._records[start:end]))  # noqa: S301
        label["im_file"] = self.path(index)
        label["shape"] = tuple(int(x) for x in self._shapes[physical])
        if self.include_class is not None:
            cls = label["cls"]
            selected = (cls == self.include_class).any(1)
            label["cls"] = cls[selected]
            label["bboxes"] = label["bboxes"][selected]
            if label.get("segments"):
                label["segments"] = [segment for segment, keep in zip(label["segments"], selected) if keep]
            if label.get("keypoints") is not None:
                label["keypoints"] = label["keypoints"][selected]
        if self.single_cls:
            label["cls"][:] = 0
        return label

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]

    @property
    def im_files(self) -> "MMapPathSequence":
        return MMapPathSequence(self)

    @property
    def shapes_array(self) -> np.ndarray:
        if self.order is None:
            return self._shapes
        return self._shapes[self.order]

    @property
    def summary(self) -> dict[str, Any]:
        return self.manifest.get("summary", {})

    def with_filter(self, include_class: Sequence[int] | None, single_cls: bool) -> "MMapLabelSequence":
        return MMapLabelSequence(
            self.cache_dir,
            order=self.order,
            include_class=include_class,
            single_cls=single_cls,
            summary_valid=self.summary_valid,
        )

    def reordered(self, order: np.ndarray) -> "MMapLabelSequence":
        physical = order if self.order is None else self.order[order]
        return MMapLabelSequence(
            self.cache_dir,
            order=np.asarray(physical, dtype=np.int64),
            include_class=None if self.include_class is None else self.include_class.ravel(),
            single_cls=self.single_cls,
            summary_valid=self.summary_valid,
        )

    def subset(self, count: int) -> "MMapLabelSequence":
        count = max(0, min(int(count), len(self)))
        order = np.arange(count, dtype=np.int64) if self.order is None else self.order[:count].copy()
        return MMapLabelSequence(
            self.cache_dir,
            order=order,
            include_class=None if self.include_class is None else self.include_class.ravel(),
            single_cls=self.single_cls,
            summary_valid=self.summary_valid and count == int(self.manifest["num_samples"]),
        )


class MMapPathSequence(Sequence[str]):
    """标签存储中的惰性图片路径视图。"""

    def __init__(self, labels: MMapLabelSequence) -> None:
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int | slice) -> str | list[str]:
        if isinstance(index, slice):
            return [self.labels.path(i) for i in range(*index.indices(len(self)))]
        return self.labels.path(index)

    def __iter__(self) -> Iterator[str]:
        for index in range(len(self)):
            yield self.labels.path(index)


class NpyPathSequence(Sequence[Path]):
    """按需推导图片对应的NPY路径。"""

    def __init__(self, image_files: Sequence[str]) -> None:
        self.image_files = image_files

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int | slice) -> Path | list[Path]:
        if isinstance(index, slice):
            return [Path(path).with_suffix(".npy") for path in self.image_files[index]]
        return Path(self.image_files[index]).with_suffix(".npy")


def write_metadata_store(
    cache_dir: str | Path,
    labels: Iterable[dict[str, Any]],
    *,
    num_samples: int,
    source_signature: dict[str, Any],
) -> Path:
    """以原子方式写入紧凑元数据缓存。"""
    cache_dir = Path(cache_dir)
    if (cache_dir / "manifest.json").is_file():
        return cache_dir
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(cache_dir) + ".lock")
    with lock:
        if (cache_dir / "manifest.json").is_file():
            return cache_dir
        temp_dir = Path(tempfile.mkdtemp(prefix=f".{cache_dir.name}.", dir=cache_dir.parent))
        path_offsets = np.lib.format.open_memmap(
            temp_dir / "path_offsets.npy", mode="w+", dtype=np.int64, shape=(num_samples + 1,)
        )
        record_offsets = np.lib.format.open_memmap(
            temp_dir / "record_offsets.npy", mode="w+", dtype=np.int64, shape=(num_samples + 1,)
        )
        shapes = np.lib.format.open_memmap(
            temp_dir / "shapes.npy", mode="w+", dtype=np.int32, shape=(num_samples, 2)
        )
        class_counts: np.ndarray | None = None
        text_counts: dict[str, int] = {}
        max_num_obj = 0
        plot_boxes: list[np.ndarray] = []
        plot_classes: list[np.ndarray] = []
        plot_count = 0
        try:
            with open(temp_dir / "paths.bin", "wb") as path_file, open(temp_dir / "records.bin", "wb") as record_file:
                path_offsets[0] = record_offsets[0] = 0
                actual = 0
                for actual, source_label in enumerate(labels, start=1):
                    if actual > num_samples:
                        raise ValueError("标签数量超过声明的样本数")
                    label = dict(source_label)
                    image_path = str(label.pop("im_file"))
                    shape = label.pop("shape")
                    path_file.write(image_path.encode("utf-8"))
                    path_offsets[actual] = path_file.tell()
                    payload = pickle.dumps(label, protocol=pickle.HIGHEST_PROTOCOL)
                    record_file.write(payload)
                    record_offsets[actual] = record_file.tell()
                    shapes[actual - 1] = shape

                    cls = np.asarray(label.get("cls", ()), dtype=np.int64).reshape(-1)
                    max_num_obj = max(max_num_obj, len(cls))
                    if cls.size:
                        needed = int(cls.max()) + 1
                        if class_counts is None:
                            class_counts = np.zeros(needed, dtype=np.int64)
                        elif needed > len(class_counts):
                            class_counts = np.pad(class_counts, (0, needed - len(class_counts)))
                        class_counts += np.bincount(cls, minlength=len(class_counts))
                    for text_group in label.get("texts", ()):
                        for text in text_group:
                            normalized_text = str(text).strip()
                            text_counts[normalized_text] = text_counts.get(normalized_text, 0) + 1
                    if plot_count < 100_000 and len(label.get("bboxes", ())):
                        take = min(100_000 - plot_count, len(label["bboxes"]))
                        plot_boxes.append(np.asarray(label["bboxes"][:take], dtype=np.float32))
                        plot_classes.append(np.asarray(label["cls"][:take], dtype=np.float32))
                        plot_count += take
                if actual != num_samples:
                    raise ValueError(f"标签数量不一致：期望{num_samples}，实际{actual}")
            path_offsets.flush()
            record_offsets.flush()
            shapes.flush()
            np.save(temp_dir / "plot_bboxes.npy", np.concatenate(plot_boxes) if plot_boxes else np.zeros((0, 4)))
            np.save(temp_dir / "plot_cls.npy", np.concatenate(plot_classes) if plot_classes else np.zeros((0, 1)))
            manifest = {
                "version": METADATA_CACHE_VERSION,
                "num_samples": num_samples,
                "source_signature": source_signature,
                "content_id": source_signature["content_id"],
                "summary": {
                    "class_counts": (class_counts.tolist() if class_counts is not None else []),
                    "max_num_obj": max_num_obj,
                    "plot_samples": plot_count,
                    "text_counts": text_counts,
                },
            }
            (temp_dir / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
            )
            try:
                os.replace(temp_dir, cache_dir)
            except OSError:
                if not (cache_dir / "manifest.json").is_file():
                    raise
        finally:
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
    return cache_dir


def migrate_legacy_metadata_store(
    legacy_path: str | Path,
    cache_dir: str | Path,
    source_signature: dict[str, Any],
    expected_version: str,
) -> bool:
    """在独立进程中把传统NumPy对象缓存转换为紧凑缓存。"""
    import gc

    gc.disable()
    try:
        legacy = np.load(str(legacy_path), allow_pickle=True).item()
    finally:
        gc.enable()
    if not str(legacy.get("version", "")).startswith(expected_version) or not legacy.get("labels"):
        return False
    write_metadata_store(
        cache_dir,
        legacy["labels"],
        num_samples=len(legacy["labels"]),
        source_signature=source_signature,
    )
    return True


def shared_metadata_dir(cache_path: str | Path, source_signature: dict[str, Any]) -> Path:
    """返回源数据旁的版本化紧凑缓存目录。"""
    cache_path = Path(cache_path)
    return cache_path.with_suffix(".metadata") / source_signature["content_id"]


def _mount_filesystem_type(path: Path) -> str | None:
    """在Linux上解析覆盖目标路径的最长挂载点。"""
    try:
        resolved = path.resolve()
        best: tuple[int, str] | None = None
        for line in Path("/proc/mounts").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            mount = Path(fields[1].replace("\\040", " ")).resolve()
            try:
                resolved.relative_to(mount)
            except ValueError:
                continue
            candidate = (len(str(mount)), fields[2].lower())
            if best is None or candidate[0] > best[0]:
                best = candidate
        return None if best is None else best[1]
    except OSError:
        return None


def is_remote_filesystem(path: str | Path) -> bool:
    """判断路径是否位于已知远程文件系统。"""
    filesystem = _mount_filesystem_type(Path(path))
    return bool(filesystem and (filesystem in REMOTE_FILESYSTEMS or filesystem.startswith("fuse.")))


def _directory_size(path: Path) -> int:
    """统计紧凑缓存中少量文件的总大小。"""
    return sum(file.stat().st_size for file in path.iterdir() if file.is_file())


def stage_metadata_cache(source_dir: str | Path, policy: str = "auto") -> Path:
    """按策略把共享元数据缓存暂存到节点本地。"""
    source_dir = Path(source_dir)
    if policy in {"", "shared", "none", "false"}:
        return source_dir
    if policy == "auto" and not is_remote_filesystem(source_dir):
        return source_dir
    if policy == "auto":
        uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
        root = Path(
            os.environ.get(
                "ULTRALYTICS_DATA_CACHE_DIR", Path(tempfile.gettempdir()) / f"ultralytics-dataset-cache-{uid}"
            )
        )
    else:
        root = Path(policy).expanduser()
    target = root / source_dir.name
    if (target / "manifest.json").is_file():
        return target
    try:
        root.mkdir(parents=True, exist_ok=True)
        required = _directory_size(source_dir)
        if shutil.disk_usage(root).free < int(required * 1.1):
            LOGGER.warning(f"本地元数据缓存空间不足，继续使用共享缓存：{source_dir}")
            return source_dir
        with FileLock(str(target) + ".lock"):
            if not (target / "manifest.json").is_file():
                temp_dir = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=root))
                try:
                    shutil.copytree(source_dir, temp_dir, dirs_exist_ok=True)
                    os.replace(temp_dir, target)
                finally:
                    if temp_dir.exists():
                        shutil.rmtree(temp_dir, ignore_errors=True)
        return target
    except OSError as error:
        LOGGER.warning(f"暂存节点本地元数据缓存失败，继续使用共享缓存：{error}")
        return source_dir


def load_metadata_store(source_dir: str | Path, policy: str = "auto") -> MMapLabelSequence:
    """按缓存策略加载紧凑标签序列。"""
    source_dir = Path(source_dir)
    staged_dir = stage_metadata_cache(source_dir, policy)
    try:
        return MMapLabelSequence(staged_dir)
    except (OSError, ValueError, KeyError, EOFError, pickle.UnpicklingError) as error:
        if staged_dir == source_dir:
            raise
        LOGGER.warning(f"节点本地元数据缓存损坏，回退共享缓存：{error}")
        with FileLock(str(staged_dir) + ".lock"):
            if staged_dir.exists():
                backup = staged_dir.with_name(f"{staged_dir.name}.corrupt-{time.time_ns()}")
                staged_dir.replace(backup)
                LOGGER.warning(f"损坏的节点本地缓存已保留到：{backup}")
        return MMapLabelSequence(source_dir)
