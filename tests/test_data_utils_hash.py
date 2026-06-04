from __future__ import annotations

import hashlib
import os

from ultralytics.data import utils as data_utils


def _sequential_hash(paths: list[str]) -> str:
    size = sum(os.stat(path).st_size if os.path.exists(path) else 0 for path in paths)
    digest = hashlib.sha256(str(size).encode())
    digest.update("".join(paths).encode())
    return digest.hexdigest()


def test_get_hash_matches_sequential_result_with_missing_file(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    missing = tmp_path / "missing.txt"
    first.write_bytes(b"a" * 7)
    second.write_bytes(b"b" * 13)
    paths = [str(first), str(missing), str(second)]

    assert data_utils.get_hash(paths) == _sequential_hash(paths)
    assert data_utils.get_hash(iter(paths)) == _sequential_hash(paths)


def test_get_hash_handles_empty_paths():
    assert data_utils.get_hash([]) == _sequential_hash([])


def test_get_hash_threads_use_half_visible_cpu_threads(monkeypatch):
    monkeypatch.setattr(data_utils.os, "sched_getaffinity", lambda pid: set(range(8)))

    assert data_utils._get_hash_threads(100) == 4
    assert data_utils._get_hash_threads(2) == 2
    assert data_utils._get_hash_threads(0) == 1


def test_get_hash_threads_fall_back_to_cpu_count(monkeypatch):
    def unavailable_affinity(pid):
        raise OSError("affinity unavailable")

    monkeypatch.setattr(data_utils.os, "sched_getaffinity", unavailable_affinity)
    monkeypatch.setattr(data_utils.os, "cpu_count", lambda: 6)

    assert data_utils._get_hash_threads(100) == 3
