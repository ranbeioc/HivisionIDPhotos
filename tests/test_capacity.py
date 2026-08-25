from contextlib import contextmanager
from pathlib import Path

import pytest

from services.capacity import ComputeBusy, MemoryPressure, RequestAdmission, available_memory_mib, require_memory_headroom
from services.model_registry import ModelRegistry


def test_available_memory_uses_the_tightest_host_or_cgroup_headroom(tmp_path: Path):
    proc = tmp_path / "meminfo"
    proc.write_text("MemTotal: 8388608 kB\nMemAvailable: 4194304 kB\n", encoding="utf-8")
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "memory.max").write_text(str(3 * 1024 * 1024 * 1024), encoding="utf-8")
    (cgroup / "memory.current").write_text(str(1 * 1024 * 1024 * 1024), encoding="utf-8")
    assert available_memory_mib(proc, cgroup) == 2048


def test_memory_pressure_fails_closed_before_model_loading():
    with pytest.raises(MemoryPressure, match="headroom is too low"):
        require_memory_headroom(1024, 2048)


def test_registry_rejects_parallel_compute_without_waiting(tmp_path: Path, monkeypatch):
    weights = tmp_path / "hivision" / "creator" / "weights"
    weights.mkdir(parents=True)
    (weights / "modnet_photographic_portrait_matting.onnx").write_bytes(b"fixture")

    class FakeCreator:
        pass

    monkeypatch.setattr("services.model_registry.IDCreator", FakeCreator)
    monkeypatch.setattr("services.model_registry.choose_handler", lambda *_args: None)
    registry = ModelRegistry(root=tmp_path, concurrency=1, memory_probe=lambda: 4096, minimum_available_mib=2048)
    first = registry.acquire("modnet_photographic_portrait_matting", "mtcnn")
    first.__enter__()
    try:
        with pytest.raises(ComputeBusy, match="concurrency is currently full"):
            with registry.acquire("modnet_photographic_portrait_matting", "mtcnn"):
                pass
    finally:
        first.__exit__(None, None, None)


def test_request_admission_caps_full_path_inflight_without_waiting():
    admission = RequestAdmission(maximum=2)
    first = admission.acquire()
    second = admission.acquire()
    first.__enter__()
    second.__enter__()
    try:
        with pytest.raises(ComputeBusy, match="Request concurrency is currently full"):
            with admission.acquire():
                pass
    finally:
        second.__exit__(None, None, None)
        first.__exit__(None, None, None)
