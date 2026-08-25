from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from threading import BoundedSemaphore


MIB = 1024 * 1024
DEFAULT_MIN_AVAILABLE_MIB = 2048
DEFAULT_MAX_INFLIGHT_REQUESTS = 4


class CapacityUnavailable(RuntimeError):
    retry_after_seconds = 30


class ComputeBusy(CapacityUnavailable):
    retry_after_seconds = 5


class MemoryPressure(CapacityUnavailable):
    retry_after_seconds = 30


def configured_max_inflight_requests() -> int:
    raw = os.environ.get("HIVISION_MAX_INFLIGHT_REQUESTS", str(DEFAULT_MAX_INFLIGHT_REQUESTS))
    try:
        value = int(raw)
    except ValueError as error:
        raise ComputeBusy("HIVISION_MAX_INFLIGHT_REQUESTS is invalid") from error
    if value < 1 or value > 32:
        raise ComputeBusy("HIVISION_MAX_INFLIGHT_REQUESTS is outside the safe range")
    return value


class RequestAdmission:
    def __init__(self, maximum: int | None = None):
        self.maximum = maximum if maximum is not None else configured_max_inflight_requests()
        if self.maximum < 1 or self.maximum > 32:
            raise ComputeBusy("Request concurrency is outside the safe range")
        self._slots = BoundedSemaphore(self.maximum)

    @contextmanager
    def acquire(self):
        if not self._slots.acquire(blocking=False):
            raise ComputeBusy("Request concurrency is currently full")
        try:
            yield
        finally:
            self._slots.release()


def configured_min_available_mib() -> int:
    raw = os.environ.get("HIVISION_MIN_AVAILABLE_MEMORY_MB", str(DEFAULT_MIN_AVAILABLE_MIB))
    try:
        value = int(raw)
    except ValueError as error:
        raise MemoryPressure("HIVISION_MIN_AVAILABLE_MEMORY_MB is invalid") from error
    if value < 256 or value > 65536:
        raise MemoryPressure("HIVISION_MIN_AVAILABLE_MEMORY_MB is outside the safe range")
    return value


def _proc_available_mib(path: Path) -> float | None:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _cgroup_headroom_mib(root: Path) -> float | None:
    try:
        maximum = (root / "memory.max").read_text(encoding="utf-8").strip()
        current = int((root / "memory.current").read_text(encoding="utf-8").strip())
        if maximum == "max":
            return None
        return max(0, int(maximum) - current) / MIB
    except (OSError, ValueError):
        return None


def available_memory_mib(
    proc_meminfo: Path = Path("/proc/meminfo"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> float:
    candidates = [value for value in (_proc_available_mib(proc_meminfo), _cgroup_headroom_mib(cgroup_root)) if value is not None]
    if not candidates:
        raise MemoryPressure("Available memory cannot be determined")
    return min(candidates)


def require_memory_headroom(available_mib: float, minimum_mib: int) -> None:
    if available_mib < minimum_mib:
        raise MemoryPressure(
            f"Compute memory headroom is too low ({available_mib:.0f} MiB available; {minimum_mib} MiB required)"
        )
