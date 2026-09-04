from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Callable

from hivision import IDCreator
from hivision.creator.choose_handler import HUMAN_MATTING_MODELS, choose_handler
from .capacity import CapacityUnavailable, ComputeBusy, available_memory_mib, configured_min_available_mib, require_memory_headroom

MATTING_MODEL_MIN_AVAILABLE_MIB = {
    # The 2026-08-28 ARM64 Preview gate peaked at 6682 MiB. Keep a bounded
    # safety margin and fail before ONNX allocation rather than risking OOM.
    "birefnet-v1-lite": 7168,
}

FACE_DETECTION_MODELS = ["mtcnn", "retinaface-resnet50"]


class ModelRegistry:
    """Shares configured creators and serializes request-scoped compute."""

    def __init__(
        self,
        root: Path | None = None,
        concurrency: int = 1,
        memory_probe: Callable[[], float] = available_memory_mib,
        minimum_available_mib: int | None = None,
    ):
        self.root = root or Path(__file__).resolve().parents[1]
        self._creators: dict[tuple[str, str], IDCreator] = {}
        self._lock = Lock()
        self._compute = BoundedSemaphore(concurrency)
        self._memory_probe = memory_probe
        self._minimum_available_mib = minimum_available_mib

    def available_matting_models(self) -> list[str]:
        weights = self.root / "hivision" / "creator" / "weights"
        installed = {path.stem for path in weights.glob("*.*") if path.suffix in {".onnx", ".mnn"}} if weights.exists() else set()
        return [model for model in HUMAN_MATTING_MODELS if model in installed]

    def available_face_models(self) -> list[str]:
        models = ["mtcnn"]
        if (self.root / "hivision" / "creator" / "retinaface" / "weights" / "retinaface-resnet50.onnx").exists():
            models.append("retinaface-resnet50")
        return models

    def ready(self) -> bool:
        try:
            self.ensure_ready()
            return True
        except CapacityUnavailable:
            return False

    def ensure_ready(self, matting_model: str | None = None) -> None:
        if not self.available_matting_models():
            raise CapacityUnavailable("No matting model is installed")
        minimum = self._minimum_available_mib if self._minimum_available_mib is not None else configured_min_available_mib()
        if matting_model is not None:
            minimum = max(minimum, MATTING_MODEL_MIN_AVAILABLE_MIB.get(matting_model, 0))
        require_memory_headroom(self._memory_probe(), minimum)

    @contextmanager
    def acquire(self, matting_model: str, face_model: str):
        if matting_model not in self.available_matting_models():
            raise RuntimeError(f"Matting model unavailable: {matting_model}")
        if face_model not in self.available_face_models():
            raise RuntimeError(f"Face model unavailable: {face_model}")
        if not self._compute.acquire(blocking=False):
            raise ComputeBusy("Compute concurrency is currently full")
        try:
            self.ensure_ready(matting_model)
            key = (matting_model, face_model)
            with self._lock:
                creator = self._creators.get(key)
                if creator is None:
                    creator = IDCreator()
                    choose_handler(creator, matting_model, face_model)
                    self._creators[key] = creator
            yield creator
        finally:
            self._compute.release()


_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    return _registry
