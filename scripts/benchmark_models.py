from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - the release benchmark runs on Linux.
    resource = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.image_validator import validate_and_decode
from services.model_registry import ModelRegistry


def percentile(values: list[float], value: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("No benchmark samples")
    rank = (len(ordered) - 1) * value
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def peak_rss_mib() -> float:
    if resource is None:
        raise RuntimeError("Peak RSS collection requires the Linux ARM release environment")
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / (1024 * 1024) if sys.platform == "darwin" else raw / 1024


def child_model_value(values: list[str], label: str) -> str:
    if len(values) != 1:
        raise ValueError(f"Child benchmark requires exactly one {label} model")
    return values[0]


def parse_child_output(output: str) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("status") in {"pass", "fail"}:
            return payload
    raise ValueError("Child benchmark did not emit a result envelope")


def child_benchmark(image_path: Path, mime: str, matting: str, face: str, iterations: int) -> dict[str, Any]:
    payload = image_path.read_bytes()
    image = validate_and_decode(payload, mime)
    registry = ModelRegistry(concurrency=1)
    if matting not in registry.available_matting_models():
        raise RuntimeError(f"Matting model unavailable: {matting}")
    if face not in registry.available_face_models():
        raise RuntimeError(f"Face model unavailable: {face}")
    wall_samples: list[float] = []
    cpu_samples: list[float] = []
    rss_before = peak_rss_mib()
    for _ in range(iterations + 1):
        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        with registry.acquire(matting, face) as creator:
            result = creator(image, size=(413, 295))
            if result.standard.shape[:2] != (413, 295):
                raise RuntimeError(f"Unexpected benchmark output shape: {result.standard.shape}")
        wall_samples.append((time.perf_counter() - wall_started) * 1000)
        cpu_samples.append((time.process_time() - cpu_started) * 1000)
    cold_wall = wall_samples.pop(0)
    cold_cpu = cpu_samples.pop(0)
    return {
        "mattingModel": matting,
        "faceModel": face,
        "iterations": iterations,
        "cold": {"wallMs": round(cold_wall, 2), "cpuMs": round(cold_cpu, 2)},
        "warm": {
            "p50WallMs": round(percentile(wall_samples, 0.50), 2),
            "p95WallMs": round(percentile(wall_samples, 0.95), 2),
            "meanWallMs": round(statistics.fmean(wall_samples), 2),
            "p50CpuMs": round(percentile(cpu_samples, 0.50), 2),
            "p95CpuMs": round(percentile(cpu_samples, 0.95), 2),
        },
        "peakRssMiB": round(peak_rss_mib(), 2),
        "peakRssDeltaMiB": round(max(0.0, peak_rss_mib() - rss_before), 2),
    }


def run_child(args: argparse.Namespace) -> int:
    try:
        # Legacy Hivision prints progress to stdout. Preserve those diagnostics on
        # stderr so the parent process receives exactly one machine-readable line.
        with redirect_stdout(sys.stderr):
            result = child_benchmark(
                args.image.resolve(),
                args.mime,
                child_model_value(args.matting_model, "matting"),
                child_model_value(args.face_model, "face"),
                args.iterations,
            )
        print(json.dumps({"status": "pass", "result": result}, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({"status": "fail", "error": f"{type(error).__name__}: {error}"}))
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Isolated ARM Hivision cold/warm model benchmark")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mime", choices=("image/jpeg", "image/png", "image/webp"), required=True)
    parser.add_argument("--matting-model", action="append", required=True)
    parser.add_argument("--face-model", action="append", required=True)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.iterations < 2 or args.iterations > 100:
        parser.error("--iterations must be between 2 and 100")
    if args.child:
        return run_child(args)
    if not args.image.is_file():
        parser.error(f"--image does not exist: {args.image}")

    combinations: list[dict[str, Any]] = []
    failed = False
    for matting in dict.fromkeys(args.matting_model):
        for face in dict.fromkeys(args.face_model):
            command = [
                sys.executable, str(Path(__file__).resolve()), "--child",
                "--image", str(args.image.resolve()), "--mime", args.mime,
                "--matting-model", matting, "--face-model", face,
                "--iterations", str(args.iterations),
            ]
            completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=False)
            try:
                child = parse_child_output(completed.stdout)
            except (ValueError, json.JSONDecodeError):
                child = {"status": "fail", "error": f"Invalid child output: {completed.stdout[-500:]} {completed.stderr[-500:]}"}
            child["mattingModel"] = matting
            child["faceModel"] = face
            combinations.append(child)
            failed = failed or completed.returncode != 0 or child.get("status") != "pass"

    report = {
        "schemaVersion": 1,
        "status": "fail" if failed else "pass",
        "environment": {
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "ompNumThreads": os.environ.get("OMP_NUM_THREADS"),
            "openblasNumThreads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "opencvNumThreads": os.environ.get("OPENCV_FOR_THREADS_NUM"),
        },
        "image": str(args.image.resolve()),
        "combinations": combinations,
    }
    output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"{output}\n", encoding="utf-8")
    print(output)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
