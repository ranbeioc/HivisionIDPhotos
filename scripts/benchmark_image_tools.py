from __future__ import annotations

import argparse
import json
import resource
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from services.image_tools import ImagePipeline, ImageToolsService
from services.image_validator import validate_and_decode


def percentile(values: list[float], ratio: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * ratio)))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bounded ARM64 benchmark for the isolated Image Tools core")
    parser.add_argument("image", type=Path)
    parser.add_argument("--tasks", type=int, default=8)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    args = parser.parse_args()
    source = args.image.read_bytes()
    image = validate_and_decode(source, "image/jpeg")
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    height, width = mask.shape
    mask[max(0, height * 3 // 4):min(height, height * 3 // 4 + max(16, height // 20)), max(0, width // 2 - max(8, width // 40)):min(width, width // 2 + max(8, width // 40))] = 255
    basic = ImagePipeline.model_validate({"version": 1, "operations": [{"schemaVersion": 1, "id": "resize", "width": 600, "height": 600, "fit": "contain", "allowUpscale": False}, {"schemaVersion": 1, "id": "convert", "format": "jpeg"}, {"schemaVersion": 1, "id": "compress", "quality": 82, "targetBytes": None}]})
    inpaint = ImagePipeline.model_validate({"version": 1, "operations": [{"schemaVersion": 1, "id": "inpaint", "method": "telea", "radius": 5}, {"schemaVersion": 1, "id": "convert", "format": "png"}]})
    service = ImageToolsService()
    results: list[dict[str, object]] = []

    for label, pipeline, operation_mask in (("server-basic", basic, None), ("opencv-fast", inpaint, mask)):
        for concurrency in args.concurrency:
            cpu_start = time.process_time()
            wall_start = time.perf_counter()

            def run_once(_: int) -> float:
                started = time.perf_counter()
                service.process(image, pipeline, operation_mask)
                return (time.perf_counter() - started) * 1000

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                durations = list(executor.map(run_once, range(args.tasks)))
            wall_ms = (time.perf_counter() - wall_start) * 1000
            cpu_ms = (time.process_time() - cpu_start) * 1000
            results.append({
                "engine": label,
                "concurrency": concurrency,
                "tasks": args.tasks,
                "p50Ms": round(statistics.median(durations), 2),
                "p95Ms": round(percentile(durations, 0.95), 2),
                "wallMs": round(wall_ms, 2),
                "cpuMs": round(cpu_ms, 2),
                "throughputPerSecond": round(args.tasks / (wall_ms / 1000), 3),
                "peakRssMiB": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
            })
    print(json.dumps({"inputBytes": len(source), "width": int(image.shape[1]), "height": int(image.shape[0]), "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
