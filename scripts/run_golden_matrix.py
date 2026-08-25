from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


REQUIRED_LANES = ("gradio", "service", "gateway", "pages", "vps")
MIME_BY_FORMAT = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class GateFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class LoadedImage:
    pixels: np.ndarray
    alpha: np.ndarray | None
    mime: str
    width: int
    height: int
    dpi: int | None
    sha256: str


def _positive_number(value: Any, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise GateFailure(f"{label} must be a finite non-negative number")
    return float(value)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GateFailure(f"Unable to read Golden manifest: {error}") from error
    if manifest.get("schemaVersion") != 1:
        raise GateFailure("Golden manifest schemaVersion must be 1")
    if manifest.get("baselineLane") != "gradio":
        raise GateFailure("Golden baselineLane must remain gradio")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise GateFailure("Golden manifest must contain at least one licensed fixture case")
    seen_cases: set[str] = set()
    for case in cases:
        case_id = case.get("id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen_cases:
            raise GateFailure("Every Golden case needs a unique non-empty id")
        seen_cases.add(case_id)
        variants = case.get("variants")
        if not isinstance(variants, dict) or not variants:
            raise GateFailure(f"Golden case {case_id} must declare variants")
        for variant_name, expected in variants.items():
            if not isinstance(variant_name, str) or not isinstance(expected, dict):
                raise GateFailure(f"Golden case {case_id} has an invalid variant")
            for key in ("path", "mime", "width", "height", "dpi"):
                if key not in expected:
                    raise GateFailure(f"Golden case {case_id}/{variant_name} is missing {key}")
            if expected.get("deterministic") is True and not isinstance(expected.get("sha256"), str):
                raise GateFailure(f"Deterministic Golden output {case_id}/{variant_name} requires sha256")
            if expected.get("deterministic") is True and (
                len(expected["sha256"]) != 64 or any(character not in "0123456789abcdef" for character in expected["sha256"])
            ):
                raise GateFailure(f"Deterministic Golden output {case_id}/{variant_name} requires a lowercase SHA-256")
            thresholds = expected.get("thresholds", {})
            if not isinstance(thresholds, dict):
                raise GateFailure(f"Golden thresholds for {case_id}/{variant_name} must be an object")
            _positive_number(thresholds.get("ssim", 0.995), "ssim")
            _positive_number(thresholds.get("alphaMaskIou", 0.995), "alphaMaskIou")
            _positive_number(thresholds.get("maxGeometryDeviationPx", 1), "maxGeometryDeviationPx")
    return manifest


def parse_lanes(values: list[str], allow_subset: bool) -> dict[str, Path]:
    lanes: dict[str, Path] = {}
    for value in values:
        name, separator, directory = value.partition("=")
        if not separator or name not in REQUIRED_LANES or not directory:
            raise GateFailure(f"Invalid lane {value!r}; use name=/absolute/output/directory")
        if name in lanes:
            raise GateFailure(f"Duplicate lane: {name}")
        path = Path(directory).resolve()
        if not path.is_dir():
            raise GateFailure(f"Lane directory does not exist: {path}")
        lanes[name] = path
    missing = [name for name in REQUIRED_LANES if name not in lanes]
    if missing and not allow_subset:
        raise GateFailure(f"Golden Matrix requires all five lanes; missing: {', '.join(missing)}")
    if "gradio" not in lanes:
        raise GateFailure("Golden Matrix requires the gradio baseline lane")
    return lanes


def load_image(path: Path) -> LoadedImage:
    if not path.is_file():
        raise GateFailure(f"Golden output is missing: {path}")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    try:
        with Image.open(path) as source:
            source.load()
            mime = MIME_BY_FORMAT.get(source.format or "")
            if mime is None:
                raise GateFailure(f"Unsupported Golden output format: {path}")
            dpi_value = source.info.get("dpi")
            dpi = round(float(dpi_value[0])) if isinstance(dpi_value, tuple) and dpi_value else None
            rgba = np.asarray(source.convert("RGBA"))
    except OSError as error:
        raise GateFailure(f"Unable to decode Golden output {path}: {error}") from error
    return LoadedImage(
        pixels=rgba[:, :, :3],
        alpha=rgba[:, :, 3] if np.any(rgba[:, :, 3] != 255) else None,
        mime=mime,
        width=rgba.shape[1],
        height=rgba.shape[0],
        dpi=dpi,
        sha256=digest,
    )


def ssim(reference: np.ndarray, candidate: np.ndarray) -> float:
    if reference.shape != candidate.shape:
        return 0.0
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float64)
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY).astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_reference = cv2.GaussianBlur(reference_gray, (11, 11), 1.5)
    mu_candidate = cv2.GaussianBlur(candidate_gray, (11, 11), 1.5)
    mu_reference_sq = mu_reference * mu_reference
    mu_candidate_sq = mu_candidate * mu_candidate
    mu_product = mu_reference * mu_candidate
    sigma_reference_sq = cv2.GaussianBlur(reference_gray * reference_gray, (11, 11), 1.5) - mu_reference_sq
    sigma_candidate_sq = cv2.GaussianBlur(candidate_gray * candidate_gray, (11, 11), 1.5) - mu_candidate_sq
    sigma_product = cv2.GaussianBlur(reference_gray * candidate_gray, (11, 11), 1.5) - mu_product
    numerator = (2 * mu_product + c1) * (2 * sigma_product + c2)
    denominator = (mu_reference_sq + mu_candidate_sq + c1) * (sigma_reference_sq + sigma_candidate_sq + c2)
    return float(np.mean(numerator / np.maximum(denominator, np.finfo(np.float64).eps)))


def alpha_metrics(reference: np.ndarray | None, candidate: np.ndarray | None) -> tuple[float, float]:
    if reference is None and candidate is None:
        return 1.0, 0.0
    if reference is None or candidate is None or reference.shape != candidate.shape:
        return 0.0, math.inf
    reference_mask = reference >= 128
    candidate_mask = candidate >= 128
    union = np.logical_or(reference_mask, candidate_mask).sum()
    intersection = np.logical_and(reference_mask, candidate_mask).sum()
    iou = 1.0 if union == 0 else float(intersection / union)

    def box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
        rows, columns = np.where(mask)
        if not len(rows):
            return None
        return int(columns.min()), int(rows.min()), int(columns.max()), int(rows.max())

    reference_box = box(reference_mask)
    candidate_box = box(candidate_mask)
    if reference_box is None and candidate_box is None:
        deviation = 0.0
    elif reference_box is None or candidate_box is None:
        deviation = math.inf
    else:
        deviation = float(max(abs(left - right) for left, right in zip(reference_box, candidate_box, strict=True)))
    return iou, deviation


def validate_contract(image: LoadedImage, expected: dict[str, Any], label: str) -> list[str]:
    failures: list[str] = []
    checks = {
        "mime": (image.mime, expected["mime"]),
        "width": (image.width, expected["width"]),
        "height": (image.height, expected["height"]),
        "dpi": (image.dpi, expected["dpi"]),
    }
    for key, (actual, wanted) in checks.items():
        if actual != wanted:
            failures.append(f"{label}: {key}={actual!r}, expected {wanted!r}")
    if expected.get("deterministic") is True and image.sha256 != expected["sha256"]:
        failures.append(f"{label}: sha256={image.sha256}, expected {expected['sha256']}")
    return failures


def run_matrix(manifest: dict[str, Any], lanes: dict[str, Path]) -> dict[str, Any]:
    failures: list[str] = []
    comparisons: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        for variant_name, expected in case["variants"].items():
            relative = Path(expected["path"])
            if relative.is_absolute() or ".." in relative.parts:
                failures.append(f"{case['id']}/{variant_name}: path must remain relative to each lane")
                continue
            loaded: dict[str, LoadedImage] = {}
            for lane_name, lane_root in lanes.items():
                label = f"{lane_name}:{case['id']}/{variant_name}"
                try:
                    loaded[lane_name] = load_image(lane_root / relative)
                    failures.extend(validate_contract(loaded[lane_name], expected, label))
                except GateFailure as error:
                    failures.append(str(error))
            reference = loaded.get("gradio")
            if reference is None:
                continue
            thresholds = expected.get("thresholds", {})
            min_ssim = float(thresholds.get("ssim", 0.995))
            min_iou = float(thresholds.get("alphaMaskIou", 0.995))
            max_geometry = float(thresholds.get("maxGeometryDeviationPx", 1))
            for lane_name, candidate in loaded.items():
                if lane_name == "gradio":
                    continue
                measured_ssim = ssim(reference.pixels, candidate.pixels)
                measured_iou, geometry = alpha_metrics(reference.alpha, candidate.alpha)
                comparisons.append({
                    "case": case["id"], "variant": variant_name, "lane": lane_name,
                    "ssim": measured_ssim, "alphaMaskIou": measured_iou,
                    "geometryDeviationPx": geometry if math.isfinite(geometry) else None,
                })
                label = f"{lane_name}:{case['id']}/{variant_name}"
                if measured_ssim < min_ssim:
                    failures.append(f"{label}: SSIM {measured_ssim:.6f} < {min_ssim:.6f}")
                if reference.alpha is not None or candidate.alpha is not None:
                    if measured_iou < min_iou:
                        failures.append(f"{label}: alpha IoU {measured_iou:.6f} < {min_iou:.6f}")
                    if geometry > max_geometry:
                        failures.append(f"{label}: geometry deviation {geometry:g}px > {max_geometry:g}px")
    return {
        "schemaVersion": 1,
        "status": "pass" if not failures else "fail",
        "lanes": list(lanes),
        "cases": len(manifest["cases"]),
        "comparisons": comparisons,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fail-closed Gradio/Service/Gateway/Pages/VPS Golden Matrix")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lane", action="append", default=[], help="lane=/absolute/output/directory")
    parser.add_argument("--allow-subset", action="store_true", help="Local harness development only; never use for release")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    try:
        manifest = load_manifest(args.manifest.resolve())
        lanes = parse_lanes(args.lane, args.allow_subset)
        report = run_matrix(manifest, lanes)
    except GateFailure as error:
        report = {"schemaVersion": 1, "status": "fail", "failures": [str(error)]}
    output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"{output}\n", encoding="utf-8")
    print(output)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
