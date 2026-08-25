import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.benchmark_models import child_model_value, parse_child_output, percentile
from scripts.run_golden_matrix import GateFailure, load_manifest, parse_lanes, run_matrix
from scripts.verify_model_provenance import ProvenanceError, validate_model_set


def _write_png(path: Path, color: tuple[int, int, int, int]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (8, 10), color)
    image.save(path, format="PNG", dpi=(300, 300))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_golden_matrix_requires_all_release_lanes(tmp_path):
    baseline = tmp_path / "gradio"
    baseline.mkdir()
    with pytest.raises(GateFailure, match="all five lanes"):
        parse_lanes([f"gradio={baseline}"], allow_subset=False)


def test_golden_manifest_refuses_empty_fixture_corpus(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schemaVersion": 1, "baselineLane": "gradio", "cases": []}), encoding="utf-8")
    with pytest.raises(GateFailure, match="licensed fixture"):
        load_manifest(manifest)


def test_golden_matrix_passes_identical_synthetic_outputs(tmp_path):
    lanes = {}
    digest = ""
    for lane in ("gradio", "service", "gateway", "pages", "vps"):
        root = tmp_path / lane
        digest = _write_png(root / "front-default" / "transparent.png", (20, 40, 60, 255))
        lanes[lane] = root
    manifest = {
        "schemaVersion": 1,
        "baselineLane": "gradio",
        "cases": [{
            "id": "front-default",
            "variants": {
                "transparentStandard": {
                    "path": "front-default/transparent.png",
                    "mime": "image/png", "width": 8, "height": 10, "dpi": 300,
                    "deterministic": True, "sha256": digest,
                    "thresholds": {"ssim": 0.995, "alphaMaskIou": 0.995, "maxGeometryDeviationPx": 1},
                }
            },
        }],
    }
    assert run_matrix(manifest, lanes)["status"] == "pass"


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert np.isclose(percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.85)


def test_child_benchmark_unwraps_exactly_one_model():
    assert child_model_value(["modnet"], "matting") == "modnet"
    with pytest.raises(ValueError, match="exactly one"):
        child_model_value(["modnet", "rmbg"], "matting")


def test_child_benchmark_parser_ignores_legacy_progress_logs():
    output = "[1] Start matting...\nlegacy progress\n{\"status\":\"pass\",\"result\":{\"warm\":1}}\n"
    assert parse_child_output(output)["status"] == "pass"
    with pytest.raises(ValueError, match="result envelope"):
        parse_child_output("legacy progress only")


def test_production_model_set_excludes_unreviewed_retinaface():
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    production = {model["id"] for model in manifest["models"] if "production-default" in model["sets"]}
    preview = {model["id"] for model in manifest["models"] if "preview-default" in model["sets"]}
    assert production == {"modnet_photographic_portrait_matting"}
    assert "retinaface-resnet50" in preview
    assert "retinaface-resnet50" not in production


def test_model_provenance_allows_reviewed_production_and_preview_only_retinaface():
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads(Path("models/provenance.json").read_text(encoding="utf-8"))
    production = validate_model_set(manifest, provenance, "production-default")
    preview = validate_model_set(manifest, provenance, "preview-default")
    assert production == [{"id": "modnet_photographic_portrait_matting", "decision": "approved", "productionAllowed": True}]
    assert {item["id"] for item in preview} == {"modnet_photographic_portrait_matting", "retinaface-resnet50"}


def test_model_provenance_refuses_unreviewed_weight_in_production():
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads(Path("models/provenance.json").read_text(encoding="utf-8"))
    tampered = deepcopy(manifest)
    retinaface = next(model for model in tampered["models"] if model["id"] == "retinaface-resnet50")
    retinaface["sets"].append("production-default")
    with pytest.raises(ProvenanceError, match="not approved for Production"):
        validate_model_set(tampered, provenance, "production-default")


def test_model_provenance_refuses_artifact_metadata_drift():
    manifest = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))
    provenance = json.loads(Path("models/provenance.json").read_text(encoding="utf-8"))
    tampered = deepcopy(provenance)
    tampered["models"][0]["artifact"]["sha256"] = "0" * 64
    with pytest.raises(ProvenanceError, match="artifact mismatch"):
        validate_model_set(manifest, tampered, "production-default")
