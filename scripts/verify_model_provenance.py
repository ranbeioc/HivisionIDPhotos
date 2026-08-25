from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProvenanceError(RuntimeError):
    pass


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _indexed(items: list[dict], label: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for item in items:
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id or model_id in result:
            raise ProvenanceError(f"{label} contains a missing or duplicate model ID")
        result[model_id] = item
    return result


def validate_model_set(manifest: dict, provenance: dict, set_name: str) -> list[dict]:
    if manifest.get("schemaVersion") != 1 or provenance.get("schemaVersion") != 1:
        raise ProvenanceError("Unsupported model manifest or provenance schema")
    manifest_models = _indexed(manifest.get("models", []), "Model manifest")
    provenance_models = _indexed(provenance.get("models", []), "Provenance manifest")
    if set(manifest_models) != set(provenance_models):
        raise ProvenanceError("Every model must have exactly one provenance record")

    selected = [model for model in manifest_models.values() if set_name in model.get("sets", [])]
    if not selected:
        raise ProvenanceError(f"No models configured for set {set_name}")

    decisions: list[dict] = []
    for model in selected:
        record = provenance_models[model["id"]]
        artifact = record.get("artifact", {})
        for field in ("url", "bytes", "sha256"):
            if artifact.get(field) != model.get(field):
                raise ProvenanceError(f"Provenance artifact mismatch for {model['id']}: {field}")
        evidence = record.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ProvenanceError(f"Provenance evidence is missing for {model['id']}")

        decision = record.get("decision")
        production_allowed = record.get("productionAllowed") is True
        weight_status = record.get("weightProvenance", {}).get("status")
        if decision == "approved":
            implementation = record.get("implementation", {})
            if not production_allowed or weight_status != "approved" or not implementation.get("licenseSpdx"):
                raise ProvenanceError(f"Approved provenance is incomplete for {model['id']}")
        elif decision == "needs-verification":
            if production_allowed or not record.get("missingEvidence"):
                raise ProvenanceError(f"Unreviewed provenance is not fail-closed for {model['id']}")
        else:
            raise ProvenanceError(f"Unknown provenance decision for {model['id']}")

        if set_name.startswith("production") and not production_allowed:
            raise ProvenanceError(f"Model {model['id']} is not approved for Production")
        decisions.append({"id": model["id"], "decision": decision, "productionAllowed": production_allowed})
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", required=True)
    args = parser.parse_args()
    decisions = validate_model_set(
        load_json(ROOT / "models" / "manifest.json"),
        load_json(ROOT / "models" / "provenance.json"),
        args.set,
    )
    print(json.dumps({"set": args.set, "models": decisions}, separators=(",", ":")))


if __name__ == "__main__":
    main()
