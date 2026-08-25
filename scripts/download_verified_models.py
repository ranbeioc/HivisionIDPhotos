from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]


def verify(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    if size != expected_bytes or digest.hexdigest() != expected_sha256:
        raise RuntimeError(f"Model checksum mismatch: {path.name}")


def install(model: dict) -> None:
    destination = ROOT / model["path"]
    if destination.exists():
        verify(destination, model["bytes"], model["sha256"])
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    request = Request(model["url"], headers={"User-Agent": "xhalo-hivision-model-installer/1"})
    with urlopen(request, timeout=120) as response, partial.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    verify(partial, model["bytes"], model["sha256"])
    partial.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="preview-default")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    selected = [model for model in manifest["models"] if args.set in model["sets"]]
    if not selected:
        raise RuntimeError(f"No models configured for set {args.set}")
    for model in selected:
        path = ROOT / model["path"]
        if args.verify_only:
            verify(path, model["bytes"], model["sha256"])
        else:
            install(model)


if __name__ == "__main__":
    main()
