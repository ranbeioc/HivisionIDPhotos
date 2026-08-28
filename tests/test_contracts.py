import base64
import asyncio
import hashlib
import hmac
import io
import json
import time

import pytest
from PIL import Image

from services.contracts import DEFAULT_OPTIONS, IdPhotoOptions, IdPhotoProcessResult


def test_default_options_round_trip():
    payload = DEFAULT_OPTIONS.model_dump(by_alias=True)
    assert IdPhotoOptions.model_validate(payload) == DEFAULT_OPTIONS
    assert payload["size"]["presetId"] == "one-inch"


def test_whitening_strength_is_an_integer_iteration_count():
    payload = DEFAULT_OPTIONS.model_dump(by_alias=True)
    payload["beauty"]["whitening"] = 5
    assert IdPhotoOptions.model_validate(payload).beauty.whitening == 5

    payload["beauty"]["whitening"] = 0.5
    with pytest.raises(ValueError):
        IdPhotoOptions.model_validate(payload)


def test_legacy_beauty_ranges_are_preserved():
    payload = DEFAULT_OPTIONS.model_dump(by_alias=True)
    payload["beauty"] = {"whitening": 15, "brightness": 25, "contrast": 50, "saturation": 50, "sharpen": 5}
    assert IdPhotoOptions.model_validate(payload).beauty.whitening == 15
    payload["beauty"]["brightness"] = 26
    with pytest.raises(ValueError):
        IdPhotoOptions.model_validate(payload)


def test_config_exposes_complete_legacy_size_and_color_catalog():
    pytest.importorskip("cv2")
    from services.id_photo import IdPhotoService

    config = IdPhotoService().config()
    sizes = {item["id"]: (item["width"], item["height"]) for item in config["sizePresets"]}
    colors = {item["colors"][0].lower() for item in config["backgroundPresets"] if item["type"] == "solid"}
    assert len(sizes) == 18
    assert sizes["teacher-qualification"] == (295, 413)
    assert sizes["us-visa"] == (600, 600)
    assert {"#628bce", "#d74532", "#4b6190", "#f2f0f0"}.issubset(colors)


def test_config_exposes_every_legacy_matting_model_with_runtime_availability(tmp_path):
    pytest.importorskip("cv2")
    from services.id_photo import IdPhotoService
    from services.model_registry import ModelRegistry

    weights = tmp_path / "hivision" / "creator" / "weights"
    weights.mkdir(parents=True)
    (weights / "modnet_photographic_portrait_matting.onnx").write_bytes(b"fixture")
    registry = ModelRegistry(root=tmp_path, memory_probe=lambda: 4096, minimum_available_mib=2048)

    models = IdPhotoService(registry).config()["mattingModels"]
    assert [model["id"] for model in models] == [
        "modnet_photographic_portrait_matting",
        "birefnet-v1-lite",
        "hivision_modnet",
        "rmbg-1.4",
    ]
    assert [model["id"] for model in models if model["available"]] == ["modnet_photographic_portrait_matting"]
    assert {model["id"] for model in models if model["heavy"]} == {"birefnet-v1-lite", "rmbg-1.4"}
    assert all(model["label"]["zh"] != model["id"] for model in models)


def test_service_preserves_rgb_background_channel_order():
    pytest.importorskip("cv2")
    import numpy as np
    from hivision.utils import add_background
    from services.id_photo import _composite_channels

    transparent = np.zeros((2, 2, 4), dtype=np.uint8)
    output = np.uint8(add_background(transparent, bgr=_composite_channels("#d74532")))
    assert tuple(output[0, 0]) == (215, 69, 50)


def _signed_headers(secret: str, method: str, path: str, request_id: str, body: bytes) -> dict[str, str]:
    timestamp = str(int(time.time() * 1000))
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = f"{method}\n{path}\n{timestamp}\n{request_id}\n{body_hash}".encode()
    signature = base64.urlsafe_b64encode(hmac.new(secret.encode(), canonical, hashlib.sha256).digest()).decode().rstrip("=")
    return {"x-request-id": request_id, "x-xhalo-timestamp": timestamp, "x-xhalo-body-sha256": body_hash, "x-xhalo-signature": signature}


def test_signed_multipart_process_contract(monkeypatch):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    secret = "test-hmac-secret-with-enough-entropy"
    monkeypatch.setenv("HIVISION_HMAC_SECRET", secret)
    from api import main
    from api.security import _seen

    _seen.clear()
    expected = {
        "requestId": "multipart-contract-request",
        "inputAssetId": "input-asset",
        "savedOriginal": False,
        "assets": {"templates": []},
        "processing": {"model": "modnet", "faceModel": "mtcnn", "durationMs": 1, "configVersion": "contract-test"},
    }

    captured = {}

    async def fake_process(*args, **_kwargs):
        captured["context"] = args[-1]
        return IdPhotoProcessResult.model_validate(expected)

    monkeypatch.setattr(main.service, "process", fake_process)
    image = Image.new("RGB", (4, 4), (20, 40, 60))
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    boundary = "xhalo-contract-boundary"
    config = json.dumps(DEFAULT_OPTIONS.model_dump(by_alias=True), separators=(",", ":"))
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"config\"\r\n\r\n{config}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"fixture.png\"\r\nContent-Type: image/png\r\n\r\n"
    ).encode() + buffer.getvalue() + f"\r\n--{boundary}--\r\n".encode()
    headers = _signed_headers(secret, "POST", "/v1/id-photo/process", expected["requestId"], body)
    headers.update({"content-type": f"multipart/form-data; boundary={boundary}", "x-xhalo-asset-broker-url": "https://gateway.test/internal/v1/asset-uploads", "x-xhalo-asset-grant": "signed-grant", "x-xhalo-input-asset-id": "existing-input", "x-xhalo-input-asset-saved": "true"})
    response = TestClient(main.app).post("/v1/id-photo/process", content=body, headers=headers)
    assert response.status_code == 200
    assert response.json() == expected
    assert captured["context"].input_asset_id == "existing-input"
    assert captured["context"].input_asset_saved is True


def test_request_id_boundary():
    from services.request_id import is_valid_request_id

    assert is_valid_request_id("gateway:request-1")
    assert is_valid_request_id("a")
    assert not is_valid_request_id("unsafe request id")
    assert not is_valid_request_id("line\nbreak")
    assert not is_valid_request_id("x" * 129)


def test_ready_returns_retry_after_under_memory_pressure(monkeypatch):
    TestClient = pytest.importorskip("fastapi.testclient").TestClient
    secret = "test-hmac-secret-with-enough-entropy"
    monkeypatch.setenv("HIVISION_HMAC_SECRET", secret)
    from api import main
    from api.security import _seen
    from services.capacity import MemoryPressure

    _seen.clear()

    def reject():
        raise MemoryPressure("Compute memory headroom is too low")

    monkeypatch.setattr(main.service.registry, "ensure_ready", reject)
    headers = _signed_headers(secret, "GET", "/v1/ready", "memory-pressure-request", b"")
    response = TestClient(main.app).get("/v1/ready", headers=headers)
    assert response.status_code == 503
    assert response.headers["retry-after"] == "30"
    assert response.json() == {
        "error": {
            "code": "MODEL_UNAVAILABLE",
            "message": "Compute memory headroom is too low",
            "requestId": "memory-pressure-request",
        }
    }


def test_service_creates_bounded_preview_and_thumbnail_derivatives(monkeypatch):
    pytest.importorskip("cv2")
    import numpy as np
    from contextlib import contextmanager
    from types import SimpleNamespace
    from services.contracts import AssetDescriptor
    from services.id_photo import IdPhotoService, ProcessContext
    import services.id_photo as service_module

    class FakeRegistry:
        @contextmanager
        def acquire(self, *_args):
            def create(_image, **_kwargs):
                standard = np.zeros((413, 295, 4), dtype=np.uint8)
                standard[:, :, 3] = 255
                hd = np.zeros((2480, 1772, 4), dtype=np.uint8)
                hd[:, :, 3] = 255
                return SimpleNamespace(standard=standard, hd=hd)
            yield create

    calls = {"uploads": [], "attached": [], "completed": []}

    class FakeBroker:
        def __init__(self, *_args):
            pass

        async def upload(self, variant, mime, image_bytes, width, height, dpi, config, parent_asset_id=None):
            calls["uploads"].append((variant, width, height, parent_asset_id, len(image_bytes)))
            return AssetDescriptor.model_validate({
                "assetId": f"asset-{variant}", "variant": variant, "mime": mime,
                "width": width, "height": height, "bytes": len(image_bytes), "dpi": dpi,
                "previewUrl": "https://assets.test/preview", "downloadUrl": "https://assets.test/download",
                "urlExpiresAt": "2026-08-24T00:05:00Z", "saved": True,
            })

        async def attach_derivatives(self, asset_id, preview_asset_id, thumbnail_asset_id):
            calls["attached"].append((asset_id, preview_asset_id, thumbnail_asset_id))
            return AssetDescriptor.model_validate({
                "assetId": asset_id, "variant": "standard", "mime": "image/png",
                "width": 295, "height": 413, "bytes": 100, "dpi": 300,
                "previewUrl": "https://assets.test/preview", "downloadUrl": "https://assets.test/download",
                "urlExpiresAt": "2026-08-24T00:05:00Z", "saved": True,
            })

        async def complete_process(self, result):
            calls["completed"].append(result)

    monkeypatch.setattr(service_module, "AssetBrokerClient", FakeBroker)
    options = DEFAULT_OPTIONS.model_copy(deep=True)
    options.background.type = "transparent"
    options.output.format = "png"
    options.output.variants = ["standard"]
    options.layout.enabled = False
    context = ProcessContext("derivative-request", "https://gateway.test/internal/v1/asset-uploads", "grant", "secret", "existing-input", True)
    image = np.zeros((640, 480, 3), dtype=np.uint8)
    result = asyncio.run(IdPhotoService(FakeRegistry()).process(image, b"original", "image/png", options, context))

    assert [(item[0], item[1], item[2]) for item in calls["uploads"]] == [
        ("standard", 295, 413),
        ("standard.preview", 295, 413),
        ("standard.thumbnail", 229, 320),
    ]
    assert calls["uploads"][1][3] == "asset-standard"
    assert calls["uploads"][2][3] == "asset-standard"
    assert calls["attached"] == [("asset-standard", "asset-standard.preview", "asset-standard.thumbnail")]
    assert result.assets.standard.asset_id == "asset-standard"
    assert len(calls["completed"]) == 1


def test_asset_descriptor_accepts_asset_lifecycle_metadata():
    from services.contracts import AssetDescriptor

    descriptor = AssetDescriptor.model_validate({
        "assetId": "asset-standard",
        "variant": "standard",
        "mime": "image/jpeg",
        "width": 295,
        "height": 413,
        "bytes": 12345,
        "dpi": 300,
        "previewUrl": "https://assets.test/preview",
        "downloadUrl": "https://assets.test/download",
        "urlExpiresAt": "2026-08-25T10:35:00Z",
        "saved": False,
        "status": "temporary",
        "filename": "standard.jpg",
        "folderId": None,
        "parentAssetId": "asset-original",
        "createdAt": "2026-08-25T10:30:00Z",
        "expiresAt": "2026-08-26T10:30:00Z",
    })

    assert descriptor.asset_id == "asset-standard"
    assert descriptor.saved is False
