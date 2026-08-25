from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import httpx
from .contracts import AssetDescriptor


class AssetBrokerClient:
    def __init__(self, base_url: str, grant: str, request_id: str, hmac_secret: str):
        self.base_url = base_url.rstrip("/")
        self.grant = grant
        self.request_id = request_id
        self.hmac_secret = hmac_secret.encode("utf-8")

    def _headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        body_hash = hashlib.sha256(body).hexdigest()
        canonical = f"{method}\n{path}\n{timestamp}\n{self.request_id}\n{body_hash}".encode("utf-8")
        signature = base64.urlsafe_b64encode(hmac.new(self.hmac_secret, canonical, hashlib.sha256).digest()).decode("ascii").rstrip("=")
        return {"content-type": "application/json", "x-request-id": self.request_id, "x-xhalo-timestamp": timestamp, "x-xhalo-signature": signature}

    async def upload(self, variant: str, mime: str, image_bytes: bytes, width: int, height: int, dpi: int | None, config: dict, parent_asset_id: str | None = None) -> AssetDescriptor:
        checksum = hashlib.sha256(image_bytes).hexdigest()
        body = json.dumps({"grant": self.grant, "variant": variant, "mime": mime, "bytes": len(image_bytes), "checksumSha256": checksum}, separators=(",", ":")).encode()
        create_path = "/internal/v1/asset-uploads"
        async with httpx.AsyncClient(timeout=20) as client:
            created = await client.post(self.base_url, content=body, headers=self._headers("POST", create_path, body))
            created.raise_for_status()
            session = created.json()
            uploaded = await client.put(session["uploadUrl"], content=image_bytes, headers=session["requiredHeaders"])
            uploaded.raise_for_status()
            finalize_url = f"{self.base_url}/{session['uploadSessionId']}/finalize"
            finalize_path = f"/internal/v1/asset-uploads/{session['uploadSessionId']}/finalize"
            extension = {"image/png": "png", "image/webp": "webp"}.get(mime, "jpg")
            finalize_body = json.dumps({"grant": self.grant, "width": width, "height": height, "dpi": dpi, "filename": f"{variant}.{extension}", "parentAssetId": parent_asset_id, "operation": "id-photo" if parent_asset_id else None, "operationVersion": "v1" if parent_asset_id else None, "config": config}, separators=(",", ":")).encode()
            finalized = await client.post(finalize_url, content=finalize_body, headers=self._headers("POST", finalize_path, finalize_body))
            finalized.raise_for_status()
            payload = finalized.json()
            return AssetDescriptor.model_validate(payload.get("descriptor", payload.get("asset", payload)))

    async def attach_derivatives(self, asset_id: str, preview_asset_id: str, thumbnail_asset_id: str) -> AssetDescriptor:
        path = "/internal/v1/assets/derivatives"
        body = json.dumps({"grant": self.grant, "assetId": asset_id, "previewAssetId": preview_asset_id, "thumbnailAssetId": thumbnail_asset_id}, separators=(",", ":")).encode()
        callback_url = f"{self.base_url.rsplit('/asset-uploads', 1)[0]}/assets/derivatives"
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(callback_url, content=body, headers=self._headers("POST", path, body))
            response.raise_for_status()
            payload = response.json()
            return AssetDescriptor.model_validate(payload.get("asset", payload))

    async def complete_process(self, result: dict) -> None:
        path = "/internal/v1/process-runs/complete"
        body = json.dumps({"grant": self.grant, "result": result}, separators=(",", ":")).encode()
        callback_url = f"{self.base_url.rsplit('/asset-uploads', 1)[0]}/process-runs/complete"
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(callback_url, content=body, headers=self._headers("POST", path, body))
            response.raise_for_status()
