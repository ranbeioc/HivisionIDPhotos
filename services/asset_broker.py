from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import time

import httpx
from .contracts import AssetDescriptor


_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class AssetBrokerError(RuntimeError):
    def __init__(self, stage: str, status_code: int | None = None):
        super().__init__(f"Asset broker {stage} failed" + (f" with status {status_code}" if status_code else ""))
        self.stage = stage
        self.status_code = status_code


def _bounded_integer_env(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(minimum, min(maximum, value))


class AssetBrokerClient:
    def __init__(self, base_url: str, grant: str, request_id: str, hmac_secret: str):
        self.base_url = base_url.rstrip("/")
        self.grant = grant
        self.request_id = request_id
        self.hmac_secret = hmac_secret.encode("utf-8")
        # A single ID-photo request can produce seven public variants and two
        # derivatives per variant. Bound callback fan-out so one successful
        # inference cannot burst R2/D1 with more writes than the asset service
        # can absorb.
        self._upload_slots = asyncio.Semaphore(_bounded_integer_env("HIVISION_ASSET_UPLOAD_CONCURRENCY", 4, 1, 8))
        self._max_attempts = _bounded_integer_env("HIVISION_ASSET_CALLBACK_ATTEMPTS", 3, 1, 4)
        self._timeout = httpx.Timeout(5.0, connect=5.0, read=5.0, write=5.0, pool=5.0)

    def _headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        body_hash = hashlib.sha256(body).hexdigest()
        canonical = f"{method}\n{path}\n{timestamp}\n{self.request_id}\n{body_hash}".encode("utf-8")
        signature = base64.urlsafe_b64encode(hmac.new(self.hmac_secret, canonical, hashlib.sha256).digest()).decode("ascii").rstrip("=")
        return {"content-type": "application/json", "x-request-id": self.request_id, "x-xhalo-timestamp": timestamp, "x-xhalo-signature": signature}

    async def _request_with_retry(self, client: httpx.AsyncClient, method: str, url: str, *, content: bytes, headers: dict[str, str], stage: str) -> httpx.Response:
        """Retry only callbacks whose server-side operation is idempotent.

        Upload PUT reuses the same one-time session, finalize is idempotent on
        uploadSessionId, and derivative/process completion are idempotent on
        their signed request identifiers. Upload-session creation deliberately
        remains single-attempt because a lost response could otherwise orphan
        an extra session.
        """
        for attempt in range(self._max_attempts):
            response: httpx.Response | None = None
            try:
                response = await client.request(method, url, content=content, headers=headers)
                if response.status_code not in _RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                    return response
                if attempt + 1 >= self._max_attempts:
                    response.raise_for_status()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as error:
                retryable = not isinstance(error, httpx.HTTPStatusError) or error.response.status_code in _RETRYABLE_STATUS_CODES
                if not retryable or attempt + 1 >= self._max_attempts:
                    status_code = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                    raise AssetBrokerError(stage, status_code) from error

            retry_after = response.headers.get("retry-after") if response is not None else None
            try:
                delay = min(2.0, max(0.0, float(retry_after))) if retry_after is not None else 0.15 * (2 ** attempt)
            except ValueError:
                delay = 0.15 * (2 ** attempt)
            await asyncio.sleep(delay)

        raise AssetBrokerError(stage)

    async def upload(self, variant: str, mime: str, image_bytes: bytes, width: int, height: int, dpi: int | None, config: dict, parent_asset_id: str | None = None, operation: str | None = None) -> AssetDescriptor:
        checksum = hashlib.sha256(image_bytes).hexdigest()
        body = json.dumps({"grant": self.grant, "variant": variant, "mime": mime, "bytes": len(image_bytes), "checksumSha256": checksum}, separators=(",", ":")).encode()
        create_path = "/internal/v1/asset-uploads"
        async with self._upload_slots:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                try:
                    created = await client.post(self.base_url, content=body, headers=self._headers("POST", create_path, body))
                    created.raise_for_status()
                except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as error:
                    status_code = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                    raise AssetBrokerError("create", status_code) from error
                session = created.json()
                await self._request_with_retry(client, "PUT", session["uploadUrl"], content=image_bytes, headers=session["requiredHeaders"], stage="upload")
                finalize_url = f"{self.base_url}/{session['uploadSessionId']}/finalize"
                finalize_path = f"/internal/v1/asset-uploads/{session['uploadSessionId']}/finalize"
                extension = {"image/png": "png", "image/webp": "webp"}.get(mime, "jpg")
                finalize_body = json.dumps({"grant": self.grant, "width": width, "height": height, "dpi": dpi, "filename": f"{variant}.{extension}", "parentAssetId": parent_asset_id, "operation": operation or ("id-photo" if parent_asset_id else None), "operationVersion": "v1" if operation or parent_asset_id else None, "config": config}, separators=(",", ":")).encode()
                finalized = await self._request_with_retry(client, "POST", finalize_url, content=finalize_body, headers=self._headers("POST", finalize_path, finalize_body), stage="finalize")
                payload = finalized.json()
                return AssetDescriptor.model_validate(payload.get("descriptor", payload.get("asset", payload)))

    async def attach_derivatives(self, asset_id: str, preview_asset_id: str, thumbnail_asset_id: str) -> AssetDescriptor:
        path = "/internal/v1/assets/derivatives"
        body = json.dumps({"grant": self.grant, "assetId": asset_id, "previewAssetId": preview_asset_id, "thumbnailAssetId": thumbnail_asset_id}, separators=(",", ":")).encode()
        callback_url = f"{self.base_url.rsplit('/asset-uploads', 1)[0]}/assets/derivatives"
        async with self._upload_slots:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await self._request_with_retry(client, "POST", callback_url, content=body, headers=self._headers("POST", path, body), stage="attach-derivatives")
                payload = response.json()
                return AssetDescriptor.model_validate(payload.get("asset", payload))

    async def complete_process(self, result: dict) -> None:
        path = "/internal/v1/process-runs/complete"
        body = json.dumps({"grant": self.grant, "result": result}, separators=(",", ":")).encode()
        callback_url = f"{self.base_url.rsplit('/asset-uploads', 1)[0]}/process-runs/complete"
        async with self._upload_slots:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                await self._request_with_retry(client, "POST", callback_url, content=body, headers=self._headers("POST", path, body), stage="complete-process")
