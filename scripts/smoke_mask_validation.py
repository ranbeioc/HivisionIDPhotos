from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
import uuid

import cv2
import httpx
import numpy as np


def multipart(boundary: str, image: bytes, mask: bytes) -> bytes:
    pipeline = json.dumps(
        {
            "version": 1,
            "operations": [
                {"schemaVersion": 1, "id": "inpaint", "method": "telea", "radius": 5},
                {"schemaVersion": 1, "id": "convert", "format": "png"},
            ],
        },
        separators=(",", ":"),
    )
    sections = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"pipeline\"\r\n\r\n{pipeline}\r\n".encode(),
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"input.png\"\r\nContent-Type: image/png\r\n\r\n".encode()
        + image
        + b"\r\n",
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"mask\"; filename=\"mask.png\"\r\nContent-Type: image/png\r\n\r\n".encode()
        + mask
        + b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(sections)


def main() -> None:
    ok, encoded = cv2.imencode(".png", np.full((32, 32, 3), 180, dtype=np.uint8))
    if not ok:
        raise RuntimeError("Could not create input fixture")
    boundary = "xhalo-mask-smoke"
    body = multipart(boundary, encoded.tobytes(), b"not-a-png")
    request_id = str(uuid.uuid4())
    timestamp = str(int(time.time() * 1000))
    body_hash = hashlib.sha256(body).hexdigest()
    canonical = f"POST\n/v1/images/inpaint\n{timestamp}\n{request_id}\n{body_hash}".encode()
    signature = base64.urlsafe_b64encode(
        hmac.new(os.environ["IMAGE_TOOLS_HMAC_SECRET"].encode(), canonical, hashlib.sha256).digest()
    ).decode().rstrip("=")
    response = httpx.post(
        "http://127.0.0.1:8091/v1/images/inpaint",
        content=body,
        headers={
            "content-type": f"multipart/form-data; boundary={boundary}",
            "x-request-id": request_id,
            "x-xhalo-timestamp": timestamp,
            "x-xhalo-body-sha256": body_hash,
            "x-xhalo-signature": signature,
            "x-xhalo-asset-broker-url": "https://invalid.local",
            "x-xhalo-asset-grant": "smoke-only",
        },
        timeout=10,
    )
    payload = response.json()
    assert response.status_code == 400, (response.status_code, payload)
    assert payload["error"]["code"] == "MASK_INVALID", payload
    print(json.dumps({"status": response.status_code, "code": payload["error"]["code"], "requestId": request_id}))


if __name__ == "__main__":
    main()
