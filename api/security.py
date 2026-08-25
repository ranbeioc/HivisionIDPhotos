from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from collections import OrderedDict
from fastapi import HTTPException, Request
from services.request_id import is_valid_request_id


_seen: OrderedDict[str, float] = OrderedDict()


async def verify_gateway_request(request: Request) -> None:
    secret = os.environ.get("HIVISION_HMAC_SECRET", "")
    if not secret:
        raise HTTPException(status_code=503, detail="HMAC secret is not configured")
    request_id = request.headers.get("x-request-id", "")
    timestamp = request.headers.get("x-xhalo-timestamp", "")
    body_hash = request.headers.get("x-xhalo-body-sha256", "")
    signature = request.headers.get("x-xhalo-signature", "")
    try:
        timestamp_value = int(timestamp)
    except ValueError as error:
        raise HTTPException(status_code=401, detail="Invalid request timestamp") from error
    if abs(int(time.time() * 1000) - timestamp_value) > 300_000:
        raise HTTPException(status_code=401, detail="Expired request")
    if not is_valid_request_id(request_id):
        raise HTTPException(status_code=401, detail="Invalid request ID")
    body = await request.body()
    actual_hash = hashlib.sha256(body).hexdigest()
    canonical = f"{request.method}\n{request.url.path}\n{timestamp}\n{request_id}\n{actual_hash}".encode()
    expected = base64.urlsafe_b64encode(hmac.new(secret.encode(), canonical, hashlib.sha256).digest()).decode().rstrip("=")
    if not hmac.compare_digest(actual_hash, body_hash) or not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=401, detail="Invalid request signature")
    current = time.time()
    while _seen and next(iter(_seen.values())) < current - 300:
        _seen.popitem(last=False)
    replay_key = f"{request_id}:{body_hash}"
    if replay_key in _seen:
        raise HTTPException(status_code=409, detail="Replayed request")
    _seen[replay_key] = current
