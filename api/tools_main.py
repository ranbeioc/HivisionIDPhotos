from __future__ import annotations

import json
import os
from contextlib import contextmanager
from threading import BoundedSemaphore

import cv2
import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from services.image_tools import ImagePipeline, ImageToolsService, ToolsProcessContext
from services.image_validator import InvalidImage, validate_and_decode
from .security import verify_tools_gateway_request


app = FastAPI(title="XHalo Image Tools Compute", version="2.0.0", docs_url=None, redoc_url=None, openapi_url=None)
service = ImageToolsService()
admission = BoundedSemaphore(int(os.environ.get("IMAGE_TOOLS_MAX_INFLIGHT_REQUESTS", "1")))


def error_body(request: Request, code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "requestId": request.headers.get("x-request-id", "unknown")}}


@contextmanager
def admitted():
    if not admission.acquire(blocking=False):
        raise HTTPException(status_code=503, detail={"code": "ENGINE_UNAVAILABLE", "message": "Image Tools compute is full"}, headers={"retry-after": "5"})
    try:
        yield
    finally:
        admission.release()


@app.exception_handler(HTTPException)
async def http_error(request: Request, error: HTTPException):
    detail = error.detail if isinstance(error.detail, dict) else {"code": "INVALID_CONFIG", "message": str(error.detail)}
    return JSONResponse(error_body(request, str(detail.get("code", "INVALID_CONFIG")), str(detail.get("message", "Request failed"))), status_code=error.status_code, headers=error.headers)


@app.exception_handler(InvalidImage)
async def invalid_image(request: Request, error: InvalidImage):
    return JSONResponse(error_body(request, "INVALID_IMAGE", str(error)), status_code=415)


@app.get("/v1/health")
async def health(request: Request):
    await verify_tools_gateway_request(request)
    return {"ok": True, "service": "xhalo-image-tools-compute"}


@app.get("/v1/ready")
async def ready(request: Request):
    await verify_tools_gateway_request(request)
    return {"ready": True, "engines": ["server-basic-pillow", "opencv-fast"]}


async def _process(request: Request, broker_url: str, broker_grant: str, request_id: str, require_mask: bool) -> dict:
    await verify_tools_gateway_request(request)
    form = await request.form()
    image_file = form.get("image")
    pipeline_value = form.get("pipeline")
    if not isinstance(image_file, UploadFile) or not isinstance(pipeline_value, str):
        raise HTTPException(status_code=400, detail={"code": "INVALID_CONFIG", "message": "image and pipeline are required"})
    original = await image_file.read()
    image = validate_and_decode(original, image_file.content_type or "")
    try:
        pipeline = ImagePipeline.model_validate(json.loads(pipeline_value))
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_CONFIG", "message": str(error)}) from error
    mask: np.ndarray | None = None
    if require_mask:
        mask_file = form.get("mask")
        if not isinstance(mask_file, UploadFile) or mask_file.content_type != "image/png":
            raise HTTPException(status_code=400, detail={"code": "MASK_INVALID", "message": "A PNG mask is required"})
        mask_bytes = await mask_file.read()
        if not mask_bytes.startswith(b"\x89PNG\r\n\x1a\n") or len(mask_bytes) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail={"code": "MASK_INVALID", "message": "Mask must be a bounded PNG"})
        mask = cv2.imdecode(np.frombuffer(mask_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.size > 40_000_000:
            raise HTTPException(status_code=400, detail={"code": "MASK_INVALID", "message": "Mask cannot be decoded safely"})
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    with admitted():
        return await service.process_and_upload(image, len(original), pipeline, ToolsProcessContext(request_id=request_id, broker_url=broker_url, broker_grant=broker_grant, broker_hmac_secret=os.environ["IMAGE_TOOLS_HMAC_SECRET"]), mask)


@app.post("/v1/images/process")
async def process(request: Request, x_xhalo_asset_broker_url: str = Header(alias="x-xhalo-asset-broker-url"), x_xhalo_asset_grant: str = Header(alias="x-xhalo-asset-grant"), x_request_id: str = Header(alias="x-request-id")):
    return await _process(request, x_xhalo_asset_broker_url, x_xhalo_asset_grant, x_request_id, False)


@app.post("/v1/images/inpaint")
async def inpaint(request: Request, x_xhalo_asset_broker_url: str = Header(alias="x-xhalo-asset-broker-url"), x_xhalo_asset_grant: str = Header(alias="x-xhalo-asset-grant"), x_request_id: str = Header(alias="x-request-id")):
    return await _process(request, x_xhalo_asset_broker_url, x_xhalo_asset_grant, x_request_id, True)
