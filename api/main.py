from __future__ import annotations

import json
import logging
import os
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from starlette.datastructures import UploadFile
from fastapi.responses import JSONResponse

from hivision.error import APIError, FaceError
from services.contracts import IdPhotoOptions
from services.capacity import CapacityUnavailable, RequestAdmission
from services.id_photo import IdPhotoService, ProcessContext
from services.background_remove import BackgroundRemoveContext, BackgroundRemoveService
from services.asset_broker import AssetBrokerError
from services.image_validator import InvalidImage, validate_and_decode
from .security import verify_gateway_request


app = FastAPI(title="XHalo Hivision Compute", version="1.0.0", docs_url=None, redoc_url=None, openapi_url=None)
logger = logging.getLogger("xhalo.hivision")
service = IdPhotoService()
background_remove_service = BackgroundRemoveService(service.registry)
request_admission = RequestAdmission()


def admit_process_request():
    with request_admission.acquire():
        yield


def error_body(request: Request, code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message, "requestId": request.headers.get("x-request-id", "unknown")}}


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, error: HTTPException):
    if isinstance(error.detail, dict):
        code = str(error.detail.get("code", "INVALID_CONFIG"))
        detail = str(error.detail.get("message", "Request failed"))
    else:
        code = "INVALID_CONFIG"
        detail = str(error.detail)
    return JSONResponse(error_body(request, code, detail), status_code=error.status_code, headers=error.headers)


@app.exception_handler(CapacityUnavailable)
async def capacity_handler(request: Request, error: CapacityUnavailable):
    return JSONResponse(
        error_body(request, "MODEL_UNAVAILABLE", str(error)),
        status_code=503,
        headers={"retry-after": str(error.retry_after_seconds)},
    )


@app.exception_handler(InvalidImage)
async def invalid_image_handler(request: Request, error: InvalidImage):
    return JSONResponse(error_body(request, "INVALID_IMAGE", str(error)), status_code=415)


@app.exception_handler(FaceError)
async def face_handler(request: Request, error: FaceError):
    if error.face_num == 0:
        return JSONResponse(error_body(request, "FACE_NOT_FOUND", "No face was detected."), status_code=422)
    return JSONResponse(error_body(request, "FACE_COUNT_INVALID", "Exactly one face is required."), status_code=422)


@app.exception_handler(APIError)
async def model_handler(request: Request, error: APIError):
    return JSONResponse(error_body(request, "MODEL_UNAVAILABLE", str(error)), status_code=503)


@app.exception_handler(AssetBrokerError)
async def asset_broker_handler(request: Request, error: AssetBrokerError):
    logger.error(
        "asset_broker_failure request_id=%s stage=%s status=%s",
        request.headers.get("x-request-id", "unknown"),
        error.stage,
        error.status_code,
    )
    return JSONResponse(
        error_body(request, "ASSET_UPLOAD_FAILED", "Asset storage is temporarily unavailable. Please retry."),
        status_code=502,
        headers={"retry-after": "2"},
    )


@app.get("/v1/health", dependencies=[Depends(verify_gateway_request)])
async def health():
    return {"ok": True, "service": "hivision-compute"}


@app.get("/v1/ready", dependencies=[Depends(verify_gateway_request)])
async def ready():
    service.registry.ensure_ready()
    return {"ready": True}


@app.get("/v1/id-photo/config", dependencies=[Depends(verify_gateway_request)])
async def config():
    return service.config()


@app.post("/v1/id-photo/process", dependencies=[Depends(admit_process_request)])
async def process(
    request: Request,
    x_xhalo_asset_broker_url: str = Header(alias="x-xhalo-asset-broker-url"),
    x_xhalo_asset_grant: str = Header(alias="x-xhalo-asset-grant"),
    x_xhalo_input_asset_id: str | None = Header(default=None, alias="x-xhalo-input-asset-id"),
    x_xhalo_input_asset_saved: bool = Header(default=False, alias="x-xhalo-input-asset-saved"),
    x_request_id: str = Header(alias="x-request-id"),
):
    # The signature must cover the exact wire body before multipart parsing consumes it.
    await verify_gateway_request(request)
    if request.headers.get("content-type", "").startswith("application/json"):
        payload = await request.json()
        raise HTTPException(status_code=501, detail={"code": "ASSET_INPUT_NOT_READY", "inputAssetId": payload.get("inputAssetId")})
    form = await request.form()
    image = form.get("image")
    config = form.get("config")
    if not isinstance(image, UploadFile) or not isinstance(config, str):
        raise HTTPException(status_code=400, detail="image and config are required")
    original_bytes = await image.read()
    original_mime = (image.content_type or "").split(";")[0].strip()
    decoded = validate_and_decode(original_bytes, original_mime)
    try:
        options = IdPhotoOptions.model_validate(json.loads(config))
    except (ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=422, detail="Invalid config") from error
    result = await service.process(decoded, original_bytes, original_mime, options, ProcessContext(request_id=x_request_id, broker_url=x_xhalo_asset_broker_url, broker_grant=x_xhalo_asset_grant, broker_hmac_secret=os.environ["HIVISION_HMAC_SECRET"], input_asset_id=x_xhalo_input_asset_id, input_asset_saved=x_xhalo_input_asset_saved))
    return result.model_dump(by_alias=True, exclude_none=True)


@app.post("/v1/images/background-remove", dependencies=[Depends(admit_process_request)])
async def background_remove(
    request: Request,
    x_xhalo_asset_broker_url: str = Header(alias="x-xhalo-asset-broker-url"),
    x_xhalo_asset_grant: str = Header(alias="x-xhalo-asset-grant"),
    x_request_id: str = Header(alias="x-request-id"),
):
    await verify_gateway_request(request)
    form = await request.form()
    image_file = form.get("image")
    pipeline_value = form.get("pipeline")
    if not isinstance(image_file, UploadFile) or not isinstance(pipeline_value, str):
        raise HTTPException(status_code=400, detail={"code": "INVALID_CONFIG", "message": "image and pipeline are required"})
    try:
        pipeline = json.loads(pipeline_value)
        operation = next(value for value in pipeline.get("operations", []) if value.get("id") == "background-remove")
        model = str(operation.get("model", "modnet_photographic_portrait_matting"))
    except (ValueError, TypeError, json.JSONDecodeError, StopIteration) as error:
        raise HTTPException(status_code=422, detail={"code": "INVALID_CONFIG", "message": "A valid background-remove operation is required"}) from error
    original = await image_file.read()
    decoded = validate_and_decode(original, image_file.content_type or "")
    return await background_remove_service.process(decoded, len(original), model, BackgroundRemoveContext(request_id=x_request_id, broker_url=x_xhalo_asset_broker_url, broker_grant=x_xhalo_asset_grant, broker_hmac_secret=os.environ["HIVISION_HMAC_SECRET"]))


if os.environ.get("ENABLE_LEGACY_GRADIO") == "true":
    import gradio as gr
    from demo.processor import IDPhotoProcessor
    from demo.ui import create_ui
    from services.model_registry import get_model_registry

    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    registry = get_model_registry()
    legacy = create_ui(IDPhotoProcessor(model_registry=registry), root_dir, registry.available_matting_models(), registry.available_face_models(), ["zh", "en", "ko", "ja"])
    app = gr.mount_gradio_app(app, legacy, path="/legacy/gradio")
