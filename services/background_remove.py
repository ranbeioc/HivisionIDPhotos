from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .asset_broker import AssetBrokerClient
from .model_registry import ModelRegistry, get_model_registry


@dataclass(frozen=True)
class BackgroundRemoveContext:
    request_id: str
    broker_url: str
    broker_grant: str
    broker_hmac_secret: str


class BackgroundRemoveService:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or get_model_registry()

    async def process(self, image: np.ndarray, input_bytes: int, model: str, context: BackgroundRemoveContext) -> dict[str, Any]:
        started = time.perf_counter()
        with self.registry.acquire(model, "mtcnn") as creator:
            result = creator(image, size=image.shape[:2], change_bg_only=True)
        rgba = Image.fromarray(np.uint8(result.matting)).convert("RGBA")
        output = io.BytesIO()
        rgba.save(output, format="PNG", optimize=True)
        encoded = output.getvalue()
        broker = AssetBrokerClient(context.broker_url, context.broker_grant, context.request_id, context.broker_hmac_secret)
        descriptor = await broker.upload("background-removed", "image/png", encoded, rgba.width, rgba.height, None, {"model": model}, operation="image.background-remove")
        public_descriptor = descriptor.model_dump(by_alias=True, exclude_none=True)
        public_descriptor["dpi"] = None
        return {
            "requestId": context.request_id,
            "engine": "background-remove",
            "asset": public_descriptor,
            "warnings": [],
            "processing": {
                "durationMs": round((time.perf_counter() - started) * 1000),
                "inputBytes": input_bytes,
                "outputBytes": len(encoded),
                "width": rgba.width,
                "height": rgba.height,
                "pipelineVersion": 1,
            },
        }
