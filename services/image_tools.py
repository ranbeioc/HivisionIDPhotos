from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .asset_broker import AssetBrokerClient


class ToolOperation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    schema_version: int = Field(alias="schemaVersion", default=1)


class ImagePipeline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    operations: list[dict[str, Any]] = Field(min_length=1, max_length=20)

    @model_validator(mode="after")
    def validate_operations(self):
        supported = {"auto-orient", "resize", "crop", "rotate", "flip", "watermark", "convert", "compress", "metadata", "inpaint"}
        ids = [str(operation.get("id", "")) for operation in self.operations]
        if any(identifier not in supported for identifier in ids):
            raise ValueError("Pipeline contains an unsupported server operation")
        for singleton in ("convert", "compress", "metadata", "inpaint"):
            if ids.count(singleton) > 1:
                raise ValueError(f"{singleton} may appear only once")
        order = {"auto-orient": 0, "inpaint": 1, "crop": 2, "rotate": 2, "flip": 2, "resize": 2, "watermark": 3, "convert": 4, "compress": 4, "metadata": 5}
        if any(order[ids[index]] > order[ids[index + 1]] for index in range(len(ids) - 1)):
            raise ValueError("Pipeline operations are out of order")
        return self


@dataclass(frozen=True)
class ToolsProcessContext:
    request_id: str
    broker_url: str
    broker_grant: str
    broker_hmac_secret: str


def _hex_color(value: str) -> tuple[int, int, int, int]:
    clean = value.lstrip("#")
    if len(clean) not in {6, 8}:
        return 255, 255, 255, 255
    channels = tuple(int(clean[index:index + 2], 16) for index in range(0, len(clean), 2))
    return (*channels, 255) if len(channels) == 3 else channels


def _format_for(mime: str) -> tuple[str, str]:
    values = {
        "image/jpeg": ("JPEG", "jpg"),
        "image/png": ("PNG", "png"),
        "image/webp": ("WEBP", "webp"),
    }
    if mime not in values:
        raise ValueError("Unsupported output format")
    return values[mime]


class ImageToolsService:
    def process(self, image: np.ndarray, pipeline: ImagePipeline, mask: np.ndarray | None = None) -> tuple[bytes, str, int, int, list[dict[str, str]]]:
        working = Image.fromarray(np.uint8(image)).convert("RGBA")
        output_mime = "image/png" if working.mode == "RGBA" else "image/jpeg"
        quality = 90
        warnings: list[dict[str, str]] = []
        for operation in pipeline.operations:
            identifier = str(operation["id"])
            if identifier == "auto-orient":
                continue
            if identifier == "inpaint":
                if mask is None:
                    raise ValueError("MASK_INVALID")
                rgb = cv2.cvtColor(np.asarray(working.convert("RGB")), cv2.COLOR_RGB2BGR)
                resized_mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
                method = cv2.INPAINT_NS if operation.get("method") == "navier-stokes" else cv2.INPAINT_TELEA
                radius = min(25.0, max(1.0, float(operation.get("radius", 3))))
                working = Image.fromarray(cv2.cvtColor(cv2.inpaint(rgb, resized_mask, radius, method), cv2.COLOR_BGR2RGB)).convert("RGBA")
            elif identifier == "crop":
                width, height = working.size
                ratio = operation.get("unit", "px") == "ratio"
                x = round(float(operation.get("x", 0)) * width) if ratio else round(float(operation.get("x", 0)))
                y = round(float(operation.get("y", 0)) * height) if ratio else round(float(operation.get("y", 0)))
                crop_width = round(float(operation.get("width", 1)) * width) if ratio else round(float(operation.get("width", width)))
                crop_height = round(float(operation.get("height", 1)) * height) if ratio else round(float(operation.get("height", height)))
                x = max(0, min(width - 1, x))
                y = max(0, min(height - 1, y))
                right = max(x + 1, min(width, x + crop_width))
                bottom = max(y + 1, min(height, y + crop_height))
                working = working.crop((x, y, right, bottom))
            elif identifier == "rotate":
                working = working.rotate(-float(operation.get("degrees", 0)), expand=bool(operation.get("expand", True)), resample=Image.Resampling.BICUBIC)
            elif identifier == "flip":
                if operation.get("horizontal"): working = ImageOps.mirror(working)
                if operation.get("vertical"): working = ImageOps.flip(working)
            elif identifier == "resize":
                width = max(1, int(operation["width"]))
                height = max(1, int(operation["height"]))
                fit = str(operation.get("fit", "contain"))
                allow_upscale = bool(operation.get("allowUpscale", False))
                if fit == "fill":
                    target = (width, height) if allow_upscale else (min(width, working.width), min(height, working.height))
                    working = working.resize(target, Image.Resampling.LANCZOS)
                else:
                    scale = (max if fit == "cover" else min)(width / working.width, height / working.height)
                    if not allow_upscale:
                        scale = min(1.0, scale)
                    resized = (max(1, round(working.width * scale)), max(1, round(working.height * scale)))
                    working = working.resize(resized, Image.Resampling.LANCZOS)
                    if fit == "cover" and working.width >= width and working.height >= height:
                        left = (working.width - width) // 2
                        top = (working.height - height) // 2
                        working = working.crop((left, top, left + width, top + height))
            elif identifier == "watermark":
                overlay = Image.new("RGBA", working.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)
                text = str(operation.get("text", "XHalo Image"))[:200]
                font = ImageFont.load_default(size=max(10, min(160, int(operation.get("fontSize", 32)))))
                bbox = draw.textbbox((0, 0), text, font=font)
                margin = max(4, int(operation.get("margin", 24)))
                x = working.width - (bbox[2] - bbox[0]) - margin
                y = working.height - (bbox[3] - bbox[1]) - margin
                color = _hex_color(str(operation.get("color", "#ffffff")))
                opacity = max(0.0, min(1.0, float(operation.get("opacity", 0.65))))
                draw.text((x, y), text, font=font, fill=(*color[:3], round(color[3] * opacity)))
                working = Image.alpha_composite(working, overlay)
            elif identifier == "convert":
                output_mime = {"jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(str(operation.get("format")), "")
                if not output_mime:
                    raise ValueError("UNSUPPORTED_FORMAT")
            elif identifier == "compress":
                quality = max(1, min(100, int(operation.get("quality", 85))))
            elif identifier == "metadata":
                # Re-encoding intentionally strips GPS and all EXIF. ICC is only
                # retained when the source adapter can guarantee a safe profile.
                if operation.get("preserveIcc"):
                    warnings.append({"code": "ICC_PRESERVATION_UNAVAILABLE", "message": "Server Basic stripped the ICC profile because safe preservation could not be guaranteed."})
        format_name, _extension = _format_for(output_mime)
        if output_mime == "image/jpeg":
            background = Image.new("RGB", working.size, (255, 255, 255))
            if working.mode == "RGBA": background.paste(working, mask=working.getchannel("A"))
            encoded_image = background
        else:
            encoded_image = working
        output = io.BytesIO()
        save_args: dict[str, Any] = {"format": format_name}
        if output_mime in {"image/jpeg", "image/webp"}: save_args["quality"] = quality
        if output_mime == "image/png": save_args["compress_level"] = max(0, min(9, round((100 - quality) / 11)))
        encoded_image.save(output, **save_args)
        return output.getvalue(), output_mime, encoded_image.width, encoded_image.height, warnings

    async def process_and_upload(self, image: np.ndarray, input_bytes: int, pipeline: ImagePipeline, context: ToolsProcessContext, mask: np.ndarray | None = None) -> dict[str, Any]:
        started = time.perf_counter()
        encoded, mime, width, height, warnings = self.process(image, pipeline, mask)
        broker = AssetBrokerClient(context.broker_url, context.broker_grant, context.request_id, context.broker_hmac_secret)
        descriptor = await broker.upload("image-tool-result", mime, encoded, width, height, None, pipeline.model_dump(by_alias=True), operation="image.inpaint" if mask is not None else "image.process")
        public_descriptor = descriptor.model_dump(by_alias=True, exclude_none=True)
        public_descriptor["dpi"] = None
        return {
            "requestId": context.request_id,
            "engine": "opencv-fast" if mask is not None else "server-basic",
            "asset": public_descriptor,
            "warnings": warnings,
            "processing": {
                "durationMs": round((time.perf_counter() - started) * 1000),
                "inputBytes": input_bytes,
                "outputBytes": len(encoded),
                "width": width,
                "height": height,
                "pipelineVersion": pipeline.version,
            },
        }
