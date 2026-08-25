from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ContractModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="forbid")


class SizeOptions(ContractModel):
    mode: Literal["preset", "only-background", "custom-px", "custom-mm"]
    preset_id: str | None = None
    width: float | None = None
    height: float | None = None


class BackgroundOptions(ContractModel):
    preset_id: str | None = None
    type: Literal["solid", "gradient-up-down", "gradient-center", "transparent"]
    colors: list[str] = Field(min_length=1, max_length=2)


class GeometryOptions(ContractModel):
    head_measure_ratio: float = Field(ge=0.05, le=0.5)
    head_height_ratio: float = Field(ge=0.1, le=0.9)
    top_distance_min: float = Field(ge=0, le=0.5)
    top_distance_max: float = Field(ge=0, le=0.5)
    alignment: bool = False
    flip: bool = False

    @model_validator(mode="after")
    def validate_range(self):
        if self.top_distance_min > self.top_distance_max:
            raise ValueError("topDistanceMin must not exceed topDistanceMax")
        return self


class BeautyOptions(ContractModel):
    # The legacy whitening implementation uses the value as an iteration count.
    # Keep the public contract aligned with the creator's integer-only parameter.
    whitening: int = Field(ge=0, le=15)
    brightness: int = Field(ge=-5, le=25)
    contrast: int = Field(ge=-10, le=50)
    saturation: int = Field(ge=-10, le=50)
    sharpen: int = Field(ge=0, le=5)


class OutputOptions(ContractModel):
    format: Literal["jpeg", "png", "transparent-png"]
    dpi: int = Field(ge=1, le=2400)
    target_kb: int | None = Field(default=None, ge=1, le=10240)
    variants: list[Literal["standard", "hd", "transparentStandard", "transparentHd", "layout", "templates"]]


class WatermarkOptions(ContractModel):
    enabled: bool = False
    text: str = Field(default="", max_length=100)
    color: str = "#000000"
    size: int = Field(default=20, ge=8, le=200)
    opacity: float = Field(default=0.5, ge=0, le=1)
    angle: float = Field(default=30, ge=-180, le=180)
    spacing: int = Field(default=25, ge=0, le=500)


class LayoutOptions(ContractModel):
    enabled: bool = True
    paper_size: Literal["6-inch", "5-inch", "A4", "3R", "4R"] = "6-inch"
    crop_lines: bool = False


class IdPhotoOptions(ContractModel):
    size: SizeOptions
    background: BackgroundOptions
    matting_model: str
    face_detection_model: str
    geometry: GeometryOptions
    beauty: BeautyOptions
    output: OutputOptions
    watermark: WatermarkOptions
    layout: LayoutOptions


class AssetDescriptor(ContractModel):
    # The Asset API may add lifecycle metadata (status, filename, timestamps)
    # that is not part of the Image process result contract. Required process
    # descriptor fields remain strict while compatible metadata is ignored.
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")

    asset_id: str
    variant: str
    mime: Literal["image/jpeg", "image/png", "image/webp"]
    width: int
    height: int
    bytes: int
    dpi: int | None
    preview_url: str
    download_url: str
    url_expires_at: str
    saved: bool


class ProcessingMetadata(ContractModel):
    model: str
    face_model: str
    duration_ms: int
    config_version: str


class ResultAssets(ContractModel):
    standard: AssetDescriptor | None = None
    hd: AssetDescriptor | None = None
    transparent_standard: AssetDescriptor | None = None
    transparent_hd: AssetDescriptor | None = None
    layout: AssetDescriptor | None = None
    templates: list[AssetDescriptor] = Field(default_factory=list)


class IdPhotoProcessResult(ContractModel):
    request_id: str
    input_asset_id: str
    saved_original: bool
    assets: ResultAssets
    processing: ProcessingMetadata


DEFAULT_OPTIONS = IdPhotoOptions.model_validate({
    "size": {"mode": "preset", "presetId": "one-inch"},
    "background": {"presetId": "blue", "type": "solid", "colors": ["#438ff5"]},
    "mattingModel": "modnet_photographic_portrait_matting", "faceDetectionModel": "mtcnn",
    "geometry": {"headMeasureRatio": 0.2, "headHeightRatio": 0.45, "topDistanceMin": 0.1, "topDistanceMax": 0.12, "alignment": False, "flip": False},
    "beauty": {"whitening": 0, "brightness": 0, "contrast": 0, "saturation": 0, "sharpen": 0},
    "output": {"format": "jpeg", "dpi": 300, "targetKb": None, "variants": ["standard", "hd", "transparentStandard", "transparentHd", "layout", "templates"]},
    "watermark": {"enabled": False, "text": "", "color": "#000000", "size": 20, "opacity": 0.5, "angle": 30, "spacing": 25},
    "layout": {"enabled": True, "paperSize": "6-inch", "cropLines": False},
})
