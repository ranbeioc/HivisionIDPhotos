from __future__ import annotations

import asyncio
import io
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from hivision.creator.choose_handler import HUMAN_MATTING_MODELS
from hivision.creator.layout_calculator import generate_layout_array, generate_layout_image
from hivision.error import APIError, FaceError
from hivision.plugin.template.template_calculator import generte_template_photo
from hivision.utils import add_background, add_watermark, resize_image_to_kb

from .asset_broker import AssetBrokerClient
from .contracts import DEFAULT_OPTIONS, IdPhotoOptions, IdPhotoProcessResult, ProcessingMetadata, ResultAssets
from .model_registry import FACE_DETECTION_MODELS, ModelRegistry, get_model_registry


SIZE_PRESETS = {
    "one-inch": (413, 295),
    "two-inch": (626, 413),
    "small-one-inch": (378, 260),
    "small-two-inch": (531, 413),
    "large-one-inch": (567, 390),
    "large-two-inch": (626, 413),
    "five-inch": (1499, 1050),
    "teacher-qualification": (413, 295),
    "national-civil-service": (413, 295),
    "junior-accounting": (413, 295),
    "cet": (192, 144),
    "computer-rank-exam": (567, 390),
    "postgraduate-exam": (709, 531),
    "social-security-card": (441, 358),
    "electronic-driving-license": (378, 260),
    "us-visa": (600, 600),
    "japan-visa": (413, 295),
    "korea-visa": (531, 413),
}
PAPER_PIXELS = {"6-inch": (1205, 1795), "5-inch": (1051, 1500), "A4": (2479, 3508), "3R": (1051, 1500), "4R": (1205, 1795)}
CONFIG_VERSION = "hivision-21e59f6-contract-v6-face-catalog"

MATTING_MODEL_LABELS = {
    "modnet_photographic_portrait_matting": {
        "zh": "MODNet 人像抠图",
        "zh-Hant": "MODNet 人像去背",
        "en": "MODNet portrait matting",
        "ja": "MODNet 人物切り抜き",
        "ko": "MODNet 인물 누끼",
        "de": "MODNet-Portraitfreistellung",
        "fr": "Détourage portrait MODNet",
        "es": "Recorte de retrato MODNet",
        "it": "Scontorno ritratto MODNet",
        "pt": "Recorte de retrato MODNet",
    },
    "birefnet-v1-lite": {
        "zh": "BiRefNet v1 Lite（高精度）",
        "zh-Hant": "BiRefNet v1 Lite（高精度）",
        "en": "BiRefNet v1 Lite (high precision)",
        "ja": "BiRefNet v1 Lite（高精度）",
        "ko": "BiRefNet v1 Lite (고정밀)",
        "de": "BiRefNet v1 Lite (hohe Präzision)",
        "fr": "BiRefNet v1 Lite (haute précision)",
        "es": "BiRefNet v1 Lite (alta precisión)",
        "it": "BiRefNet v1 Lite (alta precisione)",
        "pt": "BiRefNet v1 Lite (alta precisão)",
    },
    "hivision_modnet": {
        "zh": "Hivision MODNet（纯色换底优化）",
        "zh-Hant": "Hivision MODNet（純色換底最佳化）",
        "en": "Hivision MODNet (solid-background optimized)",
        "ja": "Hivision MODNet（単色背景向け）",
        "ko": "Hivision MODNet (단색 배경 최적화)",
        "de": "Hivision MODNet (für Volltonhintergründe)",
        "fr": "Hivision MODNet (fonds unis)",
        "es": "Hivision MODNet (fondos lisos)",
        "it": "Hivision MODNet (sfondi uniformi)",
        "pt": "Hivision MODNet (fundos sólidos)",
    },
    "rmbg-1.4": {
        "zh": "RMBG 1.4（高质量）",
        "zh-Hant": "RMBG 1.4（高品質）",
        "en": "RMBG 1.4 (high quality)",
        "ja": "RMBG 1.4（高品質）",
        "ko": "RMBG 1.4 (고품질)",
        "de": "RMBG 1.4 (hohe Qualität)",
        "fr": "RMBG 1.4 (haute qualité)",
        "es": "RMBG 1.4 (alta calidad)",
        "it": "RMBG 1.4 (alta qualità)",
        "pt": "RMBG 1.4 (alta qualidade)",
    },
}

FACE_MODEL_LABELS = {
    "mtcnn": {
        "zh": "MTCNN（本地快速）", "zh-Hant": "MTCNN（本機快速）", "en": "MTCNN (fast local)",
        "ja": "MTCNN（ローカル高速）", "ko": "MTCNN (로컬 고속)", "de": "MTCNN (lokal, schnell)",
        "fr": "MTCNN (local, rapide)", "es": "MTCNN (local, rápido)", "it": "MTCNN (locale, rapido)",
        "pt": "MTCNN (local, rápido)",
    },
    "retinaface-resnet50": {
        "zh": "RetinaFace ResNet50（高精度）", "zh-Hant": "RetinaFace ResNet50（高精度）", "en": "RetinaFace ResNet50 (high accuracy)",
        "ja": "RetinaFace ResNet50（高精度）", "ko": "RetinaFace ResNet50 (고정밀)", "de": "RetinaFace ResNet50 (hohe Genauigkeit)",
        "fr": "RetinaFace ResNet50 (haute précision)", "es": "RetinaFace ResNet50 (alta precisión)", "it": "RetinaFace ResNet50 (alta precisione)",
        "pt": "RetinaFace ResNet50 (alta precisão)",
    },
}


def _composite_channels(value: str) -> tuple[int, int, int]:
    color = value.lstrip("#")
    rgb = tuple(int(color[index:index + 2], 16) for index in (0, 2, 4))
    # IDCreator returns RGBA and the Asset encoder preserves that channel order.
    # add_background's historical parameter name says BGR, but this service sends
    # its returned channels directly to PIL.
    return rgb


def _size(options: IdPhotoOptions, input_image: np.ndarray) -> tuple[int, int]:
    size = options.size
    if size.mode == "preset":
        if size.preset_id not in SIZE_PRESETS:
            raise ValueError("Unknown size preset")
        return SIZE_PRESETS[size.preset_id]
    if size.mode == "only-background":
        return input_image.shape[0], input_image.shape[1]
    if size.width is None or size.height is None:
        raise ValueError("Custom dimensions are required")
    if size.mode == "custom-mm":
        return round(size.height / 25.4 * options.output.dpi), round(size.width / 25.4 * options.output.dpi)
    return round(size.height), round(size.width)


def _encode(image: np.ndarray, mime: str, dpi: int, target_kb: int | None = None) -> bytes:
    image = np.uint8(image)
    if mime == "image/jpeg" and target_kb:
        return resize_image_to_kb(image, None, target_kb, dpi=dpi)
    alpha = image.ndim == 3 and image.shape[2] == 4 and mime == "image/png"
    pil = Image.fromarray(image).convert("RGBA" if alpha else "RGB")
    output = io.BytesIO()
    pil.save(output, format="PNG" if mime == "image/png" else "JPEG", quality=95, dpi=(dpi, dpi))
    return output.getvalue()


def _bounded_derivative(image: np.ndarray, max_edge: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    if scale == 1.0:
        return image
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    mode = "RGBA" if image.ndim == 3 and image.shape[2] == 4 else "RGB"
    resized = Image.fromarray(np.uint8(image)).convert(mode).resize(target, Image.Resampling.LANCZOS)
    return np.asarray(resized)


@dataclass(frozen=True)
class ProcessContext:
    request_id: str
    broker_url: str
    broker_grant: str
    broker_hmac_secret: str
    input_asset_id: str | None = None
    input_asset_saved: bool = False


class IdPhotoService:
    def __init__(self, registry: ModelRegistry | None = None):
        self.registry = registry or get_model_registry()

    def config(self) -> dict:
        labels = lambda zh, en, ja, ko: {"zh": zh, "en": en, "ja": ja, "ko": ko}
        installed_matting_models = set(self.registry.available_matting_models())
        available_face_models = set(self.registry.available_face_models())
        return {
            "version": CONFIG_VERSION,
            "sizePresets": [
                {"id": preset_id, "label": labels(zh, en, ja, ko), "width": width, "height": height, "unit": "px"}
                for preset_id, zh, en, ja, ko, width, height in (
                    ("one-inch", "一寸", "1 inch", "1インチ", "1인치", 295, 413),
                    ("two-inch", "二寸", "2 inch", "2インチ", "2인치", 413, 626),
                    ("small-one-inch", "小一寸", "Small 1 inch", "小1インチ", "소형 1인치", 260, 378),
                    ("small-two-inch", "小二寸", "Small 2 inch", "小2インチ", "소형 2인치", 413, 531),
                    ("large-one-inch", "大一寸", "Large 1 inch", "大1インチ", "대형 1인치", 390, 567),
                    ("large-two-inch", "大二寸", "Large 2 inch", "大2インチ", "대형 2인치", 413, 626),
                    ("five-inch", "五寸", "5 inch", "5インチ", "5인치", 1050, 1499),
                    ("teacher-qualification", "教师资格证", "Teacher qualification", "教員資格証", "교사 자격증", 295, 413),
                    ("national-civil-service", "国家公务员考试", "National civil service exam", "国家公務員試験", "국가 공무원 시험", 295, 413),
                    ("junior-accounting", "初级会计考试", "Junior accounting exam", "初級会計試験", "초급 회계 시험", 295, 413),
                    ("cet", "英语四六级考试", "CET-4 / CET-6", "中国語英語試験 CET", "CET-4 / CET-6", 144, 192),
                    ("computer-rank-exam", "计算机等级考试", "Computer rank exam", "コンピューター等級試験", "컴퓨터 등급 시험", 390, 567),
                    ("postgraduate-exam", "研究生考试", "Postgraduate exam", "大学院試験", "대학원 시험", 531, 709),
                    ("social-security-card", "社保卡", "Social security card", "社会保障カード", "사회 보장 카드", 358, 441),
                    ("electronic-driving-license", "电子驾驶证", "Electronic driving license", "電子運転免許証", "전자 운전면허증", 260, 378),
                    ("us-visa", "美国签证", "US visa", "米国ビザ", "미국 비자", 600, 600),
                    ("japan-visa", "日本签证", "Japan visa", "日本ビザ", "일본 비자", 295, 413),
                    ("korea-visa", "韩国签证", "Korea visa", "韓国ビザ", "한국 비자", 413, 531),
                )
            ],
            "backgroundPresets": [
                {"id": "white", "label": labels("白色", "White", "白", "흰색"), "type": "solid", "colors": ["#ffffff"]},
                {"id": "bright-blue", "label": labels("亮蓝色", "Bright blue", "ブライトブルー", "밝은 파랑"), "type": "solid", "colors": ["#438ff5"]},
                {"id": "hivision-blue", "label": labels("标准蓝色", "Hivision blue", "標準青", "표준 파랑"), "type": "solid", "colors": ["#628bce"]},
                {"id": "hivision-red", "label": labels("标准红色", "Hivision red", "標準赤", "표준 빨강"), "type": "solid", "colors": ["#d74532"]},
                {"id": "black", "label": labels("黑色", "Black", "黒", "검정"), "type": "solid", "colors": ["#000000"]},
                {"id": "dark-blue", "label": labels("深蓝色", "Dark blue", "濃紺", "진한 파랑"), "type": "solid", "colors": ["#4b6190"]},
                {"id": "light-gray", "label": labels("浅灰色", "Light gray", "ライトグレー", "연한 회색"), "type": "solid", "colors": ["#f2f0f0"]},
                {"id": "transparent", "label": labels("透明", "Transparent", "透明", "투명"), "type": "transparent", "colors": ["#ffffff"]},
            ],
            "mattingModels": [
                {
                    "id": model,
                    "label": MATTING_MODEL_LABELS[model],
                    "available": model in installed_matting_models,
                    "default": model == DEFAULT_OPTIONS.matting_model,
                    "heavy": model in {"rmbg-1.4", "birefnet-v1-lite"},
                }
                for model in HUMAN_MATTING_MODELS
            ],
            "faceDetectionModels": [
                {
                    "id": model,
                    "label": FACE_MODEL_LABELS[model],
                    "available": model in available_face_models,
                    "default": model == "mtcnn",
                    "heavy": model == "retinaface-resnet50",
                }
                for model in FACE_DETECTION_MODELS
            ],
            "paperSizes": [{"id": key, "label": labels(key, key, key, key), "widthMm": width / 300 * 25.4, "heightMm": height / 300 * 25.4} for key, (height, width) in PAPER_PIXELS.items()],
            "defaults": DEFAULT_OPTIONS.model_dump(by_alias=True),
            "limits": {"maxUploadBytes": 15_728_640, "maxWidth": 8192, "maxHeight": 8192, "maxPixels": 40_000_000, "maxDecodedBytes": 160_000_000, "processingTimeoutMs": 30_000},
            "features": {"camera": True, "processing": self.registry.ready(), "anonymousTemporaryAssets": False, "anonymousManualRetention": True, "authenticatedAutoSave": True, "layouts": True, "templates": True},
        }

    async def process(self, image: np.ndarray, original_bytes: bytes, original_mime: str, options: IdPhotoOptions, context: ProcessContext) -> IdPhotoProcessResult:
        started = time.perf_counter()
        size = _size(options, image)
        try:
            with self.registry.acquire(options.matting_model, options.face_detection_model) as creator:
                result = creator(image, size=size, change_bg_only=options.size.mode == "only-background", head_measure_ratio=options.geometry.head_measure_ratio, head_height_ratio=options.geometry.head_height_ratio, head_top_range=(options.geometry.top_distance_max, options.geometry.top_distance_min), whitening_strength=options.beauty.whitening, brightness_strength=options.beauty.brightness, contrast_strength=options.beauty.contrast, sharpen_strength=options.beauty.sharpen, saturation_strength=options.beauty.saturation, face_alignment=options.geometry.alignment, horizontal_flip=options.geometry.flip)
        except (FaceError, APIError):
            raise
        transparent_standard = np.uint8(result.standard)
        transparent_hd = np.uint8(result.hd)
        start_channels = _composite_channels(options.background.colors[0])
        end_channels = _composite_channels(options.background.colors[1]) if len(options.background.colors) > 1 else (255, 255, 255)
        mode = {"solid": "pure_color", "gradient-up-down": "updown_gradient", "gradient-center": "center_gradient", "transparent": "pure_color"}[options.background.type]
        standard = transparent_standard if options.background.type == "transparent" else np.uint8(add_background(transparent_standard, bgr=start_channels, end_bgr=end_channels, mode=mode))
        hd = transparent_hd if options.background.type == "transparent" else np.uint8(add_background(transparent_hd, bgr=start_channels, end_bgr=end_channels, mode=mode))
        if options.watermark.enabled:
            values = {"text": options.watermark.text, "size": options.watermark.size, "opacity": options.watermark.opacity, "angle": options.watermark.angle, "color": options.watermark.color, "space": options.watermark.spacing}
            standard, hd = add_watermark(standard, **values), add_watermark(hd, **values)
        variants: dict[str, tuple[np.ndarray, str]] = {}
        requested = set(options.output.variants)
        normal_mime = "image/jpeg" if options.output.format == "jpeg" else "image/png"
        if "standard" in requested: variants["standard"] = (standard, normal_mime)
        if "hd" in requested: variants["hd"] = (hd, normal_mime)
        if "transparentStandard" in requested: variants["transparentStandard"] = (transparent_standard, "image/png")
        if "transparentHd" in requested: variants["transparentHd"] = (transparent_hd, "image/png")
        if "layout" in requested and options.layout.enabled and options.size.mode != "only-background":
            layout_height, layout_width = PAPER_PIXELS[options.layout.paper_size]
            typography, rotate = generate_layout_array(size[0], size[1], LAYOUT_HEIGHT=layout_height, LAYOUT_WIDTH=layout_width)
            variants["layout"] = (generate_layout_image(standard, typography, rotate, height=size[0], width=size[1], crop_line=options.layout.crop_lines, LAYOUT_HEIGHT=layout_height, LAYOUT_WIDTH=layout_width), normal_mime)
        templates: list[tuple[str, np.ndarray, str]] = []
        if "templates" in requested and options.size.mode != "only-background":
            for name in ("template_1", "template_2"):
                templates.append((name, generte_template_photo(name, hd), normal_mime))
        broker = AssetBrokerClient(context.broker_url, context.broker_grant, context.request_id, context.broker_hmac_secret)
        original_asset_id = context.input_asset_id
        saved_original = context.input_asset_saved
        if original_asset_id is None:
            original = await broker.upload("original", original_mime, original_bytes, image.shape[1], image.shape[0], None, {})
            original_asset_id = original.asset_id
            saved_original = original.saved
        config_json = options.model_dump(by_alias=True)

        async def upload_bundle(kind: str, name: str, value: np.ndarray, mime: str):
            encoded = _encode(value, mime, options.output.dpi, options.output.target_kb if name == "standard" else None)
            primary = await broker.upload(name, mime, encoded, value.shape[1], value.shape[0], options.output.dpi, config_json, original_asset_id)
            preview_value = _bounded_derivative(value, 1600)
            thumbnail_value = _bounded_derivative(value, 320)
            preview, thumbnail = await asyncio.gather(
                broker.upload(f"{name}.preview", mime, _encode(preview_value, mime, options.output.dpi), preview_value.shape[1], preview_value.shape[0], options.output.dpi, {"maxEdge": 1600}, primary.asset_id),
                broker.upload(f"{name}.thumbnail", mime, _encode(thumbnail_value, mime, options.output.dpi), thumbnail_value.shape[1], thumbnail_value.shape[0], options.output.dpi, {"maxEdge": 320}, primary.asset_id),
            )
            descriptor = await broker.attach_derivatives(primary.asset_id, preview.asset_id, thumbnail.asset_id)
            return kind, name, descriptor

        completed = await asyncio.gather(
            *(upload_bundle("asset", name, value, mime) for name, (value, mime) in variants.items()),
            *(upload_bundle("template", name, value, mime) for name, value, mime in templates),
        )
        descriptors = {name: descriptor for kind, name, descriptor in completed if kind == "asset"}
        template_descriptors = [descriptor for kind, _name, descriptor in completed if kind == "template"]
        assets = ResultAssets(standard=descriptors.get("standard"), hd=descriptors.get("hd"), transparent_standard=descriptors.get("transparentStandard"), transparent_hd=descriptors.get("transparentHd"), layout=descriptors.get("layout"), templates=template_descriptors)
        response = IdPhotoProcessResult(request_id=context.request_id, input_asset_id=original_asset_id, saved_original=saved_original, assets=assets, processing=ProcessingMetadata(model=options.matting_model, face_model=options.face_detection_model, duration_ms=round((time.perf_counter() - started) * 1000), config_version=CONFIG_VERSION))
        await broker.complete_process(response.model_dump(by_alias=True, exclude_none=True))
        return response
