from __future__ import annotations

import io
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


class InvalidImage(ValueError):
    pass


def validate_and_decode(data: bytes, declared_mime: str, *, max_bytes: int = 15_728_640, max_width: int = 8192, max_height: int = 8192, max_pixels: int = 40_000_000, max_decoded_bytes: int = 160_000_000) -> np.ndarray:
    if not data or len(data) > max_bytes:
        raise InvalidImage("Image byte length is invalid")
    magic_mime = "image/jpeg" if data.startswith(b"\xff\xd8\xff") else "image/png" if data.startswith(b"\x89PNG\r\n\x1a\n") else "image/webp" if data.startswith(b"RIFF") and data[8:12] == b"WEBP" else None
    if magic_mime is None or declared_mime.split(";")[0].strip() != magic_mime:
        raise InvalidImage("MIME and magic bytes do not match")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if width <= 0 or height <= 0 or width > max_width or height > max_height or width * height > max_pixels or width * height * 4 > max_decoded_bytes:
                raise InvalidImage("Decoded image dimensions exceed limits")
            probe.verify()
        with Image.open(io.BytesIO(data)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
            return np.asarray(image)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise InvalidImage("Image decode failed") from error
