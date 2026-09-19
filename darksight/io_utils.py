"""Image I/O that behaves identically for raw and enhanced copies."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None  # a few ExDark photos are large; we resize right after loading

EXIF_ORIENTATION_TAG = 0x0112


def exif_orientation(img: Image.Image) -> int:
    try:
        return int(img.getexif().get(EXIF_ORIENTATION_TAG, 1))
    except Exception:
        return 1


def load_rgb(path: str | Path, apply_exif: bool = True) -> np.ndarray:
    """Load any image as HxWx3 uint8 RGB. Optionally applies EXIF rotation."""
    with Image.open(path) as im:
        if apply_exif:
            im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("RGB"), dtype=np.uint8).copy()


def load_rgb_with_info(path: str | Path, apply_exif: bool = True) -> tuple[np.ndarray, dict]:
    with Image.open(path) as im:
        info = {"orientation": exif_orientation(im), "mode": im.mode, "raw_size": im.size}
        if apply_exif:
            im = ImageOps.exif_transpose(im)
        arr = np.asarray(im.convert("RGB"), dtype=np.uint8).copy()
    return arr, info


def resize_long_side(img: np.ndarray, max_side: int | None) -> np.ndarray:
    """Downscale so max(H, W) <= max_side. Never upscales."""
    if not max_side:
        return img
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    return np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS))


def save_png(img: np.ndarray, path: str | Path) -> None:
    """Save lossless PNG without metadata (no EXIF => no orientation surprises)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path, format="PNG", compress_level=3)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def list_images(folder: str | Path) -> list[Path]:
    from .constants import IMAGE_EXTS

    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMAGE_EXTS)
