"""Full-reference (PSNR/SSIM vs LOL 'high') and simple no-reference image statistics."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from ..enhance.gating import mean_luminance


def psnr(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(ref, pred, data_range=255))


def ssim(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(structural_similarity(ref, pred, channel_axis=2, data_range=255))


def image_stats(img: np.ndarray) -> dict:
    """Brightness, contrast (luma std), and colourfulness (Hasler & Suesstrunk 2003)."""
    rgb = img.astype(np.float32)
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    colorfulness = np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return {
        "luminance": mean_luminance(img),
        "contrast": float(y.std() / 255.0),
        "colorfulness": float(colorfulness),
        "clipped_white_frac": float((img >= 250).all(axis=2).mean()),
    }
