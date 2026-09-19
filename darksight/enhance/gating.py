"""Decide whether an image is dark enough to be worth enhancing (limits over-enhancement)."""

import numpy as np


def mean_luminance(img: np.ndarray) -> float:
    """Mean Rec.601 luma of an HxWx3 uint8 RGB image, in [0, 1]."""
    rgb = img.astype(np.float32) / 255.0
    y = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    return float(y.mean())


def should_enhance(img: np.ndarray, tau: float = 0.35) -> bool:
    return mean_luminance(img) < tau
