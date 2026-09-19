"""Classical baselines for experiment E6: CLAHE on luminance and gamma correction."""

import cv2
import numpy as np


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def gamma(img: np.ndarray, g: float = 0.5) -> np.ndarray:
    lut = (np.linspace(0, 1, 256) ** g * 255 + 0.5).astype(np.uint8)
    return lut[img]
