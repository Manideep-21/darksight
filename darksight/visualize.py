"""Box drawing shared by label checks, qualitative figures and the demo."""

from __future__ import annotations

import cv2
import numpy as np

from .constants import CLASS_NAMES

# 12 distinct colours (RGB), one per class.
PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
    (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
]


def draw_boxes(img: np.ndarray, xyxy, cls, conf=None, thickness: int | None = None) -> np.ndarray:
    """Return a copy of an RGB image with labelled boxes."""
    out = np.ascontiguousarray(img.copy())
    h, w = out.shape[:2]
    thickness = thickness or max(1, round(max(h, w) / 400))
    font_scale = max(0.4, max(h, w) / 1600)
    for i, (box, c) in enumerate(zip(xyxy, cls)):
        c = int(c)
        color = PALETTE[c % len(PALETTE)]
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = CLASS_NAMES[c] if conf is None else f"{CLASS_NAMES[c]} {float(conf[i]):.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        ty = max(y1, th + base + 2)
        cv2.rectangle(out, (x1, ty - th - base - 2), (x1 + tw + 2, ty), color, -1)
        cv2.putText(out, label, (x1 + 1, ty - base - 1), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out
