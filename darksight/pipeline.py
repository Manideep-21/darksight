"""End-to-end DarkSight inference: (optional gate) -> Zero-DCE -> YOLO."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .detect.detector import Detections, Detector
from .enhance.enhancer import ZeroDCEEnhancer
from .enhance.gating import mean_luminance
from .io_utils import resize_long_side


@dataclass
class PipelineResult:
    input_image: np.ndarray
    enhanced_image: np.ndarray | None = None
    raw_detections: Detections | None = None
    enhanced_detections: Detections | None = None
    luminance: float = 0.0
    enhancement_applied: bool = False
    timings_ms: dict = field(default_factory=dict)


def _sync(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


class DarkSightPipeline:
    """mode: 'raw' (detector on raw), 'enhanced' (Zero-DCE -> detector), 'both'."""

    def __init__(self, enhancer: ZeroDCEEnhancer, det_raw: Detector, det_enh: Detector | None = None,
                 max_side: int = 1280):
        self.enhancer = enhancer
        self.det_raw = det_raw
        self.det_enh = det_enh or det_raw
        self.max_side = max_side

    def run(self, img: np.ndarray, mode: str = "both", conf: float = 0.25, gate_tau: float | None = None,
            imgsz: int = 640) -> PipelineResult:
        if mode not in {"raw", "enhanced", "both"}:
            raise ValueError(f"unknown mode {mode}")
        img = resize_long_side(img, self.max_side)
        out = PipelineResult(input_image=img, luminance=mean_luminance(img))

        if mode in {"raw", "both"}:
            t = time.perf_counter()
            out.raw_detections = self.det_raw.predict(img, conf=conf, imgsz=imgsz)
            _sync(self.det_raw.device)
            out.timings_ms["detect_raw"] = (time.perf_counter() - t) * 1000

        if mode in {"enhanced", "both"}:
            t = time.perf_counter()
            if gate_tau is not None and out.luminance >= gate_tau:
                out.enhanced_image = img
            else:
                out.enhanced_image = self.enhancer.enhance(img)
                out.enhancement_applied = True
            _sync(self.enhancer.device)
            out.timings_ms["enhance"] = (time.perf_counter() - t) * 1000

            t = time.perf_counter()
            out.enhanced_detections = self.det_enh.predict(out.enhanced_image, conf=conf, imgsz=imgsz)
            _sync(self.det_enh.device)
            out.timings_ms["detect_enhanced"] = (time.perf_counter() - t) * 1000
        return out
