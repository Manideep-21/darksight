"""Thin wrapper over Ultralytics YOLO that optionally remaps COCO classes to ExDark ids."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import COCO_NAMES_EXPECTED, COCO_TO_EXDARK, CLASS_NAMES
from ..device import select_device


@dataclass
class Detections:
    """xyxy in pixels of the input image; cls are ExDark ids after remapping."""
    xyxy: np.ndarray   # (N, 4) float32
    conf: np.ndarray   # (N,) float32
    cls: np.ndarray    # (N,) int64

    def __len__(self) -> int:
        return len(self.conf)

    def as_rows(self) -> list[dict]:
        return [
            {"class": CLASS_NAMES[int(c)], "conf": round(float(s), 3),
             "box": [round(float(v), 1) for v in b]}
            for b, s, c in zip(self.xyxy, self.conf, self.cls)
        ]


class Detector:
    def __init__(self, weights: str | Path, device: str | None = None, coco_remap: bool = False):
        from ultralytics import YOLO

        from ..constants import PROJECT_ROOT

        local = PROJECT_ROOT / "weights" / Path(weights).name
        if not Path(weights).exists() and local.exists():
            weights = local  # bare names like "yolov8s.pt" resolve to the project's weights/ folder first
        self.model = YOLO(str(weights))
        self.device = select_device(device)
        self.coco_remap = coco_remap
        if coco_remap:
            names = self.model.names
            for coco_id, expected in COCO_NAMES_EXPECTED.items():
                if names.get(coco_id) != expected:
                    raise ValueError(f"COCO id {coco_id} is '{names.get(coco_id)}', expected '{expected}'")
            self._lut = np.full(len(names), -1, dtype=np.int64)
            for coco_id, ex_id in COCO_TO_EXDARK.items():
                self._lut[coco_id] = ex_id

    def predict(self, img: np.ndarray, conf: float = 0.25, iou: float = 0.7, imgsz: int = 640,
                max_det: int = 300) -> Detections:
        """img: HxWx3 uint8 RGB."""
        # Ultralytics treats numpy input as BGR (OpenCV convention).
        res = self.model.predict(np.ascontiguousarray(img[..., ::-1]), conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
                                 device=self.device, verbose=False)[0]
        return self._to_detections(res)

    def predict_paths(self, paths: list[str], conf: float = 0.001, iou: float = 0.7, imgsz: int = 640,
                      max_det: int = 300, batch: int = 16):
        """Yields (path, orig_hw, Detections) for files on disk."""
        for i in range(0, len(paths), batch):
            chunk = paths[i:i + batch]
            results = self.model.predict(chunk, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det,
                                         device=self.device, verbose=False)
            for p, res in zip(chunk, results):
                yield p, res.orig_shape, self._to_detections(res)

    def _to_detections(self, res) -> Detections:
        b = res.boxes
        xyxy = b.xyxy.cpu().numpy().astype(np.float32)
        conf = b.conf.cpu().numpy().astype(np.float32)
        cls = b.cls.cpu().numpy().astype(np.int64)
        if self.coco_remap:
            cls = self._lut[cls]
            keep = cls >= 0
            xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
        return Detections(xyxy, conf, cls)
