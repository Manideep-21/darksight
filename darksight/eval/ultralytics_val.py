"""Official-style numbers straight from Ultralytics model.val()."""

from __future__ import annotations

from pathlib import Path

from ..constants import CLASS_NAMES
from ..device import select_device


def run_val(weights: str | Path, data_yaml: str | Path, split: str = "test", imgsz: int = 640,
            batch: int = 16, device: str | None = None, project: str | Path = "runs/val",
            name: str = "val") -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, conf=0.001, iou=0.7,
                  device=select_device(device), plots=True, save_json=False, project=str(project),
                  name=name, exist_ok=True, verbose=False)
    box = m.box
    per_class = {CLASS_NAMES[i]: {"ap50": None, "ap50_95": None, "precision": None, "recall": None}
                 for i in range(len(CLASS_NAMES))}
    for k, c in enumerate(box.ap_class_index):
        p, r, ap50, ap = box.class_result(k)
        per_class[CLASS_NAMES[int(c)]] = {"ap50": float(ap50), "ap50_95": float(ap),
                                          "precision": float(p), "recall": float(r)}
    return {
        "map50": float(box.map50), "map50_95": float(box.map),
        "precision": float(box.mp), "recall": float(box.mr),
        "per_class": per_class,
        "speed_ms": {k: float(v) for k, v in m.speed.items()},
        "save_dir": str(m.save_dir),
    }
