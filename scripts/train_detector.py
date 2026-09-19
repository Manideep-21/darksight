#!/usr/bin/env python
"""Fine-tune YOLOv8 on a prepared dataset with the shared settings in configs/train.yaml.

    python scripts/train_detector.py --dataset data/exdark_yolo/raw --name det_raw
    python scripts/train_detector.py --dataset data/exdark_yolo/enhanced --name det_enh
    python scripts/train_detector.py --resume runs/detect/det_raw/weights/last.pt   # after a Colab disconnect
Smoke test: --model yolov8n.pt --epochs 3 --fraction 0.1
"""

import _bootstrap  # noqa: F401

import argparse
import json
import shutil
from pathlib import Path

from darksight.config import load_yaml
from darksight.constants import PROJECT_ROOT
from darksight.data.yolo_dataset import write_data_yaml
from darksight.device import select_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="dataset root containing images/ and labels/")
    ap.add_argument("--name", help="run name, e.g. det_raw / det_enh")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "train.yaml"))
    ap.add_argument("--project", default=str(PROJECT_ROOT / "runs" / "detect"), help="set to a Drive path on Colab")
    ap.add_argument("--resume", help="path to last.pt to resume an interrupted run")
    ap.add_argument("--device", default=None)
    ap.add_argument("--model"); ap.add_argument("--epochs", type=int); ap.add_argument("--batch", type=int)
    ap.add_argument("--seed", type=int); ap.add_argument("--fraction", type=float, default=1.0)
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.resume:
        YOLO(args.resume).train(resume=True)
        return
    if not (args.dataset and args.name):
        ap.error("--dataset and --name are required unless --resume is given")

    cfg = load_yaml(args.config)
    for k in ("model", "epochs", "batch", "seed"):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    model_name = cfg.pop("model")
    local = PROJECT_ROOT / "weights" / model_name
    if local.exists():  # pre-downloaded COCO checkpoint (GitHub release downloads can be blocked)
        model_name = str(local)
    data_yaml = write_data_yaml(args.dataset)  # refresh absolute path for this machine

    model = YOLO(model_name)
    model.train(data=str(data_yaml), project=args.project, name=args.name, exist_ok=False,
                device=select_device(args.device), fraction=args.fraction, **cfg)

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    dst = PROJECT_ROOT / "weights" / f"{args.name}.pt"
    shutil.copy2(best, dst)
    (save_dir / "darksight_train_config.json").write_text(json.dumps(
        {"model": model_name, "dataset": str(args.dataset), "fraction": args.fraction, **cfg}, indent=2))
    print(f"best weights copied to {dst}")


if __name__ == "__main__":
    main()
