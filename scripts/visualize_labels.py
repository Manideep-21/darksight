#!/usr/bin/env python
"""Draw ground-truth boxes on random images (label sanity check before any training).

    python scripts/visualize_labels.py --root data/exdark_yolo/raw --split train --n 50
    python scripts/visualize_labels.py --root data/exdark_yolo/raw --stems 2015_00001 2015_03280
"""

import _bootstrap  # noqa: F401

import argparse
import random
from pathlib import Path

from darksight.data.exdark import read_yolo_labels, yolo_to_xyxy
from darksight.io_utils import load_rgb, save_png
from darksight.visualize import draw_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/exdark_yolo/raw")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--stems", nargs="*", help="specific image stems instead of a random sample")
    ap.add_argument("--out", default="reports/label_check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    img_dir = Path(args.root) / "images" / args.split
    lbl_dir = Path(args.root) / "labels" / args.split
    paths = sorted(img_dir.glob("*.png"))
    if args.stems:
        wanted = set(args.stems)
        paths = [p for p in paths if p.stem in wanted]
    else:
        random.Random(args.seed).shuffle(paths)
        paths = paths[:args.n]

    out = Path(args.out) / args.split
    for p in paths:
        img = load_rgb(p)
        h, w = img.shape[:2]
        labels = read_yolo_labels(lbl_dir / f"{p.stem}.txt")
        boxes = [yolo_to_xyxy(*l[1:], w, h) for l in labels]
        save_png(draw_boxes(img, boxes, [l[0] for l in labels]), out / f"{p.stem}.png")
    print(f"wrote {len(paths)} images to {out}")


if __name__ == "__main__":
    main()
