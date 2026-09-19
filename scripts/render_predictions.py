#!/usr/bin/env python
"""Draw an experiment's predicted boxes on every test image and export all detections to CSV.

    python scripts/render_predictions.py --exp E0 E1 --conf 0.25

Writes results/<EXP>/detections_<split>/<stem>.jpg   (image the model saw + predicted boxes, no ground truth)
       results/<EXP>/detections_<split>.csv          (stem, class, confidence, x1, y1, x2, y2 in pixels)
With --side-by-side A B also writes results/<A>_vs_<B>_<split>/<stem>.jpg (left: A, right: B).
"""

import _bootstrap  # noqa: F401

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from darksight.config import experiment, paths
from darksight.constants import CLASS_NAMES
from darksight.eval.detection_map import load_predictions
from darksight.io_utils import load_rgb
from darksight.visualize import draw_boxes


def render(exp, stem, preds, split, conf):
    img = load_rgb(Path(exp["dataset_root"]) / "images" / split / f"{stem}.png")
    h, w = img.shape[:2]
    cls, xyxyn, c = preds.get(stem, (np.zeros(0, int), np.zeros((0, 4)), np.zeros(0)))
    keep = c >= conf
    boxes = xyxyn[keep] * [w, h, w, h]
    return draw_boxes(img, boxes, cls[keep], c[keep]), boxes, cls[keep], c[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25, help="only draw/export boxes at or above this confidence")
    ap.add_argument("--side-by-side", nargs=2, metavar=("A", "B"))
    ap.add_argument("--quality", type=int, default=90)
    args = ap.parse_args()

    loaded = {}
    for exp_id in args.exp:
        exp = experiment(exp_id)
        preds = load_predictions(Path(exp["results_dir"]) / f"predictions_{args.split}.jsonl")
        loaded[exp_id] = (exp, preds)
        out_dir = Path(exp["results_dir"]) / f"detections_{args.split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(Path(exp["results_dir"]) / f"detections_{args.split}.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["image", "class", "confidence", "x1", "y1", "x2", "y2"])
            for stem in tqdm(sorted(preds), desc=f"{exp_id} render"):
                vis, boxes, cls, conf = render(exp, stem, preds, args.split, args.conf)
                Image.fromarray(vis).save(out_dir / f"{stem}.jpg", quality=args.quality)
                for b, k, s in zip(boxes, cls, conf):
                    wr.writerow([stem, CLASS_NAMES[int(k)], f"{s:.3f}", *(f"{v:.1f}" for v in b)])
        print(f"{exp_id}: {len(preds)} images -> {out_dir}")

    if args.side_by_side:
        a, b = args.side_by_side
        for e in (a, b):
            if e not in loaded:
                exp = experiment(e)
                loaded[e] = (exp, load_predictions(Path(exp["results_dir"]) / f"predictions_{args.split}.jsonl"))
        out_dir = paths()["results_root"] / f"{a}_vs_{b}_{args.split}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for stem in tqdm(sorted(loaded[a][1]), desc=f"{a} vs {b}"):
            left = render(loaded[a][0], stem, loaded[a][1], args.split, args.conf)[0]
            right = render(loaded[b][0], stem, loaded[b][1], args.split, args.conf)[0]
            Image.fromarray(np.concatenate([left, right], axis=1)).save(out_dir / f"{stem}.jpg", quality=args.quality)
        print(f"side-by-side -> {out_dir}")


if __name__ == "__main__":
    main()
