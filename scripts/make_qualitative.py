#!/usr/bin/env python
"""Qualitative panels for the report: raw+detections | enhanced | enhanced+detections | ground truth.

    python scripts/make_qualitative.py --raw-exp E2 --enh-exp E4 --n 6 --pick gain
--pick gain  -> images where enhancement helped most (by matched TP count), 'loss' -> hurt most, 'random'.
"""

import _bootstrap  # noqa: F401

import argparse
import random
from pathlib import Path

import numpy as np

from darksight.config import experiment, paths
from darksight.data.exdark import read_yolo_labels, yolo_to_xyxy
from darksight.eval.detection_map import load_ground_truth, load_predictions, match_dataset
from darksight.io_utils import load_rgb, save_png
from darksight.visualize import draw_boxes


def dets_px(pred, w, h, conf_thr):
    cls, xyxyn, conf = pred
    keep = conf >= conf_thr
    return (xyxyn[keep] * [w, h, w, h]), cls[keep], conf[keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-exp", default="E2")
    ap.add_argument("--enh-exp", default="E4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--pick", choices=["gain", "loss", "random"], default="gain")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stems", nargs="*")
    args = ap.parse_args()

    ea, eb = experiment(args.raw_exp), experiment(args.enh_exp)
    pa = load_predictions(Path(ea["results_dir"]) / f"predictions_{args.split}.jsonl")
    pb = load_predictions(Path(eb["results_dir"]) / f"predictions_{args.split}.jsonl")
    gt = load_ground_truth(Path(ea["dataset_root"]) / "labels" / args.split)

    if args.stems:
        stems = args.stems
    elif args.pick == "random":
        stems = random.Random(0).sample(sorted(gt), args.n)
    else:
        def conf_filter(pred):
            cls, xyxyn, conf = pred
            keep = conf >= args.conf
            return cls[keep], xyxyn[keep], conf[keep]
        ma = match_dataset(gt, {s: conf_filter(v) for s, v in pa.items()})
        mb = match_dataset(gt, {s: conf_filter(v) for s, v in pb.items()})
        gain = {s: int(mb[s].tp.sum()) - int(ma[s].tp.sum()) for s in gt}
        stems = sorted(gain, key=lambda s: gain[s], reverse=args.pick == "gain")[:args.n]

    out_dir = paths()["results_root"] / "figures" / f"qualitative_{args.pick}"
    for s in stems:
        raw = load_rgb(Path(ea["dataset_root"]) / "images" / args.split / f"{s}.png")
        enh = load_rgb(Path(eb["dataset_root"]) / "images" / args.split / f"{s}.png")
        h, w = raw.shape[:2]
        labels = read_yolo_labels(Path(ea["dataset_root"]) / "labels" / args.split / f"{s}.txt")
        gt_img = draw_boxes(enh, [yolo_to_xyxy(*l[1:], w, h) for l in labels], [l[0] for l in labels])
        b, c, f = dets_px(pa[s], w, h, args.conf)
        raw_det = draw_boxes(raw, b, c, f)
        b, c, f = dets_px(pb[s], w, h, args.conf)
        enh_det = draw_boxes(enh, b, c, f)
        save_png(np.concatenate([raw_det, enh, enh_det, gt_img], axis=1), out_dir / f"{s}.png")
    print(f"wrote {len(stems)} panels (raw+det | enhanced | enhanced+det | GT) to {out_dir}")


if __name__ == "__main__":
    main()
