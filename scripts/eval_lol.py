#!/usr/bin/env python
"""Enhancer quality on LOL: PSNR/SSIM vs the normal-light reference, for input / Zero-DCE / classical baselines.

    python scripts/eval_lol.py --root data/lol --split test --save-examples 5
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from darksight.config import paths
from darksight.enhance.classical import clahe, gamma
from darksight.enhance.enhancer import ZeroDCEEnhancer
from darksight.eval.enhance_quality import image_stats, psnr, ssim
from darksight.io_utils import load_rgb, save_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/lol")
    ap.add_argument("--split", default="test")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-examples", type=int, default=5)
    args = ap.parse_args()

    enh = ZeroDCEEnhancer(**({"weights": args.weights} if args.weights else {}), device=args.device)
    methods = {"input_low": lambda x: x, "zerodce": enh.enhance, "clahe": clahe, "gamma_0.5": gamma}
    low_dir, high_dir = Path(args.root) / args.split / "low", Path(args.root) / args.split / "high"
    files = sorted(low_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"no images in {low_dir} — run scripts/prepare_lol.py first")

    res_root = paths()["results_root"]
    rows = []
    for i, p in enumerate(files):
        low, high = load_rgb(p), load_rgb(high_dir / p.name)
        panels = [low]
        for name, fn in methods.items():
            out = fn(low)
            rows.append({"image": p.name, "method": name, "psnr": psnr(out, high), "ssim": ssim(out, high),
                         **image_stats(out)})
            if name != "input_low":
                panels.append(out)
        if i < args.save_examples:
            save_png(np.concatenate(panels + [high], axis=1), res_root / "figures" / "lol" / p.name)

    df = pd.DataFrame(rows)
    summary = df.groupby("method")[["psnr", "ssim", "luminance", "contrast", "colorfulness"]].mean()
    summary = summary.reindex(list(methods))
    print(summary.round(4).to_string())
    print("example panels: low | zerodce | clahe | gamma | high")
    df.to_csv(res_root / f"lol_{args.split}_per_image.csv", index=False)
    (res_root / f"lol_{args.split}_summary.json").write_text(json.dumps(summary.round(5).to_dict(orient="index"), indent=2))


if __name__ == "__main__":
    main()
