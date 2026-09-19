#!/usr/bin/env python
"""Create an enhanced copy of a YOLO dataset: same filenames, same sizes, byte-identical labels.

    python scripts/enhance_dataset.py --src data/exdark_yolo/raw --dst data/exdark_yolo/enhanced
    python scripts/enhance_dataset.py --dst data/exdark_yolo/enhanced_gated --gate-tau 0.35   # E5
    python scripts/enhance_dataset.py --dst data/exdark_yolo/clahe --method clahe              # E6
"""

import _bootstrap  # noqa: F401

import argparse
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from darksight.data.yolo_dataset import write_data_yaml
from darksight.enhance.classical import clahe, gamma
from darksight.enhance.gating import mean_luminance
from darksight.eval.enhance_quality import image_stats
from darksight.io_utils import load_rgb, save_png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/exdark_yolo/raw")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--method", choices=["zerodce", "clahe", "gamma"], default="zerodce")
    ap.add_argument("--weights", default=None, help="Zero-DCE weights (default: weights/zerodce_epoch99.pth)")
    ap.add_argument("--gate-tau", type=float, default=None, help="only enhance images with mean luminance < tau")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--half", action="store_true", help="fp16 on CUDA")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    if args.method == "zerodce":
        from darksight.enhance.enhancer import ZeroDCEEnhancer

        kw = {"weights": args.weights} if args.weights else {}
        fn = ZeroDCEEnhancer(device=args.device, half=args.half, **kw).enhance
    else:
        fn = clahe if args.method == "clahe" else gamma

    log = {"method": args.method, "gate_tau": args.gate_tau, "splits": {}}
    for split in args.splits:
        imgs = sorted((src / "images" / split).glob("*.png"))
        (dst / "labels").mkdir(parents=True, exist_ok=True)
        if (dst / "labels" / split).exists():
            shutil.rmtree(dst / "labels" / split)
        shutil.copytree(src / "labels" / split, dst / "labels" / split)

        n_enh, lum_before, lum_after, clipped = 0, 0.0, 0.0, 0.0
        for p in tqdm(imgs, desc=f"{args.method}:{split}"):
            out = dst / "images" / split / p.name
            if out.exists() and not args.overwrite:
                continue
            img = load_rgb(p)
            lum = mean_luminance(img)
            if args.gate_tau is None or lum < args.gate_tau:
                res = fn(img)
                n_enh += 1
            else:
                res = img
            assert res.shape == img.shape, f"size changed for {p.name}: {img.shape} -> {res.shape}"
            s = image_stats(res)
            lum_before += lum; lum_after += s["luminance"]; clipped += s["clipped_white_frac"]
            save_png(res, out)
        n = max(len(imgs), 1)
        log["splits"][split] = {"images": len(imgs), "enhanced": n_enh, "mean_luminance_before": lum_before / n,
                                "mean_luminance_after": lum_after / n, "mean_clipped_white_frac": clipped / n}
        print(split, log["splits"][split])

    write_data_yaml(dst)
    (dst / "enhancement_log.json").write_text(json.dumps(log, indent=2))
    print("done — now run: python scripts/verify_dataset.py --a", src, "--b", dst)


if __name__ == "__main__":
    main()
