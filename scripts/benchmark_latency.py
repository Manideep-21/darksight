#!/usr/bin/env python
"""Per-stage latency: Zero-DCE, YOLO, and the full pipeline.

    python scripts/benchmark_latency.py --detector weights/det_enh.pt --sizes 640 1280 --device mps
"""

import _bootstrap  # noqa: F401

import argparse
import json
import platform
import time

import numpy as np

from darksight.config import paths
from darksight.detect.detector import Detector
from darksight.device import select_device
from darksight.enhance.enhancer import ZeroDCEEnhancer
from darksight.pipeline import _sync


def timeit(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    _sync(device)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t) * 1000)
    return {"mean_ms": float(np.mean(times)), "p50_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="yolov8s.pt")
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 1280], help="long side of the synthetic input")
    ap.add_argument("--device", default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    device = select_device(args.device)
    enh = ZeroDCEEnhancer(device=device)
    det = Detector(args.detector, device=device)
    rng = np.random.default_rng(0)
    out = {"device": device, "machine": platform.platform(), "detector": args.detector, "results": {}}
    for s in args.sizes:
        img = (rng.random((s * 3 // 4, s, 3)) * 60).astype(np.uint8)
        enhanced = enh.enhance(img)
        r = {
            "zerodce": timeit(lambda: enh.enhance(img), device, args.warmup, args.iters),
            "yolo": timeit(lambda: det.predict(enhanced, conf=0.25), device, args.warmup, args.iters),
            "pipeline": timeit(lambda: det.predict(enh.enhance(img), conf=0.25), device, args.warmup, args.iters),
        }
        r["pipeline_fps"] = 1000 / r["pipeline"]["mean_ms"]
        out["results"][f"{img.shape[1]}x{img.shape[0]}"] = r
        print(s, json.dumps(r, indent=1))
    dst = paths()["results_root"] / f"latency_{device.replace(':', '')}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print("wrote", dst)


if __name__ == "__main__":
    main()
