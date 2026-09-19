#!/usr/bin/env python
"""Collect results/E*/metrics_*.json (+ ultralytics_*.json) into one table and charts.

    python scripts/make_results_table.py --split test
Writes results/summary_<split>.csv, results/summary_<split>.md, results/figures/{overall,per_class}_<split>.png
"""

import _bootstrap  # noqa: F401

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darksight.config import paths
from darksight.constants import CLASS_NAMES


def fmt(v):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    res = paths()["results_root"]

    rows, per_class = [], {}
    for d in sorted(p for p in res.glob("E*") if p.is_dir()):
        mf = d / f"metrics_{args.split}.json"
        if not mf.exists():
            continue
        m = json.loads(mf.read_text())
        uf = d / f"ultralytics_{args.split}.json"
        u = json.loads(uf.read_text()) if uf.exists() else {}
        rows.append({"experiment": d.name, "description": m["description"], "map50": m["map50"],
                     "precision": m["precision"], "recall": m["recall"],
                     "ultra_map50": u.get("map50"), "ultra_map50_95": u.get("map50_95"),
                     "ultra_precision": u.get("precision"), "ultra_recall": u.get("recall")})
        per_class[d.name] = m["ap50_per_class"]
    if not rows:
        raise SystemExit(f"no results/E*/metrics_{args.split}.json found — run scripts/run_experiment.py first")

    df = pd.DataFrame(rows)
    df.to_csv(res / f"summary_{args.split}.csv", index=False)
    pc = pd.DataFrame(per_class, index=CLASS_NAMES)
    pc.to_csv(res / f"per_class_ap50_{args.split}.csv")

    md = [f"# DarkSight results ({args.split} split)", "",
          "| Exp | Description | mAP@0.5 | P | R | mAP@0.5 (Ultralytics) | mAP@0.5:0.95 (Ultralytics) |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {r.experiment} | {r.description} | {fmt(r.map50)} | {fmt(r.precision)} | {fmt(r.recall)} | "
           f"{fmt(r.ultra_map50)} | {fmt(r.ultra_map50_95)} |" for r in df.itertuples()]
    md += ["", "## AP@0.5 per class", "", "| Class | " + " | ".join(pc.columns) + " |",
           "|---" * (len(pc.columns) + 1) + "|"]
    md += [f"| {c} | " + " | ".join(fmt(v) for v in pc.loc[c]) + " |" for c in pc.index]
    for b in sorted(res.glob(f"bootstrap_*_{args.split}.json")):
        j = json.loads(b.read_text())
        md += ["", f"**{j['b']} vs {j['a']}:** ΔmAP@0.5 = {j['diff']:+.4f}, "
                   f"95% CI [{j['ci95_low']:+.4f}, {j['ci95_high']:+.4f}] ({j['n_resamples']} paired bootstrap resamples)"]
    (res / f"summary_{args.split}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    figs = res / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(5, 1.1 * len(df)), 4))
    bars = ax.bar(df.experiment, df.map50, color="#3b6ea8")
    ax.bar_label(bars, fmt="%.3f", fontsize=8)
    ax.set_ylabel("mAP@0.5"); ax.set_title(f"Overall mAP@0.5 ({args.split})")
    ax.set_ylim(0, max(1e-3, df.map50.max()) * 1.2)
    fig.tight_layout(); fig.savefig(figs / f"overall_{args.split}.png", dpi=150)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    width = 0.8 / len(pc.columns)
    x = np.arange(len(CLASS_NAMES))
    for i, col in enumerate(pc.columns):
        ax.bar(x + i * width - 0.4 + width / 2, pc[col], width, label=col)
    ax.set_xticks(x, CLASS_NAMES, rotation=30); ax.set_ylabel("AP@0.5")
    ax.set_title(f"Per-class AP@0.5 ({args.split})"); ax.legend(ncols=len(pc.columns), fontsize=8)
    fig.tight_layout(); fig.savefig(figs / f"per_class_{args.split}.png", dpi=150)
    print("wrote summary + figures to", res)


if __name__ == "__main__":
    main()
