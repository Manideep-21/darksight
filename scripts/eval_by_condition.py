#!/usr/bin/env python
"""mAP@0.5 per lighting condition and indoor/outdoor, for several experiments (reuses predictions).

    python scripts/eval_by_condition.py --exp E2 E3 E4 E5
Writes results/by_condition_<split>.csv and results/figures/by_light_<split>.png
"""

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from darksight.config import experiment, paths
from darksight.constants import LIGHT_TYPES, PROJECT_ROOT
from darksight.eval.detection_map import compute_metrics, load_ground_truth, load_predictions, match_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--meta", default=str(PROJECT_ROOT / "data" / "exdark_meta.csv"))
    args = ap.parse_args()

    meta = pd.read_csv(args.meta, dtype={"stem": str})
    meta = meta[meta.split == args.split]
    groups = [("light", name, meta[meta.light_name == name].stem.tolist()) for name in LIGHT_TYPES.values()]
    groups += [("indoor_outdoor", v, meta[meta.indoor_outdoor == v].stem.tolist()) for v in ("Indoor", "Outdoor")]
    groups += [("all", "All", meta.stem.tolist())]

    rows = []
    for exp_id in args.exp:
        exp = experiment(exp_id)
        gt = load_ground_truth(Path(exp["dataset_root"]) / "labels" / args.split)
        matches = match_dataset(gt, load_predictions(Path(exp["results_dir"]) / f"predictions_{args.split}.jsonl"))
        for kind, name, stems in groups:
            stems = [s for s in stems if s in matches]
            if not stems:
                continue
            m = compute_metrics([matches[s] for s in stems])
            rows.append({"experiment": exp_id, "group_type": kind, "group": name, "n_images": len(stems),
                         "map50": m["map50"], "precision": m["precision"], "recall": m["recall"]})
    df = pd.DataFrame(rows)
    res = paths()["results_root"]
    out_csv = res / f"by_condition_{args.split}.csv"
    df.to_csv(out_csv, index=False)
    print(df.pivot_table(index="group", columns="experiment", values="map50").round(4).to_string())

    light = df[df.group_type == "light"].pivot(index="group", columns="experiment", values="map50")
    light = light.reindex([n for n in LIGHT_TYPES.values() if n in light.index])
    fig, ax = plt.subplots(figsize=(1.2 * len(args.exp) + 3, 5))
    im = ax.imshow(light.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(light.columns)), light.columns)
    ax.set_yticks(range(len(light.index)), light.index)
    for i in range(light.shape[0]):
        for j in range(light.shape[1]):
            v = light.values[i, j]
            dark_cell = v < (np.nanmin(light.values) + np.nanmax(light.values)) / 2
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if dark_cell else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="mAP@0.5")
    ax.set_title(f"mAP@0.5 by lighting condition ({args.split})")
    fig.tight_layout()
    (res / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(res / "figures" / f"by_light_{args.split}.png", dpi=150)
    print("wrote", out_csv)


if __name__ == "__main__":
    main()
