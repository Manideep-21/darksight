#!/usr/bin/env python
"""Paired bootstrap over test images: is B really better than A?

    python scripts/compare_experiments.py --a E2 --b E4 --n 1000
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from darksight.config import experiment, paths
from darksight.eval.detection_map import load_ground_truth, load_predictions, match_dataset, paired_bootstrap


def matches_for(exp_id: str, split: str):
    exp = experiment(exp_id)
    gt = load_ground_truth(Path(exp["dataset_root"]) / "labels" / split)
    return match_dataset(gt, load_predictions(Path(exp["results_dir"]) / f"predictions_{split}.jsonl"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="reference experiment, e.g. E2")
    ap.add_argument("--b", required=True, help="candidate experiment, e.g. E4")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    res = paired_bootstrap(matches_for(args.a, args.split), matches_for(args.b, args.split), args.n, args.seed)
    res.update({"a": args.a, "b": args.b, "split": args.split})
    out = paths()["results_root"] / f"bootstrap_{args.a}_vs_{args.b}_{args.split}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"mAP50 {args.a}={res['map50_a']:.4f}  {args.b}={res['map50_b']:.4f}  "
          f"diff={res['diff']:+.4f}  95% CI [{res['ci95_low']:+.4f}, {res['ci95_high']:+.4f}]  "
          f"P(B<=A)={res['p_b_not_better']:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
