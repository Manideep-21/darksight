"""Self-contained mAP@0.5 evaluator with per-image matching, so we can
(1) score class-remapped COCO models, (2) slice by lighting condition, and
(3) run a paired bootstrap over test images. Uses the same 101-point interpolated AP
as COCO / Ultralytics, so numbers are close to (not bit-identical with) model.val().

Box convention everywhere in this module: normalized xyxy in [0, 1]. IoU is invariant to
per-axis scaling, so normalized coordinates give the same matches as pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import NUM_CLASSES


@dataclass
class ImageMatch:
    """Matching result of one image at a fixed IoU threshold."""
    conf: np.ndarray    # (M,) prediction confidences
    cls: np.ndarray     # (M,) prediction classes
    tp: np.ndarray      # (M,) bool, true positive?
    n_gt: np.ndarray    # (NUM_CLASSES,) ground-truth count per class


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4), b: (M,4) xyxy -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(axis=2)
    area_a = (a[:, 2:] - a[:, :2]).prod(axis=1)
    area_b = (b[:, 2:] - b[:, :2]).prod(axis=1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-12)


def match_image(gt_cls: np.ndarray, gt_xyxy: np.ndarray, pr_cls: np.ndarray, pr_xyxy: np.ndarray,
                pr_conf: np.ndarray, iou_thr: float = 0.5) -> ImageMatch:
    """Greedy COCO-style matching: highest confidence first, each GT used at most once, same class."""
    order = np.argsort(-pr_conf, kind="stable")
    pr_cls, pr_xyxy, pr_conf = pr_cls[order], pr_xyxy[order], pr_conf[order]
    tp = np.zeros(len(pr_conf), dtype=bool)
    ious = box_iou(pr_xyxy, gt_xyxy)
    used = np.zeros(len(gt_cls), dtype=bool)
    for i in range(len(pr_conf)):
        if len(gt_cls) == 0:
            break
        cand = np.where((gt_cls == pr_cls[i]) & ~used, ious[i], -1.0)
        j = int(np.argmax(cand))
        if cand[j] >= iou_thr:
            tp[i] = True
            used[j] = True
    n_gt = np.bincount(gt_cls.astype(np.int64), minlength=NUM_CLASSES)[:NUM_CLASSES]
    return ImageMatch(pr_conf, pr_cls, tp, n_gt)


def average_precision(tp_sorted: np.ndarray, n_gt: int) -> tuple[float, np.ndarray, np.ndarray]:
    """101-point interpolated AP. tp_sorted must be ordered by descending confidence."""
    if n_gt == 0:
        return float("nan"), np.array([]), np.array([])
    if len(tp_sorted) == 0:
        return 0.0, np.array([]), np.array([])
    tpc = np.cumsum(tp_sorted)
    fpc = np.cumsum(~tp_sorted)
    recall = tpc / n_gt
    precision = tpc / np.maximum(tpc + fpc, 1)
    # Same sentinels as ultralytics.utils.metrics.compute_ap (8.4). Note the 101-point trapezoid
    # gives a perfect detector 0.995, exactly like Ultralytics.
    mrec = np.concatenate(([0.0], recall, [recall[-1]], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = float(np.trapezoid(np.interp(x, mrec, mpre), x))
    return ap, precision, recall


def compute_metrics(matches: list[ImageMatch]) -> dict:
    """Per-class AP50 plus precision/recall at the confidence that maximises mean F1."""
    if not matches:
        raise ValueError("no images to evaluate")
    conf = np.concatenate([m.conf for m in matches])
    cls = np.concatenate([m.cls for m in matches])
    tp = np.concatenate([m.tp for m in matches])
    n_gt = np.sum([m.n_gt for m in matches], axis=0)

    grid = np.linspace(0, 1, 1000)
    ap50 = np.full(NUM_CLASSES, np.nan)
    p_curve = np.zeros((NUM_CLASSES, len(grid)))
    r_curve = np.zeros((NUM_CLASSES, len(grid)))
    for c in range(NUM_CLASSES):
        sel = cls == c
        order = np.argsort(-conf[sel], kind="stable")
        c_conf, c_tp = conf[sel][order], tp[sel][order]
        ap, prec, rec = average_precision(c_tp, int(n_gt[c]))
        ap50[c] = ap
        if n_gt[c] and len(c_conf):
            # interpolate P/R as functions of confidence (conf descending -> reverse for np.interp)
            p_curve[c] = np.interp(-grid, -c_conf, prec, left=1.0)
            r_curve[c] = np.interp(-grid, -c_conf, rec, left=0.0)
    valid = n_gt > 0
    f1 = 2 * p_curve * r_curve / np.maximum(p_curve + r_curve, 1e-16)
    best = int(np.argmax(f1[valid].mean(axis=0))) if valid.any() else 0
    return {
        "map50": float(np.nanmean(ap50)) if valid.any() else float("nan"),
        "precision": float(p_curve[valid, best].mean()) if valid.any() else float("nan"),
        "recall": float(r_curve[valid, best].mean()) if valid.any() else float("nan"),
        "best_conf": float(grid[best]),
        "ap50_per_class": ap50.tolist(),
        "precision_per_class": p_curve[:, best].tolist(),
        "recall_per_class": r_curve[:, best].tolist(),
        "n_gt_per_class": n_gt.astype(int).tolist(),
        "n_images": len(matches),
    }


# ---------------------------------------------------------------- data loading

def load_ground_truth(labels_dir: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """YOLO txt labels -> {stem: (cls (N,), normalized xyxy (N,4))}."""
    gt = {}
    for p in sorted(Path(labels_dir).glob("*.txt")):
        rows = np.loadtxt(p, ndmin=2) if p.stat().st_size else np.zeros((0, 5))
        xc, yc, w, h = rows[:, 1], rows[:, 2], rows[:, 3], rows[:, 4]
        xyxy = np.stack([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1) if len(rows) else np.zeros((0, 4))
        gt[p.stem] = (rows[:, 0].astype(np.int64), xyxy)
    return gt


def load_predictions(jsonl: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """predictions.jsonl lines: {"stem", "cls": [...], "conf": [...], "xyxyn": [[...], ...]}."""
    preds = {}
    with open(jsonl) as f:
        for line in f:
            d = json.loads(line)
            preds[d["stem"]] = (
                np.asarray(d["cls"], dtype=np.int64),
                np.asarray(d["xyxyn"], dtype=np.float64).reshape(-1, 4),
                np.asarray(d["conf"], dtype=np.float64),
            )
    return preds


def match_dataset(gt: dict, preds: dict, stems: list[str] | None = None, iou_thr: float = 0.5) -> dict[str, ImageMatch]:
    stems = sorted(gt) if stems is None else stems
    empty = (np.zeros(0, np.int64), np.zeros((0, 4)), np.zeros(0))
    missing = [s for s in stems if s not in preds]
    if len(missing) > 0.01 * len(stems):
        raise ValueError(f"{len(missing)} of {len(stems)} images have no prediction entry (e.g. {missing[:3]})")
    return {s: match_image(*gt[s], *preds.get(s, empty), iou_thr=iou_thr) for s in stems}


def paired_bootstrap(matches_a: dict[str, ImageMatch], matches_b: dict[str, ImageMatch],
                     n_resamples: int = 1000, seed: int = 0) -> dict:
    """Resample test images with replacement (same draw for both systems) -> CI of mAP50(B) - mAP50(A)."""
    stems = sorted(set(matches_a) & set(matches_b))
    rng = np.random.default_rng(seed)
    a_list = [matches_a[s] for s in stems]
    b_list = [matches_b[s] for s in stems]
    base_a = compute_metrics(a_list)["map50"]
    base_b = compute_metrics(b_list)["map50"]
    diffs = np.empty(n_resamples)
    for k in range(n_resamples):
        idx = rng.integers(0, len(stems), len(stems))
        diffs[k] = (compute_metrics([b_list[i] for i in idx])["map50"]
                    - compute_metrics([a_list[i] for i in idx])["map50"])
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {
        "n_images": len(stems), "n_resamples": n_resamples,
        "map50_a": base_a, "map50_b": base_b, "diff": base_b - base_a,
        "ci95_low": float(lo), "ci95_high": float(hi),
        "p_b_not_better": float(np.mean(diffs <= 0)),
    }
