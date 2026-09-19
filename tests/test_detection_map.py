import numpy as np
import pytest

from darksight.eval.detection_map import (average_precision, box_iou, compute_metrics, match_image,
                                          paired_bootstrap)


def arr(*rows):
    return np.array(rows, dtype=float).reshape(-1, 4)


def test_box_iou():
    iou = box_iou(arr([0, 0, 0.5, 0.5]), arr([0, 0, 0.5, 0.5], [0.25, 0, 0.75, 0.5], [0.6, 0.6, 1, 1]))
    assert iou[0] == pytest.approx([1.0, 1 / 3, 0.0])


def test_perfect_predictions_give_ap_1():
    gt_cls, gt_box = np.array([0, 4]), arr([0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.9, 0.9])
    m = match_image(gt_cls, gt_box, gt_cls, gt_box, np.array([0.9, 0.8]))
    res = compute_metrics([m])
    # 101-point trapezoid integration tops out at 0.995 (identical to Ultralytics)
    assert res["ap50_per_class"][0] == pytest.approx(0.995)
    assert res["ap50_per_class"][4] == pytest.approx(0.995)
    assert np.isnan(res["ap50_per_class"][1])  # class absent from GT -> excluded from the mean
    assert res["map50"] == pytest.approx(0.995)


def test_wrong_class_and_low_iou_are_false_positives():
    gt_cls, gt_box = np.array([0]), arr([0.1, 0.1, 0.3, 0.3])
    pr_cls = np.array([1, 0, 0])
    pr_box = arr([0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.4, 0.4], [0.1, 0.1, 0.3, 0.3])
    m = match_image(gt_cls, gt_box, pr_cls, pr_box, np.array([0.9, 0.8, 0.7]))
    assert m.tp.tolist() == [False, False, True]


def test_duplicate_detection_counts_once():
    gt_cls, gt_box = np.array([2]), arr([0.1, 0.1, 0.3, 0.3])
    m = match_image(gt_cls, gt_box, np.array([2, 2]), arr([0.1, 0.1, 0.3, 0.3], [0.1, 0.1, 0.3, 0.3]), np.array([0.9, 0.5]))
    assert m.tp.tolist() == [True, False]


def test_ap_with_fp_ranked_first():
    # FP then TP, one GT: precision at recall 1 is 0.5 -> 101-point AP = 0.5 (plus the recall-0 point)
    ap, _, _ = average_precision(np.array([False, True]), 1)
    assert ap == pytest.approx(0.5, abs=0.01)


def test_no_predictions_gives_zero():
    m = match_image(np.array([3]), arr([0, 0, 1, 1]), np.zeros(0, int), np.zeros((0, 4)), np.zeros(0))
    assert compute_metrics([m])["map50"] == 0.0


def test_paired_bootstrap_identical_systems_has_zero_diff():
    rng = np.random.default_rng(0)
    matches = {}
    for i in range(30):
        gt_box = rng.random((2, 2)); gt_box = np.concatenate([gt_box * 0.5, gt_box * 0.5 + 0.3], axis=1)
        pr_box = gt_box + rng.normal(0, 0.03, gt_box.shape)
        matches[str(i)] = match_image(np.array([0, 1]), gt_box, np.array([0, 1]), pr_box, rng.random(2))
    res = paired_bootstrap(matches, matches, n_resamples=50)
    assert res["diff"] == 0 and res["ci95_low"] == 0 and res["ci95_high"] == 0


@pytest.mark.parametrize("seed", range(5))
def test_ap_matches_ultralytics_compute_ap(seed):
    from ultralytics.utils.metrics import compute_ap

    rng = np.random.default_rng(seed)
    tp = rng.random(40) < 0.6
    n_gt = int(tp.sum()) + int(rng.integers(0, 10))
    ours, _, _ = average_precision(tp, n_gt)
    tpc, fpc = np.cumsum(tp), np.cumsum(~tp)
    ref, _, _ = compute_ap(tpc / n_gt, tpc / (tpc + fpc))
    assert ours == pytest.approx(ref, abs=1e-9)
