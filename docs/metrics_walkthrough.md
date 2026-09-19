# Metrics walkthrough (Week 1 team session)

Everything DarkSight reports comes down to four ideas. Read this before results start arriving in Week 2.

## 1. IoU — Intersection over Union
Overlap between a predicted box and a ground-truth box:

```
IoU = area(pred ∩ gt) / area(pred ∪ gt)          0 = no overlap, 1 = identical
```
A prediction "hits" a ground-truth object when **IoU ≥ 0.5** *and* the class is the same.
Each ground-truth box can be hit **once**; duplicates of the same object count as false positives.

## 2. Precision and recall
Sort all predictions by confidence and walk down the list:

| term | meaning |
|---|---|
| TP (true positive) | prediction that matched an unmatched GT box (IoU ≥ 0.5, same class) |
| FP (false positive) | wrong class, poor localisation, duplicate, or background |
| FN (false negative) | GT object nobody detected |

```
precision = TP / (TP + FP)     "of what I detected, how much was right?"
recall    = TP / (TP + FN)     "of what exists, how much did I find?"
```
Lowering the confidence threshold ⇒ recall ↑, precision usually ↓. That trade-off is the PR curve.

## 3. AP — Average Precision (per class)
Area under the precision–recall curve for one class. We use the COCO/Ultralytics 101-point version:
make precision monotonically decreasing, sample it at recall = 0.00, 0.01, …, 1.00, integrate.
(Consequence you'll notice: a perfect detector scores **0.995**, not 1.0.)

## 4. mAP@0.5 and mAP@0.5:0.95
- **mAP@0.5** = mean of the 12 per-class APs at IoU 0.5. **Our primary metric.**
- **mAP@0.5:0.95** = the same averaged over IoU thresholds 0.50, 0.55, …, 0.95 — rewards tight boxes.

Precision/recall in our tables are taken at the single confidence threshold that maximises mean F1
(same convention as Ultralytics), so P and R describe one operating point, while mAP summarises all of them.

## Why evaluation is done at conf = 0.001
mAP needs the *whole* PR curve, so evaluation keeps nearly every prediction. The demo uses conf = 0.25
because humans want to see only confident boxes. Never compare an mAP computed at conf 0.25 with one at 0.001.

## How we decide "enhancement helped"
1. ΔmAP@0.5 = mAP(E4) − mAP(E2) on the **same 2,563 test images**.
2. Paired bootstrap (`scripts/compare_experiments.py`): resample test images 1,000×, recompute both mAPs on the
   same resample, take the 2.5–97.5 percentile of the difference. If the 95% CI excludes 0, the difference is
   unlikely to be noise.
3. Slice by lighting condition (`scripts/eval_by_condition.py`) — an overall null result can hide a gain on
   "Low"/"Weak" light and a loss on "Strong"/"Screen".

## Quick self-check questions
1. A model outputs two boxes on the same person, both IoU 0.8. How many TPs/FPs?  *(1 TP, 1 FP)*
2. Class "Bus" has no predictions at all. What is its AP?  *(0 — and it still counts in the mean)*
3. Why can't we tune the gating threshold τ on the test set?  *(test-set leakage — tune on val)*
