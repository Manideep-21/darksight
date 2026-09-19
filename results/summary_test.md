# DarkSight results (test split)

| Exp | Description | mAP@0.5 | P | R | mAP@0.5 (Ultralytics) | mAP@0.5:0.95 (Ultralytics) |
|---|---|---|---|---|---|---|
| E0 | COCO YOLOv8s zero-shot on raw images | 0.609 | 0.710 | 0.572 | – | – |
| E1 | COCO YOLOv8s zero-shot on Zero-DCE enhanced images | 0.567 | 0.709 | 0.529 | – | – |

## AP@0.5 per class

| Class | E0 | E1 |
|---|---|---|
| Bicycle | 0.706 | 0.705 |
| Boat | 0.480 | 0.466 |
| Bottle | 0.649 | 0.630 |
| Bus | 0.870 | 0.874 |
| Car | 0.720 | 0.645 |
| Cat | 0.600 | 0.478 |
| Chair | 0.526 | 0.478 |
| Cup | 0.549 | 0.492 |
| Dog | 0.661 | 0.567 |
| Motorbike | 0.567 | 0.521 |
| People | 0.752 | 0.706 |
| Table | 0.229 | 0.237 |

**E1 vs E0:** ΔmAP@0.5 = -0.0427, 95% CI [-0.0527, -0.0341] (1000 paired bootstrap resamples)
