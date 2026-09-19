#!/usr/bin/env python
"""Run one experiment from configs/experiments.yaml end to end.

    python scripts/run_experiment.py --exp E2
    python scripts/run_experiment.py --exp E0 E1 --split test

For each experiment writes results/<EXP>/:
    predictions_<split>.jsonl   per-image detections (conf >= 0.001), used for slicing + bootstrap
    metrics_<split>.json        mAP50 / P / R per class from darksight's evaluator (all experiments)
    ultralytics_<split>.json    model.val() numbers incl. mAP50-95 (fine-tuned models only; COCO models
                                cannot be val()'d on ExDark ids without remapping)
"""

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from darksight.config import experiment
from darksight.constants import CLASS_NAMES
from darksight.detect.detector import Detector
from darksight.eval.detection_map import compute_metrics, load_ground_truth, load_predictions, match_dataset


def predict(exp: dict, split: str, device: str | None, batch: int, imgsz: int) -> Path:
    img_dir = Path(exp["dataset_root"]) / "images" / split
    paths = [str(p) for p in sorted(img_dir.glob("*.png"))]
    if not paths:
        raise FileNotFoundError(f"no images in {img_dir}")
    det = Detector(exp["weights_path"], device=device, coco_remap=exp["coco_remap"])
    out = Path(exp["results_dir"]) / f"predictions_{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for path, (h, w), d in tqdm(det.predict_paths(paths, conf=0.001, iou=0.7, imgsz=imgsz, batch=batch),
                                    total=len(paths), desc=f"{exp['id']} predict"):
            xyxyn = d.xyxy.copy()
            xyxyn[:, [0, 2]] /= w
            xyxyn[:, [1, 3]] /= h
            f.write(json.dumps({"stem": Path(path).stem, "cls": d.cls.tolist(),
                                "conf": [round(float(c), 5) for c in d.conf],
                                "xyxyn": [[round(float(v), 5) for v in b] for b in xyxyn]}) + "\n")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--skip-predict", action="store_true", help="reuse existing predictions file")
    ap.add_argument("--skip-ultralytics", action="store_true")
    args = ap.parse_args()

    for exp_id in args.exp:
        exp = experiment(exp_id)
        print(f"\n=== {exp_id}: {exp['description']}\n    weights={exp['weights_path']} dataset={exp['dataset_root']}")
        res_dir = Path(exp["results_dir"])
        pred_file = res_dir / f"predictions_{args.split}.jsonl"
        if not (args.skip_predict and pred_file.exists()):
            pred_file = predict(exp, args.split, args.device, args.batch, args.imgsz)

        gt = load_ground_truth(Path(exp["dataset_root"]) / "labels" / args.split)
        metrics = compute_metrics(list(match_dataset(gt, load_predictions(pred_file)).values()))
        metrics.update({"experiment": exp_id, "description": exp["description"], "split": args.split,
                        "weights": str(exp["weights_path"]), "dataset": str(exp["dataset_root"])})
        (res_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))
        print(f"    darksight eval: mAP50={metrics['map50']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f}")
        for name, ap50 in zip(CLASS_NAMES, metrics["ap50_per_class"]):
            print(f"      {name:<10} AP50={ap50:.4f}")

        if not exp["coco_remap"] and not args.skip_ultralytics:
            from darksight.data.yolo_dataset import write_data_yaml
            from darksight.eval.ultralytics_val import run_val

            data_yaml = write_data_yaml(exp["dataset_root"])
            uv = run_val(exp["weights_path"], data_yaml, split=args.split, imgsz=args.imgsz, batch=args.batch,
                         device=args.device, project=res_dir, name=f"val_{args.split}")
            uv.update({"experiment": exp_id, "split": args.split})
            (res_dir / f"ultralytics_{args.split}.json").write_text(json.dumps(uv, indent=2))
            print(f"    ultralytics val: mAP50={uv['map50']:.4f} mAP50-95={uv['map50_95']:.4f} "
                  f"P={uv['precision']:.4f} R={uv['recall']:.4f}")


if __name__ == "__main__":
    main()
