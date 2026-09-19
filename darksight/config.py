import os
from pathlib import Path

import yaml

from .constants import PROJECT_ROOT


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def paths() -> dict[str, Path]:
    cfg = load_yaml(PROJECT_ROOT / "configs" / "paths.yaml")
    return {
        "data_root": _resolve(os.environ.get("DARKSIGHT_DATA", cfg["data_root"])),
        "weights_root": _resolve(os.environ.get("DARKSIGHT_WEIGHTS", cfg["weights_root"])),
        "results_root": _resolve(cfg["results_root"]),
    }


def experiment(exp_id: str) -> dict:
    exps = load_yaml(PROJECT_ROOT / "configs" / "experiments.yaml")
    if exp_id not in exps:
        raise KeyError(f"unknown experiment {exp_id}; choose from {list(exps)}")
    e = dict(exps[exp_id], id=exp_id)
    p = paths()
    local = p["weights_root"] / e["weights"]
    # Stock Ultralytics names (yolov8s.pt) auto-download when not present locally.
    e["weights_path"] = str(local) if local.exists() or not e["weights"].startswith("yolo") else e["weights"]
    e["dataset_root"] = p["data_root"] / e["dataset"]
    e["results_dir"] = p["results_root"] / exp_id
    e.setdefault("coco_remap", False)
    return e
