# DarkSight

**Enhanced vision beyond the visible:** a two-stage low-light object detection pipeline.
**Zero-DCE** brightens a dark image → **YOLOv8** detects objects on the result.
The research question: *does the enhancement stage improve detection mAP on real low-light images (ExDark)
compared with detecting on the raw image?*

```
dark image ─► (resize ≤1280, optional luminance gate) ─► Zero-DCE (79k params) ─► YOLOv8s ─► boxes
```

Team of 3 · 4 weeks · Python 3.11 · PyTorch · Ultralytics · Gradio · Google Colab (T4)

---

## 1. Setup

### Local (macOS / Linux) — development, evaluation, demo
```bash
uv venv --python 3.11 .venv          # Python 3.14 is too new for torch wheels; use 3.11
source .venv/bin/activate
uv pip install -r requirements.txt   # or: pip install -r requirements.txt
pytest -q                            # 28 tests, ~10 s
```
Apple Silicon uses `mps` automatically (`darksight/device.py` picks cuda → mps → cpu).

### Kaggle — training (recommended: 2× T4)
Import a notebook, set Accelerator = **GPU T4 x2**, Internet = On, press Run All (no dataset to attach, no cell to edit):
- `notebooks/05_kaggle_E2_E3.ipynb` — **self-contained and beginner-readable**: every step written out in plain
  Python (no imports from `darksight/`), trains `det_raw` and produces **E2 + E3** (~1.5–2 h). Use this one to
  explain the project.
- `notebooks/04_kaggle_full_pipeline.ipynb` — both detectors and **E0–E5** end to end (~3–4 h)
Full checklist in [`docs/kaggle.md`](docs/kaggle.md).

### Colab — alternative (single T4)
`notebooks/00_colab_setup.ipynb`, then `02_train.ipynb` and `03_analysis.ipynb`.
Datasets are cached as zips on the shared Drive folder `MyDrive/darksight/` and unzipped to `/content` each session.

### Weights
| file | source | in git? |
|---|---|---|
| `weights/zerodce_epoch99.pth` | official Zero-DCE snapshot (github.com/Li-Chongyi/Zero-DCE) | yes (320 KB) |
| `weights/yolov8s.pt`, `yolov8n.pt` | COCO-pretrained; Ultralytics auto-downloads from GitHub. If GitHub release downloads are blocked on your network: `curl -L -o weights/yolov8s.pt https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8s.pt` | no |
| `weights/det_raw.pt`, `det_enh.pt` | produced by `scripts/train_detector.py` (copy from Drive) | no |

---

## 2. Datasets

### ExDark (detector training + evaluation)
- 7,363 low-light images, 12 classes, 23,710 boxes. Research / non-commercial use only.
- Download (Google Drive IDs from the official repo):
  ```bash
  gdown 1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx -O data/downloads/ExDark.zip        # images, 1.5 GB
  gdown 1P3iO3UYn7KoBi5jiUkogJq96N6maZS1i -O data/downloads/ExDark_Annno.zip  # annotations, 5 MB
  unzip -q data/downloads/ExDark.zip -d data/downloads && unzip -q data/downloads/ExDark_Annno.zip -d data/downloads
  ```
- Convert (≈3 min with 8 workers):
  ```bash
  python scripts/prepare_exdark.py --images data/downloads/ExDark --annotations data/downloads/ExDark_Annno
  python scripts/visualize_labels.py --split train --n 50          # look at reports/label_check/
  python scripts/verify_dataset.py --a data/exdark_yolo/raw --manifest   # freeze with sha256 manifest
  ```
- **Split:** official split from `configs/exdark_imageclasslist.txt` — 3,000 train / 1,800 val / 2,563 test.
- **Class ids:** `0 Bicycle, 1 Boat, 2 Bottle, 3 Bus, 4 Car, 5 Cat, 6 Chair, 7 Cup, 8 Dog, 9 Motorbike, 10 People, 11 Table`
- What the converter handles (all logged in `reports/data_stats.md`):
  mixed extensions (`.jpg/.JPG/.JPEG/.jpeg/.png`), grayscale/RGBA/palette images → RGB,
  50 boxes extending past the image border → clipped, 1 image with no objects (`2015_05894`) → kept as background,
  526 images larger than 1280 px → downscaled, and EXIF rotation (`2015_05337` was annotated on the
  *un-rotated* pixels, so `--exif auto` keeps its raw orientation). Output is PNG without EXIF so raw and
  enhanced copies are pixel-aligned and free of JPEG re-compression differences.
- Output: `data/exdark_yolo/raw/{images,labels}/{train,val,test}`, `data.yaml`, and `data/exdark_meta.csv`
  (lighting type, indoor/outdoor, sizes per image — used for per-condition analysis).

### LOL (enhancer quality check only)
Zero-DCE is trained without reference images (the official snapshot was trained on SICE), so LOL is **not**
needed for training. We use its 15 test pairs to report PSNR/SSIM of the enhancement.
```bash
# download LOLdataset.zip from the RetinexNet project page or the Kaggle mirror into data/downloads/
python scripts/prepare_lol.py --src data/downloads/LOLdataset --out data/lol
python scripts/eval_lol.py --root data/lol --split test
```

---

## 3. Experiments

Defined in `configs/experiments.yaml`; all evaluated on the same 2,563 official test images.

| ID | Detector | Test images | Question |
|---|---|---|---|
| E0 | COCO YOLOv8s zero-shot (classes remapped) | raw | off-the-shelf reference |
| E1 | COCO YOLOv8s zero-shot | Zero-DCE | does enhancement help an unadapted detector? |
| **E2** | fine-tuned on raw | raw | **baseline** |
| E3 | fine-tuned on raw | Zero-DCE | enhancement at test time only |
| **E4** | fine-tuned on Zero-DCE | Zero-DCE | **proposed pipeline** |
| E5 | fine-tuned on Zero-DCE | gated Zero-DCE | does skipping bright images help? |
| E6 | fine-tuned on raw | CLAHE | classical baseline (stretch) |

### First results (already run locally, Apple M2, 2,563 test images)
| Exp | mAP@0.5 | P | R |
|---|---|---|---|
| E0 COCO YOLOv8s zero-shot, raw | **0.609** | 0.710 | 0.572 |
| E1 COCO YOLOv8s zero-shot, Zero-DCE | **0.567** | 0.709 | 0.529 |

ΔmAP@0.5 = **−0.043**, 95% paired-bootstrap CI [−0.053, −0.034]: test-time enhancement *hurts* an unadapted
detector, most on the darkest "Low" images (0.571 → 0.491). Zero-DCE lifts mean luminance 0.13 → 0.33 but amplifies
sensor noise and shifts colours. That makes E2 vs E4 (detectors fine-tuned on each domain) the key comparison.
Full tables: `results/summary_test.md`, `results/by_condition_test.csv`, figures in `results/figures/`.

### Workflow
```bash
# Week 2 — baseline detector (Colab)
python scripts/train_detector.py --dataset data/exdark_yolo/raw --name det_raw --project /content/drive/MyDrive/darksight/runs
# smoke test first: --model yolov8n.pt --epochs 1 --fraction 0.1

# Week 3 — enhanced data + second detector
python scripts/enhance_dataset.py --src data/exdark_yolo/raw --dst data/exdark_yolo/enhanced [--half]
python scripts/verify_dataset.py  --a data/exdark_yolo/raw --b data/exdark_yolo/enhanced
python scripts/train_detector.py  --dataset data/exdark_yolo/enhanced --name det_enh
python scripts/enhance_dataset.py --dst data/exdark_yolo/enhanced_gated --gate-tau 0.35   # tune tau on val

# Evaluate
python scripts/run_experiment.py --exp E0 E1 E2 E3 E4 E5     # predictions + metrics per experiment
python scripts/compare_experiments.py --a E2 --b E4          # paired bootstrap 95% CI of ΔmAP@0.5
python scripts/eval_by_condition.py --exp E2 E3 E4 E5        # mAP per lighting type / indoor-outdoor
python scripts/make_results_table.py                         # results/summary_test.md + figures
python scripts/make_qualitative.py --pick gain --n 6         # report figures (also --pick loss)
python scripts/benchmark_latency.py --detector weights/det_enh.pt
python scripts/render_predictions.py --exp E2 E4 --side-by-side E2 E4   # detection images + CSV per test image
```

`run_experiment.py` reports two sets of numbers:
- **darksight evaluator** (`darksight/eval/detection_map.py`): mAP@0.5, P, R per class. Works for every
  experiment (incl. COCO-remapped models), supports slicing and bootstrap. AP integration is verified
  against `ultralytics.utils.metrics.compute_ap` in the tests.
- **Ultralytics `model.val()`**: mAP@0.5 and mAP@0.5:0.95 for the fine-tuned models (E2–E6).

See `docs/metrics_walkthrough.md` for IoU / precision / recall / mAP explained.

---

## 4. Demo
```bash
python app/app.py            # http://127.0.0.1:7860   (Colab: python app/app.py --share)
```
Upload an image or pick one of 6 ExDark test examples (one per lighting type), choose side-by-side / raw / enhanced,
adjust confidence and the luminance gate. Shows both detections, the enhanced image, per-stage latency and a
detection table. Falls back to COCO zero-shot weights (clearly labelled) until `det_raw.pt` / `det_enh.pt` exist.

---

## 5. Repository layout
```
configs/     train.yaml (shared hyper-params) · experiments.yaml · paths.yaml · exdark_imageclasslist.txt · coco_to_exdark.yaml
darksight/   constants · io_utils · device · config · pipeline · visualize
             data/{exdark, yolo_dataset}   enhance/{zerodce_model, enhancer, gating, classical}
             detect/detector   eval/{detection_map, ultralytics_val, enhance_quality}
scripts/     prepare_exdark · visualize_labels · verify_dataset · enhance_dataset · train_detector
             run_experiment · compare_experiments · eval_by_condition · make_results_table · make_qualitative
             render_predictions · benchmark_latency · prepare_lol · eval_lol · make_data_yaml · find_exdark
notebooks/   00_colab_setup · 01_data_exploration · 02_train · 03_analysis · 04_kaggle_full_pipeline
app/         app.py (Gradio) · examples/
tests/       converter, enhancer, evaluator, end-to-end pipeline
docs/        metrics_walkthrough.md · kaggle.md · TASKS.md
results/     metrics JSON/CSV + figures (committed; predictions are git-ignored)
reports/     data_stats.md, label checks, final report & slides
```

## 6. Team workflow
- Branch per task (`a/enhance-dataset`, `b/train-det-raw`, `c/eval-scripts`), PR reviewed by one teammate, `main` stays green (`pytest -q`).
- Commit every `results/E*/metrics_*.json` — every number in the report must be regenerable with `make_results_table.py`.
- Change `configs/train.yaml` only *before* training either detector; both detectors must share it.
- Tune anything (gate τ, confidence for the demo) on **val**; evaluate on **test** once per final configuration.

## 7. Licences & citations
- ExDark — Loh & Chan, *CVIU* 2019. Non-commercial research use.
- Zero-DCE — Guo et al., *CVPR* 2020. Code: academic research use (CC BY-NC 4.0).
- LOL — Wei et al., *BMVC* 2018. · COCO — Lin et al., *ECCV* 2014.
- Ultralytics YOLOv8 — AGPL-3.0.
