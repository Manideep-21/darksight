# DarkSight task board

Tick items in PRs. ✅ = already done in the initial scaffold.

## Week 1 — Setup & data
- [x] ✅ Repo skeleton, `requirements.txt`, `.gitignore`, pytest (28 tests)
- [x] ✅ ExDark downloaded & converted to YOLO (7,363 images · 23,710 boxes · official 3000/1800/2563 split)
- [x] ✅ Data-quality report `reports/data_stats.md` (EXIF case 2015_05337 fixed, 50 clipped boxes, 1 empty image)
- [x] ✅ Zero-DCE vendored, device-agnostic, output identical to the official code
- [x] ✅ Metrics walkthrough `docs/metrics_walkthrough.md`
- [ ] **All** — create GitHub repo, push, add teammates (Kaggle notebook clones it); phone-verify Kaggle accounts
- [ ] **All** — each member runs `pytest -q` locally and `notebooks/00_colab_setup.ipynb` on Colab
- [ ] **B** — upload `exdark_yolo_raw.zip` + `exdark_meta.csv` to Drive (setup notebook does this)
- [ ] **A** — download LOL, `prepare_lol.py`, `eval_lol.py` → `results/lol_test_summary.json`
- [ ] **C** — team reviews 50 images in `reports/label_check/`; `verify_dataset.py --manifest` to freeze
- [ ] **All** — 30-min metrics walkthrough session

## Week 2 — Individual models
- [ ] **B** — Kaggle: Run All on `05_kaggle_E2_E3.ipynb` → `det_raw` + **E2 + E3**
- [ ] **B** — download `det_raw.pt` + `darksight_E2_E3.zip` from the Kaggle Output panel; commit metrics
- [x] ✅ **C** — evaluator, experiment runner, results table, latency benchmark, Gradio app shell
- [x] ✅ **C** — E0/E1 run locally (0.609 → 0.567 mAP@0.5, CI excludes 0); re-run on Colab to confirm
- [x] ✅ Smoke training (yolov8n, 1 epoch) + `model.val()` wrapper verified on MPS
- [ ] **A** — luminance distribution on **val** (`01_data_exploration.ipynb`) → pick gate τ candidates
- [ ] **A** — decide whether LOL fine-tuning of Zero-DCE is needed (default: no)

## Week 3 — Integration
- [ ] **A** — `enhance_dataset.py` on all splits (Kaggle, `--half`) + `verify_dataset.py` (notebook cell 3)
- [ ] **B** — Kaggle: Run All on `06_kaggle_E4.ipynb` (attach 05's output) → `det_enh` + **E4**
- [ ] **A** — Kaggle: Run All on `07_kaggle_E5_E6.ipynb` (attach 06's output) → **E5** (τ from val) + **E6** CLAHE
- [ ] **C** — Kaggle: Run All on `08_kaggle_final_results.ipynb` (attach 05 + 06 outputs) → final table, bootstrap CI, report figures, demo link
- [ ] **B** (if GPU time) — second seed for det_raw/det_enh (`--seed 1 --name det_raw_s1`)
- [ ] **A** (stretch) — E6 CLAHE dataset

## Week 4 — Evaluation, demo, report
- [ ] **C** — `make_results_table.py`, `make_qualitative.py --pick gain` and `--pick loss`
- [ ] **B** — `benchmark_latency.py` on T4 and a laptop
- [ ] **C** — demo with fine-tuned weights, record 60-s backup video, optional HF Space
- [ ] **All** — report (intro, related work, method, setup, results, discussion, limitations, conclusion) + 10–12 slides
- [ ] **All** — rehearse demo twice (once offline)
