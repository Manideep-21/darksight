# Running DarkSight on Kaggle (2× T4)

Kaggle is the recommended place to train: 2× T4 (32 GB total), 12-hour sessions, ~30 GPU-hours/week,
and `/kaggle/working` survives a session restart. Everything is driven by
two notebooks — this page is the checklist around them.

| Notebook | What it does | Time |
|---|---|---|
| `notebooks/05_kaggle_E2_E3.ipynb` | trains `det_raw`, then runs **E2** (raw) and **E3** (enhanced at test time) | ~1.5–2 h |
| `notebooks/04_kaggle_full_pipeline.ipynb` | everything: both detectors, E0–E5, all figures | ~3–4 h |

Both clone https://github.com/Manideep-21/darksight.git, so no editing is needed.

## One-time setup
1. Kaggle account → **Settings → Phone verification** (required to enable Internet in notebooks).
2. Create a notebook → **File → Import Notebook** → upload the `.ipynb` you want.
3. Right-hand panel: **Accelerator = GPU T4 x2**, **Internet = On**, **Persistence = Variables and Files**.
4. Press **Run All**. Notebook 05 needs no attached dataset — it downloads ExDark itself if none is attached.

## Attaching ExDark
**Add Input → Datasets → search "ExDark"**. Public mirrors exist, e.g.
`mangosata/exclusivelydarkimagedataset-from-cschan`, `washingtongold/exdark-dataset`,
`otachiking/exdark-dataset`.

They differ in folder layout and **some contain images only, no bounding boxes**. Cell 1 of the notebook runs
`scripts/find_exdark.py /kaggle/input`, which locates the class folders wherever they are and prints how many
images and annotations it matched against the official list. You want:

```
expected 7363 images; missing images: 0, missing annotations: 0
```

If annotations are missing, upload them yourself (5 MB, 2 minutes):
```bash
pip install kaggle                     # locally; put kaggle.json in ~/.kaggle/
kaggle datasets init -p data/downloads # edit dataset-metadata.json (title, id)
kaggle datasets create -p data/downloads -u   # with ExDark_Annno.zip in that folder
```
Then attach that dataset too and re-run the cell.

## Disk layout inside a session
| Path | Size | Persists? | Used for |
|---|---|---|---|
| `/kaggle/input/...` | read-only | n/a | attached ExDark dataset |
| `/kaggle/temp/exdark_yolo/` | ~50 GB free | no | converted raw + enhanced datasets (~8 GB) |
| `/kaggle/working/` | 20 GB | yes | repo, runs, checkpoints, results, weights |

`DARKSIGHT_DATA=/kaggle/temp/exdark_yolo` (set in the notebook) tells `darksight/config.py` where the
datasets are, so `run_experiment.py` finds them without editing any config file.

## Two GPUs
```bash
python scripts/train_detector.py --dataset ... --name det_raw --device 0,1 --batch 32
```
`--batch 32` on 2 GPUs = 16 per GPU, i.e. the same per-GPU batch as `configs/train.yaml`. Ultralytics starts
DDP in a subprocess, which is why training is launched via `!python script` rather than in-notebook Python.
**Use identical flags for `det_raw` and `det_enh`** — differing batch size or epochs would invalidate the comparison.

## Session limits and recovery
- GPU session: max 12 h, and it stops if the tab is idle too long; use **Save Version → Save & Run All (Commit)**
  for unattended runs (up to 12 h, results land in the version's Output).
- Checkpoints every 5 epochs (`save_period: 5`) to `/kaggle/working/runs/<name>/weights/last.pt`.
- After a crash: re-run the setup + prepare + enhance cells (~15 min), then
  `!python scripts/train_detector.py --resume /kaggle/working/runs/det_raw/weights/last.pt`.
- Weekly GPU quota is ~30 h: the full plan (2 detectors ≈ 2.5 h + evaluation) fits comfortably, even with a rerun.

## What to bring back
`darksight_results.zip` (metrics JSON/CSV, figures, reports) and `det_raw.pt` / `det_enh.pt`.
Commit the metrics and figures; keep the `.pt` files out of git (see `.gitignore`) and store them on Drive.

## Colab instead?
`notebooks/00_colab_setup.ipynb` + `02_train.ipynb` + `03_analysis.ipynb` do the same on a single T4 with
Google Drive for storage. Kaggle is faster (2 GPUs) and has the more generous session limit; Colab is easier
for sharing datasets between teammates. The scripts are identical either way.
