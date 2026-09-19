#!/usr/bin/env python
"""Convert raw ExDark (class folders + bbGt txt) into a YOLO dataset using the official split.

Example:
    python scripts/prepare_exdark.py \
        --images data/downloads/ExDark --annotations data/downloads/ExDark_Annno \
        --out data/exdark_yolo/raw

Writes:
    <out>/images/{train,val,test}/<stem>.png     (RGB, EXIF applied then stripped, long side <= --max-side)
    <out>/labels/{train,val,test}/<stem>.txt     (cls xc yc w h, normalized)
    <out>/data.yaml
    data/exdark_meta.csv                         (one row per image: split, light, indoor/outdoor, sizes, box counts)
    reports/data_stats.md                        (counts + everything that was fixed or skipped)
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from darksight.constants import CLASS_NAMES, IMAGE_EXTS, INDOOR_OUTDOOR, LIGHT_TYPES, PROJECT_ROOT
from darksight.data.exdark import (ConversionStats, box_to_yolo, format_yolo_line, index_files,
                                   parse_bbgt, read_imageclasslist)
from darksight.data.yolo_dataset import write_data_yaml
from darksight.io_utils import load_rgb_with_info, resize_long_side, save_png


def process_one(job: dict) -> dict:
    rec, img_path, ann_path = job["rec"], Path(job["img"]), job["ann"]
    out_root = Path(job["out"])
    stats = ConversionStats()
    raw_boxes = parse_bbgt(Path(ann_path).read_text(errors="ignore")) if ann_path else []
    exif_mode = job["exif"]
    try:
        img, info = load_rgb_with_info(img_path, apply_exif=exif_mode != "ignore")
        exif_applied = exif_mode != "ignore" and info["orientation"] not in (0, 1)
        if exif_mode == "auto" and info["orientation"] in (5, 6, 7, 8) and raw_boxes:
            # EXIF swaps width/height. Some ExDark boxes were drawn on the un-rotated pixels:
            # keep the raw orientation when every box fits it but some box does not fit the rotated frame.
            rw, rh = info["raw_size"]
            fits = lambda W, H: all(b.left + b.width <= W + 2 and b.top + b.height <= H + 2 for b in raw_boxes)
            if fits(rw, rh) and not fits(img.shape[1], img.shape[0]):
                img, info = load_rgb_with_info(img_path, apply_exif=False)
                exif_applied = False
    except Exception as e:  # corrupt / unreadable
        return {"stem": rec["name"], "error": f"load failed: {e}"}

    h, w = img.shape[:2]
    lines, cls_counts = [], Counter()
    fits_rotated = 0
    for b in raw_boxes:
        y = box_to_yolo(b, w, h, min_size=job["min_box"], stats=stats)
        # diagnostic: box overflows this frame (> 5% of the side) but would fit the transposed one
        if b.left + b.width > 1.05 * w or b.top + b.height > 1.05 * h:
            if b.left + b.width <= h + 2 and b.top + b.height <= w + 2:
                fits_rotated += 1
        if y is not None:
            lines.append(format_yolo_line(*y))
            cls_counts[y[0]] += 1

    stem = Path(rec["name"]).stem
    split = rec["split"]
    out_img = resize_long_side(img, job["max_side"])
    save_png(out_img, out_root / "images" / split / f"{stem}.png")
    label_path = out_root / "labels" / split / f"{stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    return {
        "stem": stem, "split": split, "source": str(img_path),
        "primary_class": CLASS_NAMES[rec["class_id"]], "light": rec["light"],
        "light_name": LIGHT_TYPES.get(rec["light"], "Unknown"), "indoor_outdoor": INDOOR_OUTDOOR.get(rec["indoor_outdoor"]),
        "orig_w": w, "orig_h": h, "w": out_img.shape[1], "h": out_img.shape[0],
        "exif_orientation": info["orientation"], "exif_applied": exif_applied, "mode": info["mode"],
        "has_annotation": ann_path is not None, "n_boxes": len(lines),
        "class_counts": {CLASS_NAMES[k]: v for k, v in cls_counts.items()},
        "stats": asdict(stats), "fits_rotated": fits_rotated,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="ExDark image root (12 class folders)")
    ap.add_argument("--annotations", required=True, help="ExDark_Annno root (12 class folders)")
    ap.add_argument("--classlist", default=str(PROJECT_ROOT / "configs" / "exdark_imageclasslist.txt"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "exdark_yolo" / "raw"))
    ap.add_argument("--meta", default=str(PROJECT_ROOT / "data" / "exdark_meta.csv"))
    ap.add_argument("--report", default=str(PROJECT_ROOT / "reports" / "data_stats.md"))
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--min-box", type=float, default=2.0, help="drop boxes thinner than this (px)")
    ap.add_argument("--exif", choices=["auto", "apply", "ignore"], default="auto",
                    help="auto = apply EXIF rotation unless the boxes only fit the un-rotated image")
    ap.add_argument("--limit", type=int, default=0, help="only first N images per split (smoke test)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    records = read_imageclasslist(args.classlist)
    print(f"imageclasslist: {len(records)} records, split counts {Counter(r.split for r in records)}")

    img_index = index_files(args.images, lambda p: p.suffix.lower() in IMAGE_EXTS)
    ann_index = index_files(args.annotations, lambda p: p.suffix.lower() == ".txt")
    print(f"found {len(img_index)} images, {len(ann_index)} annotation files")

    jobs, missing_img = [], []
    per_split_seen = Counter()
    for r in records:
        key = Path(r.name).stem.lower()
        if key not in img_index:
            missing_img.append(r.name)
            continue
        if args.limit and per_split_seen[r.split] >= args.limit:
            continue
        per_split_seen[r.split] += 1
        ann = ann_index.get(key)
        jobs.append({"rec": {"name": r.name, "class_id": r.class_id, "light": r.light,
                             "indoor_outdoor": r.indoor_outdoor, "split": r.split},
                     "img": str(img_index[key]), "ann": str(ann) if ann else None, "out": args.out,
                     "max_side": args.max_side, "min_box": args.min_box, "exif": args.exif})
    listed = {Path(r.name).stem.lower() for r in records}
    unlisted = sorted(set(img_index) - listed)

    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for res in tqdm(pool.map(process_one, jobs, chunksize=16), total=len(jobs), desc="converting"):
            results.append(res)

    errors = [r for r in results if "error" in r]
    ok = [r for r in results if "error" not in r]

    # ---- metadata csv
    Path(args.meta).parent.mkdir(parents=True, exist_ok=True)
    fields = ["stem", "split", "primary_class", "light", "light_name", "indoor_outdoor", "orig_w", "orig_h",
              "w", "h", "exif_orientation", "exif_applied", "mode", "has_annotation", "n_boxes", "source"]
    with open(args.meta, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(sorted(ok, key=lambda r: r["stem"]))

    write_data_yaml(args.out)

    # ---- report
    tot = ConversionStats()
    unknown = Counter()
    for r in ok:
        s = r["stats"]
        tot.boxes_in += s["boxes_in"]; tot.boxes_out += s["boxes_out"]
        tot.clipped += s["clipped"]; tot.dropped_small += s["dropped_small"]
        unknown.update(s["unknown_classes"])
    split_imgs = Counter(r["split"] for r in ok)
    split_boxes = Counter()
    class_split = {c: Counter() for c in CLASS_NAMES}
    for r in ok:
        split_boxes[r["split"]] += r["n_boxes"]
        for c, n in r["class_counts"].items():
            class_split[c][r["split"]] += n
    light_split = Counter((r["light_name"], r["split"]) for r in ok)
    rotated = [r for r in ok if r["exif_orientation"] not in (0, 1)]
    rotation_kept_raw = [r["stem"] for r in rotated if not r["exif_applied"]]
    suspicious_rot = [r for r in ok if r["fits_rotated"]]
    no_ann = [r["stem"] for r in ok if not r["has_annotation"]]
    empty = [r["stem"] for r in ok if r["n_boxes"] == 0]
    modes = Counter(r["mode"] for r in ok)
    resized = sum(1 for r in ok if (r["w"], r["h"]) != (r["orig_w"], r["orig_h"]))

    S = ["train", "val", "test"]
    md = ["# ExDark → YOLO conversion report", "",
          f"Settings: max_side={args.max_side}, min_box={args.min_box}px, exif={args.exif}, limit={args.limit or 'none'}", "",
          "## Images and boxes per split", "", "| split | images | boxes |", "|---|---|---|"]
    md += [f"| {s} | {split_imgs[s]} | {split_boxes[s]} |" for s in S]
    md += [f"| **total** | {sum(split_imgs.values())} | {sum(split_boxes.values())} |", "",
           "## Boxes per class", "", "| id | class | train | val | test | total |", "|---|---|---|---|---|---|"]
    md += [f"| {i} | {c} | " + " | ".join(str(class_split[c][s]) for s in S) + f" | {sum(class_split[c].values())} |"
           for i, c in enumerate(CLASS_NAMES)]
    md += ["", "## Images per lighting condition", "", "| light | train | val | test |", "|---|---|---|---|"]
    md += [f"| {LIGHT_TYPES[k]} | " + " | ".join(str(light_split[(LIGHT_TYPES[k], s)]) for s in S) + " |"
           for k in sorted(LIGHT_TYPES)]
    md += ["", "## Data-quality log", "",
           f"- listed in imageclasslist but image file missing: **{len(missing_img)}** {missing_img[:10]}",
           f"- image files not in imageclasslist (ignored): **{len(unlisted)}** {unlisted[:10]}",
           f"- unreadable images: **{len(errors)}** {[e['stem'] + ': ' + e['error'] for e in errors[:5]]}",
           f"- images without annotation file: **{len(no_ann)}** {no_ann[:10]}",
           f"- images with zero boxes after conversion: **{len(empty)}** {empty[:10]}",
           f"- boxes read: {tot.boxes_in}, kept: {tot.boxes_out}, clipped to image: {tot.clipped}, "
           f"dropped (<{args.min_box}px): {tot.dropped_small}, unknown class: {dict(unknown)}",
           f"- source colour modes: {dict(modes)}",
           f"- images downscaled to max side {args.max_side}: {resized}",
           f"- images with EXIF rotation tag: **{len(rotated)}** {[r['stem'] for r in rotated[:15]]}; "
           f"kept un-rotated because boxes match raw pixels: {rotation_kept_raw}",
           f"- images whose boxes only fit the *un-rotated* frame (check these!): **{len(suspicious_rot)}** "
           f"{[r['stem'] for r in suspicious_rot[:15]]}", ""]
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(md))
    (Path(args.out) / "conversion_log.json").write_text(json.dumps(
        {"missing_images": missing_img, "unlisted_images": unlisted, "errors": errors, "no_annotation": no_ann,
         "exif_rotated": [r["stem"] for r in rotated], "exif_kept_raw": rotation_kept_raw, "suspicious_rotation": [r["stem"] for r in suspicious_rot]},
        indent=1))
    print("\n".join(md))
    if errors or missing_img:
        print(f"WARNING: {len(errors)} errors, {len(missing_img)} missing images — see report")


if __name__ == "__main__":
    main()
