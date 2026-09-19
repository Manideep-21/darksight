#!/usr/bin/env python
"""Locate ExDark images + bbGt annotations anywhere under a search root, and validate them.

Kaggle/Colab mirrors nest the folders differently, so instead of hard-coding a path we look for
the 12 class folders and check the files against the official imageclasslist.

    python scripts/find_exdark.py /kaggle/input          # prints the two paths, exits 1 if incomplete
    IMAGES=$(python scripts/find_exdark.py /kaggle/input --print images)
"""

import _bootstrap  # noqa: F401

import argparse
import sys
from collections import Counter
from pathlib import Path

from darksight.constants import CLASS_NAMES, IMAGE_EXTS, PROJECT_ROOT
from darksight.data.exdark import read_imageclasslist


def candidate_roots(search_root: Path) -> list[Path]:
    """Directories that directly contain at least 8 of the 12 class folders."""
    wanted = {c.lower() for c in CLASS_NAMES}
    roots = []
    for p in [search_root, *search_root.rglob("*")]:
        if not p.is_dir():
            continue
        try:
            names = {c.name.lower() for c in p.iterdir() if c.is_dir()}
        except PermissionError:
            continue
        if len(wanted & names) >= 8:
            roots.append(p)
    return roots


def score(root: Path) -> tuple[int, int]:
    """(number of images, number of bbGt .txt files) directly under this root's class folders."""
    imgs = sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    txts = sum(1 for p in root.rglob("*.txt") if p.is_file())
    return imgs, txts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("search_root", nargs="?", default="/kaggle/input")
    ap.add_argument("--classlist", default=str(PROJECT_ROOT / "configs" / "exdark_imageclasslist.txt"))
    ap.add_argument("--print", choices=["both", "images", "annotations"], default="both")
    args = ap.parse_args()

    root = Path(args.search_root)
    if not root.exists():
        sys.exit(f"search root {root} does not exist")

    images_root = annotations_root = None
    best_imgs = best_txts = (None, -1)
    for cand in candidate_roots(root):
        n_img, n_txt = score(cand)
        if n_img > best_imgs[1]:
            best_imgs = (cand, n_img)
        if n_txt > best_txts[1]:
            best_txts = (cand, n_txt)
    images_root, n_img = best_imgs
    annotations_root, n_txt = best_txts

    if images_root is None:
        sys.exit(f"no ExDark class folders (Bicycle, Boat, ...) found under {root}")

    records = read_imageclasslist(args.classlist)
    listed = {Path(r.name).stem.lower() for r in records}
    found = {p.stem.lower() for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
    ann = {p.name.lower().removesuffix(".txt") for p in annotations_root.rglob("*.txt")} if annotations_root else set()
    ann = {Path(a).stem for a in ann}
    missing_img, missing_ann = listed - found, listed - ann

    if args.print == "images":
        print(images_root); return
    if args.print == "annotations":
        print(annotations_root); return

    print(f"images root      : {images_root}   ({n_img} image files)")
    print(f"annotations root : {annotations_root}   ({n_txt} txt files)")
    print(f"expected {len(listed)} images; missing images: {len(missing_img)}, missing annotations: {len(missing_ann)}")
    if missing_img:
        print("  e.g. missing images:", sorted(missing_img)[:5])
    if missing_ann:
        print("  e.g. missing annotations:", sorted(missing_ann)[:5])
        print("  -> this mirror has no (or partial) bounding boxes; attach a mirror that includes ExDark_Annno,")
        print("     or upload the official annotation zip (5 MB) as your own Kaggle dataset.")
    per_class = Counter(p.parent.name for p in images_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    print("images per class folder:", dict(sorted(per_class.items())))
    sys.exit(0 if not (missing_img or missing_ann) else 1)


if __name__ == "__main__":
    main()
