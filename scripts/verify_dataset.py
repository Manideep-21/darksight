#!/usr/bin/env python
"""Freeze / compare dataset trees.

    python scripts/verify_dataset.py --a data/exdark_yolo/raw --manifest      # write MANIFEST.csv (Week 1 freeze)
    python scripts/verify_dataset.py --a data/exdark_yolo/raw --b data/exdark_yolo/enhanced
Checks for the comparison: identical file lists, identical image sizes, byte-identical labels.
"""

import _bootstrap  # noqa: F401

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from darksight.io_utils import sha256_file


def tree(root: Path, splits: list[str]) -> dict[str, Path]:
    files = []
    for s in splits:
        files += sorted(root.glob(f"images/{s}/*.png")) + sorted(root.glob(f"labels/{s}/*.txt"))
    return {str(p.relative_to(root)): p for p in files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--manifest", action="store_true", help="write <a>/MANIFEST.csv with sha256 of every file")
    args = ap.parse_args()
    a = tree(Path(args.a), args.splits)
    print(f"{args.a}: {sum(k.startswith('images') for k in a)} images, {sum(k.startswith('labels') for k in a)} labels")

    if args.manifest:
        with open(Path(args.a) / "MANIFEST.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["path", "sha256"])
            for k, p in tqdm(a.items(), desc="hashing"):
                wr.writerow([k, sha256_file(p)])
        print("wrote", Path(args.a) / "MANIFEST.csv")

    if not args.b:
        return
    b = tree(Path(args.b), args.splits)
    problems = []
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_a or only_b:
        problems.append(f"file lists differ: {len(only_a)} only in A (e.g. {only_a[:3]}), {len(only_b)} only in B (e.g. {only_b[:3]})")
    for k in tqdm(sorted(set(a) & set(b)), desc="comparing"):
        if k.startswith("labels"):
            if a[k].read_bytes() != b[k].read_bytes():
                problems.append(f"label differs: {k}")
        else:
            with Image.open(a[k]) as ia, Image.open(b[k]) as ib:
                if ia.size != ib.size:
                    problems.append(f"size differs: {k} {ia.size} vs {ib.size}")
    if problems:
        print(f"FAILED ({len(problems)} problems)")
        print("\n".join(problems[:30]))
        sys.exit(1)
    print("OK — trees match (names, image sizes, label bytes)")


if __name__ == "__main__":
    main()
