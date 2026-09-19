#!/usr/bin/env python
"""Organise the LOL dataset into data/lol/{train,test}/{low,high}.

The official zip unpacks to  LOLdataset/our485/{low,high}  and  LOLdataset/eval15/{low,high}
(some mirrors use lol_dataset/ or capitalised names — the folder search below is case-insensitive).

    python scripts/prepare_lol.py --src data/downloads/LOLdataset --out data/lol
"""

import _bootstrap  # noqa: F401

import argparse
import shutil
from pathlib import Path


def find_dir(root: Path, name: str) -> Path:
    hits = [p for p in root.rglob("*") if p.is_dir() and p.name.lower() == name and "__macosx" not in str(p).lower()]
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one '{name}' folder under {root}, found {hits}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="data/lol")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    for split, folder in (("train", "our485"), ("test", "eval15")):
        base = find_dir(src, folder)
        for kind in ("low", "high"):
            files = sorted(p for p in (base / kind).iterdir() if p.suffix.lower() == ".png")
            dst = out / split / kind
            dst.mkdir(parents=True, exist_ok=True)
            for p in files:
                shutil.copy2(p, dst / p.name)
            print(f"{split}/{kind}: {len(files)} images")
        low = {p.name for p in (out / split / "low").iterdir()}
        high = {p.name for p in (out / split / "high").iterdir()}
        if low != high:
            raise SystemExit(f"{split}: low/high file names differ: {sorted(low ^ high)[:5]}")
    print("expected: train 485 pairs, test 15 pairs")


if __name__ == "__main__":
    main()
