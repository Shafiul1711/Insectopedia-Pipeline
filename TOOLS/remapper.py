#!/usr/bin/env python3
"""
Remap YOLO label files from old 10-class schema to new 23-class schema.

Old -> New class ID mapping:
  0  alfalfa_weevil              -> 0
  1  black_blister_beetle        -> 18
  2  brown_marmorated_stinkbug   -> 19
  3  colorado_potato_beetle      -> 20
  4  grape_flea_beetle           -> 17
  5  green_stink_bug             -> 21
  6  striped_blister_beetle      -> 22
  7  striped_flea_beetle         -> 23
  8  tarnished_plant_bug         -> 13
  9  two_spotted_spider_mite     -> 12

Usage:
  python TOOLS/remapper.py \
    --labels /home/silvermoon/Music/GrowLiv/Dataset/Insects4.v1-version-1--without-moths.yolov11/train/labels \
    --dry_run

  Remove --dry_run to actually rewrite files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tqdm import tqdm

# Old class ID -> New class ID
REMAP = {
    0: 0,   # alfalfa_weevil
    1: 18,  # black_blister_beetle
    2: 19,  # brown_marmorated_stinkbug
    3: 20,  # colorado_potato_beetle
    4: 17,  # grape_flea_beetle
    5: 21,  # green_stink_bug
    6: 22,  # striped_blister_beetle
    7: 23,  # striped_flea_beetle
    8: 13,  # tarnished_plant_bug
    9: 12,  # two_spotted_spider_mite / red_spider
}


def remap_file(lbl_path: Path, dry_run: bool) -> tuple[int, int]:
    """
    Remap class IDs in a single label file.
    Returns (lines_remapped, lines_skipped).
    """
    lines = lbl_path.read_text().strip().splitlines()
    new_lines = []
    remapped = 0
    skipped = 0

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        old_cls = int(float(parts[0]))
        if old_cls not in REMAP:
            skipped += 1
            new_lines.append(line)  # keep as-is, shouldn't happen
            continue
        new_cls = REMAP[old_cls]
        new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
        remapped += 1

    if not dry_run:
        lbl_path.write_text("\n".join(new_lines) + "\n")

    return remapped, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True,
                    help="Root labels folder (will recurse into train/ and val/ subfolders)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print what would be changed without modifying files")
    args = ap.parse_args()

    labels_root = Path(args.labels)
    if not labels_root.exists():
        raise FileNotFoundError(f"Labels folder not found: {labels_root}")

    txt_files = list(labels_root.rglob("*.txt"))
    if not txt_files:
        print(f"[WARN] No .txt files found under {labels_root}")
        return

    print(f"[INFO] Found {len(txt_files)} label files | dry_run={args.dry_run}")

    total_remapped = 0
    total_skipped = 0

    for lbl in tqdm(txt_files, unit="file"):
        r, s = remap_file(lbl, dry_run=args.dry_run)
        total_remapped += r
        total_skipped += s

    print(f"\n[DONE] lines remapped={total_remapped} lines_unchanged={total_skipped}")
    if args.dry_run:
        print("Dry run — no files modified. Remove --dry_run to apply.")


if __name__ == "__main__":
    main()