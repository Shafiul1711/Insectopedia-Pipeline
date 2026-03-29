#!/usr/bin/env python3
"""
Post-process SAM crop output — flag crops where the masked foreground
pixels are mostly black (likely inverted mask / SAM failure).

Logic:
  - Load each crop PNG
  - Find non-black pixels (the mask region, since bg is black)
  - If >black_frac of those pixels are near-black, the insect itself
    is black = likely a silhouette/bad mask
  - Move flagged crops to review/ subfolder

Usage:
  python sam_crop_filter.py \
    --crops /home/silvermoon/Music/GrowLiv/samOutTemp \
    --black_thresh 30 \
    --black_frac 0.85 \
    --dry_run

  Remove --dry_run to actually move files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import cv2
import numpy as np
from tqdm import tqdm


def is_bad_crop(img_bgr: np.ndarray, black_thresh: int, black_frac: float) -> bool:
    """
    Returns True if the crop is likely a silhouette/bad mask.

    Strategy: look only at pixels that are NOT background black.
    Background black = all channels < black_thresh.
    Of the remaining foreground pixels, if >black_frac are still
    near-black, the insect itself is black = bad crop.
    """
    # Background mask: pixels where all channels are very dark
    bg_mask = np.all(img_bgr < black_thresh, axis=2)
    fg_mask = ~bg_mask

    fg_count = fg_mask.sum()
    if fg_count == 0:
        # Entirely black image — definitely bad
        return True

    # Of foreground pixels, how many are near-black?
    fg_pixels = img_bgr[fg_mask]  # shape (N, 3)
    near_black = np.all(fg_pixels < black_thresh, axis=1)
    near_black_frac = near_black.sum() / fg_count

    return float(near_black_frac) > black_frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", required=True,
                    help="SAM output root (contains train/ and valid/ subfolders)")
    ap.add_argument("--black_thresh", type=int, default=30,
                    help="Pixel channel value below which a pixel is considered 'near-black' (default 30)")
    ap.add_argument("--black_frac", type=float, default=0.85,
                    help="If >this fraction of foreground pixels are near-black, flag the crop (default 0.85)")
    ap.add_argument("--dry_run", action="store_true",
                    help="Print what would be moved without actually moving anything")
    args = ap.parse_args()

    crops_root = Path(args.crops)
    if not crops_root.exists():
        raise FileNotFoundError(f"Crops folder not found: {crops_root}")

    png_files = list(crops_root.rglob("*.png"))
    if not png_files:
        print(f"[WARN] No PNG files found under {crops_root}")
        return

    print(f"[INFO] Scanning {len(png_files)} crops | "
          f"black_thresh={args.black_thresh} black_frac={args.black_frac} dry_run={args.dry_run}")

    flagged = 0
    kept = 0

    pbar = tqdm(png_files, unit="crop")
    for png in pbar:
        # Skip files already in a review folder
        if "review" in png.parts:
            continue

        img = cv2.imread(str(png))
        if img is None:
            continue

        if is_bad_crop(img, args.black_thresh, args.black_frac):
            flagged += 1
            # Mirror path under a review/ sibling folder
            # e.g. samOutTemp/train/weevils/foo.png
            #   -> samOutTemp/review/train/weevils/foo.png
            rel = png.relative_to(crops_root)
            review_path = crops_root / "review" / rel
            review_path.parent.mkdir(parents=True, exist_ok=True)

            if not args.dry_run:
                shutil.move(str(png), str(review_path))
                # Also move label if it exists alongside
                lbl = png.with_suffix(".txt")
                if lbl.exists():
                    shutil.move(str(lbl), str(review_path.with_suffix(".txt")))
        else:
            kept += 1

        pbar.set_postfix(flagged=flagged, kept=kept)

    print(f"\n[DONE] flagged={flagged} kept={kept} "
          f"({'dry run — nothing moved' if args.dry_run else f'flagged crops moved to {crops_root}/review/'})")


if __name__ == "__main__":
    main()