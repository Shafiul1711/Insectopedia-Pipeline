#!/usr/bin/env python3
"""
YOLO labels -> RepViT-SAM box-prompt (+optional points) -> cleaned mask -> masked crops per class folder.

Input expected (YOUR structure):
  ROOT/
    classification/
      class.yaml
      images/
        train/<species_name>/*.jpg|png...
        val/<species_name>/*.jpg|png...
      labels/
        train/<species_name>/*.txt
        val/<species_name>/*.txt

Output:
  OUT/
    train/<class_name>/*.png
    valid/<class_name>/*.png

Notes:
  - Input splits: "train", "val"
  - Output splits: "train", "valid" (val -> valid)
  - Adds:
      * --use_points : 1 positive center + 2 negative near-corners of box
      * mask cleanup: keep largest connected component + morphological close
      * --min_mask_frac : skip tiny/garbage masks within the crop
      * centroid-based mask selection: rejects masks whose centroid is near crop edge

  python PIPELINE_TOOLS/sam_export_specialized.py \
  --root /home/silvermoon/Music/GrowLiv/Dataset \
  --out /home/silvermoon/Music/GrowLiv/samOutTemp \
  --ckpt /home/silvermoon/Music/GrowLiv/RepViT/sam/weights/repvit_sam.pt \
  --device cuda \
  --use_points \
  --min_mask_frac 0.01 \
  --mask_min_area 200 \
  --bg black \
  --limit 10 \
  --box_pad 8 \

"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import yaml
import cv2
import numpy as np
import torch
from tqdm import tqdm

from repvit_sam import SamPredictor, sam_model_registry

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def load_names_from_yaml(yaml_path: Path) -> list[str]:
    data = yaml.safe_load(yaml_path.read_text())
    names = data.get("names", None)
    if names is None:
        raise ValueError(f"No 'names' key found in {yaml_path}")

    if isinstance(names, dict):
        out = [None] * (max(int(k) for k in names.keys()) + 1)
        for k, v in names.items():
            out[int(k)] = str(v)
        if any(x is None for x in out):
            raise ValueError("names dict has missing indices.")
        return out

    if isinstance(names, list):
        return [str(x) for x in names]

    raise ValueError(f"Unsupported names type in {yaml_path}: {type(names)}")


def find_image_for_stem(images_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXTS:
        p = images_dir / f"{stem}{ext}"
        if p.exists():
            return p
    for p in images_dir.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS and p.stem == stem:
            return p
    return None


def yolo_line_to_xyxy(line: str, w: int, h: int) -> tuple[int, float, float, float, float] | None:
    line = line.strip()
    if not line:
        return None
    parts = re.split(r"\s+", line)
    if len(parts) < 5:
        return None

    cls = int(float(parts[0]))
    xc, yc, bw, bh = map(float, parts[1:5])

    x1 = (xc - bw / 2.0) * w
    y1 = (yc - bh / 2.0) * h
    x2 = (xc + bw / 2.0) * w
    y2 = (yc + bh / 2.0) * h

    x1 = max(0.0, min(x1, w - 1.0))
    y1 = max(0.0, min(y1, h - 1.0))
    x2 = max(0.0, min(x2, w - 1.0))
    y2 = max(0.0, min(y2, h - 1.0))

    if x2 <= x1 + 1 or y2 <= y1 + 1:
        return None

    return cls, x1, y1, x2, y2


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sanitize_class(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


def clean_mask(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    """
    Keep largest connected component + close small holes.
    mask: HxW bool/0-1/0-255
    returns: HxW uint8 in {0,1}
    """
    m = (mask > 0).astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m

    areas = stats[:, cv2.CC_STAT_AREA]
    areas[0] = 0  # ignore background
    best = int(np.argmax(areas))

    if areas[best] < min_area:
        return np.zeros_like(m)

    m = (labels == best).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m


def select_best_mask(masks: np.ndarray, scores: np.ndarray,
                     ix1: int, iy1: int, ix2: int, iy2: int,
                     try_invert: bool = False) -> np.ndarray:
    """
    Pick the best mask by scoring all masks and optionally their inverses.
    Use try_invert=True for dark insects on complex backgrounds (weevils etc).
    Omit for thin insects like wireworms where inversion hurts selection.
    """
    crop_h = max(iy2 - iy1, 1)
    crop_w = max(ix2 - ix1, 1)
    order = np.argsort(scores)[::-1]

    candidates = []
    for i in order:
        m = masks[i]
        variants = [(m, False)]
        if try_invert:
            variants.append((~m, True))
        for mask, inverted in variants:
            frac = float(mask.mean())
            if not (0.05 <= frac <= 0.85):
                continue
            crop_m = mask[iy1:iy2, ix1:ix2].astype(np.uint8)
            if crop_m.sum() == 0:
                continue

            # Edge density — how much of each edge strip is filled
            edge_w = max(int(crop_w * 0.10), 1)
            edge_h = max(int(crop_h * 0.10), 1)
            max_edge = max(
                float(crop_m[:, :edge_w].mean()),
                float(crop_m[:, -edge_w:].mean()),
                float(crop_m[:edge_h, :].mean()),
                float(crop_m[-edge_h:, :].mean())
            )

            # Centroid offset from center
            ys, xs = np.where(crop_m)
            centroid_x = xs.mean() / crop_w
            centroid_y = ys.mean() / crop_h
            off_center = abs(centroid_x - 0.5) + abs(centroid_y - 0.5)

            candidates.append((max_edge, off_center, inverted, mask))

    if candidates:
        # Sort by edge bleed first, then centroid offset
        candidates.sort(key=lambda x: (x[0], x[1]))
        return candidates[0][3]

    # Absolute fallback
    return masks[int(np.argmax(scores))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Dataset root containing classification/")
    ap.add_argument("--out", type=str, required=True, help="Output root (e.g., samOut)")
    ap.add_argument("--ckpt", type=str, required=True, help="RepViT-SAM checkpoint path (repvit_sam.pt)")
    ap.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    ap.add_argument("--min_box_px", type=int, default=10, help="Skip bboxes smaller than this (min side in pixels)")
    ap.add_argument("--box_pad", type=int, default=0, help="Expand each bounding box by this many pixels on each side before SAM (useful for tight IP102 boxes)")
    ap.add_argument("--limit", type=int, default=0, help="Debug: limit #label files per species per split (0 = no limit)")
    ap.add_argument(
        "--yaml",
        type=str,
        default="",
        help="Optional explicit path to class.yaml. If omitted, uses ROOT/classification/class.yaml",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        help="Input splits to process (default: train val).",
    )
    ap.add_argument("--use_points", action="store_true",
                    help="Use 1 positive point at box center + 2 negative points near box corners")
    ap.add_argument("--try_invert", action="store_true",
                    help="Also consider inverted masks (helps dark insects on complex backgrounds, avoid for thin insects like wireworms)")
    ap.add_argument("--min_mask_frac", type=float, default=0.01,
                    help="Skip saving crop if mask covers < this fraction of the crop")
    ap.add_argument("--mask_min_area", type=int, default=200,
                    help="Minimum connected-component area for mask cleanup (in pixels)")
    ap.add_argument("--bg", choices=["black", "white"], default="black",
                    help="Background color for masked-out pixels")

    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out)
    ckpt = Path(args.ckpt)

    cls_root = root / "classification"
    class_yaml = Path(args.yaml) if args.yaml else (cls_root / "class.yaml")
    if not class_yaml.exists():
        raise FileNotFoundError(f"Missing {class_yaml}")

    names = load_names_from_yaml(class_yaml)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU.")
        device = "cpu"

    sam = sam_model_registry["repvit"](checkpoint=str(ckpt)).to(device)
    predictor = SamPredictor(sam)

    bg_value = 0 if args.bg == "black" else 255

    for in_split in args.splits:
        out_split = "valid" if in_split == "val" else in_split

        images_split_root = cls_root / "images" / in_split
        labels_split_root = cls_root / "labels" / in_split

        if not images_split_root.exists() or not labels_split_root.exists():
            raise FileNotFoundError(f"Expected {images_split_root} and {labels_split_root}")

        species_dirs = sorted([p for p in images_split_root.iterdir() if p.is_dir()])
        if not species_dirs:
            print(f"[WARN] No species folders found under {images_split_root}")
            continue

        print(f"[INFO] Split {in_split} -> output {out_split} | species_folders={len(species_dirs)}")

        split_crops = 0
        split_missing = 0
        split_bad = 0
        split_skipped_tiny_mask = 0

        for sp_dir in species_dirs:
            sp = sp_dir.name
            lbl_dir = labels_split_root / sp
            if not lbl_dir.exists():
                print(f"[WARN] Missing labels folder for {in_split}/{sp}: {lbl_dir} (skipping)")
                continue

            label_files = sorted(lbl_dir.glob("*.txt"))
            if args.limit and args.limit > 0:
                label_files = label_files[: args.limit]
            if not label_files:
                continue

            pbar = tqdm(label_files, desc=f"{in_split}:{sp}", unit="img", leave=False)
            for lf in pbar:
                stem = lf.stem
                img_path = find_image_for_stem(sp_dir, stem)
                if img_path is None:
                    split_missing += 1
                    pbar.set_postfix(crops=split_crops, missing=split_missing, bad=split_bad, tiny_mask=split_skipped_tiny_mask)
                    continue

                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None:
                    split_bad += 1
                    pbar.set_postfix(crops=split_crops, missing=split_missing, bad=split_bad, tiny_mask=split_skipped_tiny_mask)
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                h, w = img_rgb.shape[:2]
                predictor.set_image(img_rgb)

                lines = lf.read_text().strip().splitlines()
                obj_i = 0
                for line in lines:
                    parsed = yolo_line_to_xyxy(line, w, h)
                    if parsed is None:
                        continue

                    cls, x1, y1, x2, y2 = parsed
                    if cls < 0 or cls >= len(names):
                        continue

                    if (x2 - x1) < args.min_box_px or (y2 - y1) < args.min_box_px:
                        continue

                    class_name = names[cls]
                    safe_class = sanitize_class(class_name)
                    out_dir = out_root / out_split / safe_class
                    ensure_dir(out_dir)

                    # Extract crop coords early — needed for mask selection
                    # Apply optional box padding (useful for tight IP102 bboxes)
                    if args.box_pad > 0:
                        x1 = max(0.0, x1 - args.box_pad)
                        y1 = max(0.0, y1 - args.box_pad)
                        x2 = min(float(w - 1), x2 + args.box_pad)
                        y2 = min(float(h - 1), y2 + args.box_pad)
                    ix1, iy1, ix2, iy2 = map(int, [x1, y1, x2, y2])

                    box = np.array([x1, y1, x2, y2], dtype=np.float32)

                    if args.use_points:
                        cx = (x1 + x2) * 0.5
                        cy = (y1 + y2) * 0.5
                        # 15% of box size — scales with insect size, more aggressive than fixed 2px
                        margin_x = (x2 - x1) * 0.15
                        margin_y = (y2 - y1) * 0.15
                        nx1, ny1 = x1 + margin_x, y1 + margin_y
                        nx2, ny2 = x2 - margin_x, y2 - margin_y
                        point_coords = np.array([[cx, cy], [nx1, ny1], [nx2, ny2]], dtype=np.float32)
                        point_labels = np.array([1, 0, 0], dtype=np.int32)
                        masks, scores, _ = predictor.predict(
                            box=box,
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True,
                        )
                    else:
                        masks, scores, _ = predictor.predict(box=box, multimask_output=True)

                    if masks is None or len(masks) == 0:
                        continue

                    # Centroid-based mask selection — rejects edge-bleeding masks
                    mask_full = select_best_mask(masks, scores, ix1, iy1, ix2, iy2,
                                                 try_invert=args.try_invert)

                    # Clean mask on full image then crop
                    mask_full = clean_mask(mask_full, min_area=args.mask_min_area)

                    crop_bgr = img_bgr[iy1:iy2, ix1:ix2]
                    crop_mask01 = mask_full[iy1:iy2, ix1:ix2].astype(np.uint8)

                    if crop_bgr.size == 0 or crop_mask01.size == 0:
                        continue

                    if float(crop_mask01.mean()) < args.min_mask_frac:
                        split_skipped_tiny_mask += 1
                        continue

                    crop_mask255 = crop_mask01 * 255
                    masked_crop = cv2.bitwise_and(crop_bgr, crop_bgr, mask=crop_mask255)

                    if bg_value == 255:
                        bg = np.full_like(masked_crop, 255)
                        inv = cv2.bitwise_not(crop_mask255)
                        bg_part = cv2.bitwise_and(bg, bg, mask=inv)
                        masked_crop = cv2.add(masked_crop, bg_part)

                    out_name = f"{sp}__{stem}_obj{obj_i:02d}_c{cls}.png"
                    cv2.imwrite(str(out_dir / out_name), masked_crop)

                    split_crops += 1
                    obj_i += 1

                pbar.set_postfix(crops=split_crops, missing=split_missing, bad=split_bad, tiny_mask=split_skipped_tiny_mask)

        print(
            f"[OK] Finished {in_split} -> {out_split} | "
            f"crops={split_crops} missing_images={split_missing} bad_images={split_bad} tiny_mask_skips={split_skipped_tiny_mask}"
        )

    print("[DONE] RepViT-SAM crop export complete.")


if __name__ == "__main__":
    main()