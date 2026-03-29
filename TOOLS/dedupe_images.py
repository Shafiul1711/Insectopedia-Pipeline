#!/usr/bin/env python3
"""
Find and remove duplicate images using MD5 hashing.
Keeps the first occurrence of each image, removes the rest.

Modes:
  1) Single folder deduplication (original behaviour)
  2) Cross-folder deduplication with --against
  3) YOLO train/valid deduplication with --yolo_root
     Scans Dataset/YOLO/train and valid, removes duplicates found in valid
     that already exist in train (train wins). Also removes matching .txt labels.

Usage:
  # Dry run on a single folder
  python dedupe_images.py --images ToAdd/alfalfa_weevil

  # Actually delete
  python dedupe_images.py --images ToAdd/alfalfa_weevil --delete

  # Dedupe across a second folder
  python dedupe_images.py --images ToAdd/alfalfa_weevil --against Dataset/alfalfa_weevil --delete

  # Dedupe YOLO train vs valid (train wins, removes dupes from valid)
  python dedupe_images.py --yolo_root Dataset/YOLO --delete

  python TOOLS/dedupe_images.py \
  --yolo_root /home/silvermoon/Music/GrowLiv/Dataset/YOLO \
  --dry_run
"""

import argparse
import hashlib
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def hash_file(p: Path) -> str:
    h = hashlib.md5()
    h.update(p.read_bytes())
    return h.hexdigest()


def iter_images(root: Path):
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


def remove_image_and_label(img_path: Path, split_images_root: Path, delete: bool):
    """
    Remove an image and its corresponding label file.
    Mirrors path from images/ -> labels/.
    e.g. Dataset/YOLO/valid/images/foo/bar.jpg
      -> Dataset/YOLO/valid/labels/foo/bar.txt
    """
    try:
        rel = img_path.relative_to(split_images_root)
        labels_root = split_images_root.parent / "labels"
        label_path = labels_root / rel.parent / (img_path.stem + ".txt")
    except ValueError:
        label_path = img_path.parent / (img_path.stem + ".txt")

    if delete:
        img_path.unlink()
        print(f"    deleted image: {img_path}")
        if label_path.exists():
            label_path.unlink()
            print(f"    deleted label: {label_path}")
    else:
        print(f"    would delete image: {img_path}")
        if label_path.exists():
            print(f"    would delete label: {label_path}")


def dedupe_yolo(yolo_root: Path, delete: bool):
    """
    Deduplicate between YOLO train and valid splits.
    Train wins — duplicates are removed from valid.
    """
    train_images = yolo_root / "train" / "images"
    valid_images = yolo_root / "valid" / "images"

    if not train_images.exists():
        raise FileNotFoundError(f"Expected: {train_images}")
    if not valid_images.exists():
        raise FileNotFoundError(f"Expected: {valid_images}")

    train_imgs = iter_images(train_images)
    valid_imgs = iter_images(valid_images)

    print(f"Train images: {len(train_imgs)}")
    print(f"Valid images: {len(valid_imgs)}")
    print(f"Mode: {'DELETE' if delete else 'DRY RUN (use --delete to actually remove)'}")
    print()

    print("Hashing train images...")
    train_hashes: dict[str, Path] = {}
    for p in train_imgs:
        h = hash_file(p)
        if h not in train_hashes:
            train_hashes[h] = p

    print("Scanning valid for duplicates...")
    duplicates = []
    for p in valid_imgs:
        h = hash_file(p)
        if h in train_hashes:
            duplicates.append((p, train_hashes[h]))

    if not duplicates:
        print("No duplicates found between train and valid.")
        return

    print(f"\nFound {len(duplicates)} duplicate(s) in valid that exist in train:\n")
    for dup, original in duplicates:
        print(f"  ORIGINAL (train): {original}")
        print(f"  DUPLICATE (valid):")
        remove_image_and_label(dup, valid_images, delete)
        print()

    if delete:
        print(f"Deleted {len(duplicates)} duplicate(s) from valid.")
    else:
        print(f"Dry run: {len(duplicates)} would be deleted from valid. Run with --delete to remove.")


def dedupe_single(images_root: Path, against, delete: bool):
    """Original single-folder / cross-folder deduplication."""
    seen: dict[str, Path] = {}

    if against:
        against_imgs = iter_images(against)
        print(f"Pre-loading hashes from: {against} ({len(against_imgs)} images)")
        for p in against_imgs:
            h = hash_file(p)
            if h not in seen:
                seen[h] = p

    img_paths = iter_images(images_root)
    print(f"Scanning: {images_root} ({len(img_paths)} images)")
    print(f"Mode: {'DELETE' if delete else 'DRY RUN (use --delete to actually remove)'}")
    print()

    duplicates = []
    for p in img_paths:
        h = hash_file(p)
        if h in seen:
            duplicates.append((p, seen[h]))
        else:
            seen[h] = p

    if not duplicates:
        print("No duplicates found.")
        return

    print(f"Found {len(duplicates)} duplicate(s):\n")
    for dup, original in duplicates:
        print(f"  DUPLICATE: {dup}")
        print(f"  ORIGINAL:  {original}")
        if delete:
            dup.unlink()
            label = dup.parent / (dup.stem + ".txt")
            if label.exists():
                label.unlink()
                print(f"  (also deleted label: {label.name})")
        print()

    if delete:
        print(f"Deleted {len(duplicates)} duplicate image(s).")
    else:
        print(f"Dry run: {len(duplicates)} would be deleted. Run with --delete to remove them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None,
                    help="Folder to deduplicate (single folder mode)")
    ap.add_argument("--against", default=None,
                    help="Optional second folder to cross-check against")
    ap.add_argument("--yolo_root", default=None,
                    help="YOLO dataset root containing train/ and valid/ — dedupes valid against train")
    ap.add_argument("--delete", action="store_true",
                    help="Actually delete duplicates (default: dry run only)")
    args = ap.parse_args()

    if args.yolo_root:
        dedupe_yolo(Path(args.yolo_root), delete=args.delete)
    elif args.images:
        dedupe_single(
            Path(args.images),
            against=Path(args.against) if args.against else None,
            delete=args.delete
        )
    else:
        ap.error("Provide either --images or --yolo_root")


if __name__ == "__main__":
    main()