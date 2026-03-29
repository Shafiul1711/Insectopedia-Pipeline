#!/usr/bin/env python3
"""
Hardcoded YOLO image-per-class counter for GrowLiv.

Scans:
  Dataset/classification/train/labels
  Dataset/classification/valid/labels

Counts how many images contain each class.
Each image is counted once per class even if multiple boxes exist.
Also prints combined totals.
"""

from pathlib import Path
from collections import defaultdict

# ======== HARD CODED PATHS ========
TRAIN_LABELS = Path("Dataset/classification/train/labels")
VALID_LABELS = Path("Dataset/classification/valid/labels")

# ======== HARD CODED CLASS MAP ========
CLASS_NAMES = {
    0: "alfalfa_weevil",
    1: "aphids",
    2: "army_worm",
    3: "black_cutworm",
    4: "blister_beetle",
    5: "corn_borer",
    6: "flea_beetle",
    7: "green_bug",
    8: "grub",
    9: "miridae",
    10: "oides_decempunctata",
    11: "peach_borer",
    12: "red_spider",
    13: "tarnished_plant_bug",
    14: "thrips",
    15: "wireworm",
}
# =======================================


def count_images_per_class(labels_path: Path):
    counts = defaultdict(int)
    total_images = 0

    if not labels_path.exists():
        print(f"[ERROR] Path not found: {labels_path}")
        return counts, total_images

    for txt_file in labels_path.glob("*.txt"):
        total_images += 1
        classes_in_image = set()

        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                classes_in_image.add(class_id)

        for cid in classes_in_image:
            counts[cid] += 1

    return counts, total_images


def print_split_report(split_name, counts, total_images):
    print(f"\n===== {split_name.upper()} =====")
    for cid in sorted(CLASS_NAMES.keys()):
        print(f"{cid:2d} | {CLASS_NAMES[cid]:<22} | {counts[cid]:4d} images")
    print(f"\nTotal images in {split_name}: {total_images}\n")


def print_combined_report(train_counts, valid_counts, train_total, valid_total):
    print("\n===== COMBINED TOTALS =====")

    combined_counts = defaultdict(int)

    for cid in CLASS_NAMES.keys():
        combined_counts[cid] = train_counts[cid] + valid_counts[cid]

    for cid in sorted(CLASS_NAMES.keys()):
        print(f"{cid:2d} | {CLASS_NAMES[cid]:<22} | {combined_counts[cid]:4d} images")

    print("\nGrand total images (train + valid):", train_total + valid_total)
    print()


def main():
    train_counts, train_total = count_images_per_class(TRAIN_LABELS)
    valid_counts, valid_total = count_images_per_class(VALID_LABELS)

    print_split_report("train", train_counts, train_total)
    print_split_report("valid", valid_counts, valid_total)
    print_combined_report(train_counts, valid_counts, train_total, valid_total)


if __name__ == "__main__":
    main()
