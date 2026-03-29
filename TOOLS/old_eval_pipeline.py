#!/usr/bin/env python3
"""
End-to-end test runner for your GrowLiv pipeline (YOLO -> SAM -> classifier),
with optional "fusion" when YOLO isn't confident:

- If top-1 YOLO confidence >= --yolo_fusion_thresh:
    use top-1 detection normally (fast path)

- Else (YOLO unsure):
    evaluate top-K detections (K = --fusion_topk),
    run the bucket-specific classifier for each,
    and pick the candidate with the best joint score:

        joint_score = yolo_conf * clf_conf

GT conventions:
- Ground-truth species comes from the parent folder name under --images.
- Ground-truth YOLO bucket is derived from gt_species via SPECIES_TO_BUCKET.

Outputs:
- CSV row per image
- Summary accuracy: YOLO bucket acc, final species acc (misses/low-conf counted as wrong)
- Per-bucket + per-species breakdown
- Wrong predictions list
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import csv
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import timm
import yaml

from ultralytics import YOLO
from mobile_sam import sam_model_registry, SamPredictor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# -----------------------------
# YOLO BUCKET → CLASSIFIER GROUP
# (MUST match your trained classifier filenames: mnv3_<group>.pt)
# -----------------------------
COARSE_TO_GROUP = {
    "tiny_pests":   "tiny_pests",     # mnv3_tiny_pests.pt
    "beetles":      "beetles",        # mnv3_beetles.pt
    "borers":       "borers",         # mnv3_borers.pt
    "caterpillars": "caterpillars",   # mnv3_caterpillars.pt
    "plant_bugs":   "plant_bugs",     # mnv3_plant_bugs.pt
    "soil_larvae":  "soil_larvae",    # mnv3_soil_larvae.pt
    "weevils":      "weevils",        # mnv3_weevils.pt
}

# -----------------------------
# Species label normalization / aliases
# -----------------------------
SPECIES_ALIASES = {
    "alfalfaweevil": "alfalfa_weevil",
    "alfalfa_weevil": "alfalfa_weevil",

    "aphid": "aphids",
    "aphids": "aphids",

    "armyworm": "army_worm",
    "army_worm": "army_worm",

    "blackcutworm": "black_cutworm",
    "black_cutworm": "black_cutworm",

    "blisterbeetle": "blister_beetle",
    "blister_beetle": "blister_beetle",

    "cornborer": "corn_borer",
    "corn_borer": "corn_borer",

    "fleabeetle": "flea_beetle",
    "flea_beetle": "flea_beetle",

    "grub": "grub",
    "wireworm": "wireworm",

    "miridae": "miridae",

    "oides": "oides_decempunctata",
    "oides_decempunctata": "oides_decempunctata",
    "leafbeetle": "oides_decempunctata",

    "peachborer": "peach_borer",
    "peach_borer": "peach_borer",

    "redspider": "red_spider",
    "red_spider": "red_spider",

    "tarnishedplantbug": "tarnished_plant_bug",
    "tarnished_plant_bug": "tarnished_plant_bug",

    "thrips": "thrips",

    "strawberryrootweevil": "strawberry_root_weevil",
    "strawberry_root_weevil": "strawberry_root_weevil",

    "fourlinedplantbug": "four_lined_plant_bug",
    "four_lined_plant_bug": "four_lined_plant_bug",
}

# -----------------------------
# Canonical species -> YOLO bucket (7-bucket layout)
# -----------------------------
SPECIES_TO_BUCKET: dict[str, str] = {
    # tiny_pests
    "aphids":      "tiny_pests",
    "thrips":      "tiny_pests",
    "red_spider":  "tiny_pests",

    # beetles
    "blister_beetle":      "beetles",
    "flea_beetle":         "beetles",
    "oides_decempunctata": "beetles",

    # borers
    "corn_borer":  "borers",
    "peach_borer": "borers",

    # caterpillars
    "army_worm":     "caterpillars",
    "black_cutworm": "caterpillars",

    # plant_bugs
    "miridae":              "plant_bugs",
    "tarnished_plant_bug":  "plant_bugs",
    "four_lined_plant_bug": "plant_bugs",

    # soil_larvae
    "grub":     "soil_larvae",
    "wireworm": "soil_larvae",

    # weevils
    "alfalfa_weevil":         "weevils",
    "strawberry_root_weevil": "weevils",
}


def load_names_from_data_yaml(data_yaml: Path):
    data = yaml.safe_load(data_yaml.read_text())
    names = data["names"]
    return list(names.values()) if isinstance(names, dict) else names


def clamp_xyxy(x1, y1, x2, y2, w, h):
    return (
        int(max(0, min(x1, w - 1))),
        int(max(0, min(y1, h - 1))),
        int(max(1, min(x2, w))),
        int(max(1, min(y2, h))),
    )


def pad_xyxy(x1, y1, x2, y2, pad, w, h):
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad * 0.5, bh * pad * 0.5
    return clamp_xyxy(x1 - px, y1 - py, x2 + px, y2 + py, w, h)


def clean_mask(mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m

    areas = stats[:, cv2.CC_STAT_AREA].copy()
    areas[0] = 0
    best = int(np.argmax(areas))
    if areas[best] < min_area:
        return np.zeros_like(m)

    m = (labels == best).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    return m


def apply_mode(crop, mask, mode, bg_alpha):
    if mode == "box":
        return crop
    m = mask.astype(np.float32)
    if mode == "mask":
        return (crop * m[:, :, None]).astype(np.uint8)
    return (crop * (m[:, :, None] + (1 - m[:, :, None]) * bg_alpha)).astype(np.uint8)


def normalize_species(label: str) -> str:
    s = label.strip().lower()
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    s2 = s.replace("_", "")
    return SPECIES_ALIASES.get(s, SPECIES_ALIASES.get(s2, s))


def gt_from_folder(img_path: Path) -> tuple[str, str]:
    raw_folder = img_path.parent.name
    gt_species = normalize_species(raw_folder)
    gt_bucket = SPECIES_TO_BUCKET.get(gt_species, "unknown")
    return gt_species, gt_bucket


class ClassifierBank:
    def __init__(self, root: Path, device):
        self.root = root
        self.device = device
        self.cache = {}

    def load(self, group: str):
        if group in self.cache:
            return self.cache[group]

        ckpt_path = self.root / f"mnv3_{group}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing classifier ckpt: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = timm.create_model(
            ckpt["arch"],
            pretrained=False,
            num_classes=len(ckpt["classes"])
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(self.device)

        self.cache[group] = (model, ckpt["classes"], int(ckpt.get("imgsz", 224)))
        return self.cache[group]

    @torch.no_grad()
    def predict(self, group: str, bgr: np.ndarray):
        model, classes, sz = self.load(group)
        rgb = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (sz, sz))
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        x = ((x - mean) / std).unsqueeze(0).to(self.device)
        p = F.softmax(model(x), dim=1)[0]
        i = int(torch.argmax(p))
        return str(classes[i]), float(p[i])


def run_sam(predictor: SamPredictor, box: np.ndarray, use_points: bool):
    x1, y1, x2, y2 = box.tolist()

    if use_points:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        nx1, ny1 = x1 + 2.0, y1 + 2.0
        nx2, ny2 = x2 - 2.0, y2 - 2.0

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
        return None, None

    return masks, scores


def get_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    crop_slice_ys,
    crop_slice_xs,
    mask_min_area: int,
    min_mask_frac: float,
    max_mask_frac: float,
) -> np.ndarray | None:
    order = np.argsort(scores)[::-1]
    for i in order:
        raw = masks[i]
        crop_raw = raw[crop_slice_ys, crop_slice_xs]
        frac = float(crop_raw.mean())
        if not (min_mask_frac <= frac <= max_mask_frac):
            continue
        return clean_mask(raw, min_area=mask_min_area)
    return None


def evaluate_candidate(
    *,
    img: np.ndarray,
    w: int,
    h: int,
    box_obj,
    yolo_names: list[str],
    predictor: SamPredictor,
    clf_bank: ClassifierBank,
    args,
):
    """
    Evaluate ONE YOLO detection candidate:
      - derive pred_bucket + yolo_conf
      - map to classifier group
      - crop (+pad)
      - run SAM mask (optional points)
      - run classifier
      - compute joint_score = yolo_conf * clf_conf

    Returns dict with fields; returns None if it can't be evaluated (no group map).
    """
    pred_bucket = yolo_names[int(box_obj.cls)]
    yolo_conf = float(box_obj.conf)

    if pred_bucket not in COARSE_TO_GROUP:
        return {
            "pred_bucket": pred_bucket,
            "yolo_conf": yolo_conf,
            "group": "",
            "pred_species": "NO_GROUP_MAP",
            "clf_conf": 0.0,
            "joint": 0.0,
        }

    group = COARSE_TO_GROUP[pred_bucket]

    x1, y1, x2, y2 = map(float, box_obj.xyxy[0])
    px1, py1, px2, py2 = pad_xyxy(x1, y1, x2, y2, args.pad, w, h)
    ys = slice(py1, py2)
    xs = slice(px1, px2)
    crop = img[ys, xs]

    sam_box = np.array([px1, py1, px2, py2], dtype=np.float32)
    masks, scores = run_sam(predictor, sam_box, use_points=args.use_points)

    if masks is not None:
        mask = get_best_mask(
            masks, scores,
            ys, xs,
            mask_min_area=args.mask_min_area,
            min_mask_frac=args.min_mask_frac,
            max_mask_frac=args.max_mask_frac,
        )
    else:
        mask = None

    if mask is None:
        mask_crop = np.ones(crop.shape[:2], dtype=np.uint8)
    else:
        mask_crop = mask[ys, xs]

    crop2 = apply_mode(crop, mask_crop, args.mode, args.bg_alpha)

    pred_species, clf_conf = clf_bank.predict(group, crop2)
    pred_species = normalize_species(pred_species)

    joint = yolo_conf * float(clf_conf)

    return {
        "pred_bucket": pred_bucket,
        "yolo_conf": yolo_conf,
        "group": group,
        "pred_species": pred_species,
        "clf_conf": float(clf_conf),
        "joint": float(joint),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="Root folder containing species subfolders (GT from folder name)")
    ap.add_argument("--yolo", required=True, help="Path to YOLO weights .pt")
    ap.add_argument("--data", required=True, help="Path to YOLO data.yaml (7 buckets)")
    ap.add_argument("--sam_ckpt", required=True, help="MobileSAM checkpoint")
    ap.add_argument("--clf_dir", required=True, help="Folder containing mnv3_<group>.pt files")
    ap.add_argument("--out_csv", default="pipeline_results.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pad", type=float, default=0.03)
    ap.add_argument("--mode", choices=["mask", "box", "hybrid"], default="hybrid")
    ap.add_argument("--bg_alpha", type=float, default=0.2)

    # Thresholds / fusion controls
    ap.add_argument("--yolo_low_conf", type=float, default=0.4,
                    help="Below this, treat as NO_DETECTION (skip fusion entirely)")
    ap.add_argument("--yolo_fusion_thresh", type=float, default=0.6,
                    help="If top-1 YOLO conf is BELOW this, run fusion over top-K candidates")
    ap.add_argument("--fusion_topk", type=int, default=3,
                    help="How many YOLO candidates to consider during fusion (only used when YOLO is unsure)")

    ap.add_argument("--imgsz", type=int, default=896,
                    help="Inference resolution for YOLO + SAM. Images are resized to imgsz x imgsz.")

    # SAM knobs
    ap.add_argument("--use_points", action="store_true",
                    help="Add 1 positive center + 2 negative near-corner point prompts to SAM")
    ap.add_argument("--mask_min_area", type=int, default=200,
                    help="Minimum CC area kept by clean_mask (pixels)")
    ap.add_argument("--min_mask_frac", type=float, default=0.08,
                    help="Skip masks covering less than this fraction of the crop")
    ap.add_argument("--max_mask_frac", type=float, default=0.85,
                    help="Skip masks covering more than this fraction of the crop")

    args = ap.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("Device:", device)

    yolo_names = load_names_from_data_yaml(Path(args.data))
    yolo = YOLO(args.yolo)

    sam = sam_model_registry["vit_t"](checkpoint=args.sam_ckpt).to(device)
    predictor = SamPredictor(sam)

    clf_bank = ClassifierBank(Path(args.clf_dir), device)

    root = Path(args.images)
    rows = []

    # accuracy counters (count every image)
    yolo_correct = 0
    yolo_total = 0
    final_correct = 0
    final_total = 0

    per_bucket = defaultdict(lambda: {"yolo_ok": 0, "yolo_n": 0, "final_ok": 0, "final_n": 0})
    per_species = defaultdict(lambda: {"ok": 0, "n": 0})
    wrong_preds: list[tuple[str, str, str]] = []

    for img_path in sorted(root.rglob("*")):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue

        gt_species, gt_bucket = gt_from_folder(img_path)

        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            continue

        img = cv2.resize(img_orig, (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # count every image
        yolo_total += 1
        final_total += 1
        per_bucket[gt_bucket]["yolo_n"] += 1
        per_bucket[gt_bucket]["final_n"] += 1
        per_species[gt_species]["n"] += 1

        res = yolo.predict(img, imgsz=args.imgsz, verbose=False)[0]
        if not res.boxes or len(res.boxes) == 0:
            rows.append([str(img_path), gt_bucket, gt_species, "", 0.0, "", "NO_DETECTION", 0.0, 0.0, 0, 0, "TOP1"])
            wrong_preds.append((str(img_path), gt_species, "NO_DETECTION"))
            continue

        # Sort by YOLO conf
        boxes_sorted = sorted(res.boxes, key=lambda bb: float(bb.conf), reverse=True)
        top1 = boxes_sorted[0]
        top1_conf = float(top1.conf)

        # Low-conf treated as no detection (skip everything)
        if top1_conf < args.yolo_low_conf:
            pred_bucket = yolo_names[int(top1.cls)]
            rows.append([str(img_path), gt_bucket, gt_species, pred_bucket, top1_conf, "", "LOW_CONF", 0.0, 0.0, 0, 0, "TOP1"])
            wrong_preds.append((str(img_path), gt_species, "LOW_CONF"))
            continue

        # Decide whether to fuse
        use_fusion = top1_conf < args.yolo_fusion_thresh

        if not use_fusion:
            # ---- Normal top-1 path ----
            cand = evaluate_candidate(
                img=img, w=w, h=h, box_obj=top1,
                yolo_names=yolo_names,
                predictor=predictor,
                clf_bank=clf_bank,
                args=args,
            )
            pred_bucket = cand["pred_bucket"]
            yolo_conf = cand["yolo_conf"]
            group = cand["group"]
            pred_species = cand["pred_species"]
            clf_conf = cand["clf_conf"]
            joint = cand["joint"]

            yolo_ok = int(pred_bucket == gt_bucket)
            yolo_correct += yolo_ok
            per_bucket[gt_bucket]["yolo_ok"] += yolo_ok

            final_ok = int(pred_species == gt_species)
            final_correct += final_ok
            per_bucket[gt_bucket]["final_ok"] += final_ok
            per_species[gt_species]["ok"] += final_ok

            if not final_ok:
                wrong_preds.append((str(img_path), gt_species, pred_species))

            rows.append([
                str(img_path),
                gt_bucket, gt_species,
                pred_bucket, yolo_conf,
                group, pred_species, clf_conf, joint,
                yolo_ok, final_ok,
                "TOP1"
            ])
            continue

        # ---- Fusion path (YOLO unsure): evaluate top-K candidates and pick best joint score ----
        topk = max(1, int(args.fusion_topk))
        candidates = boxes_sorted[:topk]

        best = None
        for bb in candidates:
            cand = evaluate_candidate(
                img=img, w=w, h=h, box_obj=bb,
                yolo_names=yolo_names,
                predictor=predictor,
                clf_bank=clf_bank,
                args=args,
            )
            if best is None or cand["joint"] > best["joint"]:
                best = cand

        # If all candidates were NO_GROUP_MAP, joint may be 0.0; still record best attempt.
        pred_bucket = best["pred_bucket"]
        yolo_conf = best["yolo_conf"]
        group = best["group"]
        pred_species = best["pred_species"]
        clf_conf = best["clf_conf"]
        joint = best["joint"]

        yolo_ok = int(pred_bucket == gt_bucket)
        yolo_correct += yolo_ok
        per_bucket[gt_bucket]["yolo_ok"] += yolo_ok

        final_ok = int(pred_species == gt_species)
        final_correct += final_ok
        per_bucket[gt_bucket]["final_ok"] += final_ok
        per_species[gt_species]["ok"] += final_ok

        if not final_ok:
            wrong_preds.append((str(img_path), gt_species, pred_species))

        rows.append([
            str(img_path),
            gt_bucket, gt_species,
            pred_bucket, yolo_conf,
            group, pred_species, clf_conf, joint,
            yolo_ok, final_ok,
            f"FUSION_TOP{topk}"
        ])

    # write CSV
    out_csv = Path(args.out_csv)
    with out_csv.open("w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow([
            "image_path", "gt_bucket", "gt_species",
            "pred_bucket", "pred_bucket_conf",
            "group", "pred_species", "pred_species_conf",
            "joint_score",
            "yolo_bucket_correct", "final_species_correct",
            "decision_mode"
        ])
        wri.writerows(rows)

    def pct(a, b):
        return 0.0 if b == 0 else 100.0 * a / b

    print("\n=== SUMMARY ===")
    print(f"YOLO bucket acc:   {pct(yolo_correct, yolo_total):6.2f}%  ({yolo_correct}/{yolo_total})")
    print(f"Final species acc: {pct(final_correct, final_total):6.2f}%  ({final_correct}/{final_total})")
    print(f"CSV saved: {out_csv}")

    print("\n=== PER-BUCKET ===")
    for bname in sorted(per_bucket.keys()):
        d = per_bucket[bname]
        print(
            f"{bname:20s} | "
            f"yolo {pct(d['yolo_ok'], d['yolo_n']):6.2f}% ({d['yolo_ok']}/{d['yolo_n']}) | "
            f"final {pct(d['final_ok'], d['final_n']):6.2f}% ({d['final_ok']}/{d['final_n']})"
        )

    print("\n=== PER-SPECIES ===")
    for sname in sorted(per_species.keys()):
        d = per_species[sname]
        bar = "█" * d["ok"] + "░" * (d["n"] - d["ok"])
        print(f"{sname:30s} | {pct(d['ok'], d['n']):6.2f}% ({d['ok']}/{d['n']})  {bar}")

    if wrong_preds:
        print(f"\n=== WRONG PREDICTIONS ({len(wrong_preds)}) ===")
        by_gt: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for fpath, gt, pred in wrong_preds:
            by_gt[gt].append((fpath, pred))

        for gt_sp in sorted(by_gt.keys()):
            entries = by_gt[gt_sp]
            print(f"\n  GT: {gt_sp}  ({len(entries)} wrong)")
            for fpath, pred in entries:
                fname = Path(fpath).name
                print(f"    WRONG | {fname:40s} | gt={gt_sp:30s} | pred={pred}")
    else:
        print("\n✓ No wrong predictions!")


if __name__ == "__main__":
    main()