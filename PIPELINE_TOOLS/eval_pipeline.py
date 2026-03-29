#!/usr/bin/env python3
"""
End-to-end test runner for the GrowLiv pipeline (YOLO -> SAM -> classifier),
with optional fusion, tiling, low-conf rescue, and per-bucket confidence thresholds.

Routing logic (per image):
  1. YOLO runs on the full image at imgsz (default 896).
  2. If NO_DETECTION or top-1 bucket is 'tiny_pests' or 'weevils' →
     re-run YOLO at retry_imgsz (default 1280) and use those results instead.
  3. If top-1 conf < effective low-conf threshold → LOW_CONF_RESCUE (if enabled) or discard.
  4. If image is large (size trigger) → TILED_SIZE: tiling runs unconditionally.
  5. If top-1 bucket is in --force_tile_buckets → TILED_FORCED: tiling runs unconditionally.
  6. If top-1 conf >= yolo_fusion_thresh → TOP1: fast single-candidate path.
  7. Else → FUSION_TOP<k>: evaluate top-K candidates, pick best joint score.
     If conf/joint still weak after fusion → TILED_FUSION: tiling may override.

GT conventions:
  Ground-truth species = parent folder name under --images.
  Ground-truth YOLO bucket = derived via SPECIES_TO_BUCKET map.

Decision modes in output CSV:
  TOP1             - YOLO confident; single candidate used
  TOP1_FALLBACK    - Large image, tiling found nothing; fell back to TOP1
  FUSION_TOP<k>    - YOLO unsure; fusion over top-k (tiling didn't improve)
  TILED_FUSION     - YOLO unsure; tiling beat fusion result
  TILED_SIZE       - Large image; tiling fired unconditionally
  TILED_FORCED     - Bucket in --force_tile_buckets; tiling fired unconditionally
  OVERRIDE_BUCKET  - YOLO bucket ignored; routed to --override_bucket classifier instead
  LOW_CONF_RESCUE  - YOLO below low-conf floor; classifier voted and accepted
  LOW_CONF         - YOLO below low-conf floor; rescue failed or disabled
  NO_DETECTION     - YOLO found no boxes at all
  YOLO_ONLY        - Bucket has no classifier (e.g. potato_beetle); YOLO result used directly

================================================================================
RECOMMENDED RUN COMMAND (tuned for GrowLiv dataset, all features enabled)
================================================================================

python eval_pipeline.py \
  --images test_suite \
  --yolo Yolo+Classifier+Info/YOLO.pt \
  --data Dataset/YOLO/yolo.yaml \
  --sam_ckpt RepViT/sam/weights/repvit_sam.pt \
  --clf_dir Yolo+Classifier+Info/classifiers/ \
  --out_csv results.csv \
  --device cuda \
  --imgsz 896 \
  --retry_imgsz 1280 \
  --retry_buckets tiny_pests weevils \
  --pad 0.03 \
  --mode hybrid \
  --bg_alpha 0.2 \
  --yolo_low_conf 0.4 \
  --yolo_fusion_thresh 0.6 \
  --fusion_topk 3 \
  --use_points \
  --mask_min_area 200 \
  --min_mask_frac 0.02 \
  --max_mask_frac 0.85 \
  --clf_prescale 224 \
  --tile_size_mult 1.5 \
  --tile_overlap 0.25 \
  --tile_conf_thresh 0.50 \
  --tile_joint_thresh 0.35 \
  --tile_per_tile_topk 3 \
  --tile_topk 3 \
  --low_conf_rescue \
  --low_conf_rescue_topk 3 \
  --low_conf_weight_clf 0.7 \
  --low_conf_weight_yolo 0.3 \
  --low_conf_accept_thresh 0.5 \
  --tiny_pest_low_conf


  #1280
  python PIPELINE_TOOLS/eval_pipeline.py \
  --images test_suite \
  --yolo YOLOModels/YOLO26_GrowLiv_V4.pt \
  --data Dataset/YOLO/data.yaml \
  --sam_ckpt RepViT/sam/weights/repvit_sam.pt \
  --clf_dir ClfModelsRes \
  --out_csv results.csv \
  --device cuda \
  --imgsz 896 \
  --retry_imgsz 1280 \
  --retry_buckets caterpillars flea_beetle weevils tiny_pests \
  --pad 0.05 \
  --mode hybrid \
  --bg_alpha 0.2 \
  --yolo_low_conf 0.4 \
  --yolo_fusion_thresh 0.6 \
  --fusion_topk 3 \
  --use_points \
  --mask_min_area 200 \
  --min_mask_frac 0.005 \
  --max_mask_frac 0.85 \
  --clf_prescale 224 \
  --tile_size_mult 1.5 \
  --tile_overlap 0.25 \
  --tile_conf_thresh 0.50 \
  --tile_joint_thresh 0.35 \
  --tile_per_tile_topk 3 \
  --tile_topk 3 \
  --low_conf_rescue \
  --low_conf_rescue_topk 3 \
  --low_conf_weight_clf 0.7 \
  --low_conf_weight_yolo 0.3 \
  --low_conf_accept_thresh 0.5 \
  --tiny_pest_low_conf 0.2 \
  --bucket_mode_overrides blister_beetle:box caterpillars:mask tiny_pests:mask


  --override_bucket tiny_pests \
  --force_tile_buckets tiny_pests

  default tiling 0.5 and 0.35 tiny conf

================================================================================
CHANGES FROM PREVIOUS VERSION
================================================================================

  1. red_spider / redspider aliases → spider_mite (tiny_pests bucket unchanged)
  2. borers bucket REMOVED entirely from COARSE_TO_GROUP and SPECIES_TO_BUCKET
  3. corn_borer and peach_borer moved from borers → caterpillars bucket
  4. potato_beetle moved OUT of YOLO_ONLY_BUCKETS; now uses rn18_potato_beetle.pt classifier
  5. striped_cucumber_beetle added to potato_beetle bucket
  6. Comprehensive alias coverage added for all species (camelCase, snake_case, no-sep, plural)

================================================================================
NOTES
================================================================================

Install dependencies:
    pip install ultralytics timm torch torchvision opencv-python pyyaml tqdm
    pip install git+https://github.com/ChaoningZhang/MobileSAM.git

force_tile_buckets:
    Any bucket listed here will unconditionally trigger tiling, bypassing the
    confidence and joint-score checks entirely. The tiled result is used if it
    beats the fusion result; otherwise falls back to TOP1.
    Example: --force_tile_buckets tiny_pests weevils

Resolution retry logic:
    After the initial YOLO pass at --imgsz (default 896), if the result is:
      - NO_DETECTION (no boxes found), OR
      - top-1 bucket is in --retry_buckets (default: tiny_pests, weevils)
    then YOLO is re-run at --retry_imgsz (default 1280). The higher resolution
    helps catch small insects (aphids, thrips, spider mites, weevils) that may
    be missed or ambiguous at lower resolution. The retry result replaces the
    original for all downstream processing.
    Use --no_resolution_retry to disable this behaviour entirely.

clf_prescale:
    Classifiers were trained on raw crops at natural size. --clf_prescale 224
    downscales large crops before the classifier's own resize, matching the
    effective scale of training crops. Set to 0 to disable.

Two-pass tiling:
    Pass 1 (cheap)  — YOLO only on every tile, collect up to tile_per_tile_topk per tile.
    Pass 2 (expensive) — SAM+classifier on top tile_topk candidates by YOLO conf only.

Low-conf rescue:
    When YOLO conf < effective floor, instead of discarding, run SAM+classifier
    on top low_conf_rescue_topk candidates. Accept if:
        low_conf_weight_yolo * yolo_conf + low_conf_weight_clf * clf_conf >= low_conf_accept_thresh

tiny_pest_low_conf:
    Overrides --yolo_low_conf to the specified value for the tiny_pests bucket
    (aphids, thrips, spider_mite) only. All other buckets use --yolo_low_conf.
    Example: --tiny_pest_low_conf 0.15

YOLO-only buckets (no classifier):
    (none currently — potato_beetle now uses rn18_potato_beetle.pt)

Expected folder structure for --images:
    images/
      aphids/
        aphids001.jpg
      army_worm/
        army_worm001.jpg
      ...
    Folder names must match species keys in SPECIES_ALIASES.

Expected --clf_dir contents:
    classifiers/
      rn18_tiny_pests.pt
      rn18_flea_beetle.pt
      rn18_caterpillars.pt        # includes corn_borer + peach_borer
      rn18_plant_bugs.pt
      rn18_soil_larvae.pt
      rn18_weevils.pt
      rn18_stink_bugs.pt
      rn18_blister_beetle.pt
      rn18_potato_beetle.pt       # colorado_potato_beetle + striped_cucumber_beetle
"""

from __future__ import annotations

import argparse
import math
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
from tqdm import tqdm

from ultralytics import YOLO
from repvit_sam import sam_model_registry, SamPredictor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
clf_prefix = "rn18"  # change this for model

# Default buckets that trigger a resolution retry at retry_imgsz
DEFAULT_RETRY_BUCKETS = {"tiny_pests", "weevils"}

# Default buckets that always trigger tiling (can be overridden via --force_tile_buckets)
DEFAULT_FORCE_TILE_BUCKETS: set[str] = set()

# -----------------------------
# YOLO BUCKET → CLASSIFIER GROUP
# MUST match your trained classifier filenames: rn18_<group>.pt
#
# CHANGES:
#   - borers REMOVED
#   - caterpillars now handles corn_borer + peach_borer (previously in borers)
#   - potato_beetle ADDED (rn18_potato_beetle.pt; previously YOLO-only)
# -----------------------------
COARSE_TO_GROUP = {
    "tiny_pests":     "tiny_pests",     # rn18_tiny_pests.pt
    "flea_beetle":    "flea_beetle",    # rn18_flea_beetle.pt
    "caterpillars":   "caterpillars",   # rn18_caterpillars.pt  (army_worm, black_cutworm, corn_borer, peach_borer)
    "plant_bugs":     "plant_bugs",     # rn18_plant_bugs.pt
    "soil_larvae":    "soil_larvae",    # rn18_soil_larvae.pt
    "weevils":        "weevils",        # rn18_weevils.pt
    "stink_bugs":     "stink_bugs",     # rn18_stink_bugs.pt
    "blister_beetle": "blister_beetle", # rn18_blister_beetle.pt
    "potato_beetle":  "potato_beetle",  # rn18_potato_beetle.pt  (colorado_potato_beetle, striped_cucumber_beetle)
}

# Buckets that have no classifier — YOLO prediction is the final answer.
# potato_beetle has been REMOVED from here; it now uses rn18_potato_beetle.pt
YOLO_ONLY_BUCKETS: dict[str, str] = {}

# -----------------------------
# Species label normalization / aliases
#
# Rules applied in normalize_species():
#   1. Strip, lowercase, replace non-alphanumeric runs with '_', dedupe '_'
#   2. Look up in SPECIES_ALIASES by snake_case key
#   3. Also try no-separator key (s2 = s.replace("_",""))
#
# Coverage for each species includes:
#   - canonical snake_case          e.g. "army_worm"
#   - no separator / run-together   e.g. "armyworm"
#   - common spelling variants      e.g. "armyworms" (plural)
#   - camelCase (collapsed to no-sep after lowercasing)
# -----------------------------
SPECIES_ALIASES: dict[str, str] = {

    # ── alfalfa_weevil ──────────────────────────────────────────────────────
    "alfalfa_weevil":           "alfalfa_weevil",
    "alfalfaweevil":            "alfalfa_weevil",
    "alfalfa_weevils":          "alfalfa_weevil",
    "alfalfaweevils":           "alfalfa_weevil",

    # ── aphids ──────────────────────────────────────────────────────────────
    "aphid":                    "aphids",
    "aphids":                   "aphids",
    "plant_louse":              "aphids",
    "plant_lice":               "aphids",
    "plantlouse":               "aphids",
    "plantlice":                "aphids",
    "greenfly":                 "aphids",
    "blackfly":                 "aphids",

    # ── army_worm ────────────────────────────────────────────────────────────
    "army_worm":                "army_worm",
    "armyworm":                 "army_worm",
    "army_worms":               "army_worm",
    "armyworms":                "army_worm",
    "fall_armyworm":            "army_worm",
    "fallarmyworm":             "army_worm",
    "western_armyworm":         "army_worm",
    "westernarmyworm":          "army_worm",

    # ── black_cutworm ────────────────────────────────────────────────────────
    "black_cutworm":            "black_cutworm",
    "blackcutworm":             "black_cutworm",
    "black_cutworms":           "black_cutworm",
    "blackcutworms":            "black_cutworm",
    "cutworm":                  "black_cutworm",
    "cutworms":                 "black_cutworm",

    # ── blister_beetle ───────────────────────────────────────────────────────
    "blister_beetle":           "blister_beetle",
    "blisterbeetle":            "blister_beetle",
    "blister_beetles":          "blister_beetle",
    "blisterbeetles":           "blister_beetle",

    # ── black_blister_beetle ─────────────────────────────────────────────────
    "black_blister_beetle":     "black_blister_beetle",
    "blackblisterbeetle":       "black_blister_beetle",
    "black_blister_beetles":    "black_blister_beetle",
    "blackblisterbeetles":      "black_blister_beetle",

    # ── striped_blister_beetle ───────────────────────────────────────────────
    "striped_blister_beetle":   "striped_blister_beetle",
    "stripedblisterbeetle":     "striped_blister_beetle",
    "striped_blister_beetles":  "striped_blister_beetle",
    "stripedblisterbeetles":    "striped_blister_beetle",

    # ── colorado_potato_beetle ───────────────────────────────────────────────
    "colorado_potato_beetle":   "colorado_potato_beetle",
    "coloradopotatobeetle":     "colorado_potato_beetle",
    "colorado_potato_beetles":  "colorado_potato_beetle",
    "coloradopotatobeetles":    "colorado_potato_beetle",
    "potato_beetle":            "colorado_potato_beetle",
    "potatobeetle":             "colorado_potato_beetle",
    "potato_beetles":           "colorado_potato_beetle",
    "potatobeetles":            "colorado_potato_beetle",
    "cpb":                      "colorado_potato_beetle",

    # ── striped_cucumber_beetle ──────────────────────────────────────────────
    "striped_cucumber_beetle":  "striped_cucumber_beetle",
    "stripedcucumberbeetle":    "striped_cucumber_beetle",
    "striped_cucumber_beetles": "striped_cucumber_beetle",
    "stripedcucumberbeetles":   "striped_cucumber_beetle",
    "cucumber_beetle":          "striped_cucumber_beetle",
    "cucumberbeetle":           "striped_cucumber_beetle",
    "cucumber_beetles":         "striped_cucumber_beetle",
    "cucumberbeetles":          "striped_cucumber_beetle",
    "scb":                      "striped_cucumber_beetle",

    # ── corn_borer ───────────────────────────────────────────────────────────
    # NOTE: corn_borer now routes to caterpillars bucket (borers removed)
    "corn_borer":               "corn_borer",
    "cornborer":                "corn_borer",
    "corn_borers":              "corn_borer",
    "cornborers":               "corn_borer",
    "european_corn_borer":      "corn_borer",
    "europeancornborer":        "corn_borer",
    "ecb":                      "corn_borer",

    # ── peach_borer ──────────────────────────────────────────────────────────
    # NOTE: peach_borer now routes to caterpillars bucket (borers removed)
    "peach_borer":              "peach_borer",
    "peachborer":               "peach_borer",
    "peach_borers":             "peach_borer",
    "peachborers":              "peach_borer",
    "lesser_peach_borer":       "peach_borer",
    "lesserpeachborer":         "peach_borer",

    # ── flea_beetle ──────────────────────────────────────────────────────────
    "flea_beetle":              "flea_beetle",
    "fleabeetle":               "flea_beetle",
    "flea_beetles":             "flea_beetle",
    "fleabeetles":              "flea_beetle",

    # ── grape_flea_beetle ────────────────────────────────────────────────────
    "grape_flea_beetle":        "grape_flea_beetle",
    "grapefleabeetle":          "grape_flea_beetle",
    "grape_flea_beetles":       "grape_flea_beetle",
    "grapefleabeetles":         "grape_flea_beetle",

    # ── striped_flea_beetle ──────────────────────────────────────────────────
    "striped_flea_beetle":      "striped_flea_beetle",
    "stripedfleabeetle":        "striped_flea_beetle",
    "striped_flea_beetles":     "striped_flea_beetle",
    "stripedfleabeetles":       "striped_flea_beetle",

    # ── four_lined_plant_bug ─────────────────────────────────────────────────
    "four_lined_plant_bug":     "four_lined_plant_bug",
    "fourlinedplantbug":        "four_lined_plant_bug",
    "four_lined_plant_bugs":    "four_lined_plant_bug",
    "fourlinedplantbugs":       "four_lined_plant_bug",
    "4_lined_plant_bug":        "four_lined_plant_bug",
    "4linedplantbug":           "four_lined_plant_bug",

    # ── green_stink_bug ──────────────────────────────────────────────────────
    "green_stink_bug":          "green_stink_bug",
    "greenstinkbug":            "green_stink_bug",
    "green_stink_bugs":         "green_stink_bug",
    "greenstinkbugs":           "green_stink_bug",
    "green_shield_bug":         "green_stink_bug",
    "greenshieldbug":           "green_stink_bug",

    # ── brown_marmorated_stink_bug ───────────────────────────────────────────
    "brown_marmorated_stink_bug":  "brown_marmorated_stink_bug",
    "brownmarmoratedstinkbug":     "brown_marmorated_stink_bug",
    "brown_marmorated_stink_bugs": "brown_marmorated_stink_bug",
    "brownmarmoratedstinkbugs":    "brown_marmorated_stink_bug",
    "bmsb":                        "brown_marmorated_stink_bug",
    "marmorated_stink_bug":        "brown_marmorated_stink_bug",
    "marmoratedstinkbug":          "brown_marmorated_stink_bug",

    # ── grub ─────────────────────────────────────────────────────────────────
    "grub":                     "grub",
    "grubs":                    "grub",
    "white_grub":               "grub",
    "whitegrub":                "grub",
    "white_grubs":              "grub",
    "whitegrubs":               "grub",
    "soil_grub":                "grub",
    "soilgrub":                 "grub",

    # ── miridae ──────────────────────────────────────────────────────────────
    "miridae":                  "miridae",
    "mirid":                    "miridae",
    "mirids":                   "miridae",
    "mirid_bug":                "miridae",
    "miridbug":                 "miridae",
    "capsid_bug":               "miridae",
    "capsidbug":                "miridae",

    # ── spider_mite ──────────────────────────────────────────────────────────
    # CHANGED: was red_spider / red_spider; canonical name is now spider_mite
    "spider_mite":                  "spider_mite",
    "spidermite":                   "spider_mite",
    "spider_mites":                 "spider_mite",
    "spidermites":                  "spider_mite",
    "red_spider":                   "spider_mite",   # legacy alias kept
    "redspider":                    "spider_mite",   # legacy alias kept
    "red_spider_mite":              "spider_mite",
    "redspidermite":                "spider_mite",
    "red_spider_mites":             "spider_mite",
    "redspidermites":               "spider_mite",
    "two_spotted_spider_mite":      "spider_mite",
    "twospottedspidermite":         "spider_mite",
    "two_spotted_spider_mites":     "spider_mite",
    "twospottedspidermites":        "spider_mite",
    "twospotted_spider_mite":       "spider_mite",
    "twospottedmite":               "spider_mite",
    "tssm":                         "spider_mite",
    "european_red_mite":            "spider_mite",
    "europeanredmite":              "spider_mite",

    # ── tarnished_plant_bug ──────────────────────────────────────────────────
    "tarnished_plant_bug":      "tarnished_plant_bug",
    "tarnishedplantbug":        "tarnished_plant_bug",
    "tarnished_plant_bugs":     "tarnished_plant_bug",
    "tarnishedplantbugs":       "tarnished_plant_bug",
    "lygus_bug":                "tarnished_plant_bug",
    "lygusbug":                 "tarnished_plant_bug",

    # ── thrips ───────────────────────────────────────────────────────────────
    "thrips":                   "thrips",
    "thrip":                    "thrips",
    "western_flower_thrips":    "thrips",
    "westernflowerthrips":      "thrips",
    "onion_thrips":             "thrips",
    "onionthrips":              "thrips",

    # ── strawberry_root_weevil ───────────────────────────────────────────────
    "strawberry_root_weevil":   "strawberry_root_weevil",
    "strawberryrootweevil":     "strawberry_root_weevil",
    "strawberry_root_weevils":  "strawberry_root_weevil",
    "strawberryrootweevils":    "strawberry_root_weevil",
    "root_weevil":              "strawberry_root_weevil",
    "rootweevil":               "strawberry_root_weevil",

    # ── wireworm ─────────────────────────────────────────────────────────────
    "wireworm":                 "wireworm",
    "wireworms":                "wireworm",
    "click_beetle_larva":       "wireworm",
    "clickbeetlelarva":         "wireworm",
    "elaterid_larva":           "wireworm",
    "elateridlarva":            "wireworm",
}

# -----------------------------
# Canonical species -> YOLO bucket (9-bucket layout)
#
# CHANGES vs previous version:
#   - red_spider  → spider_mite  (same bucket: tiny_pests)
#   - borers bucket REMOVED; corn_borer + peach_borer → caterpillars
#   - potato_beetle bucket gains striped_cucumber_beetle
# -----------------------------
SPECIES_TO_BUCKET: dict[str, str] = {

    # ── tiny_pests ───────────────────────────────────────────────────────────
    "aphids":       "tiny_pests",
    "thrips":       "tiny_pests",
    "spider_mite":  "tiny_pests",   # formerly red_spider

    # ── flea_beetle ──────────────────────────────────────────────────────────
    "flea_beetle":          "flea_beetle",
    "grape_flea_beetle":    "flea_beetle",
    "striped_flea_beetle":  "flea_beetle",

    # ── caterpillars ─────────────────────────────────────────────────────────
    # corn_borer + peach_borer moved here from the now-removed borers bucket
    "army_worm":     "caterpillars",
    "black_cutworm": "caterpillars",
    "corn_borer":    "caterpillars",   # MOVED from borers
    "peach_borer":   "caterpillars",   # MOVED from borers

    # ── plant_bugs ───────────────────────────────────────────────────────────
    "miridae":              "plant_bugs",
    "tarnished_plant_bug":  "plant_bugs",
    "four_lined_plant_bug": "plant_bugs",

    # ── soil_larvae ──────────────────────────────────────────────────────────
    "grub":      "soil_larvae",
    "wireworm":  "soil_larvae",

    # ── weevils ──────────────────────────────────────────────────────────────
    "alfalfa_weevil":         "weevils",
    "strawberry_root_weevil": "weevils",

    # ── stink_bugs ───────────────────────────────────────────────────────────
    "green_stink_bug":            "stink_bugs",
    "brown_marmorated_stink_bug": "stink_bugs",

    # ── blister_beetle ───────────────────────────────────────────────────────
    "blister_beetle":         "blister_beetle",
    "black_blister_beetle":   "blister_beetle",
    "striped_blister_beetle": "blister_beetle",

    # ── potato_beetle ────────────────────────────────────────────────────────
    # Now classifier-backed via rn18_potato_beetle.pt (no longer YOLO-only)
    "colorado_potato_beetle":  "potato_beetle",
    "striped_cucumber_beetle": "potato_beetle",   # NEW
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

        ckpt_path = self.root / f"{clf_prefix}_{group}.pt"
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
        # Match training val transform: resize shortest side to sz*1.14, center crop to sz
        rgb_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb_full.shape[:2]
        short = min(h, w)
        prescale = int(sz * 1.14)
        scale = prescale / short
        new_w = max(sz, int(w * scale))
        new_h = max(sz, int(h * scale))
        rgb_scaled = cv2.resize(rgb_full, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cy = (new_h - sz) // 2
        cx = (new_w - sz) // 2
        rgb = rgb_scaled[cy:cy+sz, cx:cx+sz]
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


def image_is_large(orig_w: int, orig_h: int, imgsz: int, size_mult: float = 1.5) -> bool:
    return max(orig_w, orig_h) >= int(math.ceil(size_mult * imgsz))


def should_tile(
    orig_w: int,
    orig_h: int,
    imgsz: int,
    topk_confs: list[float],
    best_joint: float,
    *,
    size_mult: float = 1.5,
    conf_tile_thresh: float = 0.50,
    joint_tile_thresh: float = 0.35,
) -> bool:
    size_trigger = image_is_large(orig_w, orig_h, imgsz, size_mult)
    conf_trigger = (max(topk_confs) < conf_tile_thresh) or (best_joint < joint_tile_thresh)
    return bool(size_trigger or conf_trigger)


def make_tiles(img: np.ndarray, tile_size: int, overlap: float = 0.25):
    h, w = img.shape[:2]
    stride = int(tile_size * (1.0 - overlap))
    stride = max(1, stride)

    tiles = []
    ys = list(range(0, max(1, h - tile_size + 1), stride))
    xs = list(range(0, max(1, w - tile_size + 1), stride))
    if ys[-1] != max(0, h - tile_size):
        ys.append(max(0, h - tile_size))
    if xs[-1] != max(0, w - tile_size):
        xs.append(max(0, w - tile_size))

    for y0 in ys:
        for x0 in xs:
            y1 = min(h, y0 + tile_size)
            x1 = min(w, x0 + tile_size)
            tile = img[y0:y1, x0:x1]
            tiles.append((tile, x0, y0))
    return tiles


def lift_xyxy(tile_xyxy: np.ndarray, xoff: int, yoff: int) -> np.ndarray:
    x1, y1, x2, y2 = map(float, tile_xyxy)
    return np.array([x1 + xoff, y1 + yoff, x2 + xoff, y2 + yoff], dtype=np.float32)


def run_tiled_fusion(
    *,
    img_full: np.ndarray,
    yolo: YOLO,
    yolo_names: list[str],
    predictor: SamPredictor,
    clf_bank: ClassifierBank,
    args,
    tile_overlap: float = 0.25,
    per_tile_topk: int = 2,
    tile_topk: int = 3,
):
    H, W = img_full.shape[:2]
    tiles = make_tiles(img_full, tile_size=args.imgsz, overlap=tile_overlap)

    # Pass 1: YOLO only
    all_candidates: list[tuple[float, object, np.ndarray]] = []

    for tile, xoff, yoff in tiles:
        res_t = yolo.predict(tile, imgsz=args.imgsz, verbose=False)[0]
        if not res_t.boxes or len(res_t.boxes) == 0:
            continue

        boxes_sorted = sorted(res_t.boxes, key=lambda bb: float(bb.conf), reverse=True)
        for bb in boxes_sorted[:per_tile_topk]:
            lifted = lift_xyxy(bb.xyxy[0], xoff, yoff)
            all_candidates.append((float(bb.conf), bb, lifted))

    tile_top_confs = [c for c, _, _ in all_candidates]

    if not all_candidates:
        return None, tile_top_confs

    # Pass 2: SAM + classifier on top-N
    all_candidates.sort(key=lambda t: t[0], reverse=True)
    if tile_topk > 0:
        all_candidates = all_candidates[:tile_topk]

    predictor.set_image(cv2.cvtColor(img_full, cv2.COLOR_BGR2RGB))

    best = None
    for _, bb, lifted in all_candidates:
        pred_bucket = yolo_names[int(bb.cls)]

        # Skip SAM+classifier for YOLO-only buckets in tiling too
        if pred_bucket in YOLO_ONLY_BUCKETS:
            cand = {
                "pred_bucket": pred_bucket,
                "yolo_conf": float(bb.conf),
                "group": "",
                "pred_species": YOLO_ONLY_BUCKETS[pred_bucket],
                "clf_conf": 0.0,
                "joint": float(bb.conf),
            }
        else:
            cand = evaluate_candidate(
                img=img_full,
                w=W,
                h=H,
                box_obj=bb,
                xyxy_override=lifted,
                yolo_names=yolo_names,
                predictor=predictor,
                clf_bank=clf_bank,
                args=args,
            )

        if best is None or cand["joint"] > best["joint"]:
            best = cand

    return best, tile_top_confs


def evaluate_candidate(
    *,
    img: np.ndarray,
    w: int,
    h: int,
    box_obj,
    xyxy_override: np.ndarray | None = None,
    yolo_names: list[str],
    predictor: SamPredictor,
    clf_bank: ClassifierBank,
    args,
):
    pred_bucket = yolo_names[int(box_obj.cls)]
    yolo_conf = float(box_obj.conf)

    # YOLO-only bucket: no classifier needed
    if pred_bucket in YOLO_ONLY_BUCKETS:
        return {
            "pred_bucket": pred_bucket,
            "yolo_conf": yolo_conf,
            "group": "",
            "pred_species": YOLO_ONLY_BUCKETS[pred_bucket],
            "clf_conf": 0.0,
            "joint": yolo_conf,
        }

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

    if xyxy_override is None:
        x1, y1, x2, y2 = map(float, box_obj.xyxy[0])
    else:
        x1, y1, x2, y2 = map(float, xyxy_override)
    px1, py1, px2, py2 = pad_xyxy(x1, y1, x2, y2, args.pad, w, h)
    ys = slice(py1, py2)
    xs = slice(px1, px2)
    crop = img[ys, xs]

    # Per-bucket mode override: check args.bucket_mode_overrides before falling
    # back to the global args.mode. If the effective mode is "box", skip SAM
    # entirely since the raw crop is used regardless.
    bucket_mode_overrides: dict = getattr(args, "bucket_mode_overrides", {})
    effective_mode = bucket_mode_overrides.get(pred_bucket, args.mode)

    if effective_mode == "box":
        crop2 = crop
    else:
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

        crop2 = apply_mode(crop, mask_crop, effective_mode, args.bg_alpha)

    if getattr(args, "clf_prescale", 0) > 0:
        ch, cw = crop2.shape[:2]
        scale = args.clf_prescale / max(ch, cw)
        if scale < 1.0:
            crop2 = cv2.resize(
                crop2,
                (max(1, int(cw * scale)), max(1, int(ch * scale))),
                interpolation=cv2.INTER_AREA,
            )

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


def yolo_predict_with_retry(
    yolo: YOLO,
    img: np.ndarray,
    imgsz: int,
    retry_imgsz: int,
    retry_buckets: set[str],
    yolo_names: list[str],
    enabled: bool = True,
) -> tuple[object, int, bool]:
    res = yolo.predict(img, imgsz=imgsz, verbose=False)[0]

    if not enabled or retry_imgsz == imgsz:
        return res, imgsz, False

    no_detection = not res.boxes or len(res.boxes) == 0
    bucket_triggers = False
    if not no_detection:
        top1_bucket = yolo_names[int(
            max(res.boxes, key=lambda bb: float(bb.conf)).cls
        )]
        bucket_triggers = top1_bucket in retry_buckets

    if no_detection or bucket_triggers:
        res_retry = yolo.predict(img, imgsz=retry_imgsz, verbose=False)[0]
        return res_retry, retry_imgsz, True

    return res, imgsz, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--sam_ckpt", required=True)
    ap.add_argument("--clf_dir", required=True)
    ap.add_argument("--out_csv", default="pipeline_results.csv")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pad", type=float, default=0.05)
    ap.add_argument("--mode", choices=["mask", "box", "hybrid"], default="hybrid")
    ap.add_argument("--bg_alpha", type=float, default=0.2)
    ap.add_argument(
        "--bucket_mode_overrides",
        nargs="+",
        default=[],
        metavar="BUCKET:MODE",
        help=(
            "Per-bucket SAM mode overrides in BUCKET:MODE format. Overrides --mode for "
            "specific buckets. MODE must be one of: mask, box, hybrid. "
            "'box' skips SAM entirely for that bucket. "
            "Example: --bucket_mode_overrides caterpillars:box borers:box stink_bugs:mask"
        ),
    )

    # ---- Resolution retry ----
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--retry_imgsz", type=int, default=1280)
    ap.add_argument("--retry_buckets", nargs="+", default=list(DEFAULT_RETRY_BUCKETS))
    ap.add_argument("--no_resolution_retry", action="store_true")

    # ---- Force tiling ----
    ap.add_argument(
        "--force_tile_buckets",
        nargs="+",
        default=[],
        metavar="BUCKET",
        help=(
            "Buckets that unconditionally trigger tiling, bypassing confidence "
            "thresholds entirely. Tiled result is used if it beats the fusion "
            "result, otherwise falls back to TOP1. "
            "Example: --force_tile_buckets tiny_pests weevils"
        ),
    )

    # ---- Bucket override (diagnostic) ----
    ap.add_argument(
        "--override_bucket",
        default=None,
        metavar="BUCKET",
        help=(
            "Force ALL detections to be routed through this bucket's classifier, "
            "ignoring YOLO's predicted bucket. Useful for diagnosing whether "
            "misrouting or the classifier itself is the problem. "
            "Example: --override_bucket tiny_pests"
        ),
    )

    ap.add_argument("--tile_size_mult", type=float, default=1.5)
    ap.add_argument("--tile_overlap", type=float, default=0.25)
    ap.add_argument("--tile_conf_thresh", type=float, default=0.50)
    ap.add_argument("--tile_joint_thresh", type=float, default=0.35)
    ap.add_argument("--tile_per_tile_topk", type=int, default=2)
    ap.add_argument("--tile_topk", type=int, default=3)

    ap.add_argument("--yolo_low_conf", type=float, default=0.4)
    ap.add_argument("--yolo_fusion_thresh", type=float, default=0.6)
    ap.add_argument("--fusion_topk", type=int, default=3)

    ap.add_argument("--low_conf_rescue", action="store_true")
    ap.add_argument("--low_conf_rescue_topk", type=int, default=3)
    ap.add_argument("--low_conf_weight_clf", type=float, default=0.7)
    ap.add_argument("--low_conf_weight_yolo", type=float, default=0.3)
    ap.add_argument("--low_conf_accept_thresh", type=float, default=0.5)
    ap.add_argument("--tiny_pest_low_conf", type=float, default=None,
                    metavar="CONF",
                    help="Override low-conf floor for tiny_pests bucket only (e.g. 0.15). "
                         "If omitted, uses --yolo_low_conf for all buckets.")

    ap.add_argument("--clf_prescale", type=int, default=448)

    ap.add_argument("--use_points", action="store_true")
    ap.add_argument("--mask_min_area", type=int, default=200)
    ap.add_argument("--min_mask_frac", type=float, default=0.005)
    ap.add_argument("--max_mask_frac", type=float, default=0.85)

    args = ap.parse_args()

    # Parse --bucket_mode_overrides ["caterpillars:box", "stink_bugs:mask"] → dict
    valid_modes = {"mask", "box", "hybrid"}
    parsed_overrides: dict[str, str] = {}
    for entry in args.bucket_mode_overrides:
        if ":" not in entry:
            raise ValueError(f"--bucket_mode_overrides entry '{entry}' must be in BUCKET:MODE format")
        bucket, mode = entry.split(":", 1)
        if mode not in valid_modes:
            raise ValueError(f"--bucket_mode_overrides mode '{mode}' must be one of {valid_modes}")
        parsed_overrides[bucket] = mode
    args.bucket_mode_overrides = parsed_overrides

    retry_buckets: set[str] = set(args.retry_buckets)
    force_tile_buckets: set[str] = set(args.force_tile_buckets)
    resolution_retry_enabled = not args.no_resolution_retry

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(
        f"Resolution retry: {'disabled' if not resolution_retry_enabled else 'enabled'} "
        f"| base={args.imgsz} → retry={args.retry_imgsz} "
        f"| trigger buckets: {sorted(retry_buckets)}"
    )
    if force_tile_buckets:
        print(f"Force-tile buckets: {sorted(force_tile_buckets)}")
    else:
        print("Force-tile buckets: none")
    if args.override_bucket:
        print(f"⚠️  Override bucket: ALL detections routed to '{args.override_bucket}' classifier")

    yolo_names = load_names_from_data_yaml(Path(args.data))
    yolo = YOLO(args.yolo)

    sam = sam_model_registry["repvit"](checkpoint=args.sam_ckpt).to(device)
    predictor = SamPredictor(sam)

    clf_bank = ClassifierBank(Path(args.clf_dir), device)

    root = Path(args.images)
    rows = []

    all_images = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )

    yolo_correct = 0
    yolo_total = 0
    final_correct = 0
    final_total = 0

    per_bucket = defaultdict(lambda: {"yolo_ok": 0, "yolo_n": 0, "final_ok": 0, "final_n": 0})
    per_species = defaultdict(lambda: {"ok": 0, "n": 0})
    wrong_preds: list[tuple[str, str, str]] = []

    retry_count = 0
    force_tile_count = 0  # track how often force-tiling fires

    for img_path in (pbar := tqdm(all_images, desc="Evaluating", unit="img", dynamic_ncols=True)):
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue

        gt_species, gt_bucket = gt_from_folder(img_path)

        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            continue

        img = img_orig
        h, w = img.shape[:2]

        _sam_set = False
        def ensure_sam_set():
            nonlocal _sam_set
            if not _sam_set:
                predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                _sam_set = True

        yolo_total += 1
        final_total += 1
        per_bucket[gt_bucket]["yolo_n"] += 1
        per_bucket[gt_bucket]["final_n"] += 1
        per_species[gt_species]["n"] += 1

        # ----------------------------------------------------------------
        # YOLO inference with optional resolution retry
        # ----------------------------------------------------------------
        res, effective_imgsz, retried = yolo_predict_with_retry(
            yolo=yolo,
            img=img,
            imgsz=args.imgsz,
            retry_imgsz=args.retry_imgsz,
            retry_buckets=retry_buckets,
            yolo_names=yolo_names,
            enabled=resolution_retry_enabled,
        )
        if retried:
            retry_count += 1
            _sam_set = False

        _orig_imgsz = args.imgsz
        args.imgsz = effective_imgsz

        if not res.boxes or len(res.boxes) == 0:
            rows.append([str(img_path), gt_bucket, gt_species, "", 0.0, "", "NO_DETECTION", 0.0, 0.0, 0, 0, "NO_DETECTION"])
            wrong_preds.append((str(img_path), gt_species, "NO_DETECTION"))
            args.imgsz = _orig_imgsz
            pbar.set_postfix({"acc": f"{100*final_correct/final_total:.1f}%", "ok": f"{final_correct}/{final_total}", "mode": "NO_DET"}, refresh=False)
            continue

        boxes_sorted = sorted(res.boxes, key=lambda bb: float(bb.conf), reverse=True)
        top1 = boxes_sorted[0]
        top1_conf = float(top1.conf)
        top1_bucket = yolo_names[int(top1.cls)]

        # --- YOLO-only bucket fast path ---
        if top1_bucket in YOLO_ONLY_BUCKETS:
            pred_bucket = top1_bucket
            yolo_conf = top1_conf
            pred_species = YOLO_ONLY_BUCKETS[top1_bucket]
            clf_conf = 0.0
            joint = yolo_conf
            decision_mode = "YOLO_ONLY"
            group = ""

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
                decision_mode
            ])
            args.imgsz = _orig_imgsz
            pbar.set_postfix({"acc": f"{100*final_correct/final_total:.1f}%", "ok": f"{final_correct}/{final_total}", "mode": decision_mode}, refresh=False)
            continue

        # --- Override bucket: force all detections through a specific classifier ---
        if args.override_bucket and args.override_bucket in COARSE_TO_GROUP:
            ensure_sam_set()
            group = COARSE_TO_GROUP[args.override_bucket]
            x1, y1, x2, y2 = map(float, top1.xyxy[0])
            px1, py1, px2, py2 = pad_xyxy(x1, y1, x2, y2, args.pad, w, h)
            ys = slice(py1, py2)
            xs = slice(px1, px2)
            crop = img[ys, xs]
            sam_box = np.array([px1, py1, px2, py2], dtype=np.float32)
            masks, scores = run_sam(predictor, sam_box, use_points=args.use_points)
            if masks is not None:
                mask = get_best_mask(masks, scores, ys, xs,
                                     mask_min_area=args.mask_min_area,
                                     min_mask_frac=args.min_mask_frac,
                                     max_mask_frac=args.max_mask_frac)
            else:
                mask = None
            mask_crop = mask[ys, xs] if mask is not None else np.ones(crop.shape[:2], dtype=np.uint8)
            crop2 = apply_mode(crop, mask_crop, args.mode, args.bg_alpha)
            if getattr(args, "clf_prescale", 0) > 0:
                ch, cw = crop2.shape[:2]
                scale = args.clf_prescale / max(ch, cw)
                if scale < 1.0:
                    crop2 = cv2.resize(crop2, (max(1, int(cw * scale)), max(1, int(ch * scale))),
                                       interpolation=cv2.INTER_AREA)
            pred_species, clf_conf = clf_bank.predict(group, crop2)
            pred_species = normalize_species(pred_species)
            pred_bucket = args.override_bucket
            yolo_conf = top1_conf
            joint = yolo_conf * clf_conf
            decision_mode = "OVERRIDE_BUCKET"

            yolo_ok = int(pred_bucket == gt_bucket)
            yolo_correct += yolo_ok
            per_bucket[gt_bucket]["yolo_ok"] += yolo_ok
            final_ok = int(pred_species == gt_species)
            final_correct += final_ok
            per_bucket[gt_bucket]["final_ok"] += final_ok
            per_species[gt_species]["ok"] += final_ok
            if not final_ok:
                wrong_preds.append((str(img_path), gt_species, pred_species))
            rows.append([str(img_path), gt_bucket, gt_species,
                         pred_bucket, yolo_conf, group, pred_species, clf_conf, joint,
                         yolo_ok, final_ok, decision_mode])
            args.imgsz = _orig_imgsz
            pbar.set_postfix({"acc": f"{100*final_correct/final_total:.1f}%",
                              "ok": f"{final_correct}/{final_total}",
                              "mode": decision_mode}, refresh=False)
            continue

        # Determine effective low-conf threshold
        effective_low_conf = args.yolo_low_conf
        if args.tiny_pest_low_conf is not None and top1_bucket == "tiny_pests":
            effective_low_conf = args.tiny_pest_low_conf

        # Low-conf handling
        if top1_conf < effective_low_conf:
            rescued = False
            if getattr(args, "low_conf_rescue", False):
                rescue_topk = max(1, args.low_conf_rescue_topk)
                candidates = boxes_sorted[:rescue_topk]
                ensure_sam_set()
                best_rescue = None
                best_rescue_score = -1.0
                for bb in candidates:
                    bb_bucket = yolo_names[int(bb.cls)]
                    if bb_bucket in YOLO_ONLY_BUCKETS:
                        continue
                    cand = evaluate_candidate(
                        img=img, w=w, h=h, box_obj=bb,
                        yolo_names=yolo_names,
                        predictor=predictor,
                        clf_bank=clf_bank,
                        args=args,
                    )
                    if cand["pred_species"] in ("NO_GROUP_MAP",):
                        continue
                    weighted = (args.low_conf_weight_yolo * cand["yolo_conf"] +
                                args.low_conf_weight_clf * cand["clf_conf"])
                    if weighted > best_rescue_score:
                        best_rescue_score = weighted
                        best_rescue = cand

                if best_rescue is not None and best_rescue_score >= args.low_conf_accept_thresh:
                    rescued = True
                    best = best_rescue
                    decision_mode = "LOW_CONF_RESCUE"

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
                        decision_mode
                    ])
                    args.imgsz = _orig_imgsz
                    pbar.set_postfix({"acc": f"{100*final_correct/final_total:.1f}%", "ok": f"{final_correct}/{final_total}", "mode": decision_mode}, refresh=False)

            if not rescued:
                pred_bucket = yolo_names[int(top1.cls)]
                rows.append([str(img_path), gt_bucket, gt_species, pred_bucket, top1_conf, "", "LOW_CONF", 0.0, 0.0, 0, 0, "LOW_CONF"])
                wrong_preds.append((str(img_path), gt_species, "LOW_CONF"))
                args.imgsz = _orig_imgsz
                pbar.set_postfix({"acc": f"{100*final_correct/final_total:.1f}%", "ok": f"{final_correct}/{final_total}", "mode": "LOW_CONF"}, refresh=False)
            continue

        orig_h, orig_w = img_orig.shape[:2]

        # ----------------------------------------------------------------
        # Tiling decision: size trigger, force-tile bucket, or conf-based
        # ----------------------------------------------------------------
        bucket_force_tile = top1_bucket in force_tile_buckets

        if image_is_large(orig_w, orig_h, args.imgsz, size_mult=args.tile_size_mult):
            # SIZE trigger — always tile regardless of bucket
            tile_best, _tile_confs = run_tiled_fusion(
                img_full=img_orig,
                yolo=yolo,
                yolo_names=yolo_names,
                predictor=predictor,
                clf_bank=clf_bank,
                args=args,
                tile_overlap=args.tile_overlap,
                per_tile_topk=args.tile_per_tile_topk,
                tile_topk=args.tile_topk,
            )
            if tile_best is not None:
                best = tile_best
                decision_mode = "TILED_SIZE"
            else:
                ensure_sam_set()
                best = evaluate_candidate(
                    img=img, w=w, h=h, box_obj=top1,
                    yolo_names=yolo_names,
                    predictor=predictor,
                    clf_bank=clf_bank,
                    args=args,
                )
                decision_mode = "TOP1_FALLBACK"

        elif bucket_force_tile:
            # FORCE-TILE trigger — bucket is in --force_tile_buckets
            force_tile_count += 1
            tile_best, _tile_confs = run_tiled_fusion(
                img_full=img_orig,
                yolo=yolo,
                yolo_names=yolo_names,
                predictor=predictor,
                clf_bank=clf_bank,
                args=args,
                tile_overlap=args.tile_overlap,
                per_tile_topk=args.tile_per_tile_topk,
                tile_topk=args.tile_topk,
            )
            if tile_best is not None:
                # Always prefer tiled result for force-tile buckets
                best = tile_best
                decision_mode = "TILED_FORCED"
            else:
                # Tiling found nothing — fall back to normal TOP1/fusion path
                ensure_sam_set()
                best = evaluate_candidate(
                    img=img, w=w, h=h, box_obj=top1,
                    yolo_names=yolo_names,
                    predictor=predictor,
                    clf_bank=clf_bank,
                    args=args,
                )
                decision_mode = "TOP1_FALLBACK"

        elif top1_conf >= args.yolo_fusion_thresh:
            ensure_sam_set()
            best = evaluate_candidate(
                img=img, w=w, h=h, box_obj=top1,
                yolo_names=yolo_names,
                predictor=predictor,
                clf_bank=clf_bank,
                args=args,
            )
            decision_mode = "TOP1"

        else:
            topk = max(1, int(args.fusion_topk))
            candidates = boxes_sorted[:topk]

            ensure_sam_set()
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

            topk_confs = [float(bb.conf) for bb in candidates]
            best_joint = float(best["joint"])

            if should_tile(
                orig_w, orig_h, args.imgsz,
                topk_confs=topk_confs,
                best_joint=best_joint,
                size_mult=args.tile_size_mult,
                conf_tile_thresh=args.tile_conf_thresh,
                joint_tile_thresh=args.tile_joint_thresh,
            ):
                tile_best, _tile_confs = run_tiled_fusion(
                    img_full=img_orig,
                    yolo=yolo,
                    yolo_names=yolo_names,
                    predictor=predictor,
                    clf_bank=clf_bank,
                    args=args,
                    tile_overlap=args.tile_overlap,
                    per_tile_topk=args.tile_per_tile_topk,
                    tile_topk=args.tile_topk,
                )
                if tile_best is not None and float(tile_best["joint"]) > float(best["joint"]):
                    best = tile_best
                    decision_mode = "TILED_FUSION"
                else:
                    decision_mode = f"FUSION_TOP{topk}"
            else:
                decision_mode = f"FUSION_TOP{topk}"

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
            decision_mode
        ])

        args.imgsz = _orig_imgsz
        pbar.set_postfix({
            "acc": f"{100*final_correct/final_total:.1f}%",
            "ok": f"{final_correct}/{final_total}",
            "mode": decision_mode,
        }, refresh=False)

    # Write CSV
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
    print(f"Resolution retries fired: {retry_count}/{final_total}")
    print(f"Force-tile fired:         {force_tile_count}/{final_total}")
    print(f"CSV saved: {out_csv}")

    print("\n=== PER-BUCKET ===")
    for bname in sorted(per_bucket.keys()):
        d = per_bucket[bname]
        print(
            f"{bname:25s} | "
            f"yolo {pct(d['yolo_ok'], d['yolo_n']):6.2f}% ({d['yolo_ok']}/{d['yolo_n']}) | "
            f"final {pct(d['final_ok'], d['final_n']):6.2f}% ({d['final_ok']}/{d['final_n']})"
        )

    print("\n=== PER-SPECIES ===")
    for sname in sorted(per_species.keys()):
        d = per_species[sname]
        bar = "█" * d["ok"] + "░" * (d["n"] - d["ok"])
        print(f"{sname:35s} | {pct(d['ok'], d['n']):6.2f}% ({d['ok']}/{d['n']})  {bar}")

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