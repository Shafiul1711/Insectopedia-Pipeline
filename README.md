# Insectopedia Pipeline

Training pipeline, inference scripts, dataset utilities, and model weights for the Insectopedia agricultural pest identification system.

The mobile app and annotated dataset are maintained in separate repositories.

---

## Contents

- [Repository Structure](#repository-structure)
- [Models](#models)
- [Scripts](#scripts)
- [Pipeline Overview](#pipeline-overview)
- [Related Components](#related-components)

---

## Repository Structure

```
Insectopedia-Pipeline/
│
├── models/
│   ├── detector/
│   │   └── YOLO26_Insectopedia_V5.pt
│   └── classifier/
│       ├── rn18_tiny_pests.pt
│       ├── rn18_flea_beetle.pt
│       ├── rn18_caterpillars.pt
│       ├── rn18_plant_bugs.pt
│       ├── rn18_soil_larvae.pt
│       ├── rn18_weevils.pt
│       ├── rn18_stink_bugs.pt
│       ├── rn18_blister_beetle.pt
│       └── rn18_potato_beetle.pt
│
├── eval_pipeline.py
├── train_yolo.py
├── train_mnv4.py
├── sam_export_specialized.py
├── sam_export_specialized_mobilesam.py
├── sam_crop_filter.py
├── tile_yolo_dataset.py
├── INaturalistDownloader.py
├── dedupe_images.py
├── remapper.py
├── renamer.py
├── Bounding_Box_Visualizer.py
├── ImageCounter.py
├── livecounter.py
└── old_eval_pipeline.py
```

---

## Models

### Detector

| File | Description |
|---|---|
| `YOLO26_Insectopedia_V5.pt` | YOLO26 coarse bucket detector, 9 classes |

### Classifiers

One ResNet-18 classifier per detection bucket. Each model takes a segmentation crop as input and returns a species-level prediction.

| File | Bucket | Species |
|---|---|---|
| `rn18_tiny_pests.pt` | tiny_pests | aphids, thrips, spider_mite |
| `rn18_flea_beetle.pt` | flea_beetle | flea_beetle, grape_flea_beetle, striped_flea_beetle |
| `rn18_caterpillars.pt` | caterpillars | army_worm, black_cutworm, corn_borer |
| `rn18_plant_bugs.pt` | plant_bugs | miridae, tarnished_plant_bug, four_lined_plant_bug |
| `rn18_soil_larvae.pt` | soil_larvae | grub, wireworm |
| `rn18_weevils.pt` | weevils | alfalfa_weevil, strawberry_root_weevil |
| `rn18_stink_bugs.pt` | stink_bugs | green_stink_bug, brown_marmorated_stink_bug |
| `rn18_blister_beetle.pt` | blister_beetle | blister_beetle, black_blister_beetle, striped_blister_beetle |
| `rn18_potato_beetle.pt` | potato_beetle | colorado_potato_beetle, striped_cucumber_beetle |

---

## Scripts

### Training

| Script | Description |
|---|---|
| `train_yolo.py` | Trains the YOLO26 bucket detector |
| `train_mnv4.py` | Trains per-bucket ResNet-18 classifiers |

### Inference and Evaluation

| Script | Description |
|---|---|
| `eval_pipeline.py` | Runs end-to-end pipeline evaluation on the held-out test suite |
| `old_eval_pipeline.py` | Legacy evaluation script, retained for reference |

### SAM Export

| Script | Description |
|---|---|
| `sam_export_specialized.py` | Exports RepViT-SAM encoder and decoder to ONNX |
| `sam_export_specialized_mobilesam.py` | Exports MobileSAM variant to ONNX |
| `sam_crop_filter.py` | Filters segmentation crops by quality before classifier training |

### Dataset Utilities

| Script | Description |
|---|---|
| `INaturalistDownloader.py` | Downloads images from iNaturalist by species and observation filters |
| `dedupe_images.py` | Removes duplicate images across dataset splits |
| `remapper.py` | Remaps class IDs across annotation files |
| `renamer.py` | Batch renames image and annotation files |
| `tile_yolo_dataset.py` | Tiles large images into YOLO-compatible patches |
| `Bounding_Box_Visualizer.py` | Renders bounding box annotations over images for inspection |
| `ImageCounter.py` | Reports per-class image counts across dataset splits |
| `livecounter.py` | Live dataset count monitor during annotation or download |

---

## Pipeline Overview

```
Image → YOLO26 Detection → RepViT-SAM Segmentation → ResNet-18 Classification
```

The detector predicts one of 9 coarse buckets. The segmentation model isolates the pest region. The per-bucket classifier returns a species-level prediction. The full pipeline runs on-device without a network call.

See [Insectopedia Dataset](https://github.com/Shafiul1711/Insectopedia-Dataset) for dataset statistics, species classes, and benchmark results.

---

## Related Components

| Component | Description |
|---|---|
| **Insectopedia Dataset** | Annotated image dataset for training and evaluation |
| **Insectopedia Pipeline** (this repo) | Training scripts, inference tools, and model weights |
| **Insectopedia App** | Flutter mobile app with on-device inference and HITL correction workflow |

---

Dataset and models produced as part of a computer vision capstone project at the University of Windsor in collaboration with Local Greenhous (2026).
