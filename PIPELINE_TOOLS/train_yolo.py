#!/usr/bin/env python3
"""
Train a YOLO detector with Ultralytics (ROCm-compatible via torch "cuda").

Config (matches what we discussed):
- yolo11s
- 120 epochs
- imgsz 640 hopefully 640
- batch 24
- patience 30
- workers 8
- AMP on
- close_mosaic 10 (turn mosaic off near the end)

Run:
 python3 /home/silvermoon/Music/GrowLiv/PIPELINE_TOOLS/train_yolo.py
"""

from ultralytics import YOLO
import torch


def main():
    # ---- Sanity check: ROCm shows up as "cuda" in PyTorch ----
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        try:
            print("device 0:", torch.cuda.get_device_name(0))
        except Exception as e:
            print("device name lookup failed:", e)
    else:
        print("WARNING: No GPU detected by torch; this will train on CPU.")

    # ---- Train ----
    model = YOLO("yolo26s.pt")  # or a local path to your weights

    results = model.train(
    data="/home/silvermoon/Music/GrowLiv/Dataset/YOLO/data.yaml",
    epochs=120,
    imgsz=896,
    batch=12,
    patience=30,
    device=0,
    workers=8,
    cache="ram",     # now worth trying with 48GB
    amp=True,
    close_mosaic=10,
    project="GrowLivRuns",
    name="yolo26s_Run_Feb_17",
    hsv_h=0.015,      # small hue shift
    hsv_s=0.6,        # moderate saturation
    hsv_v=0.4,        # moderate brightness
    degrees=0.0,      # avoid rotations (keeps insects realistic)
    translate=0.10,   # small shifts
    scale=0.50,       # moderate zoom in/out
    shear=0.0,        # avoid shear artifacts
    perspective=0.0005,# tiny perspective (subtle realism)
    fliplr=0.5,       # left/right flip is usually fine
    flipud=0.0,       # avoid upside-down insects/leaves
    mosaic=1.0,       # keep mosaic early
    mixup=0.0,        # keep off for detection (often not worth it)
)


    print("Training complete.")
    print(results)


if __name__ == "__main__":
    main()
