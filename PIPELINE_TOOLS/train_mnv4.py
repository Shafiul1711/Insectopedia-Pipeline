#!/usr/bin/env python3
"""
Train a MobileNetV3 classifier using timm + torchvision ImageFolder.

Expected folder layout:
  DATA_ROOT/
    train/
      class_a/*.png|jpg
      class_b/*.png|jpg
      ...
    valid/      <-- (this script uses 'valid' to match your SAM exporter)
      class_a/*.png|jpg
      class_b/*.png|jpg
      ...

Example:
  python3 train_mnv4.py --data ClfDatasets/WeevilClf --out mnv4_weevil.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root containing train/ and valid/")
    ap.add_argument("--arch", default="mobilenetv4_conv_large", help="timm model name")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda", help="cuda or cpu (ROCm shows as cuda)")
    ap.add_argument("--out", default="mnv3_best.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    data_root = Path(args.data)
    train_dir = data_root / "train"
    val_dir = data_root / "valid"  # <-- patched: match your SAM exporter

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected train/ and valid/ under {data_root}")

    use_cuda = torch.cuda.is_available() and args.device == "cuda"
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device:", device)

    # Augmentations: safe defaults for SAM crops / object crops
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.imgsz, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.imgsz * 1.14)),
        transforms.CenterCrop(args.imgsz),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds = datasets.ImageFolder(val_dir, transform=val_tf)

    # ---- patched: warn if class folders differ between train and valid ----
    train_set = set(train_ds.classes)
    val_set = set(val_ds.classes)
    if train_set != val_set:
        print("[WARN] Train/valid class folders differ!")
        print("  Only in train:", sorted(train_set - val_set))
        print("  Only in valid:", sorted(val_set - train_set))
        print("This can skew validation or hide missing classes.")

    num_classes = len(train_ds.classes)
    if num_classes < 2:
        raise ValueError(f"Need >= 2 classes for training. Found {num_classes}: {train_ds.classes}")

    print("Num classes:", num_classes)
    print("Classes:", train_ds.classes)

    train_dl = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=use_cuda
    )
    val_dl = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=use_cuda
    )

    # Model
    model = timm.create_model(args.arch, pretrained=True, num_classes=num_classes)
    model.to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    amp_enabled = (device.type == "cuda") and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        tr_loss_sum = 0.0
        tr_acc_sum = 0.0
        tr_n = 0

        pbar = tqdm(train_dl, desc=f"Train {epoch:02d}/{args.epochs}", leave=False)
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(xb)
                loss = crit(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            bs = xb.size(0)
            batch_acc = accuracy(logits.detach(), yb)

            tr_loss_sum += loss.item() * bs
            tr_acc_sum += batch_acc * bs
            tr_n += bs

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}", lr=f"{opt.param_groups[0]['lr']:.2e}")

        sched.step()

        tr_loss = tr_loss_sum / max(1, tr_n)
        tr_acc = tr_acc_sum / max(1, tr_n)

        # ---- val ----
        model.eval()
        va_loss_sum = 0.0
        va_acc_sum = 0.0
        va_n = 0

        vbar = tqdm(val_dl, desc=f"Val   {epoch:02d}/{args.epochs}", leave=False)
        with torch.no_grad():
            for xb, yb in vbar:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = model(xb)
                loss = crit(logits, yb)

                bs = xb.size(0)
                batch_acc = accuracy(logits, yb)

                va_loss_sum += loss.item() * bs
                va_acc_sum += batch_acc * bs
                va_n += bs

                vbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.3f}")

        va_loss = va_loss_sum / max(1, va_n)
        va_acc = va_acc_sum / max(1, va_n)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
              f"time {elapsed/60:.1f} min")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt = {
                "arch": args.arch,
                "imgsz": args.imgsz,
                "classes": train_ds.classes,
                "state_dict": model.state_dict(),
            }
            torch.save(ckpt, args.out)
            print(f"  ✔ saved best: {args.out} (val_acc={best_val_acc:.3f})")

    print("Done. Best val acc:", best_val_acc)


if __name__ == "__main__":
    main()