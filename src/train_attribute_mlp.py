"""
COMPOSE - Attribute MLP Training Script
Trains the 3-head attribute classifier (shape / colour / size)
on synthetic data generated from PyBullet scene renders.

Usage:
    python train_attribute_mlp.py
    python train_attribute_mlp.py --samples 1200 --epochs 50

Outputs:
    mlp_weights.pth  — load with Perceptor(mlp_weights="mlp_weights.pth")
    training_log.csv — loss curve for reproducibility
"""

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np

from perception import AttributeMLP, generate_synthetic_data


def train(n_samples: int = 800, epochs: int = 40,
          save_path: str = "mlp_weights.pth",
          log_path: str = "training_log.csv"):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    print(f"Samples: {n_samples}  Epochs: {epochs}")
    print("-" * 50)

    # Data
    X, ys, yc, yz = generate_synthetic_data(n_samples)

    split = int(0.8 * n_samples)
    X_train, X_val = X[:split], X[split:]
    ys_train, ys_val = ys[:split], ys[split:]
    yc_train, yc_val = yc[:split], yc[split:]
    yz_train, yz_val = yz[:split], yz[split:]

    to_t = lambda arr, dtype=torch.float32: torch.tensor(arr, dtype=dtype).to(device)

    X_train_t  = to_t(X_train)
    X_val_t    = to_t(X_val)
    ys_train_t = to_t(ys_train, torch.long)
    yc_train_t = to_t(yc_train, torch.long)
    yz_train_t = to_t(yz_train, torch.long)
    ys_val_t   = to_t(ys_val,   torch.long)
    yc_val_t   = to_t(yc_val,   torch.long)
    yz_val_t   = to_t(yz_val,   torch.long)

    model   = AttributeMLP().to(device)
    opt     = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    log_rows = []
    best_val = float("inf")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        opt.zero_grad()
        ls, lc, lz = model(X_train_t)
        loss = loss_fn(ls, ys_train_t) + loss_fn(lc, yc_train_t) + loss_fn(lz, yz_train_t)
        loss.backward()
        opt.step()
        sched.step()

        # Validate
        model.eval()
        with torch.no_grad():
            vls, vlc, vlz = model(X_val_t)
            val_loss = (loss_fn(vls, ys_val_t) +
                        loss_fn(vlc, yc_val_t) +
                        loss_fn(vlz, yz_val_t)).item()

            # Per-head accuracy
            shape_acc  = (vls.argmax(1) == ys_val_t).float().mean().item()
            colour_acc = (vlc.argmax(1) == yc_val_t).float().mean().item()
            size_acc   = (vlz.argmax(1) == yz_val_t).float().mean().item()

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(loss.item(), 4),
            "val_loss":   round(val_loss, 4),
            "shape_acc":  round(shape_acc, 3),
            "colour_acc": round(colour_acc, 3),
            "size_acc":   round(size_acc, 3),
        })

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train={loss.item():.4f}  val={val_loss:.4f}  "
                  f"shape={shape_acc:.2%}  colour={colour_acc:.2%}  "
                  f"size={size_acc:.2%}  ({elapsed:.1f}s)")

    # Save log
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)

    print("-" * 50)
    print(f"Best val loss : {best_val:.4f}")
    print(f"Weights saved : {save_path}")
    print(f"Training log  : {log_path}")
    print(f"Total time    : {time.time() - t0:.1f}s")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train COMPOSE attribute MLP")
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--epochs",  type=int, default=40)
    parser.add_argument("--save",    type=str, default="mlp_weights.pth")
    args = parser.parse_args()

    train(n_samples=args.samples, epochs=args.epochs, save_path=args.save)
