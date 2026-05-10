"""
Training script for Amyloid PET Centiloid Prediction.

Usage:
    python train.py --train_csv /projectnb/medaihack/ABPET/data/train.csv --val_csv /projectnb/medaihack/ABPET/data/val.csv --patience 10
"""

import argparse
import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import PETDataset
from model import BaselineCNN
from losses import get_criterion

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logger(log_dir: Path, results_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_{timestamp}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger, log_dir, results_dir, timestamp


# ---------------------------------------------------------------------------
# Training & Validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    n = 0

    for images, centiloids, tracers in loader:
        images = images.to(device, non_blocking=True)
        centiloids = centiloids.to(device, non_blocking=True)
        tracers = tracers.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type):
            preds = model(images, tracers)
            loss = criterion(preds, centiloids)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        n += images.size(0)

    return total_loss / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds, all_targets, all_tracers = [], [], []

    for images, centiloids, tracers in loader:
        images = images.to(device, non_blocking=True)
        tracers = tracers.to(device, non_blocking=True)

        with torch.amp.autocast(device.type):
            preds = model(images, tracers)

        all_preds.append(preds.cpu())
        all_targets.append(centiloids)
        all_tracers.append(tracers.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    tracer_ids = torch.cat(all_tracers)
    mae = (preds - targets).abs().mean().item()
    corr = torch.corrcoef(torch.stack([preds, targets]))[0, 1].item()
    return mae, corr, preds, targets, tracer_ids


def save_val_report(preds, targets, tracer_ids, tracer_map, results_dir, timestamp):
    """Write overall + per-tracer MAE and Pearson r to a CSV."""
    id_to_name = {v: k for k, v in tracer_map.items()}

    rows = []

    def metrics(p, t):
        mae = (p - t).abs().mean().item()
        corr = torch.corrcoef(torch.stack([p, t]))[0, 1].item() if len(p) > 1 else float("nan")
        return mae, corr, len(p)

    mae, corr, n = metrics(preds, targets)
    rows.append({"tracer": "ALL", "n": n, "mae": mae, "pearson_r": corr})

    for tid in sorted(tracer_ids.unique().tolist()):
        mask = tracer_ids == tid
        mae, corr, n = metrics(preds[mask], targets[mask])
        rows.append({"tracer": id_to_name[tid], "n": n, "mae": mae, "pearson_r": corr})

    report_path = results_dir / f"val_report_{timestamp}.csv"
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tracer", "n", "mae", "pearson_r"])
        writer.writeheader()
        writer.writerows(rows)

    return report_path


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plots(history, results_dir, timestamp):
    if not HAS_MPL:
        return

    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, [h["train_loss"] for h in history], label="Train Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].plot(epochs, [h["val_mae"] for h in history], label="Val MAE", color="tab:orange")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (CL)")
    axes[1].set_title("Validation MAE")
    axes[1].legend()

    axes[2].plot(epochs, [h["val_corr"] for h in history], label="Val r", color="tab:green")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Pearson r")
    axes[2].set_title("Validation Correlation")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(results_dir / f"curves_{timestamp}.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--cache", action="store_true",
                        help="Cache all volumes in RAM (~16GB for 2000 samples)")
    parser.add_argument("--loss", type=str, default="mae",
                        choices=["mse", "mae"],
                        help="Loss function to use")
    parser.add_argument("--patience", type=int, default=7,
                        help="Early stopping patience (0 to disable)")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for training log files")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="Directory for metrics CSV and plots")
    args = parser.parse_args()

    # --- Directories ---
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    # --- Logging ---
    logger, log_dir, results_dir, timestamp = setup_logger(Path(args.log_dir), Path(args.results_dir))
    logger.info(f"Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Fixed input size -> let cudnn auto-tune convolution algorithms
    torch.backends.cudnn.benchmark = True

    # --- Data ---
    train_ds = PETDataset(args.train_csv, cache=args.cache)
    val_ds = PETDataset(args.val_csv, tracer_map=train_ds.tracer_map, cache=args.cache)

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    # --- Model ---
    mean_cl = train_ds.centiloids.mean()
    model = BaselineCNN(
        num_tracers=len(train_ds.tracer_map),
        mean_centiloid=float(mean_cl),
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,}")

    # torch.compile for fused kernels (PyTorch 2.0+, no-op if unavailable)
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # --- Training setup ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = get_criterion(args.loss)
    scaler = torch.amp.GradScaler(device.type)

    # --- Metrics CSV ---
    metrics_path = results_dir / f"metrics_{timestamp}.csv"
    metrics_file = open(metrics_path, "w", newline="")
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(["epoch", "train_loss", "val_mae", "val_corr", "lr", "epoch_time_s", "is_best"])


    best_mae = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_mae, val_corr, val_preds, val_targets, val_tracer_ids = validate(model, val_loader, device)
        epoch_time = time.time() - t0

        is_best = val_mae < best_mae
        tag = ""
        if is_best:
            best_mae = val_mae
            best_preds, best_targets, best_tracer_ids = val_preds, val_targets, val_tracer_ids
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "tracer_map": train_ds.tracer_map,
                "num_tracers": len(train_ds.tracer_map),
            }, checkpoint_path)
            tag = " *"
        else:
            epochs_without_improvement += 1

        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"MAE: {val_mae:.2f} CL | "
            f"r: {val_corr:.4f} | "
            f"LR: {lr:.2e} | "
            f"{epoch_time:.1f}s{tag}"
        )

        metrics_writer.writerow([epoch, f"{train_loss:.6f}", f"{val_mae:.4f}",
                                 f"{val_corr:.6f}", f"{lr:.2e}", f"{epoch_time:.1f}",
                                 int(is_best)])
        metrics_file.flush()

        history.append(dict(epoch=epoch, train_loss=train_loss,
                            val_mae=val_mae, val_corr=val_corr))

        if args.patience > 0 and epochs_without_improvement >= args.patience:
            logger.info(f"Early stopping: no improvement for {args.patience} epochs")
            break

    metrics_file.close()

    # --- Plots ---
    save_plots(history, results_dir, timestamp)

    # --- Validation report for best checkpoint ---
    report_path = save_val_report(best_preds, best_targets, best_tracer_ids,
                                  train_ds.tracer_map, results_dir, timestamp)

    logger.info(f"Best validation MAE: {best_mae:.2f} centiloid units")
    logger.info(f"Model saved to {checkpoint_path}")
    logger.info(f"Metrics: {metrics_path}")
    logger.info(f"Val report: {report_path}")
    if HAS_MPL:
        logger.info(f"Plots: {results_dir / f'curves_{timestamp}.png'}")


if __name__ == "__main__":
    main()
