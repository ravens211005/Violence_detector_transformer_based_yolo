"""
train.py  —  Violence Detection Training
=========================================
MODIFICATIONS (vs original):
  [REQ-1] Uses build_dataloaders() from dataset.py which loads per-video .npy files.
           No monolithic X.npy / y.npy is read.
  [REQ-2] Learning rate is FIXED for all epochs.
           - ReduceLROnPlateau, CosineAnnealing, OneCycleLR, StepLR,
             ExponentialLR, and any warmup scheduler are REMOVED.
           - Optimizer: torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
           - No scheduler.step() calls anywhere.
           - Training logs report constant LR value every epoch.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import build_dataloaders
from transformer_model import ViolenceTransformer

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Training loop (one epoch)
# ══════════════════════════════════════════════════════════════════════════════
def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss   = 0.0
    correct      = 0
    total        = 0

    for x, y in tqdm(loader, desc="  train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss   = criterion(logits, y)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        # [REQ-2] No scheduler.step() here — LR remains constant

        total_loss += loss.item() * x.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════════════
# Validation loop
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on a DataLoader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for x, y in tqdm(loader, desc="  val  ", leave=False):
        x, y   = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════════════════════════════
# Main training entry point
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    # ── Device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log.info(f"Training on device: {device}")

    # ── Directories ───────────────────────────────────────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    log_dir  = Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── DataLoaders [REQ-1] ───────────────────────────────────────────────────
    log.info(f"Loading per-video .npy files from: {args.processed_path}")
    train_loader, val_loader, _, feature_dim = build_dataloaders(
        processed_path=args.processed_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=0.15,
        test_ratio=0.15,
    )
    log.info(f"Feature dim detected: {feature_dim}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ViolenceTransformer(
        feature_dim=feature_dim,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=2,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model parameters: {total_params:,}")

    # ── Optimizer [REQ-2] ─────────────────────────────────────────────────────
    # Fixed learning rate — no scheduler of any kind is created or stepped.
    LEARNING_RATE = args.learning_rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,          # [REQ-2] constant LR, never modified
    )
    log.info(f"Optimizer: AdamW  |  Fixed LR: {LEARNING_RATE}")
    # [REQ-2] REMOVED: ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR,
    #                   StepLR, ExponentialLR, warmup schedulers — all gone.

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Resume from checkpoint if requested ───────────────────────────────────
    start_epoch  = 0
    best_val_acc = 0.0
    history: list[dict] = []

    if args.resume and Path(args.resume).exists():
        log.info(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        history      = ckpt.get("history", [])
        log.info(f"Resumed at epoch {start_epoch}, best val acc so far: {best_val_acc:.4f}")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(log_dir))

    # ── Training loop ─────────────────────────────────────────────────────────
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        epoch_elapsed = time.time() - epoch_start

        # [REQ-2] LR is always LEARNING_RATE — log the constant value
        current_lr = optimizer.param_groups[0]["lr"]  # always == LEARNING_RATE
        assert current_lr == LEARNING_RATE, \
            "LR changed unexpectedly — check for accidental scheduler usage"

        log.info(
            f"Epoch {epoch+1:03d}/{args.epochs}  |  "
            f"LR={current_lr:.2e}  |  "       # [REQ-2] constant LR reported
            f"Train loss={train_loss:.4f}  acc={train_acc:.4f}  |  "
            f"Val loss={val_loss:.4f}  acc={val_acc:.4f}  |  "
            f"({epoch_elapsed:.1f}s)"
        )

        # TensorBoard logging
        writer.add_scalar("Loss/train",  train_loss, epoch)
        writer.add_scalar("Loss/val",    val_loss,   epoch)
        writer.add_scalar("Acc/train",   train_acc,  epoch)
        writer.add_scalar("Acc/val",     val_acc,    epoch)
        writer.add_scalar("LR",          current_lr, epoch)  # flat line in TB

        history.append({
            "epoch":      epoch + 1,
            "train_loss": train_loss,
            "train_acc":  train_acc,
            "val_loss":   val_loss,
            "val_acc":    val_acc,
            "lr":         current_lr,
        })

        # ── Checkpoint (best model) ───────────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            save_path = ckpt_dir / "best_model.pth"
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc":        best_val_acc,
                    "feature_dim":         feature_dim,
                    "model_config": {
                        "feature_dim":   feature_dim,
                        "embedding_dim": args.embedding_dim,
                        "num_heads":     args.num_heads,
                        "num_layers":    args.num_layers,
                        "dropout":       args.dropout,
                    },
                    "history": history,
                },
                str(save_path),
            )
            log.info(f"  ✅ New best val acc {best_val_acc:.4f} — saved {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log.info(f"Early stopping triggered after {args.patience} epochs without improvement.")
                break

    writer.close()

    # ── Save training history ──────────────────────────────────────────────────
    history_path = log_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    log.info(f"Training history saved to {history_path}")
    log.info(f"Best validation accuracy: {best_val_acc:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Violence Detection — Training")

    # Paths
    parser.add_argument("--processed_path",  default="./processed",   help="Path to processed .npy files")
    parser.add_argument("--checkpoint_dir",  default="./checkpoints", help="Checkpoint output directory")
    parser.add_argument("--log_dir",         default="./logs",        help="TensorBoard log directory")
    parser.add_argument("--resume",          default=None,            help="Resume from checkpoint path")

    # Training hyperparameters
    parser.add_argument("--epochs",          type=int,   default=50,   help="Number of training epochs")
    parser.add_argument("--batch_size",      type=int,   default=64,   help="Batch size")
    parser.add_argument("--learning_rate",   type=float, default=1e-4, help="Fixed learning rate [REQ-2]")
    parser.add_argument("--patience",        type=int,   default=10,   help="Early stopping patience")
    parser.add_argument("--num_workers",     type=int,   default=2,    help="DataLoader workers")
    parser.add_argument("--device",          default="auto",           help="Device: auto | cpu | cuda")

    # Model architecture
    parser.add_argument("--embedding_dim",   type=int,   default=256,  help="Transformer embedding dimension")
    parser.add_argument("--num_heads",       type=int,   default=8,    help="Attention heads")
    parser.add_argument("--num_layers",      type=int,   default=4,    help="Transformer encoder layers")
    parser.add_argument("--dropout",         type=float, default=0.3,  help="Dropout rate")

    args = parser.parse_args()
    main(args)
