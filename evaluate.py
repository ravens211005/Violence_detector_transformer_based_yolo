"""
evaluate.py  —  Violence Detection Evaluation
===============================================
MODIFICATIONS (vs original):
  [REQ-1] Uses build_test_loader() from dataset.py, which discovers per-video
           .npy files and infers labels from folder structure.
           No monolithic X.npy / y.npy is read.
"""

import argparse
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from tqdm import tqdm

from dataset import build_test_loader
from transformer_model import ViolenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(model, loader, device, threshold=0.5):
    model.eval()
    all_labels = []
    all_probs  = []

    for x, y in tqdm(loader, desc="Evaluating"):
        x       = x.to(device)
        logits  = model(x)
        probs   = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # P(violence)
        all_probs.extend(probs.tolist())
        all_labels.extend(y.numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= threshold).astype(int)
    return all_labels, all_probs, all_preds


def plot_confusion_matrix(cm, output_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Violence", "Violence"],
        yticklabels=["Non-Violence", "Violence"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path}")


def plot_roc_curve(labels, probs, auc, output_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path}")


def plot_pr_curve(labels, probs, ap, output_dir):
    precision, recall, _ = precision_recall_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(output_dir, "precision_recall_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log.info(f"Saved: {path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    log.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt["model_config"]

    # ── Build model ───────────────────────────────────────────────────────────
    model = ViolenceTransformer(
        feature_dim=cfg["feature_dim"],
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        num_classes=2,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    log.info("Model loaded")

    # ── Build test DataLoader [REQ-1] ─────────────────────────────────────────
    test_loader, feature_dim = build_test_loader(
        processed_path=args.processed_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    assert feature_dim == cfg["feature_dim"], \
        f"Feature dim mismatch: data={feature_dim}, model={cfg['feature_dim']}"

    # ── Inference ─────────────────────────────────────────────────────────────
    labels, probs, preds = run_inference(model, test_loader, device, args.threshold)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    auc  = roc_auc_score(labels, probs)
    ap   = average_precision_score(labels, probs)
    cm   = confusion_matrix(labels, preds)

    log.info("=== TEST SET RESULTS ===")
    log.info(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    log.info(f"  Precision : {prec:.4f}")
    log.info(f"  Recall    : {rec:.4f}")
    log.info(f"  F1 Score  : {f1:.4f}")
    log.info(f"  ROC-AUC   : {auc:.4f}")
    log.info(f"  Avg Prec  : {ap:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(cm, str(output_dir))
    plot_roc_curve(labels, probs, auc, str(output_dir))
    plot_pr_curve(labels, probs, ap, str(output_dir))

    # ── Save JSON results ─────────────────────────────────────────────────────
    results = {
        "metrics": {
            "accuracy":  acc,
            "precision": prec,
            "recall":    rec,
            "f1":        f1,
            "roc_auc":   auc,
            "avg_precision": ap,
        },
        "confusion_matrix": cm.tolist(),
        "threshold": args.threshold,
    }
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Violence Detection — Evaluation")
    parser.add_argument("--checkpoint",      required=True,           help="Path to best_model.pth")
    parser.add_argument("--processed_path",  default="./processed",   help="Path to processed .npy files")
    parser.add_argument("--output_dir",      default="./outputs",     help="Output directory for plots/JSON")
    parser.add_argument("--threshold",       type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--num_workers",     type=int,   default=2)
    args = parser.parse_args()
    main(args)
