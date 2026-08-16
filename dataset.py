"""
dataset.py  —  Violence Detection Dataset
==========================================
MODIFICATIONS (vs original):
  [REQ-1] Dataset now loads individual per-video .npy files dynamically
           instead of reading a monolithic X.npy / y.npy.
  [REQ-1] Labels are inferred from folder structure (violence/ → 1, non_violence/ → 0).
  Preserves train/validation/test split logic via sklearn stratified split.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset class — loads individual per-video .npy files
# ══════════════════════════════════════════════════════════════════════════════
class PoseSequenceDataset(Dataset):
    """
    [REQ-1] Each item corresponds to exactly one video's .npy feature file.

    Labels are inferred from the parent directory name:
      processed/violence/     → label 1
      processed/non_violence/ → label 0

    No monolithic X.npy / y.npy is used or expected.
    """

    def __init__(self, file_paths: list[Path], labels: list[int]):
        """
        Args:
            file_paths: list of Path objects to individual .npy files
            labels:     matching list of integer labels (0 or 1)
        """
        assert len(file_paths) == len(labels), \
            "file_paths and labels must have the same length"
        self.file_paths = file_paths
        self.labels     = labels

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        """
        [REQ-1] Load exactly ONE .npy file on demand during training.
        Returns (tensor of shape (seq_len, feature_dim), label_tensor).
        """
        npy_path = self.file_paths[idx]
        label    = self.labels[idx]

        # Dynamically load the per-video feature array
        seq = np.load(str(npy_path)).astype(np.float32)  # (seq_len, feature_dim)

        x = torch.from_numpy(seq)            # → (seq_len, feature_dim)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ══════════════════════════════════════════════════════════════════════════════
# Helper: discover all per-video .npy files and infer labels from folder names
# ══════════════════════════════════════════════════════════════════════════════
def discover_samples(processed_path: Path) -> tuple[list[Path], list[int]]:
    """
    [REQ-1] Walk processed/violence/ and processed/non_violence/ and collect
    all .npy files.  Labels are inferred from the parent folder name — no
    separate y.npy is needed.

    Returns:
        (file_paths, labels) — parallel lists of Path and int.
    """
    label_map = {"violence": 1, "non_violence": 0}
    file_paths: list[Path] = []
    labels:     list[int]  = []

    for cat_name, label in label_map.items():
        cat_dir = processed_path / cat_name
        if not cat_dir.exists():
            log.warning(f"Category directory not found: {cat_dir}")
            continue
        npy_files = sorted(cat_dir.glob("*.npy"))
        file_paths.extend(npy_files)
        labels.extend([label] * len(npy_files))
        log.info(f"  [{cat_name}] found {len(npy_files)} .npy files  (label={label})")

    log.info(f"Total samples discovered: {len(file_paths)}")
    return file_paths, labels


# ══════════════════════════════════════════════════════════════════════════════
# Public API: build train / val / test DataLoaders
# ══════════════════════════════════════════════════════════════════════════════
def build_dataloaders(
    processed_path: str | Path,
    batch_size:     int   = 32,
    num_workers:    int   = 2,
    val_ratio:      float = 0.15,
    test_ratio:     float = 0.15,
    random_state:   int   = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    [REQ-1] Scans processed/ for per-video .npy files, performs a stratified
    train/val/test split, and returns three DataLoaders.

    Preserves the original split ratios and stratified-split logic.

    Returns:
        (train_loader, val_loader, test_loader, feature_dim)
    """
    processed_path = Path(processed_path)

    # ── Discover per-video .npy files ────────────────────────────────────────
    file_paths, labels = discover_samples(processed_path)

    if len(file_paths) == 0:
        raise RuntimeError(
            f"No .npy files found under {processed_path}/violence/ or "
            f"{processed_path}/non_violence/.  Run preprocess.py first."
        )

    # ── Infer feature_dim from first file ────────────────────────────────────
    sample_seq   = np.load(str(file_paths[0]))
    feature_dim  = sample_seq.shape[-1]
    sequence_len = sample_seq.shape[0]
    log.info(f"Feature shape per video: seq_len={sequence_len}, feature_dim={feature_dim}")

    # ── Stratified train / val+test split ────────────────────────────────────
    fp_array = np.array(file_paths, dtype=object)
    lb_array = np.array(labels, dtype=int)

    fp_train, fp_tmp, lb_train, lb_tmp = train_test_split(
        fp_array, lb_array,
        test_size=val_ratio + test_ratio,
        stratify=lb_array,
        random_state=random_state,
    )

    relative_test = test_ratio / (val_ratio + test_ratio)
    fp_val, fp_test, lb_val, lb_test = train_test_split(
        fp_tmp, lb_tmp,
        test_size=relative_test,
        stratify=lb_tmp,
        random_state=random_state,
    )

    log.info(f"Split — train: {len(fp_train)}, val: {len(fp_val)}, test: {len(fp_test)}")

    # ── Build Dataset objects ─────────────────────────────────────────────────
    train_ds = PoseSequenceDataset(list(fp_train), list(lb_train))
    val_ds   = PoseSequenceDataset(list(fp_val),   list(lb_val))
    test_ds  = PoseSequenceDataset(list(fp_test),  list(lb_test))

    # ── Build DataLoaders ─────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, feature_dim


# ══════════════════════════════════════════════════════════════════════════════
# Backward-compat shim: load test split alone (used by evaluate.py)
# ══════════════════════════════════════════════════════════════════════════════
def build_test_loader(
    processed_path: str | Path,
    batch_size:  int = 32,
    num_workers: int = 2,
    random_state: int = 42,
    val_ratio:  float = 0.15,
    test_ratio: float = 0.15,
) -> tuple[DataLoader, int]:
    """Return (test_loader, feature_dim) using the same split as training."""
    _, _, test_loader, feature_dim = build_dataloaders(
        processed_path=processed_path,
        batch_size=batch_size,
        num_workers=num_workers,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state,
    )
    return test_loader, feature_dim
