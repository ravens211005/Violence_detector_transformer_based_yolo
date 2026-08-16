"""
=============================================================
utils/visualization.py
=============================================================
Two types of visualization:

1. REAL-TIME INFERENCE:
   - draw_pose_skeleton()  — color-coded MediaPipe skeleton overlay
   - draw_person_box()     — bounding box + ID + violence score
   - draw_fps_counter()    — FPS in top-left corner
   - draw_info_panel()     — person count + alert status
   - FPSCounter class      — rolling FPS calculation

2. EVALUATION PLOTS (auto-saved to outputs/):
   - plot_confusion_matrix()
   - plot_roc_curve()
   - plot_precision_recall_curve()
   - plot_training_history()
=============================================================
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time

# ── BGR color constants for OpenCV drawing ────────────────────
COLOR_VIOLENCE     = (0, 0, 255)      # Red   — violent person
COLOR_NON_VIOLENCE = (0, 255, 0)      # Green — non-violent person
COLOR_UNKNOWN      = (0, 165, 255)    # Orange — buffering (not enough frames yet)
COLOR_TEXT_BG      = (0, 0, 0)        # Black background behind text
COLOR_WHITE        = (255, 255, 255)

# ── MediaPipe Pose skeleton connections ───────────────────────
# Each tuple (a, b) means "draw a line from landmark a to landmark b"
MP_POSE_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,7),           # nose → left eye
    (0,4),(4,5),(5,6),(6,8),           # nose → right eye
    (9,10),                             # mouth
    (11,12),                            # shoulders
    (11,13),(13,15),                    # left arm
    (12,14),(14,16),                    # right arm
    (15,17),(15,19),(15,21),(17,19),    # left hand
    (16,18),(16,20),(16,22),(18,20),    # right hand
    (11,23),(12,24),(23,24),            # torso
    (23,25),(25,27),(27,29),(27,31),(29,31),  # left leg
    (24,26),(26,28),(28,30),(28,32),(30,32),  # right leg
]

# Color per connection (BGR) — color-coded by body part
CONNECTION_COLOR_MAP = {
    # Face — yellow
    (0,1):(0,215,255),(1,2):(0,215,255),(2,3):(0,215,255),(3,7):(0,215,255),
    (0,4):(0,215,255),(4,5):(0,215,255),(5,6):(0,215,255),(6,8):(0,215,255),
    (9,10):(0,215,255),
    # Torso — cyan
    (11,12):(255,255,0),(11,23):(255,255,0),(12,24):(255,255,0),(23,24):(255,255,0),
    # Left arm — orange
    (11,13):(0,165,255),(13,15):(0,165,255),
    (15,17):(0,165,255),(15,19):(0,165,255),(15,21):(0,165,255),(17,19):(0,165,255),
    # Right arm — magenta
    (12,14):(255,0,255),(14,16):(255,0,255),
    (16,18):(255,0,255),(16,20):(255,0,255),(16,22):(255,0,255),(18,20):(255,0,255),
    # Left leg — green
    (23,25):(0,255,0),(25,27):(0,255,0),(27,29):(0,255,0),(27,31):(0,255,0),(29,31):(0,255,0),
    # Right leg — blue
    (24,26):(255,100,0),(26,28):(255,100,0),(28,30):(255,100,0),
    (28,32):(255,100,0),(30,32):(255,100,0),
}


# ═══════════════════════════════════════════════════════════════
# POSE SKELETON DRAWING
# ═══════════════════════════════════════════════════════════════

def draw_pose_skeleton(
    frame: np.ndarray,
    pose_features: np.ndarray,
    bbox: Tuple[int, int, int, int],
    is_violent: bool,
    visibility_threshold: float = 0.5,
    landmark_radius: int = 4,
    connection_thickness: int = 2,
    draw_landmarks: bool = True,
    draw_connections: bool = True,
) -> np.ndarray:
    """
    Draw the MediaPipe pose skeleton on the video frame.

    Pose features store normalized coordinates (0→1) relative to the crop.
    We convert them back to full-frame pixel coordinates using the bbox.

    Color scheme when NOT violent:
        Face=Yellow, Torso=Cyan, Left arm=Orange,
        Right arm=Magenta, Left leg=Green, Right leg=Blue

    When VIOLENT: entire skeleton turns red.

    Args:
        frame:                full BGR video frame (modified in place)
        pose_features:        (132,) array from PoseExtractor
        bbox:                 (x1, y1, x2, y2) in full-frame pixels
        is_violent:           True = draw skeleton red
        visibility_threshold: skip landmarks with visibility below this
        landmark_radius:      dot size in pixels
        connection_thickness: line thickness in pixels
        draw_landmarks:       draw landmark dots
        draw_connections:     draw connection lines

    Returns:
        Modified frame (same object)
    """
    if pose_features is None or len(pose_features) != 132:
        return frame

    x1, y1, x2, y2 = [int(c) for c in bbox]
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)

    # Parse landmarks: convert normalized crop coords → full frame pixels
    landmarks_px = []
    visibilities  = []

    for i in range(33):
        base  = i * 4
        lm_x  = float(pose_features[base + 0])
        lm_y  = float(pose_features[base + 1])
        lm_vis= float(pose_features[base + 3])

        px = int(x1 + lm_x * box_w)
        py = int(y1 + lm_y * box_h)

        landmarks_px.append((px, py))
        visibilities.append(lm_vis)

    # Draw bone connections
    if draw_connections:
        for (a, b) in MP_POSE_CONNECTIONS:
            if visibilities[a] < visibility_threshold or visibilities[b] < visibility_threshold:
                continue
            color = (0, 0, 200) if is_violent else CONNECTION_COLOR_MAP.get((a, b), (180, 180, 180))
            cv2.line(frame, landmarks_px[a], landmarks_px[b], color, connection_thickness, cv2.LINE_AA)

    # Draw landmark dots
    if draw_landmarks:
        for (px, py), vis in zip(landmarks_px, visibilities):
            if vis < visibility_threshold:
                continue
            dot_color = (0, 0, 255) if is_violent else (255, 255, 255)
            cv2.circle(frame, (px, py), landmark_radius,     dot_color, -1,  cv2.LINE_AA)
            cv2.circle(frame, (px, py), landmark_radius + 1, (0, 0, 0),  1,  cv2.LINE_AA)

    return frame


# ═══════════════════════════════════════════════════════════════
# BOUNDING BOX + LABEL DRAWING
# ═══════════════════════════════════════════════════════════════

def draw_person_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int,
    violence_score: float,
    is_violent: bool,
    is_ready: bool = True,
    box_thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding box with track ID and violence score.

    Box colors:
        Red    = Violence detected
        Green  = Non-violence
        Orange = Still buffering (< 30 frames seen)

    Args:
        frame:          BGR image
        bbox:           (x1, y1, x2, y2)
        track_id:       ByteTrack person ID
        violence_score: model probability (0.0 to 1.0)
        is_violent:     True if score > threshold
        is_ready:       False if buffer not full yet
        box_thickness:  border width in pixels
    """
    x1, y1, x2, y2 = [int(c) for c in bbox]

    if not is_ready:
        color = COLOR_UNKNOWN
        label = f"ID:{track_id} | Buffering..."
    elif is_violent:
        color = COLOR_VIOLENCE
        label = f"ID:{track_id} | VIOLENCE {violence_score:.0%}"
    else:
        color = COLOR_NON_VIOLENCE
        label = f"ID:{track_id} | Safe {(1 - violence_score):.0%}"

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)

    # Label with black background
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_y = max(y1 - 5, th + 5)

    cv2.rectangle(
        frame,
        (x1, text_y - th - baseline - 2),
        (x1 + tw + 4, text_y + baseline),
        COLOR_TEXT_BG, cv2.FILLED,
    )
    cv2.putText(frame, label, (x1 + 2, text_y - baseline),
                font, font_scale, color, thickness, cv2.LINE_AA)

    return frame


def draw_fps_counter(frame: np.ndarray, fps: float) -> np.ndarray:
    """Draw FPS in top-left corner."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_NON_VIOLENCE, 2, cv2.LINE_AA)
    return frame


def draw_info_panel(
    frame: np.ndarray,
    num_persons: int,
    violence_detected: bool,
) -> np.ndarray:
    """Draw status panel in top-right corner."""
    h, w = frame.shape[:2]
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6

    status_text = "! VIOLENCE DETECTED !" if violence_detected else "Scene Clear"
    status_color = COLOR_VIOLENCE if violence_detected else COLOR_NON_VIOLENCE
    persons_text = f"Persons: {num_persons}"

    (sw, _), _ = cv2.getTextSize(status_text, font, scale, 2)
    cv2.putText(frame, status_text, (w - sw - 10, 30),
                font, scale, status_color, 2, cv2.LINE_AA)

    (pw, _), _ = cv2.getTextSize(persons_text, font, scale, 1)
    cv2.putText(frame, persons_text, (w - pw - 10, 60),
                font, scale, COLOR_WHITE, 1, cv2.LINE_AA)

    return frame


# ═══════════════════════════════════════════════════════════════
# FPS COUNTER
# ═══════════════════════════════════════════════════════════════

class FPSCounter:
    """Rolling average FPS counter."""

    def __init__(self, window: int = 30):
        self.timestamps = []
        self.window     = window

    def tick(self) -> float:
        now = time.perf_counter()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)
        if len(self.timestamps) >= 2:
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed > 0:
                return (len(self.timestamps) - 1) / elapsed
        return 0.0


# ═══════════════════════════════════════════════════════════════
# EVALUATION PLOTS
# ═══════════════════════════════════════════════════════════════

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(title, fontsize=16, fontweight="bold", pad=15)
    plt.ylabel("True Label", fontsize=13)
    plt.xlabel("Predicted Label", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved → {save_path}")


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_path: str,
) -> None:
    """Plot and save ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc_score:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random")
    plt.fill_between(fpr, tpr, alpha=0.1, color="darkorange")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate", fontsize=13)
    plt.title("ROC Curve — Violence Detection", fontsize=16, fontweight="bold")
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved → {save_path}")


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    avg_precision: float,
    save_path: str,
) -> None:
    """Plot and save Precision-Recall curve."""
    plt.figure(figsize=(8, 6))
    plt.step(recall, precision, color="steelblue", lw=2, where="post",
             label=f"AP = {avg_precision:.4f}")
    plt.fill_between(recall, precision, alpha=0.1, color="steelblue", step="post")
    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.title("Precision-Recall Curve — Violence Detection", fontsize=16, fontweight="bold")
    plt.legend(loc="upper right", fontsize=12)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved → {save_path}")


def plot_training_history(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    save_path: str,
) -> None:
    """Plot and save training/validation loss and accuracy curves."""
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "b-o", label="Train Loss",     markersize=4)
    ax1.plot(epochs, val_losses,   "r-o", label="Val Loss",       markersize=4)
    ax1.set_title("Loss", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("BCE Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_accs, "b-o", label="Train Accuracy",   markersize=4)
    ax2.plot(epochs, val_accs,   "r-o", label="Val Accuracy",     markersize=4)
    ax2.set_title("Accuracy", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Violence Detection — Training History", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Visualization] Saved → {save_path}")
