# Violence Detection System

A real-time violence detection system using **YOLOv8/YOLOv11**, **MediaPipe Pose**, and a **Transformer Encoder** for temporal action recognition. Detects violence in video files, webcams, and phone cameras with confidence scores and auto-screenshots.
This model is a 4-layer temporal transformer which detects violent movements by using the coordinates of 33 land marks of the body from human pose detection models i.e. mediapose, and classifying the movement as violent or not. Multiple people are detected in real time using YOLO algorithm and ByteTrack. Using this program, security forces can be quick to react to certain violent situations, leading to less casualties. This program can be implemented in malls, prisons and even schools. 

---

## How It Works

```
Video / Camera
      ↓
YOLO  →  Detect person, crop bounding box ROI
      ↓
MediaPipe  →  Extract 33 pose landmarks from ROI (x, y, z, visibility) = 132 features/frame
      ↓
Sliding window buffer  →  last N frames of landmarks
      ↓
ViolenceTransformer  →  Violence probability score
      ↓
Overlay label on frame + auto-screenshot if violence detected
```

---

## Project Structure

```
violence_detection/
│
├── detect_violence_2.py       Real-time inference (webcam / video / phone camera)
├── preprocess.py              Convert videos → sliding window pose sequences
├── dataset.py                 PyTorch Dataset + train/val/test splits
├── transformer_model.py       Transformer Encoder architecture
├── train.py                   Training pipeline
├── evaluate.py                Evaluation + plots
│
├── utils/
│   ├── pose_extractor.py      MediaPipe Pose wrapper — 132 features per frame
│   └── tracker.py             ByteTrack multi-person tracker + per-ID pose buffers
│
├── processed/                 Output of preprocess.py (.npy windows)
├── checkpoints/               Saved model weights (best_model.pth)
├── screenshots/               Auto-captured violence frames
├── outputs/                   Annotated videos + evaluation plots
└── requirements.txt
```

---

## Installation

### Requirements
- Python 3.11 / 3.12
- NVIDIA GPU recommended (runs on CPU but training will be slow)
- Google Colab T4 fully supported

### 1. Create a virtual environment

```bash
python -m venv venv

# Activate
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac / Linux
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. GPU support (optional but recommended)

```bash
# Check your CUDA version: nvidia-smi
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 4. Verify

```bash
python -c "import cv2, mediapipe, ultralytics, torch; print('All imports OK')"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

---

## Quick Start

```bash
# Step 1 — Preprocess videos into sliding-window pose sequences
python preprocess.py --dataset_path ./dataset --output_path ./processed
### Recommended settings for 5+ second clips

```bash
python preprocess.py \
    --dataset_path ./dataset \
    --output_path  ./processed \
    --window_size   30 \
    --window_stride 15 \
    --yolo_model    yolo11n.pt \
    --device        auto

# Step 2 — Train the Transformer
python train.py --epochs 50


python train.py \
    --processed_path ./processed/sliding \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --patience 10 \
    --num_workers 2 \
    --embedding_dim 256 \
    --num_heads 8 \
    --num_layers 4 \
    --dropout 0.3

# Step 3 — Evaluate on test set
python evaluate.py --checkpoint ./checkpoints/best_model.pth

# Step 4 — Run inference on a video file
python detect_violence.py --video input.mp4 --checkpoint ./checkpoints/best_model.pth --show

# Step 5 — Run on webcam
python detect_violence.py --checkpoint ./checkpoints/best_model.pth
```

---

## Real-Time Inference (`detect_violence_2.py`)

### Video file
```bash
python detect_violence.py \
  --video path/to/video.mp4 \
  --checkpoint ./checkpoints/best_model.pth \
  --show
```

### Webcam
```bash
python detect_violence_2.py \
  --checkpoint ./checkpoints/best_model.pth
```

### Phone camera via DroidCam (WiFi)
```bash
python detect_violence_2.py \
  --camera http://192.168.1.5:4747/video \
  --checkpoint ./checkpoints/best_model.pth
```

### Save annotated output video
```bash
python detect_violence_2.py \
  --video input.mp4 \
  --checkpoint ./checkpoints/best_model.pth \
  --output ./outputs/annotated.mp4 \
  --show
```

Press **`q`** to quit the preview window.

### All inference arguments

| Argument | Default | Description |
|---|---|---|
| `--video` | None | Input video file path. Cannot be used with `--camera` |
| `--camera` | `0` | Camera source: device index, `/dev/videoN`, or URL |
| `--checkpoint` | *(required)* | Path to `best_model.pth` |
| `--yolo_model` | `yolov8n.pt` | YOLO weights file |
| `--yolo_conf` | `0.4` | YOLO person detection confidence threshold |
| `--threshold` | `0.5` | Violence probability threshold |
| `--output` | None | Save annotated video to this path |
| `--screenshot_dir` | `./screenshots` | Folder for auto-captured violence screenshots |
| `--screenshot_cooldown` | `3.0` | Minimum seconds between screenshots |
| `--disable_screenshots` | Off | Pass this flag to turn off auto-screenshots |
| `--show` | Off | Show preview window when using `--video` mode |

### Recommended threshold

Use the value from your model's `evaluation_results.json`. For the default trained model:

```bash
--threshold 0.598
```

| Goal | Threshold |
|---|---|
| Catch all violence (fewer misses) | `0.2` – `0.3` |
| Balanced | `0.3` – `0.4` |
| Reduce false alarms | `0.5` – `0.6` |

---

## Phone Camera Setup

Both your phone and laptop must be on the **same Wi-Fi network**.

| App | Platform | Command |
|---|---|---|
| DroidCam (free) | Android / iOS | `--camera http://<phone-ip>:4747/video` |
| IP Webcam (free) | Android | `--camera http://<phone-ip>:8080/video` |
| EpocCam (paid) | iOS | `--camera http://<phone-ip>:2431/live` |
| RTSP stream | Any | `--camera rtsp://<phone-ip>:8554/stream` |
| DroidCam USB | Android / iOS | `--camera 1` or `--camera 2` |

Find your phone's IP in its WiFi settings and replace `<phone-ip>`.

---

## Auto-Screenshots

When violence is detected, a PNG is saved automatically:

```
screenshots/
└── violence_20260617_143201_042_frame001234.png
```

- Timestamped to millisecond precision
- Cooldown (default 3s) prevents duplicate saves in a burst
- Disable with `--disable_screenshots`
- Change save folder with `--screenshot_dir ./my_alerts`

---

## Preprocessing — Sliding Window (`preprocess.py`)

This project uses a **sliding window** approach rather than uniform frame sampling. This is important for longer clips where violent actions may happen in short bursts.

### Why sliding windows?

The alternative — sampling 30 evenly-spaced frames from a 5-second clip — throws away 80% of temporal detail. A punch or shove that falls between two sampled frames is never seen by the model.

Sliding windows fix this:

```
Video (150 valid frames)
    │
    ├── window_000:  frames   0–29
    ├── window_001:  frames  15–44   (50% overlap)
    ├── window_002:  frames  30–59
    │   ...
    └── window_009:  frames 120–149
```

A single 5-second clip produces 8–10 training samples instead of 1 — multiplying your effective dataset size for free.

### Dataset structure required

```
dataset/
├── violence/
│   ├── vid1.mp4
│   └── ...
└── non_violence/
    ├── vid1.mp4
    └── ...
```

Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`

### Basic run

```bash
python preprocess.py --dataset_path ./dataset --output_path ./processed
```

### Recommended for 5+ second clips

```bash
python preprocess.py \
  --dataset_path ./dataset \
  --output_path  ./processed \
  --window_size   30 \
  --window_stride 15 \
  --yolo_model    yolo11n.pt \
  --device        auto
```

### Preprocessing arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_path` | `./dataset` | Root folder with `violence/` and `non_violence/` |
| `--output_path` | `./processed` | Where to save `.npy` window files |
| `--window_size` | `30` | Frames per window |
| `--window_stride` | `15` | Step between windows. Smaller = more overlap = more samples |
| `--yolo_model` | `yolo11n.pt` | YOLO weights |
| `--yolo_conf` | `0.4` | Minimum YOLO confidence for person detection |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--overwrite` | Off | Reprocess videos even if windows already exist |

### Window stride guide

| `window_stride` | Overlap | Effect |
|---|---|---|
| `= window_size` | 0% | Fewest samples, fastest preprocessing |
| `= window_size / 2` | 50% | **Recommended** — good balance |
| `< window_size / 2` | >50% | Maximum samples, slower, more redundant |

### Output structure

```
processed/
├── violence/
│   ├── vid1_w000.npy       shape: (window_size, 132)
│   ├── vid1_w001.npy
│   └── ...
├── non_violence/
│   └── ...
└── metadata.json
```

---

## Training (`train.py`)

```bash
python train.py \
  --processed_path ./processed \
  --epochs 50 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --patience 10
```

| Argument | Default | Description |
|---|---|---|
| `--processed_path` | `./processed` | Path to processed windows |
| `--epochs` | `50` | Maximum training epochs |
| `--batch_size` | `32` | Samples per batch (use 64 on Colab T4) |
| `--learning_rate` | `1e-4` | Initial learning rate |
| `--patience` | `10` | Early stopping patience |
| `--embedding_dim` | `256` | Transformer hidden dimension |
| `--num_heads` | `8` | Attention heads |
| `--num_layers` | `4` | Encoder layers |
| `--dropout` | `0.3` | Dropout rate |

Monitor with TensorBoard:
```bash
tensorboard --logdir ./logs
```

---

## Evaluation (`evaluate.py`)

```bash
python evaluate.py --checkpoint ./checkpoints/best_model.pth --threshold 0.5
```

Output files saved to `./outputs/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `precision_recall_curve.png`
- `evaluation_results.json`

---

## Model Architecture

```
Input (batch, 30, 132)
    ↓
Linear Projection + LayerNorm  →  (batch, 30, 256)
    ↓
Sinusoidal Positional Encoding
    ↓
Transformer Encoder × 4 layers
    each: Multi-Head Attention (8 heads) + FFN (1024 dim) + LayerNorm
    ↓
Global Average Pooling  →  (batch, 256)
    ↓
Linear(256 → 128) + GELU + Dropout(0.3)
    ↓
Linear(128 → 2) + Softmax
    ↓
Violence Probability (0.0 – 1.0)
```

---

## Google Colab Setup

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install -r requirements.txt

import torch
print(torch.cuda.get_device_name(0))   # e.g. "Tesla T4"

!python preprocess.py --dataset_path /content/drive/MyDrive/dataset
!python train.py --epochs 50 --batch_size 64
!cp checkpoints/best_model.pth /content/drive/MyDrive/
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: transformer_model` | Place `transformer_model.py` in the same folder as the script |
| `Cannot open source: 0` | Try `--camera 1` or `--camera 2` |
| Phone camera not connecting | Check IP, ensure same Wi-Fi, disable firewall temporarily |
| No preview window with `--video` | Add `--show` |
| Slow on CPU | Set `model_complexity=0` in `MP_POSE.Pose(...)` inside the script; use `yolo11n.pt` |
| `No videos found` during preprocessing | Check `dataset/violence/` and `dataset/non_violence/` exist with supported video files |
| Zero windows generated | Lower `--yolo_conf 0.3`; ensure people are clearly visible in the videos |
| `CUDA out of memory` | Reduce `--batch_size 16`; use `--yolo_model yolo11n.pt` |
| Training loss not decreasing | Try `--learning_rate 5e-5`; check class balance in `processed/metadata.json` |
| `MediaPipe not detecting poses` | Lower `--yolo_conf 0.3`; use a larger YOLO model (`yolo11s.pt`) |

---

## Utils Reference (`utils/`)

### `pose_extractor.py`
Clean wrapper around MediaPipe Pose. Accepts a person crop (BGR), returns 132 features.

```python
from utils.pose_extractor import PoseExtractor

extractor = PoseExtractor(model_complexity=1, min_detection_confidence=0.5)
features = extractor.extract(person_crop_bgr)   # returns (132,) array or None
extractor.close()

# Or use as context manager
with PoseExtractor() as extractor:
    features = extractor.extract(crop)
```

- Returns `None` if no pose is detected or the crop is smaller than 50×50 px
- Features are `[x, y, z, visibility]` × 33 landmarks, normalized to the crop dimensions

### `tracker.py`
Multi-person tracking with per-ID pose history buffers. Each tracked person maintains their own 30-frame rolling window for independent violence scoring.

```python
from utils.tracker import MultiPersonTracker

tracker = MultiPersonTracker(sequence_length=30, max_lost_frames=30)

# Each frame: update with YOLO+ByteTrack detections
tracker.update(track_id, bbox, pose_features, confidence)
tracker.tick()   # advance frame counter, prune lost tracks

# Get tracks ready for transformer inference
ready = tracker.get_ready_tracks()   # list of PersonTrackBuffer
for buf in ready:
    seq = buf.get_sequence()         # (30, 132) numpy array

# Batch-update violence scores after inference
tracker.update_violence_scores(track_ids, scores, threshold=0.5)
```
