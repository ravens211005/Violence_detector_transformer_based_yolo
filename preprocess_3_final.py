"""
preprocess.py — Violence Detection Pipeline (Sliding Window Edition)
======================================================================
MODIFICATIONS (vs single-window original):

  [SLIDING WINDOW] Instead of sampling ONE 30-frame sequence per video,
                    this version extracts MULTIPLE overlapping windows
                    from each video and saves each as its own .npy file.

                    Why: your clips are 5+ seconds (~150+ frames at 30fps).
                    A single uniform 30-frame sample only keeps 1-in-5 frames,
                    discarding most of the temporal detail and risking missing
                    the actual violent moment if it falls between samples.

                    Sliding windows fix this by:
                      (a) covering the WHOLE video, not just 30 sparse samples
                      (b) multiplying your effective dataset size for free
                      (c) forcing the model to recognize violence regardless
                          of WHERE in the window it occurs

  [REQ-1] One .npy file per WINDOW (not per video) saved under
           processed/violence/ or processed/non_violence/, named:
               <video_stem>_w000.npy, <video_stem>_w001.npy, ...

  [REQ-3] YOLO person detection runs first; only the best-confidence
           bounding box ROI is cropped — never the full frame.

  [REQ-4] MediaPipe pose estimation runs exclusively on the cropped ROI.

  Logging: detected bbox, crop dimensions, window boundaries, saved path.

  Edge cases handled: no person detected, empty crop, missing landmarks,
                       videos shorter than one window, videos shorter than
                       window_stride.

OUTPUT STRUCTURE:
    processed/
    ├── violence/
    │   ├── vid1_w000.npy      shape: (window_size, 132)
    │   ├── vid1_w001.npy
    │   ├── vid2_w000.npy
    │   └── ...
    ├── non_violence/
    │   ├── vid1_w000.npy
    │   └── ...
    └── metadata.json          per-window + per-video processing record

RUN:
    python preprocess.py --dataset_path ./dataset --output_path ./processed
    python preprocess.py --window_size 30 --window_stride 15   (50% overlap)
    python preprocess.py --window_size 60 --window_stride 30   (longer context)
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

# ── Logging setup ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── MediaPipe constants ──────────────────────────────────────────────────
MP_POSE       = mp.solutions.pose
NUM_LANDMARKS = 33                  # MediaPipe Pose landmark count
FEATURE_DIM   = NUM_LANDMARKS * 4   # x, y, z, visibility per landmark


# ════════════════════════════════════════════════════════════════════════
# Helper: extract landmark vector from a MediaPipe pose result
# ════════════════════════════════════════════════════════════════════════
def extract_landmarks(results) -> Optional[np.ndarray]:
    """
    Return a flat (NUM_LANDMARKS * 4,) float32 array of pose features,
    or None if no landmarks were detected.

    [REQ-4] Landmarks come from ROI-relative pose estimation; coordinates
    are already normalised by MediaPipe within the crop dimensions.
    """
    if not results or not results.pose_landmarks:
        return None  # Edge case: missing landmarks

    lm = results.pose_landmarks.landmark
    features = []
    for point in lm:
        features.extend([point.x, point.y, point.z, point.visibility])
    return np.array(features, dtype=np.float32)


# ════════════════════════════════════════════════════════════════════════
# Helper: get best-confidence person bounding box from YOLO results
# ════════════════════════════════════════════════════════════════════════
def get_best_person_box(yolo_result, frame_h: int, frame_w: int):
    """
    [REQ-3] Extract the single highest-confidence PERSON bounding box.

    Returns (x1, y1, x2, y2) clipped to frame dimensions, or None if no
    person is detected.

    Edge cases handled:
      - No detections at all → returns None
      - Multiple persons    → picks highest confidence one
    """
    boxes = yolo_result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    # Filter class 0 = 'person'
    person_mask = boxes.cls.cpu().numpy() == 0
    if not person_mask.any():
        return None

    confs = boxes.conf.cpu().numpy()[person_mask]
    xyxys = boxes.xyxy.cpu().numpy()[person_mask]   # shape (N, 4)

    # [REQ-3] Pick highest-confidence person
    best_idx = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxys[best_idx]

    # Clip to frame bounds and convert to int
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame_w, int(x2))
    y2 = min(frame_h, int(y2))

    log.debug(
        f"  Detected bbox: ({x1},{y1}) -> ({x2},{y2})  "
        f"conf={confs[best_idx]:.2f}  ({len(confs)} person(s) found)"
    )
    return x1, y1, x2, y2


# ════════════════════════════════════════════════════════════════════════
# Core: extract pose features for EVERY frame of a video (once)
# ════════════════════════════════════════════════════════════════════════
def extract_all_frame_features(
    video_path: str,
    yolo_model: YOLO,
    pose_model,
    yolo_conf: float = 0.4,
) -> np.ndarray:
    """
    Run YOLO + MediaPipe on every frame of the video ONCE, returning a
    full per-frame feature array. Sliding windows are then cut from
    this array — far cheaper than re-running detection per window.

    Pipeline (per frame):
        1. Run YOLO person detection on full frame             [REQ-3]
        2. Select highest-confidence person bounding box       [REQ-3]
        3. Crop ONLY the bounding-box ROI                       [REQ-3]
        4. Run MediaPipe pose estimation on the ROI crop        [REQ-4]
        5. Collect landmark vector (ROI-relative, normalised)   [REQ-4]

    Frames where no person/landmarks are found get a zero vector —
    this preserves frame alignment so window slicing stays correct.

    Returns:
        np.ndarray of shape (total_valid_frames, FEATURE_DIM).
        Frames with no detection are DROPPED (not zero-padded) so that
        windows are built from frames that actually contain pose data.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log.warning(f"Cannot open video: {video_path}")
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    frame_features = []
    frame_idx = 0
    skipped_no_person = 0
    skipped_empty_crop = 0
    skipped_no_landmarks = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]

        # ── Step 1: YOLO detection on full frame ────────────────────────
        yolo_results = yolo_model(frame, conf=yolo_conf, verbose=False)
        bbox = get_best_person_box(yolo_results[0], frame_h, frame_w)

        if bbox is None:
            # [EDGE CASE] No person detected in this frame — skip frame
            skipped_no_person += 1
            frame_idx += 1
            continue

        x1, y1, x2, y2 = bbox

        # ── Step 2: Crop ONLY the bounding-box ROI ──────────────────────
        crop = frame[y1:y2, x1:x2]  # [REQ-3] crop = frame[y1:y2, x1:x2]

        # [EDGE CASE] Empty or degenerate crop
        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            skipped_empty_crop += 1
            frame_idx += 1
            continue

        # ── Step 3: Pose estimation on ROI crop only ────────────────────
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results  = pose_model.process(crop_rgb)  # [REQ-4] only the ROI

        features = extract_landmarks(results)    # [REQ-4] ROI-relative landmarks

        if features is None:
            # [EDGE CASE] MediaPipe found no landmarks in this crop
            skipped_no_landmarks += 1
            frame_idx += 1
            continue

        frame_features.append(features)
        frame_idx += 1

    cap.release()

    total_frames = frame_idx
    valid_frames = len(frame_features)
    log.info(
        f"    Frames: {total_frames} total, {valid_frames} valid  "
        f"(no_person={skipped_no_person}, empty_crop={skipped_empty_crop}, "
        f"no_landmarks={skipped_no_landmarks})"
    )

    if not frame_features:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)

    return np.array(frame_features, dtype=np.float32)  # (valid_frames, FEATURE_DIM)


# ════════════════════════════════════════════════════════════════════════
# Sliding window slicing
# ════════════════════════════════════════════════════════════════════════
def make_sliding_windows(
    frame_features: np.ndarray,
    window_size: int,
    window_stride: int,
    min_windows: int = 1,
) -> list:
    """
    Slice a (T, FEATURE_DIM) array into overlapping windows of
    (window_size, FEATURE_DIM) each.

    Args:
        frame_features: (T, FEATURE_DIM) array of all valid frame poses
        window_size:    frames per window (e.g. 30)
        window_stride:  step between window starts (e.g. 15 = 50% overlap)
        min_windows:    always produce at least this many windows, even
                         for short videos (via padding) — keeps every
                         video contributing at least one training sample

    Returns:
        list of (window_size, FEATURE_DIM) numpy arrays
    """
    T = frame_features.shape[0]
    windows = []

    if T == 0:
        # [EDGE CASE] No valid frames at all — return one zero window
        # so the video still produces a (label-only) training sample
        # rather than disappearing from the dataset entirely.
        windows.append(np.zeros((window_size, FEATURE_DIM), dtype=np.float32))
        return windows

    if T <= window_size:
        # [EDGE CASE] Video shorter than one window — pad to window_size
        pad = np.zeros((window_size - T, FEATURE_DIM), dtype=np.float32)
        windows.append(np.vstack([frame_features, pad]))
        return windows

    # Normal case: slide across the full sequence
    start = 0
    while start + window_size <= T:
        windows.append(frame_features[start : start + window_size].copy())
        start += window_stride

    # Ensure the tail of the video isn't dropped if stride doesn't divide evenly
    last_start = T - window_size
    if windows and not np.array_equal(windows[-1], frame_features[last_start:T]):
        windows.append(frame_features[last_start:T].copy())

    # Guarantee at least `min_windows` windows
    while len(windows) < min_windows:
        windows.append(frame_features[-window_size:].copy())

    return windows


# ════════════════════════════════════════════════════════════════════════
# Per-video processing — produces N windows
# ════════════════════════════════════════════════════════════════════════
def process_video(
    video_path: str,
    yolo_model: YOLO,
    pose_model,
    window_size: int,
    window_stride: int,
    yolo_conf: float = 0.4,
) -> list:
    """
    Full per-video pipeline:
        1. Extract pose features for every valid frame (once)
        2. Slice into overlapping sliding windows

    Returns:
        list of (window_size, FEATURE_DIM) numpy arrays — one per window.
        Empty list only if the video file itself could not be opened.
    """
    frame_features = extract_all_frame_features(
        video_path, yolo_model, pose_model, yolo_conf=yolo_conf
    )

    windows = make_sliding_windows(frame_features, window_size, window_stride)
    return windows


# ════════════════════════════════════════════════════════════════════════
# Resume support — crash-safe progress tracking
# ════════════════════════════════════════════════════════════════════════
#
# WHY NOT just check "_w*.npy exists"?
#   If the script is interrupted (Ctrl+C, power loss, disconnect) WHILE a
#   video is being processed, that video may have written SOME windows
#   but not all of them. A simple "_w*.npy exists -> skip" check would
#   wrongly treat that partial video as fully done.
#
#   This progress log records a video as "done" only AFTER all of its
#   windows have been written successfully -- so resuming is always safe.
#
PROGRESS_FILENAME = "_progress.json"


def load_progress(output_path: Path) -> set:
    """
    Load the set of video paths that were FULLY completed in a previous run.

    Returns:
        set of video path strings already fully processed
    """
    progress_path = output_path / PROGRESS_FILENAME
    if not progress_path.exists():
        return set()

    try:
        with open(progress_path, "r") as f:
            data = json.load(f)
        completed = set(data.get("completed_videos", []))
        log.info(f"Resume: found progress log with {len(completed)} "
                  f"video(s) already completed")
        return completed
    except (json.JSONDecodeError, OSError) as e:
        # [EDGE CASE] Corrupt progress file (e.g. crash mid-write) —
        # treat as empty rather than crashing the whole run
        log.warning(f"Progress log unreadable ({e}) — starting fresh")
        return set()


def mark_video_complete(output_path: Path, video_path: str, completed: set):
    """
    Append one finished video to the progress log and write it to disk
    IMMEDIATELY. This makes the log crash-safe: if the script dies on the
    very next video, everything finished so far is still recorded.

    Args:
        output_path: processed/ directory
        video_path:  the video that just finished successfully
        completed:   the in-memory set of completed videos (updated in place)
    """
    completed.add(video_path)
    progress_path = output_path / PROGRESS_FILENAME

    # Write to a temp file first, then atomically replace — avoids a
    # half-written/corrupt progress file if the process dies mid-save
    tmp_path = progress_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump({"completed_videos": sorted(completed)}, f, indent=2)
    tmp_path.replace(progress_path)


def load_prior_metadata(output_path: Path):
    """
    Load metadata records from a previous (interrupted) run, so resuming
    doesn't lose history of videos already processed in earlier sessions.

    Returns:
        (records, processed_count, skipped_count, total_windows) from the
        last saved metadata.json, or empty/zero values if none exists.
    """
    meta_path = output_path / "metadata.json"
    if not meta_path.exists():
        return [], 0, 0, 0

    try:
        with open(meta_path, "r") as f:
            data = json.load(f)
        records   = data.get("records", [])
        processed = data.get("videos_processed", len(records))
        skipped   = data.get("videos_skipped", 0)
        windows   = data.get("total_windows_saved",
                              sum(r.get("num_windows", 0) for r in records))
        log.info(f"Resume: loaded prior metadata "
                  f"({processed} processed, {windows} windows)")
        return records, processed, skipped, windows
    except (json.JSONDecodeError, OSError) as e:
        # [EDGE CASE] Corrupt metadata.json from a crash mid-write —
        # start fresh rather than failing the whole run
        log.warning(f"Prior metadata.json unreadable ({e}) — starting fresh")
        return [], 0, 0, 0


def _write_metadata(
    output_path: Path,
    args,
    metadata_records: list,
    processed_videos: int,
    skipped_videos: int,
    total_windows: int,
    elapsed: float,
) -> dict:
    """
    Build and write metadata.json. Called both incrementally (after every
    video, so a crash never loses more than one video's progress) and once
    more at the very end with the final elapsed time.

    Returns:
        the metadata dict that was written (used for end-of-run logging)
    """
    metadata = {
        "mode":                    "sliding_window",
        "processing_time_seconds": elapsed,
        "window_size":             args.window_size,
        "window_stride":           args.window_stride,
        "feature_dim":             FEATURE_DIM,
        "num_landmarks":           NUM_LANDMARKS,
        "videos_processed":        processed_videos,
        "videos_skipped":          skipped_videos,
        "total_windows_saved":     total_windows,
        "avg_windows_per_video":   round(total_windows / max(processed_videos, 1), 2),
        "label_map":               {"violence": 1, "non_violence": 0},
        "records":                 metadata_records,
    }

    meta_path = output_path / "metadata.json"
    tmp_path  = meta_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(metadata, f, indent=2)
    tmp_path.replace(meta_path)  # atomic write — avoids corrupt metadata.json

    return metadata


# ════════════════════════════════════════════════════════════════════════
# Main preprocessing loop
# ════════════════════════════════════════════════════════════════════════
def main(args):
    dataset_path = Path(args.dataset_path)
    output_path  = Path(args.output_path)

    # [REQ-1] Create per-class subdirectories inside processed/
    (output_path / "violence").mkdir(parents=True, exist_ok=True)
    (output_path / "non_violence").mkdir(parents=True, exist_ok=True)

    # ── Load models ────────────────────────────────────────────────────
    device = args.device
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Using device: {device}")

    log.info(f"Loading YOLO model: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    yolo_model.to(device)

    log.info("Initialising MediaPipe Pose")
    pose_model = MP_POSE.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    log.info(
        f"Sliding window config: window_size={args.window_size}  "
        f"window_stride={args.window_stride}  "
        f"overlap={100 * (1 - args.window_stride / args.window_size):.0f}%"
    )

    # ── Discover videos ───────────────────────────────────────────────────
    categories = {"violence": 1, "non_violence": 0}
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

    video_list = []
    for cat_name, label in categories.items():
        cat_dir = dataset_path / cat_name
        if not cat_dir.exists():
            log.warning(f"Category folder not found: {cat_dir}")
            continue
        for vp in sorted(cat_dir.iterdir()):
            if vp.suffix.lower() in video_extensions:
                video_list.append((vp, label, cat_name))

    log.info(f"Found {len(video_list)} videos to process")

    # ── Resume support: load previously completed videos ────────────────
    # [RESUME] If this is a re-run after a disconnect/crash, skip videos
    # that were FULLY completed last time. Partially-completed videos
    # (interrupted mid-processing) are NOT in this set, so they will be
    # reprocessed from scratch -- this guarantees no incomplete data.
    completed_videos = load_progress(output_path) if not args.overwrite else set()

    # [RESUME] Also reload prior metadata records so the final metadata.json
    # still contains full history across multiple resumed runs, not just
    # the videos processed in this particular session.
    prior_metadata_records, prior_processed, prior_skipped, prior_windows = (
        load_prior_metadata(output_path) if not args.overwrite else ([], 0, 0, 0)
    )

    if completed_videos:
        remaining = [v for v in video_list if str(v[0]) not in completed_videos]
        log.info(f"Resume: {len(video_list) - len(remaining)} video(s) already "
                  f"done, {len(remaining)} remaining")
        video_list = remaining

    # ── Process each video ──────────────────────────────────────────────
    start_time = time.time()
    metadata_records  = list(prior_metadata_records)   # [RESUME] carry forward
    processed_videos  = prior_processed                 # [RESUME] carry forward
    skipped_videos    = prior_skipped                    # [RESUME] carry forward
    total_windows     = prior_windows                    # [RESUME] carry forward

    for video_path, label, cat_name in tqdm(video_list, desc="Processing videos"):
        log.info(f"Processing [{cat_name}] {video_path.name}")
        stem = video_path.stem

        # [RESUME] If --overwrite, clear any partial windows from a
        # previous interrupted run before reprocessing this video, so
        # stale window files don't linger alongside fresh ones.
        if args.overwrite:
            for stale in (output_path / cat_name).glob(f"{stem}_w*.npy"):
                stale.unlink()

        windows = process_video(
            str(video_path),
            yolo_model,
            pose_model,
            window_size=args.window_size,
            window_stride=args.window_stride,
            yolo_conf=args.yolo_conf,
        )

        if not windows:
            log.warning(f"  SKIPPED (could not open video): {video_path.name}")
            skipped_videos += 1
            continue

        # [REQ-1] Save ONE .npy per WINDOW in the appropriate subdirectory
        window_paths = []
        for w_idx, window in enumerate(windows):
            npy_path = output_path / cat_name / f"{stem}_w{w_idx:03d}.npy"
            np.save(str(npy_path), window)
            window_paths.append(str(npy_path))
            log.info(f"  Saved window {w_idx} -> {npy_path}  shape={window.shape}")

        metadata_records.append({
            "video":         str(video_path),
            "category":      cat_name,
            "label":         label,
            "num_windows":   len(windows),
            "window_files":  window_paths,
            "window_size":   args.window_size,
            "window_stride": args.window_stride,
        })

        processed_videos += 1
        total_windows     += len(windows)

        # [RESUME] Mark this video as fully done and persist immediately.
        # If the script is interrupted on the NEXT video, this one will
        # correctly be skipped on the next run instead of reprocessed.
        mark_video_complete(output_path, str(video_path), completed_videos)

        # [RESUME] Also write metadata.json incrementally (not just at the
        # very end) so detailed per-video records survive a crash too.
        _write_metadata(
            output_path, args, metadata_records,
            processed_videos, skipped_videos, total_windows,
            elapsed=time.time() - start_time,
        )

    pose_model.close()
    elapsed = time.time() - start_time

    # ── Final metadata.json write (with accurate total elapsed time) ─────
    # [REQ-1] Labels are inferred from folder structure, not a single y.npy
    metadata = _write_metadata(
        output_path, args, metadata_records,
        processed_videos, skipped_videos, total_windows,
        elapsed=elapsed,
    )
    meta_path = output_path / "metadata.json"

    log.info("=" * 60)
    log.info(f"Preprocessing complete in {elapsed:.1f}s")
    log.info(f"  Videos processed     : {processed_videos}")
    log.info(f"  Videos skipped       : {skipped_videos}")
    log.info(f"  Total windows saved  : {total_windows}")
    log.info(f"  Avg windows/video    : {metadata['avg_windows_per_video']}")
    log.info(f"  Metadata             : {meta_path}")
    log.info("=" * 60)


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Violence Detection — Sliding Window Preprocessing"
    )
    parser.add_argument("--dataset_path",  default="./dataset",
                        help="Root dataset folder (contains violence/ and non_violence/)")
    parser.add_argument("--output_path",   default="./processed",
                        help="Output folder for per-window .npy files")
    parser.add_argument("--window_size",   type=int, default=30,
                        help="Frames per window/sequence (default: 30)")
    parser.add_argument("--window_stride", type=int, default=15,
                        help="Step between window starts — smaller = more overlap "
                             "(default: 15, i.e. 50%% overlap with window_size=30)")
    parser.add_argument("--yolo_model",    default="yolo11n.pt",
                        help="YOLO model weights (default: yolo11n.pt)")
    parser.add_argument("--yolo_conf",     type=float, default=0.4,
                        help="YOLO confidence threshold (default: 0.4)")
    parser.add_argument("--device",        default="auto",
                        help="Device: auto | cpu | cuda (default: auto)")
    parser.add_argument("--overwrite",     action="store_true",
                        help="Reprocess videos even if windows already exist")
    args = parser.parse_args()

    if args.window_stride > args.window_size:
        log.warning(
            f"window_stride ({args.window_stride}) > window_size "
            f"({args.window_size}) — windows will have GAPS, not overlap. "
            f"This is unusual; double-check this is intended."
        )

    main(args)
