"""
detect_violence.py  —  Real-Time Multi-Person Violence Detection Inference
============================================================================
MODIFICATIONS (vs original):
  [REQ-3] YOLO + ByteTrack run on each frame; EVERY detected person gets
           their own bounding box ROI cropped and tracked with a stable ID.
  [REQ-4] Pose landmarks are computed from each person's ROI crop, not the
           full frame.
  Edge cases handled: no person detected, empty crop, missing landmarks,
                       tracks lost when a person leaves the frame.

NEW FEATURES:
  [MULTI-PERSON] Every person in frame is tracked independently via
                 ByteTrack. Each gets their own rolling pose buffer and
                 violence score — e.g. "ID:1 VIOLENCE 92%", "ID:2 Safe 97%"
                 — rather than only the single most-confident detection.
  [PHONE-CAM] --camera accepts a device index (int), /dev/videoN path, or a
              full URL for IP-camera apps (DroidCam, EpocCam, Camo, etc.).
  [SCREENSHOT] When ANY tracked person is flagged violent, a timestamped
               PNG of the full annotated frame is saved automatically to
               --screenshot_dir. A --screenshot_cooldown (default 3 s)
               prevents duplicate saves within bursts.
               Pass --disable_screenshots to turn this off entirely.
  [TEXT-VIS]  Both per-person labels and the overall status banner scale
              their font size, thickness, and margins relative to the
              actual frame resolution, so percentages stay readable on
              everything from a small webcam feed to a 4K phone stream.
"""

import argparse
import collections
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO

from transformer_model import ViolenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MP_POSE       = mp.solutions.pose
MP_DRAWING    = mp.solutions.drawing_utils
NUM_LANDMARKS = 33
FEATURE_DIM   = NUM_LANDMARKS * 4


# ══════════════════════════════════════════════════════════════════════════════
# Camera source resolver
# ══════════════════════════════════════════════════════════════════════════════
def resolve_camera_source(camera_arg: str):
    """
    [PHONE-CAM] Convert the --camera argument to the right OpenCV source.

    Accepted forms
    ──────────────
    Integer string  "0", "1", "2"  → device index (built-in / USB webcam)
    /dev/videoN     "/dev/video2"  → Linux V4L2 device path
    URL             "http://192.168.1.5:4747/video"   (DroidCam HTTP)
                    "rtsp://192.168.1.5:8554/stream"  (RTSP — EpocCam, etc.)

    Phone-app quick-reference
    ─────────────────────────
    DroidCam (Android/iOS, free)
        USB mode  : usually /dev/video2 on Linux, device index 1 or 2 on Win
        Wi-Fi mode: http://<phone-ip>:4747/video
    EpocCam (iOS, paid)
        Wi-Fi/USB : http://<phone-ip>:2431/live   or RTSP port 8554
    Camo (iOS, paid)
        USB mode  : appears as a regular webcam device (index 1 or 2)
    IP Webcam (Android, free)
        Wi-Fi mode: http://<phone-ip>:8080/video

    Returns the value to pass to cv2.VideoCapture().
    """
    # Pure integer → device index
    try:
        return int(camera_arg)
    except ValueError:
        pass

    # Everything else (URL, /dev/videoN) → pass as string
    return camera_arg


# ══════════════════════════════════════════════════════════════════════════════
# Screenshot helper
# ══════════════════════════════════════════════════════════════════════════════
def save_screenshot(frame: np.ndarray, output_dir: Path, frame_idx: int) -> str:
    """
    [SCREENSHOT] Save *frame* as a timestamped PNG inside *output_dir*.
    Returns the file path written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]   # ms precision
    name = f"violence_{ts}_frame{frame_idx:06d}.png"
    path = output_dir / name
    cv2.imwrite(str(path), frame)
    return str(path)


# ══════════════════════════════════════════════════════════════════════════════
# [MULTI-PERSON] Per-track pose buffer
# ══════════════════════════════════════════════════════════════════════════════
class PersonTrackBuffer:
    """
    Maintains an independent rolling pose-history buffer for ONE tracked
    person. Each person detected by ByteTrack gets one of these, keyed by
    their persistent track ID. This is what makes per-person violence
    scoring possible instead of collapsing everyone into a single buffer.
    """

    def __init__(self, track_id: int, sequence_length: int):
        self.track_id        = track_id
        self.sequence_length = sequence_length
        self.buffer: collections.deque[np.ndarray] = collections.deque(maxlen=sequence_length)

        self.last_bbox          = None   # (x1, y1, x2, y2), most recent frame
        self.last_seen_frame    = 0       # frame_idx this track was last updated
        self.violence_prob      = 0.0
        self.last_screenshot    = 0.0     # epoch-seconds, for per-person cooldown

    def add_pose(self, features: np.ndarray):
        self.buffer.append(features)

    def is_ready(self) -> bool:
        return len(self.buffer) == self.sequence_length

    def get_sequence(self) -> np.ndarray:
        return np.array(list(self.buffer), dtype=np.float32)


class MultiPersonTracker:
    """
    [MULTI-PERSON] Owns one PersonTrackBuffer per active ByteTrack ID.
    Handles creating new buffers for new people, and pruning tracks that
    haven't been seen for a while (e.g. someone walked out of frame).
    """

    def __init__(self, sequence_length: int, max_lost_frames: int = 30):
        self.sequence_length  = sequence_length
        self.max_lost_frames  = max_lost_frames
        self.tracks: dict[int, PersonTrackBuffer] = {}

    def get_or_create(self, track_id: int) -> PersonTrackBuffer:
        if track_id not in self.tracks:
            self.tracks[track_id] = PersonTrackBuffer(track_id, self.sequence_length)
        return self.tracks[track_id]

    def prune(self, current_frame: int):
        """Drop tracks not updated within max_lost_frames — person left frame."""
        stale = [
            tid for tid, buf in self.tracks.items()
            if (current_frame - buf.last_seen_frame) > self.max_lost_frames
        ]
        for tid in stale:
            del self.tracks[tid]


# ══════════════════════════════════════════════════════════════════════════════
# Pose landmark extraction
# ══════════════════════════════════════════════════════════════════════════════
def extract_landmarks(results) -> np.ndarray | None:
    """[REQ-4] Landmark vector from ROI-based pose results."""
    if not results or not results.pose_landmarks:
        return None
    features = []
    for lm in results.pose_landmarks.landmark:
        features.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(features, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Main inference loop
# ══════════════════════════════════════════════════════════════════════════════
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cpu":
        log.warning(
            "Running on CPU. YOLO + per-person MediaPipe + transformer "
            "calls are all CPU-bound here — multi-person lag is expected "
            "to be significantly worse than single-person mode. Consider "
            "--max_persons 3-5, a smaller --yolo_model, or a CUDA GPU."
        )

    # ── Load models ───────────────────────────────────────────────────────────
    log.info(f"Loading YOLO: {args.yolo_model}")
    yolo_model = YOLO(args.yolo_model)
    yolo_model.to(device)

    log.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt["model_config"]

    transformer = ViolenceTransformer(
        feature_dim=cfg["feature_dim"],
        embedding_dim=cfg["embedding_dim"],
        num_heads=cfg["num_heads"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        num_classes=2,
    ).to(device)
    transformer.load_state_dict(ckpt["model_state_dict"])
    transformer.eval()
    log.info("Transformer loaded")

    # [MULTI-PERSON] static_image_mode=True because we run pose estimation
    # independently on EACH person's crop every frame — these crops aren't
    # a continuous single-subject stream, so MediaPipe's internal temporal
    # smoothing (designed for one tracked subject) would do more harm than
    # good when applied across different people's crops in sequence.
    pose_model = MP_POSE.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── [PHONE-CAM] Camera / Video I/O ───────────────────────────────────────
    if args.video:
        source = args.video
        log.info(f"Opening video file: {source}")
    else:
        source = resolve_camera_source(args.camera)
        log.info(f"Opening camera source: {source!r}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        log.error(f"Cannot open source: {source}")
        log.error(
            "Phone-cam tips:\n"
            "  DroidCam USB  → use device index e.g. --camera 1 or --camera 2\n"
            "  DroidCam WiFi → --camera http://<phone-ip>:4747/video\n"
            "  IP Webcam     → --camera http://<phone-ip>:8080/video\n"
            "  EpocCam       → --camera http://<phone-ip>:2431/live\n"
            "  RTSP stream   → --camera rtsp://<phone-ip>:8554/stream"
        )
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info(f"Stream: {width}×{height} @ {fps:.1f} fps")

    out_writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        log.info(f"Writing annotated output to: {args.output}")

    # ── [SCREENSHOT] Setup ────────────────────────────────────────────────────
    screenshot_dir = Path(args.screenshot_dir)

    # [MULTI-PERSON] One tracker manages a separate buffer per person ID,
    # instead of a single global frame_buffer shared by everyone in frame.
    seq_length = cfg.get("sequence_length", 30)
    tracker = MultiPersonTracker(sequence_length=seq_length, max_lost_frames=int(fps * 1.5))

    # [SCREENSHOT] cooldown is now tracked globally across all people, so a
    # frame with multiple simultaneous violent detections doesn't spam one
    # screenshot per person — one shot per cooldown window is enough evidence.
    last_screenshot = 0.0

    frame_idx = 0
    fps_timestamps: collections.deque[float] = collections.deque(maxlen=30)

    log.info("Starting inference loop — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        annotated = frame.copy()

        # ── [REQ-3, MULTI-PERSON] YOLO detection + ByteTrack on full frame ────
        # persist=True keeps track IDs stable across frames; tracker="bytetrack.yaml"
        # selects the ByteTrack algorithm bundled with ultralytics.
        yolo_results = yolo_model.track(
            frame,
            conf=args.yolo_conf,
            persist=True,
            classes=[0],                 # person class only
            tracker="bytetrack.yaml",
            verbose=False,
        )
        result = yolo_results[0]

        any_violence_this_frame = False

        if result.boxes is not None and len(result.boxes) > 0:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            confs      = result.boxes.conf.cpu().numpy()

            # ByteTrack may not have assigned IDs yet on the very first
            # frame(s) — fall back to per-detection index in that case so
            # the pipeline doesn't crash, even though tracking won't be
            # stable until IDs are assigned on a later frame.
            if result.boxes.id is not None:
                track_ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                track_ids = np.arange(len(boxes_xyxy))

            # [SAFETY] Cap simultaneous tracking to the highest-confidence
            # detections if a crowd exceeds --max_persons, to keep per-frame
            # cost (one MediaPipe + one transformer call per person) bounded.
            if len(boxes_xyxy) > args.max_persons:
                order      = np.argsort(confs)[::-1][:args.max_persons]
                boxes_xyxy = boxes_xyxy[order]
                confs      = confs[order]
                track_ids  = track_ids[order]

            for (x1, y1, x2, y2), conf, tid in zip(boxes_xyxy, confs, track_ids):
                tid = int(tid)

                x1 = max(0, int(x1)); y1 = max(0, int(y1))
                x2 = min(frame_w, int(x2)); y2 = min(frame_h, int(y2))

                buf = tracker.get_or_create(tid)
                buf.last_bbox       = (x1, y1, x2, y2)
                buf.last_seen_frame = frame_idx

                # ── [REQ-3] Crop ONLY this person's bounding-box ROI ─────────
                crop = frame[y1:y2, x1:x2]

                # [PERF] Optionally skip pose extraction on some frames to
                # cut MediaPipe's CPU cost roughly in half (or more) when
                # --pose_every_n_frames > 1. The buffer simply gets updated
                # less often; the transformer still sees a valid (slightly
                # less time-dense) sequence once the buffer fills.
                run_pose = (frame_idx % args.pose_every_n_frames) == 0

                if run_pose and crop.size > 0 and crop.shape[0] >= 4 and crop.shape[1] >= 4:
                    # ── [REQ-4] Pose estimation on ROI only ──────────────────
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pose_results = pose_model.process(crop_rgb)
                    features = extract_landmarks(pose_results)

                    if features is not None:
                        buf.add_pose(features)

            # ── [PERF] Batch transformer inference across ALL ready people ───
            # Instead of one model(x) call per person, stack every ready
            # person's sequence into a single batch and run ONE forward pass.
            # This is the single biggest fix for multi-person lag: 5 people
            # ready at once goes from 5 separate GPU/CPU round-trips to 1.
            ready_ids  = []
            ready_seqs = []
            for tid in [int(t) for t in track_ids]:
                buf = tracker.tracks.get(tid)
                if buf is not None and buf.is_ready():
                    ready_ids.append(tid)
                    ready_seqs.append(buf.get_sequence())

            if ready_seqs:
                batch = torch.from_numpy(np.stack(ready_seqs, axis=0)).to(device)  # (N, T, F)
                with torch.no_grad():
                    logits = transformer(batch)
                    probs  = torch.softmax(logits, dim=1)[:, 1]  # (N,) violence prob per person
                for tid, p in zip(ready_ids, probs.tolist()):
                    tracker.tracks[tid].violence_prob = p

            for (x1, y1, x2, y2), conf, tid in zip(boxes_xyxy, confs, track_ids):
                tid = int(tid)
                buf = tracker.tracks[tid]
                x1, y1, x2, y2 = buf.last_bbox

                is_violent = buf.is_ready() and buf.violence_prob >= args.threshold

                if is_violent:
                    any_violence_this_frame = True
                    colour = (0, 0, 255)   # Red
                    label  = f"ID:{tid} VIOLENCE {buf.violence_prob*100:.0f}%"
                elif buf.is_ready():
                    colour = (0, 200, 0)   # Green
                    label  = f"ID:{tid} Safe {(1 - buf.violence_prob)*100:.0f}%"
                else:
                    colour = (0, 165, 255)  # Orange — still buffering
                    filled = len(buf.buffer)
                    label  = f"ID:{tid} Buffering {filled}/{seq_length}"

                # ── Per-person resolution-aware box + label ──────────────────
                cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

                ref_dim    = min(frame_w, frame_h)
                font_scale = max(0.4, min(1.2, (ref_dim / 480) * 0.6))
                thickness  = max(1, round(font_scale * 1.8))

                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
                )
                label_y = max(y1 - 6, text_h + 6)
                cv2.rectangle(
                    annotated,
                    (x1, label_y - text_h - baseline - 4),
                    (x1 + text_w + 6, label_y + baseline),
                    (0, 0, 0),
                    cv2.FILLED,
                )
                cv2.putText(annotated, label, (x1 + 3, label_y - baseline),
                            cv2.FONT_HERSHEY_DUPLEX, font_scale, colour, thickness)

        # [MULTI-PERSON] Drop tracks for people who left the frame a while ago
        tracker.prune(frame_idx)

        # ── [SCREENSHOT] One shot per cooldown window if ANYONE is violent ────
        if any_violence_this_frame and not args.disable_screenshots:
            now = time.time()
            if now - last_screenshot >= args.screenshot_cooldown:
                saved_path = save_screenshot(annotated, screenshot_dir, frame_idx)
                log.info(f"[SCREENSHOT] Violence detected — saved: {saved_path}")
                last_screenshot = now

        # ── [PERF] FPS counter (top-left, separate from status banner) ────────
        fps_timestamps.append(time.perf_counter())
        if len(fps_timestamps) >= 2:
            elapsed_fps = fps_timestamps[-1] - fps_timestamps[0]
            current_fps = (len(fps_timestamps) - 1) / elapsed_fps if elapsed_fps > 0 else 0.0
        else:
            current_fps = 0.0
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, frame_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.5, min(1.0, frame_w / 1000)),
                    (0, 255, 255), 2)

        # ── Overall frame status banner (resolution-aware) ─────────────────────
        status_text  = "! VIOLENCE DETECTED !" if any_violence_this_frame else "Scene Clear"
        status_colour = (0, 0, 255) if any_violence_this_frame else (0, 200, 0)

        ref_dim    = min(frame_w, frame_h)
        font_scale = max(0.5, min(2.5, (ref_dim / 480) * 1.2))
        thickness  = max(1, round(font_scale * 1.8))
        margin_x   = max(10, int(frame_w * 0.015))
        margin_y   = max(30, int(frame_h * 0.06))

        (text_w, text_h), baseline = cv2.getTextSize(
            status_text, cv2.FONT_HERSHEY_DUPLEX, font_scale, thickness
        )
        cv2.rectangle(
            annotated,
            (margin_x - 6, margin_y - text_h - 6),
            (margin_x + text_w + 6, margin_y + baseline + 4),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(annotated, status_text, (margin_x, margin_y),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, status_colour, thickness)

        if out_writer:
            out_writer.write(annotated)

        # ── [PHONE-CAM] Live preview (skip when processing a video file) ──────
        if not args.video or args.show:
            cv2.imshow("Violence Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                log.info("User quit")
                break

        frame_idx += 1

    cap.release()
    if out_writer:
        out_writer.release()
    pose_model.close()
    cv2.destroyAllWindows()
    log.info(f"Inference complete — {frame_idx} frames processed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Violence Detection — Inference")

    # ── Source (mutually exclusive: file OR camera) ───────────────────────────
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--video",
        default=None,
        help="Input video FILE path.  Omit to use a live camera.",
    )
    src.add_argument(
        "--camera",
        default="0",
        metavar="ID_OR_URL",
        help=(
            "Camera source.  Can be:\n"
            "  Integer        : device index (0 = default webcam, 1 = next, …)\n"
            "  /dev/videoN    : Linux V4L2 device path\n"
            "  URL            : IP-camera stream\n"
            "                   DroidCam WiFi  → http://<phone-ip>:4747/video\n"
            "                   IP Webcam      → http://<phone-ip>:8080/video\n"
            "                   EpocCam        → http://<phone-ip>:2431/live\n"
            "                   RTSP           → rtsp://<phone-ip>:8554/stream\n"
            "Default: 0 (built-in webcam)."
        ),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument("--checkpoint",  required=True,           help="Path to best_model.pth")
    parser.add_argument("--yolo_model",  default="yolov8n.pt",    help="YOLO model weights")
    parser.add_argument("--yolo_conf",   type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--threshold",   type=float, default=0.598, help="Violence probability threshold")
    parser.add_argument("--max_persons", type=int, default=15,
                        help="Safety cap on simultaneously tracked people (default: 15)")
    parser.add_argument("--pose_every_n_frames", type=int, default=1,
                        help="Run MediaPipe pose extraction every Nth frame instead of "
                             "every frame (default: 1 = every frame). Set to 2 or 3 to "
                             "roughly halve/third pose-extraction cost on CPU at the cost "
                             "of slightly coarser temporal resolution.")

    # ── Output ────────────────────────────────────────────────────────────────
    parser.add_argument("--output",      default=None,            help="Annotated output video path")
    parser.add_argument(
        "--screenshot_dir",
        default="screenshots",
        help="Directory to save auto-captured violence screenshots (default: ./screenshots)",
    )
    parser.add_argument(
        "--screenshot_cooldown",
        type=float,
        default=3.0,
        help="Minimum seconds between screenshots to avoid duplicates (default: 3.0)",
    )
    parser.add_argument(
        "--disable_screenshots",
        action="store_true",
        help="Disable automatic screenshots on violence detection (screenshots are ON by default)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window even when --video is given",
    )

    args = parser.parse_args()
    main(args)
