"""
detect_violence.py  —  Real-Time Violence Detection Inference
==============================================================
MODIFICATIONS (vs original):
  [REQ-3] YOLO runs on each frame first; only the best-confidence person
           bounding box ROI is passed to MediaPipe for pose extraction.
  [REQ-4] Pose landmarks are computed from the ROI crop, not the full frame.
  Edge cases handled: no person detected, empty crop, missing landmarks.

NEW FEATURES:
  [PHONE-CAM] --camera accepts a device index (int), /dev/videoN path, or a
              full URL for IP-camera apps (DroidCam, EpocCam, Camo, etc.).
  [SCREENSHOT] When violence is detected, a timestamped PNG is saved
               automatically to --screenshot_dir.  A --screenshot_cooldown
               (default 3 s) prevents duplicate saves within bursts.
               Pass --disable_screenshots to turn this off entirely.
  [ALARM]      When violence is detected, an audible alarm sound plays
               automatically (on a background thread, so it never blocks
               video processing). A --alarm_cooldown (default 5 s) avoids
               a continuous siren during a sustained violent event.
               Pass --disable_alarm to turn this off entirely.
  [TEXT-VIS]  Overlay text uses a dark shadow for readability at any
              resolution.  Font scale reduced to 0.7, thickness to 1.
"""

import argparse
import collections
import logging
import platform
import threading
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
# [ALARM] Audible alert helper
# ══════════════════════════════════════════════════════════════════════════════
def play_alarm_sound():
    """
    [ALARM] Play a short audible alert tone.

    Runs synchronously within the calling thread — callers should invoke
    this via a background thread (see play_alarm_async) so the video
    processing loop never blocks waiting for the sound to finish.

    Platform behavior:
        Windows   → winsound.Beep(), built into the standard library,
                    no extra dependency or audio file needed.
        Mac/Linux → falls back to the terminal bell character. This is
                    deliberately simple rather than pulling in a third-party
                    audio library (e.g. simpleaudio, playsound) as a hard
                    dependency just for an alert beep.
    """
    try:
        if platform.system() == "Windows":
            import winsound
            # Three quick beeps — frequency (Hz), duration (ms) each
            for _ in range(3):
                winsound.Beep(1500, 200)
        else:
            # Terminal bell — works in most terminals on Mac/Linux without
            # any extra dependency.
            for _ in range(3):
                print("\a", end="", flush=True)
                time.sleep(0.2)
    except Exception as e:
        # [EDGE CASE] Some environments (headless servers, certain SSH
        # sessions, containers without an audio device) can't play sound
        # at all. Never let an alarm failure crash the detection pipeline.
        log.warning(f"[ALARM] Could not play alarm sound: {e}")


def play_alarm_async():
    """
    [ALARM] Fire play_alarm_sound() on a background daemon thread so the
    main video loop is never blocked waiting for beeps to finish playing.
    """
    t = threading.Thread(target=play_alarm_sound, daemon=True)
    t.start()


# ══════════════════════════════════════════════════════════════════════════════
# Reuse helpers from preprocess.py (duplicated here so the script is standalone)
# ══════════════════════════════════════════════════════════════════════════════
def get_best_person_box(yolo_result, frame_h: int, frame_w: int):
    """[REQ-3] Return highest-confidence person bbox or None."""
    boxes = yolo_result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    person_mask = boxes.cls.cpu().numpy() == 0
    if not person_mask.any():
        return None
    confs = boxes.conf.cpu().numpy()[person_mask]
    xyxys = boxes.xyxy.cpu().numpy()[person_mask]
    best  = int(np.argmax(confs))
    x1, y1, x2, y2 = xyxys[best]
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(frame_w, int(x2)); y2 = min(frame_h, int(y2))
    return x1, y1, x2, y2


def get_sticky_person_box(yolo_result, frame_h: int, frame_w: int, prev_bbox):
    """
    [FIX: identity-jump false positives] Pick a person box that's
    physically close to the previously tracked box, instead of always
    jumping to whichever detection YOLO is most confident about THIS frame.

    WHY THIS MATTERS in single-person mode (no real multi-object tracker):
    get_best_person_box() re-picks "most confident" every frame with zero
    memory of who was being tracked. If multiple people are in the scene,
    the "most confident" detection can silently switch between two
    DIFFERENT physical people frame to frame. Even if both are sitting
    perfectly still, that switch creates a sudden large jump in the pose
    coordinates fed into the model — which looks like fast, erratic motion
    and can trigger a false "violence" classification, even though no one
    actually moved.

    Strategy: if a previous box exists, prefer the candidate whose CENTER
    is closest to the previous box's center, as long as it's within a
    reasonable distance (relative to frame size). Only fall back to
    "highest confidence, no matter where" if there's no previous box yet,
    or nothing is close enough to be plausibly the same person.

    Args:
        yolo_result: single YOLO result for this frame
        frame_h, frame_w: frame dimensions, for clipping and distance scaling
        prev_bbox: (x1, y1, x2, y2) from the previous frame, or None

    Returns:
        (x1, y1, x2, y2) or None
    """
    boxes = yolo_result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    person_mask = boxes.cls.cpu().numpy() == 0
    if not person_mask.any():
        return None

    confs = boxes.conf.cpu().numpy()[person_mask]
    xyxys = boxes.xyxy.cpu().numpy()[person_mask]

    def clip(x1, y1, x2, y2):
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(frame_w, int(x2)); y2 = min(frame_h, int(y2))
        return x1, y1, x2, y2

    # No previous track yet — just take the most confident detection,
    # same behavior as the original get_best_person_box().
    if prev_bbox is None:
        best = int(np.argmax(confs))
        return clip(*xyxys[best])

    px1, py1, px2, py2 = prev_bbox
    prev_cx, prev_cy = (px1 + px2) / 2, (py1 + py2) / 2

    # Max allowed jump between frames, relative to frame size. A real
    # person moving normally between consecutive frames won't cross a
    # large fraction of the frame; a different person standing/sitting
    # elsewhere in the scene typically will.
    max_jump = 0.25 * max(frame_w, frame_h)

    best_idx, best_dist = None, float("inf")
    for i, (x1, y1, x2, y2) in enumerate(xyxys):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best_idx = dist, i

    if best_idx is not None and best_dist <= max_jump:
        # Closest candidate is plausibly the same person — stick with them
        # even if a DIFFERENT person elsewhere has higher confidence.
        return clip(*xyxys[best_idx])

    # Nothing nearby — the tracked person likely left frame, or this is
    # too large a jump to trust. Fall back to highest confidence (this
    # may pick up a different person, but that's unavoidable without a
    # real re-identification model; at least it won't silently happen
    # every other frame when the original person is still right there).
    best = int(np.argmax(confs))
    return clip(*xyxys[best])


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

    pose_model = MP_POSE.Pose(
        static_image_mode=False,
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
    screenshot_dir  = Path(args.screenshot_dir)
    last_screenshot = 0.0          # epoch-seconds of last saved screenshot

    # ── [ALARM] Setup ─────────────────────────────────────────────────────────
    last_alarm = 0.0               # epoch-seconds of last alarm trigger

    # Sliding window buffer of pose frames
    seq_length = cfg.get("sequence_length", 30)
    frame_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=seq_length)

    # ── [SMOOTHING] Rolling buffer of recent violence probabilities ───────────
    # Classifying on every single frame re-evaluates an almost-identical
    # 30-frame window 30+ times per second. Near the decision threshold,
    # normal pose-estimation jitter (worse over DroidCam/Wi-Fi due to video
    # compression) is enough to flip the raw prediction back and forth even
    # when nothing meaningful changed. Averaging the last N raw probabilities
    # before deciding the DISPLAYED label smooths out that frame-to-frame
    # noise without dulling genuine, sustained changes in motion.
    prob_history: collections.deque[float] = collections.deque(maxlen=args.smoothing_window)

    prediction_label = "Initialising…"
    prediction_conf  = 0.0
    frame_idx        = 0
    prev_bbox        = None   # [FIX] tracked person's last box, for stickiness

    log.info("Starting inference loop — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        annotated = frame.copy()

        # ── [REQ-3] YOLO person detection on full frame ───────────────────────
        yolo_results = yolo_model(frame, conf=args.yolo_conf, verbose=False)
        # [FIX] Prefer whoever is closest to the previously tracked person,
        # instead of always jumping to "most confident this frame" — this
        # prevents identity switches between different people from looking
        # like sudden violent motion to the model. See get_sticky_person_box.
        bbox = get_sticky_person_box(yolo_results[0], frame_h, frame_w, prev_bbox)
        prev_bbox = bbox if bbox is not None else prev_bbox

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            log.debug(f"Frame {frame_idx}: bbox=({x1},{y1},{x2},{y2})")

            # ── [REQ-3] Crop ONLY the bounding-box ROI ────────────────────────
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0 and crop.shape[0] >= 4 and crop.shape[1] >= 4:
                crop_h, crop_w = crop.shape[:2]
                log.debug(f"Frame {frame_idx}: crop dims {crop_w}×{crop_h}")

                # ── [REQ-4] Pose estimation on ROI only ───────────────────────
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                results  = pose_model.process(crop_rgb)
                features = extract_landmarks(results)   # ROI-relative landmarks

                if features is not None:
                    frame_buffer.append(features)

                # Draw bbox on annotated frame
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, "Person", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ── Classify when buffer is full ──────────────────────────────────────
        if len(frame_buffer) == seq_length:
            seq = np.array(list(frame_buffer), dtype=np.float32)  # (T, F)
            x_t = torch.from_numpy(seq).unsqueeze(0).to(device)   # (1, T, F)

            with torch.no_grad():
                logits = transformer(x_t)
                probs  = torch.softmax(logits, dim=1)[0]
                raw_violence_prob = probs[1].item()

            # ── [SMOOTHING] Average recent raw probabilities, decide on THAT ──
            # The raw per-frame probability is still computed every frame
            # (so the model always sees fresh data), but the displayed label,
            # screenshot trigger, and alarm trigger all act on the smoothed
            # average instead — this is what actually kills the flicker.
            prob_history.append(raw_violence_prob)
            violence_prob = sum(prob_history) / len(prob_history)

            if violence_prob >= args.threshold:
                prediction_label = "VIOLENCE"
                prediction_conf  = violence_prob
                colour = (0, 0, 255)   # Red

                # ── [SCREENSHOT] Auto-capture on violence ─────────────────────
                if not args.disable_screenshots:
                    now = time.time()
                    if now - last_screenshot >= args.screenshot_cooldown:
                        saved_path = save_screenshot(annotated, screenshot_dir, frame_idx)
                        log.info(f"[SCREENSHOT] Violence detected — saved: {saved_path}")
                        last_screenshot = now

                # ── [ALARM] Auto-trigger audible alert on violence ────────────
                # Runs on a background thread (play_alarm_async) so the
                # beep/tone never blocks the next frame from being read.
                if not args.disable_alarm:
                    now = time.time()
                    if now - last_alarm >= args.alarm_cooldown:
                        play_alarm_async()
                        log.info("[ALARM] Violence detected — alarm triggered")
                        last_alarm = now

            else:
                prediction_label = "Non-Violence"
                prediction_conf  = 1.0 - violence_prob
                colour = (0, 200, 0)   # Green

            # Overlay prediction  [TEXT-VIS] font scales with frame width
            text       = f"{prediction_label}  ({prediction_conf*100:.1f}%)"
            font       = cv2.FONT_HERSHEY_DUPLEX
            font_scale = max(0.35, min(0.9, frame_w / 1280 * 0.6))
            thickness  = 1
            origin     = (12, 50)
            # Black shadow one pixel down-right for contrast on any background
            cv2.putText(annotated, text, (origin[0] + 1, origin[1] + 1),
                        font, font_scale, (0, 0, 0), thickness + 1)
            cv2.putText(annotated, text, origin,
                        font, font_scale, colour, thickness)

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
    parser.add_argument(
        "--smoothing_window",
        type=int,
        default=5,
        help="Number of recent per-frame probabilities to average before "
             "deciding the displayed label. Higher = more stable but slower "
             "to react to genuine changes; 1 = no smoothing, raw per-frame "
             "behavior (default: 5)",
    )

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
        "--alarm_cooldown",
        type=float,
        default=5.0,
        help="Minimum seconds between alarm triggers to avoid a continuous "
             "siren during a sustained violent event (default: 5.0)",
    )
    parser.add_argument(
        "--disable_alarm",
        action="store_true",
        help="Disable the audible alarm on violence detection (alarm is ON by default)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show live preview window even when --video is given",
    )

    args = parser.parse_args()
    main(args)
