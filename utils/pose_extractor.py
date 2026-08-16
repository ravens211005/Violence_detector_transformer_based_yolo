"""
=============================================================
utils/pose_extractor.py
=============================================================
Extracts 33 MediaPipe Pose landmarks from a person crop.
Returns 132 features per frame: [x, y, z, visibility] × 33
=============================================================
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional

NUM_LANDMARKS      = 33
VALUES_PER_LANDMARK = 4
POSE_FEATURE_DIM   = NUM_LANDMARKS * VALUES_PER_LANDMARK  # 132


class PoseExtractor:
    """
    Wraps MediaPipe Pose for clean, reusable pose extraction.

    Usage:
        extractor = PoseExtractor()
        features = extractor.extract(person_crop_bgr)
        # returns numpy array (132,) or None if no pose detected
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        print(f"[PoseExtractor] Initialized | complexity={model_complexity} | "
              f"min_conf={min_detection_confidence}")

    def extract(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose landmarks from a single person crop.

        Args:
            image_bgr: OpenCV BGR image cropped to one person

        Returns:
            numpy array (132,) with [x,y,z,vis] for all 33 landmarks,
            or None if no pose detected.
        """
        if image_bgr is None or image_bgr.size == 0:
            return None

        h, w = image_bgr.shape[:2]
        if h < 50 or w < 50:
            return None

        # MediaPipe needs RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results   = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return None

        features = np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
        for i, lm in enumerate(results.pose_landmarks.landmark):
            base = i * VALUES_PER_LANDMARK
            features[base + 0] = lm.x
            features[base + 1] = lm.y
            features[base + 2] = lm.z
            features[base + 3] = lm.visibility

        return features

    def extract_batch(self, images: list) -> list:
        """Extract poses from a list of crops. Returns list of arrays or Nones."""
        return [self.extract(img) for img in images]

    def close(self):
        self.pose.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def get_zero_pose_features() -> np.ndarray:
    """Return 132 zeros — used for padding when no pose is detected."""
    return np.zeros(POSE_FEATURE_DIM, dtype=np.float32)


def get_landmark_names() -> list:
    """Names of all 33 MediaPipe Pose landmarks."""
    return [
        "nose", "left_eye_inner", "left_eye", "left_eye_outer",
        "right_eye_inner", "right_eye", "right_eye_outer",
        "left_ear", "right_ear", "mouth_left", "mouth_right",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_pinky", "right_pinky",
        "left_index", "right_index", "left_thumb", "right_thumb",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_heel", "right_heel",
        "left_foot_index", "right_foot_index",
    ]
