"""
=============================================================
utils/tracker.py
=============================================================
Multi-person tracking using ByteTrack (via Ultralytics).
Each person gets a unique persistent ID across frames.
Each ID maintains its own 30-frame pose history buffer.
Violence is scored independently per tracked person.
=============================================================
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import time


class PersonTrackBuffer:
    """
    Maintains pose history for a single tracked person.

    Holds the last `sequence_length` pose feature vectors (each 132 floats).
    When full, the transformer can run a violence prediction.
    """

    def __init__(self, track_id: int, sequence_length: int = 30):
        self.track_id        = track_id
        self.sequence_length = sequence_length

        # Rolling buffer — automatically drops oldest when full
        self.pose_buffer = deque(maxlen=sequence_length)

        # Latest bounding box for drawing
        self.last_bbox        = None     # (x1, y1, x2, y2)
        self.last_confidence  = 0.0

        # Latest raw pose features for skeleton drawing
        self.last_pose_features = None   # numpy array (132,) or None

        # Violence prediction
        self.violence_score          = 0.0
        self.is_violent              = False
        self.last_prediction_time    = 0

        # For pruning lost tracks
        self.last_seen_frame = 0

    def add_pose(self, pose_features: np.ndarray):
        """Add a pose feature vector (132,) to the rolling buffer."""
        self.pose_buffer.append(pose_features.copy())
        self.last_pose_features = pose_features.copy()

    def add_empty_pose(self):
        """Add zeros when person is detected but pose extraction failed."""
        zeros = np.zeros(132, dtype=np.float32)
        self.pose_buffer.append(zeros)
        # Don't overwrite last_pose_features with zeros — keep last real pose

    def is_ready_for_prediction(self) -> bool:
        """True when buffer has enough frames for transformer input."""
        return len(self.pose_buffer) >= self.sequence_length

    def get_sequence(self) -> Optional[np.ndarray]:
        """
        Return pose sequence as numpy array (sequence_length, 132).
        Returns None if buffer not full yet.
        """
        if not self.is_ready_for_prediction():
            return None
        return np.array(list(self.pose_buffer), dtype=np.float32)

    def update_prediction(self, violence_score: float, threshold: float = 0.5):
        """Update violence score and binary flag."""
        self.violence_score       = float(violence_score)
        self.is_violent           = self.violence_score > threshold
        self.last_prediction_time = time.time()

    def __repr__(self):
        return (f"PersonTrack(id={self.track_id}, "
                f"frames={len(self.pose_buffer)}/{self.sequence_length}, "
                f"violence={self.violence_score:.2f})")


class MultiPersonTracker:
    """
    Manages pose history buffers for ALL persons in a video.

    Receives YOLO+ByteTrack detections each frame, routes each
    detection to the correct PersonTrackBuffer, and prunes
    tracks that haven't been seen recently.
    """

    def __init__(
        self,
        sequence_length: int = 30,
        max_lost_frames: int = 30,
    ):
        """
        Args:
            sequence_length: frames needed per person for prediction
            max_lost_frames: remove track after this many frames without detection
        """
        self.sequence_length  = sequence_length
        self.max_lost_frames  = max_lost_frames
        self.tracks: Dict[int, PersonTrackBuffer] = {}
        self.current_frame    = 0
        self.total_tracks_created = 0

    def update(
        self,
        track_id: int,
        bbox: Tuple[int, int, int, int],
        pose_features: Optional[np.ndarray],
        confidence: float = 1.0,
    ):
        """
        Update tracker with a new detection for one person.

        Args:
            track_id:      unique ID from ByteTrack
            bbox:          (x1, y1, x2, y2)
            pose_features: (132,) array or None if pose failed
            confidence:    YOLO detection confidence
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = PersonTrackBuffer(
                track_id=track_id,
                sequence_length=self.sequence_length,
            )
            self.total_tracks_created += 1

        buf = self.tracks[track_id]
        buf.last_bbox        = bbox
        buf.last_confidence  = confidence
        buf.last_seen_frame  = self.current_frame

        if pose_features is not None:
            buf.add_pose(pose_features)
        else:
            buf.add_empty_pose()

    def tick(self):
        """Advance frame counter and prune lost tracks. Call once per frame."""
        self.current_frame += 1
        self._prune_lost_tracks()

    def _prune_lost_tracks(self):
        lost = [
            tid for tid, buf in self.tracks.items()
            if (self.current_frame - buf.last_seen_frame) > self.max_lost_frames
        ]
        for tid in lost:
            del self.tracks[tid]

    def get_ready_tracks(self) -> List[PersonTrackBuffer]:
        """All tracks with enough frames for violence prediction."""
        return [b for b in self.tracks.values() if b.is_ready_for_prediction()]

    def get_all_tracks(self) -> List[PersonTrackBuffer]:
        """All currently active tracks."""
        return list(self.tracks.values())

    def update_violence_scores(
        self,
        track_ids: List[int],
        scores: List[float],
        threshold: float = 0.5,
    ):
        """Batch-update violence scores after model inference."""
        for tid, score in zip(track_ids, scores):
            if tid in self.tracks:
                self.tracks[tid].update_prediction(score, threshold)

    def get_stats(self) -> dict:
        return {
            "active_tracks":        len(self.tracks),
            "total_created":        self.total_tracks_created,
            "current_frame":        self.current_frame,
            "ready_for_prediction": len(self.get_ready_tracks()),
        }

    def reset(self):
        """Clear all tracks (use between videos)."""
        self.tracks.clear()
        self.current_frame = 0
