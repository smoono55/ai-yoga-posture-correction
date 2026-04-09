# pose_estimator.py — MediaPipe-based human pose estimator

from __future__ import annotations
import numpy as np
import mediapipe as mp
import cv2
from dataclasses import dataclass, field
from typing import Optional

import config


@dataclass
class Landmark:
    """Normalised (0-1) landmark coordinates + visibility score."""
    x: float
    y: float
    z: float
    visibility: float


@dataclass
class PoseResult:
    """Result of a single pose-estimation pass."""
    landmarks: dict[str, Landmark] = field(default_factory=dict)
    detected: bool = False

    def get(self, name: str) -> Optional[Landmark]:
        return self.landmarks.get(name)


class PoseEstimator:
    """
    Wraps MediaPipe Pose to extract named landmarks from a BGR frame.

    Usage
    -----
    estimator = PoseEstimator()
    result: PoseResult = estimator.estimate(frame)
    """

    _mp_pose = mp.solutions.pose

    def __init__(self) -> None:
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MEDIAPIPE_MIN_TRACKING_CONFIDENCE,
        )
        # Build a reverse mapping: index → landmark name
        self._index_to_name: dict[int, str] = {
            v: k for k, v in config.CRITICAL_LANDMARKS.items()
        }

    def estimate(self, frame_bgr: np.ndarray) -> PoseResult:
        """
        Run pose estimation on a single BGR frame.

        Parameters
        ----------
        frame_bgr : np.ndarray
            OpenCV BGR image.

        Returns
        -------
        PoseResult
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return PoseResult(detected=False)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        rgb.flags.writeable = True

        if not results.pose_landmarks:
            return PoseResult(detected=False)

        landmarks: dict[str, Landmark] = {}
        all_lm = results.pose_landmarks.landmark

        for name, idx in config.CRITICAL_LANDMARKS.items():
            lm = all_lm[idx]
            landmarks[name] = Landmark(
                x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility
            )

        return PoseResult(landmarks=landmarks, detected=True)

    def close(self) -> None:
        self._pose.close()

    # ------------------------------------------------------------------
    # Convenience: draw skeleton on a frame (for debug windows)
    # ------------------------------------------------------------------
    def draw_landmarks(
        self, frame_bgr: np.ndarray, result: PoseResult
    ) -> np.ndarray:
        """Return a copy of frame with landmark dots drawn (debug use)."""
        if not result.detected:
            return frame_bgr
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]
        for lm in result.landmarks.values():
            if lm.visibility >= config.MIN_LANDMARK_VISIBILITY:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
        return vis
