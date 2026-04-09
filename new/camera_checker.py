# camera_checker.py — Validates webcam framing before pose comparison
# FIX: Reduced sideways false positives — now requires BOTH X separation
#      being too small AND Y asymmetry to confirm a sideways orientation.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

import config
from pose_estimator import PoseResult

try:
    from voice_strings import t as _t
except ImportError:
    def _t(k, **kw): return k  # fallback: return key


@dataclass
class CameraCheckResult:
    ok: bool
    issue:   Optional[str]
    message: Optional[str]


_BODY_BOX_LANDMARKS = [
    "nose", "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
_MIN_VISIBLE_FRACTION = 0.65   # relaxed from 0.70


class CameraChecker:
    def check(self, result: PoseResult) -> CameraCheckResult:
        if not result.detected:
            return CameraCheckResult(
                ok=False, issue="no_person_detected",
                message=_t("cam_cant_see"),
            )
        for fn in (
            self._check_landmark_visibility,
            self._check_body_size,
            self._check_body_within_frame,
            self._check_sideways,
        ):
            r = fn(result)
            if not r.ok:
                return r
        return CameraCheckResult(ok=True, issue=None, message=None)

    # ------------------------------------------------------------------ #

    def _check_landmark_visibility(self, result: PoseResult) -> CameraCheckResult:
        critical = list(config.CRITICAL_LANDMARKS.keys())
        visible  = sum(
            1 for n in critical
            if (lm := result.get(n)) and lm.visibility >= config.MIN_LANDMARK_VISIBILITY
        )
        if visible / len(critical) < _MIN_VISIBLE_FRACTION:
            return CameraCheckResult(
                ok=False, issue="insufficient_landmarks",
                message=_t("cam_parts_missing"),
            )
        return CameraCheckResult(ok=True, issue=None, message=None)

    def _check_body_size(self, result: PoseResult) -> CameraCheckResult:
        ys = [
            result.get(n).y for n in _BODY_BOX_LANDMARKS
            if result.get(n) and result.get(n).visibility >= config.MIN_LANDMARK_VISIBILITY
        ]
        if len(ys) < 4:
            return CameraCheckResult(ok=True, issue=None, message=None)
        ratio = max(ys) - min(ys)
        if ratio < config.MIN_BODY_FRAME_RATIO:
            return CameraCheckResult(
                ok=False, issue="too_far",
                message=_t("cam_too_far"),
            )
        if ratio > config.MAX_BODY_FRAME_RATIO:
            return CameraCheckResult(
                ok=False, issue="too_close",
                message=_t("cam_too_close"),
            )
        return CameraCheckResult(ok=True, issue=None, message=None)

    def _check_body_within_frame(self, result: PoseResult) -> CameraCheckResult:
        margin = 0.04
        sides  = []
        for name in _BODY_BOX_LANDMARKS:
            lm = result.get(name)
            if not lm or lm.visibility < config.MIN_LANDMARK_VISIBILITY:
                continue
            if lm.x < margin:         sides.append("left")
            elif lm.x > 1 - margin:   sides.append("right")
            if lm.y < margin:         sides.append("top")
            elif lm.y > 1 - margin:   sides.append("bottom")
        if not sides:
            return CameraCheckResult(ok=True, issue=None, message=None)
        dominant = max(set(sides), key=sides.count)
        msgs = {
            "left":   "Please shift a little to the right so your full body is visible.",
            "right":  "Please shift a little to the left so your full body is visible.",
            "top":    "Your head is cut off — lower the camera or move down slightly.",
            "bottom": "Your feet are cut off — raise the camera or move up slightly.",
        }
        return CameraCheckResult(ok=False, issue=f"cropped_{dominant}", message=msgs[dominant])

    def _check_sideways(self, result: PoseResult) -> CameraCheckResult:
        """
        FIX: Only flag sideways if BOTH conditions hold:
          1. Shoulder X-separation is very small (nearly overlapping)
          2. One shoulder is noticeably higher than the other (Y asymmetry)
        This prevents false positives when the user is facing the camera
        but their shoulders are close together (e.g., Mountain pose).
        """
        ls = result.get("left_shoulder")
        rs = result.get("right_shoulder")
        if not ls or not rs:
            return CameraCheckResult(ok=True, issue=None, message=None)
        if (ls.visibility < config.MIN_LANDMARK_VISIBILITY or
                rs.visibility < config.MIN_LANDMARK_VISIBILITY):
            return CameraCheckResult(ok=True, issue=None, message=None)

        x_dist = abs(ls.x - rs.x)
        y_diff = abs(ls.y - rs.y)

        # Must satisfy BOTH: nearly overlapping AND significant Y asymmetry
        if x_dist < config.SIDEWAYS_X_THRESHOLD and y_diff > config.SIDEWAYS_Y_THRESHOLD:
            return CameraCheckResult(
                ok=False, issue="sideways_orientation",
                message=_t("cam_sideways"),
            )
        return CameraCheckResult(ok=True, issue=None, message=None)
