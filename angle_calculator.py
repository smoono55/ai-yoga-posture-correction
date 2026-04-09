# angle_calculator.py — Compute joint angles from PoseResult landmarks

from __future__ import annotations
import numpy as np
from typing import Optional

import config
from pose_estimator import PoseResult, Landmark


def _angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the angle (in degrees) at *vertex* formed by rays vertex→a and vertex→b.

    Uses the dot-product formula: θ = arccos( (u · v) / (|u||v|) )
    Works in any dimensionality (2-D or 3-D).
    """
    u = a - vertex
    v = b - vertex

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u < 1e-9 or norm_v < 1e-9:
        return 0.0

    cos_theta = np.clip(np.dot(u, v) / (norm_u * norm_v), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _lm_to_array(lm: Landmark) -> np.ndarray:
    """Convert a Landmark to a 3-D numpy array."""
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def compute_joint_angles(result: PoseResult) -> dict[str, Optional[float]]:
    """
    Compute all joint angles defined in ``config.JOINT_DEFINITIONS``.

    Parameters
    ----------
    result : PoseResult
        Detected pose landmarks.

    Returns
    -------
    dict[str, Optional[float]]
        Mapping of joint name → angle in degrees, or None if any landmark
        needed for that joint is missing / invisible.
    """
    angles: dict[str, Optional[float]] = {}

    for joint_name, (side_a_name, vertex_name, side_b_name) in config.JOINT_DEFINITIONS.items():
        lm_a      = result.get(side_a_name)
        lm_vertex = result.get(vertex_name)
        lm_b      = result.get(side_b_name)

        # Skip if any landmark is absent or has low visibility
        if (
            lm_a is None or lm_vertex is None or lm_b is None
            or lm_a.visibility      < config.MIN_LANDMARK_VISIBILITY
            or lm_vertex.visibility < config.MIN_LANDMARK_VISIBILITY
            or lm_b.visibility      < config.MIN_LANDMARK_VISIBILITY
        ):
            angles[joint_name] = None
            continue

        a      = _lm_to_array(lm_a)
        vertex = _lm_to_array(lm_vertex)
        b      = _lm_to_array(lm_b)

        angles[joint_name] = _angle_between(a, vertex, b)

    return angles


def compare_angles(
    user_angles: dict[str, Optional[float]],
    ref_angles:  dict[str, Optional[float]],
    tolerance:   float = config.ANGLE_TOLERANCE_DEGREES,
) -> dict[str, dict]:
    """
    Compare the user's joint angles against the reference (instructor) angles.

    Parameters
    ----------
    user_angles : dict
        Angles from the user's webcam pose.
    ref_angles : dict
        Angles from the instructor's screen pose.
    tolerance : float
        Maximum angle difference (degrees) before a joint is flagged.

    Returns
    -------
    dict[str, dict]
        For each joint:
            "user"      – user angle (or None)
            "ref"       – reference angle (or None)
            "diff"      – absolute difference (or None)
            "misaligned"– True if |diff| > tolerance
            "direction" – "increase" | "decrease" | None
    """
    comparison: dict[str, dict] = {}

    all_joints = set(user_angles.keys()) | set(ref_angles.keys())

    for joint in all_joints:
        u = user_angles.get(joint)
        r = ref_angles.get(joint)

        if u is None or r is None:
            comparison[joint] = {
                "user": u, "ref": r, "diff": None,
                "misaligned": False, "direction": None,
            }
            continue

        diff = abs(u - r)
        misaligned = diff > tolerance
        direction  = None
        if misaligned:
            direction = "increase" if r > u else "decrease"

        comparison[joint] = {
            "user": round(u, 1),
            "ref":  round(r, 1),
            "diff": round(diff, 1),
            "misaligned": misaligned,
            "direction": direction,
        }

    return comparison
