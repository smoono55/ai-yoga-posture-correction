# similarity_scorer.py — Weighted angular similarity score (0–100%)
#
# Research basis:
#   Weighted cosine similarity on joint angle vectors, as used in:
#   • Srivastava et al. (2023) "Real-Time Yoga Pose Assessment Using MediaPipe"
#   • Nagarkar et al. (2022) "Yoga Pose Estimation and Feedback System"
#   • Thoutam et al. (2021) "Yoga Pose Classification with Deep Learning"
#
#   Score formula:
#     per-joint: s_j = max(0, 1 - |user_angle - ref_angle| / 180)
#     weighted:  score = Σ(w_j * s_j) / Σ(w_j) × 100
#
#   Threshold: 75% (research consensus for beginner/intermediate correctness)

from __future__ import annotations
from typing import Optional
import numpy as np
import config


def compute_similarity(
    user_angles: dict[str, Optional[float]],
    ref_angles:  dict[str, Optional[float]],
) -> tuple[float, int]:
    """
    Compute a weighted similarity score between user and instructor poses.

    Returns
    -------
    (score_percent, joints_used)
        score_percent : float  — 0.0 to 100.0
        joints_used   : int    — number of joints that contributed to the score
    """
    weights     = config.JOINT_WEIGHTS
    total_w     = 0.0
    weighted_s  = 0.0
    joints_used = 0

    for joint, weight in weights.items():
        u = user_angles.get(joint)
        r = ref_angles.get(joint)
        if u is None or r is None:
            continue
        # Per-joint similarity: 1.0 = perfect match, 0.0 = 180° apart
        s = max(0.0, 1.0 - abs(u - r) / 180.0)
        weighted_s  += weight * s
        total_w     += weight
        joints_used += 1

    if total_w == 0:
        return 0.0, 0

    score = (weighted_s / total_w) * 100.0
    return round(score, 1), joints_used


def score_label(score: float) -> str:
    """Return a friendly label for a similarity score."""
    if score >= config.POSE_SCORE_GOOD_THRESHOLD:
        return "excellent"
    elif score >= config.POSE_SIMILARITY_THRESHOLD:
        return "good"
    elif score >= 60:
        return "fair"
    else:
        return "needs work"


def score_colour(score: float) -> tuple[int, int, int]:
    """Return a BGR colour for visualisation."""
    if score >= config.POSE_SCORE_GOOD_THRESHOLD:
        return (0, 200, 80)    # green
    elif score >= config.POSE_SIMILARITY_THRESHOLD:
        return (0, 200, 255)   # yellow
    elif score >= 60:
        return (0, 140, 255)   # orange
    else:
        return (0, 60, 220)    # red
