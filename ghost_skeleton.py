# ghost_skeleton.py — Instructor body stencil overlaid on user webcam frame
#
# Renders the instructor's body as a filled + outlined "chalk stencil":
#   • Each body SEGMENT (upper arm, forearm, thigh, shin, torso) is drawn
#     as a thick filled capsule/polygon, coloured by its alignment status.
#   • A bright glowing outline surrounds each segment.
#   • Joint dots are drawn on top with a highlight ring.
#   • The whole stencil is body-aligned: scaled and translated to the user's
#     torso so the ghost sits directly over the user's body.
#
# Colour coding (per segment / joint):
#   GREEN  (#00D250) — within tolerance (matched)
#   CYAN   (#00C8FF) — within 2× tolerance (close)
#   ORANGE (#FF8C00) — within 3× tolerance (needs work)
#   RED    (#E02020) — beyond 3× tolerance (far off)
#   GREY   (#909090) — no user joint to compare
#
# The stencil is drawn at ~40% opacity so the user's own body and webcam
# feed remain clearly visible underneath.

from __future__ import annotations
import cv2
import numpy as np
from typing import Optional

import config
from pose_estimator import PoseResult, Landmark

# ── Colour palette ────────────────────────────────────────────────────────────
_C_MATCH   = (80,  210,   0)    # green
_C_CLOSE   = (255, 200,   0)    # cyan-yellow
_C_WORK    = (0,   140, 255)    # orange (BGR)
_C_WRONG   = (30,   30, 220)    # red
_C_UNKNOWN = (144, 144, 144)    # grey
_C_GLOW    = (255, 255, 255)    # white glow outline
_C_FILL_A  = 0.30               # stencil fill alpha
_C_LINE_A  = 0.70               # outline alpha
_DOT_R     = 8                  # joint dot radius

# ── Body segments: (proximal_joint, distal_joint, controlling_joint, half_width_px) ──
# controlling_joint = which joint's comparison result colours this segment
_SEGMENTS = [
    # Torso
    ("left_shoulder",  "right_shoulder", None,             12),
    ("left_shoulder",  "left_hip",       "left_shoulder",   9),
    ("right_shoulder", "right_hip",      "right_shoulder",  9),
    ("left_hip",       "right_hip",       None,             12),
    # Left arm
    ("left_shoulder",  "left_elbow",     "left_shoulder",   7),
    ("left_elbow",     "left_wrist",     "left_elbow",      5),
    # Right arm
    ("right_shoulder", "right_elbow",    "right_shoulder",  7),
    ("right_elbow",    "right_wrist",    "right_elbow",     5),
    # Left leg
    ("left_hip",       "left_knee",      "left_hip",        9),
    ("left_knee",      "left_ankle",     "left_knee",       7),
    # Right leg
    ("right_hip",      "right_knee",     "right_hip",       9),
    ("right_knee",     "right_ankle",    "right_knee",      7),
    # Head
    ("nose",           "left_shoulder",  None,              4),
    ("nose",           "right_shoulder", None,              4),
]


def _segment_colour(ctrl_joint: Optional[str], comparison: dict) -> tuple:
    if ctrl_joint is None:
        return _C_UNKNOWN
    info = comparison.get(ctrl_joint)
    if not info or info.get("diff") is None:
        return _C_UNKNOWN
    diff = info["diff"]
    tol  = config.ANGLE_TOLERANCE_DEGREES
    if diff <= tol:
        return _C_MATCH
    elif diff <= tol * 2:
        return _C_CLOSE
    elif diff <= tol * 3:
        return _C_WORK
    else:
        return _C_WRONG


def _draw_capsule(
    layer: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    colour: tuple,
    half_w: int,
    alpha: float,
    outline: bool = False,
) -> None:
    """Draw a thick rounded capsule (filled polygon + end circles) between p1 and p2."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = max(np.hypot(dx, dy), 1e-6)
    nx = -dy / length * half_w
    ny =  dx / length * half_w

    pts = np.array([
        [int(p1[0] + nx), int(p1[1] + ny)],
        [int(p1[0] - nx), int(p1[1] - ny)],
        [int(p2[0] - nx), int(p2[1] - ny)],
        [int(p2[0] + nx), int(p2[1] + ny)],
    ], dtype=np.int32)

    thickness = -1 if not outline else max(2, half_w // 3)

    tmp = layer.copy() if alpha < 1.0 else layer
    cv2.fillPoly(tmp, [pts], colour)
    cv2.circle(tmp, p1, half_w, colour, thickness, cv2.LINE_AA)
    cv2.circle(tmp, p2, half_w, colour, thickness, cv2.LINE_AA)

    if alpha < 1.0:
        cv2.addWeighted(tmp, alpha, layer, 1 - alpha, 0, layer)


def _project(
    lm: Landmark,
    tx: float, ty: float,
    sx: float, sy: float,
    w: int, h: int,
) -> tuple[int, int]:
    px = int(tx + lm.x * sx)
    py = int(ty + lm.y * sy)
    return (max(0, min(w - 1, px)), max(0, min(h - 1, py)))


def _compute_alignment(
    inst_pose: PoseResult,
    user_pose: PoseResult,
    frame_w: int,
    frame_h: int,
) -> dict:
    """Scale + translate instructor skeleton to sit on user's torso."""
    def mid(a, b):
        if a and b:
            return ((a.x + b.x) / 2, (a.y + b.y) / 2)
        return None

    i_lh = inst_pose.get("left_hip");  i_rh = inst_pose.get("right_hip")
    i_ls = inst_pose.get("left_shoulder"); i_rs = inst_pose.get("right_shoulder")
    u_lh = user_pose.get("left_hip");  u_rh = user_pose.get("right_hip")
    u_ls = user_pose.get("left_shoulder"); u_rs = user_pose.get("right_shoulder")

    sx, sy = float(frame_w), float(frame_h)
    tx, ty = 0.0, 0.0

    i_hip = mid(i_lh, i_rh)
    u_hip = mid(u_lh, u_rh)

    if i_hip and u_hip:
        i_sho = mid(i_ls, i_rs)
        u_sho = mid(u_ls, u_rs)

        if i_sho and u_sho:
            i_torso = abs(i_hip[1] - i_sho[1])
            u_torso = abs(u_hip[1] - u_sho[1])
            if i_torso > 0.01 and u_torso > 0.01:
                scale = u_torso / i_torso
                sx    = scale * frame_w
                sy    = scale * frame_h

        tx = u_hip[0] * frame_w - i_hip[0] * sx
        ty = u_hip[1] * frame_h - i_hip[1] * sy

    return {"tx": tx, "ty": ty, "sx": sx, "sy": sy}


def draw_stencil(
    frame:      np.ndarray,
    inst_pose:  PoseResult,
    user_pose:  PoseResult,
    comparison: dict,
    alpha:      float = 0.42,
) -> np.ndarray:
    """
    Draw the instructor body stencil on `frame` (modified in-place).

    Layers drawn (back to front):
      1. Filled capsule body segments (low alpha)  — shows body shape
      2. Bright outlined capsule segments          — shows exact edges
      3. Joint dots with colour rings
      4. Legend
    """
    if not inst_pose.detected:
        return frame

    h, w = frame.shape[:2]
    tf = _compute_alignment(inst_pose, user_pose, w, h)

    # Project all landmarks
    pts: dict[str, tuple[int, int]] = {}
    for name, lm in inst_pose.landmarks.items():
        if lm.visibility >= config.MIN_LANDMARK_VISIBILITY:
            pts[name] = _project(lm, tf["tx"], tf["ty"], tf["sx"], tf["sy"], w, h)

    if len(pts) < 4:
        return frame

    # ── Layer 1: filled segments ──────────────────────────────────────
    fill_layer = frame.copy()
    for prox, dist, ctrl, hw in _SEGMENTS:
        if prox not in pts or dist not in pts:
            continue
        col = _segment_colour(ctrl, comparison)
        _draw_capsule(fill_layer, pts[prox], pts[dist], col, hw, alpha=1.0)

    cv2.addWeighted(fill_layer, alpha * 0.55, frame, 1 - alpha * 0.55, 0, frame)

    # ── Layer 2: glowing outline ──────────────────────────────────────
    for prox, dist, ctrl, hw in _SEGMENTS:
        if prox not in pts or dist not in pts:
            continue
        col = _segment_colour(ctrl, comparison)
        # White glow (wider)
        cv2.line(frame, pts[prox], pts[dist], _C_GLOW,    hw * 2 + 4, cv2.LINE_AA)
        # Colour inner line
        cv2.line(frame, pts[prox], pts[dist], col,        hw * 2,     cv2.LINE_AA)
        # End circles
        cv2.circle(frame, pts[prox], hw + 1, _C_GLOW,  2, cv2.LINE_AA)
        cv2.circle(frame, pts[dist], hw + 1, _C_GLOW,  2, cv2.LINE_AA)
        cv2.circle(frame, pts[prox], hw,     col,      -1, cv2.LINE_AA)
        cv2.circle(frame, pts[dist], hw,     col,      -1, cv2.LINE_AA)

    # Re-apply overall alpha blend for the outlines
    # (outlines are drawn directly to keep crispness — only fill uses addWeighted)

    # ── Layer 3: joint dots ───────────────────────────────────────────
    for name, pt in pts.items():
        col = _segment_colour(name, comparison)
        cv2.circle(frame, pt, _DOT_R + 2, _C_GLOW, -1, cv2.LINE_AA)   # white halo
        cv2.circle(frame, pt, _DOT_R,     col,      -1, cv2.LINE_AA)   # coloured fill
        cv2.circle(frame, pt, _DOT_R,     (30,30,30), 1, cv2.LINE_AA)  # dark ring

    # ── Layer 4: Guidance indicators ──────────────────────────────────
    for name, pt in pts.items():
        info = comparison.get(name)
        if info and info.get("misaligned"):
            diff = info.get("diff", 0)
            if diff > config.ANGLE_TOLERANCE_DEGREES * 2:
                direction = info.get("direction")
                if direction:
                    text = "OPEN" if direction == "increase" else "BEND"
                    cv2.putText(frame, text, (pt[0] + 14, pt[1] - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20,20,20), 2, cv2.LINE_AA)
                    cv2.putText(frame, text, (pt[0] + 14, pt[1] - 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _C_GLOW, 1, cv2.LINE_AA)

    # ── Layer 5: legend ───────────────────────────────────────────────
    _draw_legend(frame, comparison)

    return frame


def _draw_legend(frame: np.ndarray, comparison: dict) -> None:
    h, w = frame.shape[:2]

    items = [
        (_C_MATCH,   "Matched"),
        (_C_CLOSE,   "Close"),
        (_C_WORK,    "Adjust"),
        (_C_WRONG,   "Fix this"),
        (_C_UNKNOWN, "No data"),
    ]

    # Count how many joints are in each category
    counts = {label: 0 for _, label in items}
    label_map = {
        "Matched": lambda d: d <= config.ANGLE_TOLERANCE_DEGREES,
        "Close":   lambda d: config.ANGLE_TOLERANCE_DEGREES < d <= config.ANGLE_TOLERANCE_DEGREES * 2,
        "Adjust":  lambda d: config.ANGLE_TOLERANCE_DEGREES * 2 < d <= config.ANGLE_TOLERANCE_DEGREES * 3,
        "Fix this":lambda d: d > config.ANGLE_TOLERANCE_DEGREES * 3,
        "No data": lambda d: False,
    }
    no_data = 0
    for info in comparison.values():
        diff = info.get("diff")
        if diff is None:
            no_data += 1
            continue
        for label, fn in label_map.items():
            if fn(diff):
                counts[label] += 1
                break

    counts["No data"] = no_data

    row_h  = 18
    box_h  = len(items) * row_h + 14
    box_w  = 115
    bx     = w - box_w - 6
    by     = h - box_h - 80

    # Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (bx - 6, by - 6), (bx + box_w + 2, by + box_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    cv2.putText(frame, "Stencil guide", (bx, by + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

    for i, (col, label) in enumerate(items):
        cy = by + 12 + (i + 1) * row_h
        cv2.circle(frame, (bx + 7, cy - 3), 5, col, -1, cv2.LINE_AA)
        text = f"{label} ({counts[label]})"
        cv2.putText(frame, text, (bx + 17, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
