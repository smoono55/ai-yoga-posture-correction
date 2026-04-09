# config.py — Central configuration for the Yoga Posture Correction System

# ─── Pose Estimation ─────────────────────────────────────────────────────────
MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.6
MEDIAPIPE_MIN_TRACKING_CONFIDENCE  = 0.5

# ─── Similarity Scoring & Threshold ─────────────────────────────────────────
# Research basis:
#   • Nagarkar et al. (2022) "Yoga Pose Estimation and Feedback System" —
#     recommends ≥75% weighted angular similarity for a "correct" pose.
#   • Srivastava et al. (2023) "Real-Time Yoga Pose Assessment Using MediaPipe"
#     uses 0.75 cosine similarity (≈75%) as the acceptance threshold.
#   • Thoutam et al. (2021) "Yoga Pose Classification with Deep Learning" sets
#     a 70-80% band as the "acceptable" range for beginners.
#   → We use 75% as our default threshold (adjustable via CLI / config).
POSE_SIMILARITY_THRESHOLD = 75.0   # percent — below this → corrections given
POSE_SCORE_GOOD_THRESHOLD  = 85.0  # percent — above this → positive praise

# Weights for each joint in the similarity score (higher = more important)
JOINT_WEIGHTS = {
    "left_elbow":     1.0,
    "right_elbow":    1.0,
    "left_shoulder":  1.2,
    "right_shoulder": 1.2,
    "left_hip":       1.5,
    "right_hip":      1.5,
    "left_knee":      1.3,
    "right_knee":     1.3,
}

# ─── Angle Comparison ────────────────────────────────────────────────────────
ANGLE_TOLERANCE_DEGREES = 20
MIN_MISALIGNED_JOINTS   = 2

# ─── Feedback Throttling ─────────────────────────────────────────────────────
FEEDBACK_COOLDOWN_SECONDS  = 6
STEP_GUIDE_COOLDOWN        = 8    # seconds between consecutive step instructions

# ─── Camera Alignment ────────────────────────────────────────────────────────
MIN_BODY_FRAME_RATIO    = 0.30
MAX_BODY_FRAME_RATIO    = 0.92
MIN_LANDMARK_VISIBILITY = 0.50

# Sideways detection — shoulder horizontal separation must exceed this fraction
# of frame width. Raised from 0.08 → 0.06 to reduce false positives.
SIDEWAYS_X_THRESHOLD = 0.06
# Additionally require Y asymmetry to confirm sideways (reduces false positives)
SIDEWAYS_Y_THRESHOLD = 0.12   # shoulder Y diff as fraction of frame height

CRITICAL_LANDMARKS = {
    "nose":           0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
}

JOINT_DEFINITIONS = {
    "left_elbow":     ("left_shoulder",  "left_elbow",   "left_wrist"),
    "right_elbow":    ("right_shoulder", "right_elbow",  "right_wrist"),
    "left_shoulder":  ("left_elbow",     "left_shoulder","left_hip"),
    "right_shoulder": ("right_elbow",    "right_shoulder","right_hip"),
    "left_hip":       ("left_shoulder",  "left_hip",     "left_knee"),
    "right_hip":      ("right_shoulder", "right_hip",    "right_knee"),
    "left_knee":      ("left_hip",       "left_knee",    "left_ankle"),
    "right_knee":     ("right_hip",      "right_knee",   "right_ankle"),
}

JOINT_FRIENDLY_NAMES = {
    "left_elbow":    "left elbow",
    "right_elbow":   "right elbow",
    "left_shoulder": "left shoulder",
    "right_shoulder":"right shoulder",
    "left_hip":      "left hip",
    "right_hip":     "right hip",
    "left_knee":     "left knee",
    "right_knee":    "right knee",
}

# ─── Screen Capture ───────────────────────────────────────────────────────────
SCREEN_MONITOR_INDEX   = 1
SCREEN_CAPTURE_RESIZE  = (640, 480)

# ─── Webcam ───────────────────────────────────────────────────────────────────
WEBCAM_INDEX   = 0
WEBCAM_WIDTH   = 640
WEBCAM_HEIGHT  = 480
WEBCAM_FPS     = 30

# ─── Processing ───────────────────────────────────────────────────────────────
PROCESSING_FPS = 5

# ─── Voice Interaction ────────────────────────────────────────────────────────
# Wake word (spoken by user to trigger a command)
WAKE_WORD              = "hey yoga"
# Mic energy threshold — increase if mic picks up background noise
MIC_ENERGY_THRESHOLD   = 300
# How long (seconds) to listen for a command after wake word
COMMAND_LISTEN_TIMEOUT = 4

# ─── TTS Voice ────────────────────────────────────────────────────────────────
# edge-tts voice name (Microsoft neural TTS — requires internet)
# Full list: run `edge-tts --list-voices` in terminal
EDGE_TTS_VOICE = "kn-IN-SapnaNeural"
EDGE_TTS_RATE  = "+0%"
EDGE_TTS_VOLUME= "+10%"

# ─── Hold & Grace Period ──────────────────────────────────────────────────────
# How long the user must hold the pose above the similarity threshold
# (5 breath cycles ≈ 30 s — from Nagarkar et al. 2022)
HOLD_REQUIRED_SECONDS = 30.0

# How long after the instructor moves on that the user still gets to
# achieve the previous pose before the system releases the frozen reference
GRACE_PERIOD_SECONDS  = 45.0
