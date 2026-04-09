# pose_classifier.py — Score-based yoga pose classifier
#
# Each pose has a set of angle constraints with a required range and
# an importance weight. We score every pose and pick the best match
# above a minimum confidence threshold.
#
# This approach avoids if/else cascades where early conditions shadow
# later ones, and handles ambiguous frames more gracefully.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class YogaPose:
    name:        str
    sanskrit:    str
    description: str
    steps:       list[str] = field(default_factory=list)


# ── Pose library ──────────────────────────────────────────────────────────────
POSES: dict[str, YogaPose] = {
    "mountain": YogaPose(
        name="Mountain Pose", sanskrit="Tadasana",
        description="Standing tall, feet together, arms at your sides.",
        steps=[
            "Stand with your feet together or hip-width apart.",
            "Press all four corners of your feet firmly into the ground.",
            "Engage your thighs and lift your kneecaps slightly.",
            "Lengthen your tailbone down and lift your chest.",
            "Relax your shoulders away from your ears, arms at your sides.",
            "Breathe steadily and hold for five to ten breaths.",
        ],
    ),
    "forward_fold": YogaPose(
        name="Standing Forward Fold", sanskrit="Uttanasana",
        description="Standing, folding forward with a long spine.",
        steps=[
            "Stand with feet hip-width apart.",
            "Inhale and lengthen your spine.",
            "Exhale and hinge forward from your hips.",
            "Let your hands hang toward the floor or hold your elbows.",
            "Bend your knees generously if your hamstrings are tight.",
            "Relax your head and neck completely.",
            "Hold for five breaths, deepening with each exhale.",
        ],
    ),
    "halfway_lift": YogaPose(
        name="Halfway Lift", sanskrit="Ardha Uttanasana",
        description="Flat back, hands on shins, torso parallel to floor.",
        steps=[
            "From Forward Fold, place your hands on your shins.",
            "Inhale and lift your chest halfway up.",
            "Make your back as flat as a tabletop.",
            "Draw your shoulder blades together and down.",
            "Keep your neck in line with your spine.",
            "Hold for one breath before moving on.",
        ],
    ),
    "warrior_i": YogaPose(
        name="Warrior I", sanskrit="Virabhadrasana I",
        description="Lunge with both arms raised straight overhead.",
        steps=[
            "Step your left foot back about three to four feet.",
            "Turn your left foot out to about forty-five degrees.",
            "Bend your right knee directly over your right ankle.",
            "Square your hips to face the front of your mat.",
            "Inhale and raise both arms straight overhead.",
            "Draw your shoulder blades down your back.",
            "Gaze forward or slightly upward and hold for five breaths.",
        ],
    ),
    "warrior_ii": YogaPose(
        name="Warrior II", sanskrit="Virabhadrasana II",
        description="Wide stance, front knee bent, arms extended to the sides.",
        steps=[
            "Step your feet wide apart, about four feet.",
            "Turn your right foot out ninety degrees, left foot in slightly.",
            "Bend your right knee to ninety degrees over your ankle.",
            "Extend both arms out to the sides at shoulder height — palms down.",
            "Keep your torso directly over your hips, not leaning forward.",
            "Turn your head to gaze over your right fingertips.",
            "Hold for five steady breaths, then switch sides.",
        ],
    ),
    "warrior_iii": YogaPose(
        name="Warrior III", sanskrit="Virabhadrasana III",
        description="One-leg balance, torso and raised leg parallel to the floor.",
        steps=[
            "Begin standing on your right foot.",
            "Hinge forward at your hip, lifting your left leg behind you.",
            "Reach your torso and left leg parallel to the floor.",
            "Extend your arms forward or keep them by your sides.",
            "Flex your raised foot and keep your hips level.",
            "Hold for three to five breaths, then switch sides.",
        ],
    ),
    "chair": YogaPose(
        name="Chair Pose", sanskrit="Utkatasana",
        description="Sitting into an imaginary chair, arms raised overhead.",
        steps=[
            "Stand with feet together.",
            "Inhale and raise your arms straight overhead.",
            "Exhale and bend your knees, lowering your hips as if sitting.",
            "Bring your thighs as parallel to the floor as comfortable.",
            "Keep your knees over your toes.",
            "Draw your lower belly in and lengthen your spine.",
            "Hold for five breaths.",
        ],
    ),
    "tree": YogaPose(
        name="Tree Pose", sanskrit="Vrksasana",
        description="Single-leg balance, one foot pressed to the inner thigh.",
        steps=[
            "Shift your weight onto your left foot.",
            "Bend your right knee and place your right foot on your inner left thigh.",
            "Press foot and thigh against each other for stability.",
            "Bring your palms together at your chest or raise arms overhead.",
            "Fix your gaze on a steady point in front of you.",
            "Hold for five breaths, then repeat on the other side.",
        ],
    ),
    "triangle": YogaPose(
        name="Triangle Pose", sanskrit="Trikonasana",
        description="Wide stance, one hand reaching down, other arm raised to the sky.",
        steps=[
            "Step your feet wide apart, about three to four feet.",
            "Turn your right foot out ninety degrees.",
            "Extend both arms out to the sides.",
            "Reach your right hand toward your right shin.",
            "Stack your left arm directly above your right.",
            "Keep both legs straight and your torso long.",
            "Gaze up toward your raised hand. Hold five breaths.",
        ],
    ),
    "downward_dog": YogaPose(
        name="Downward Dog", sanskrit="Adho Mukha Svanasana",
        description="Inverted V-shape, hands and feet pressing into the mat.",
        steps=[
            "Begin on hands and knees.",
            "Tuck your toes and press your hips up and back.",
            "Straighten your legs as much as comfortable.",
            "Press your hands firmly into the mat, fingers spread wide.",
            "Let your head hang between your upper arms.",
            "Draw your navel toward your spine.",
            "Hold for five to ten breaths.",
        ],
    ),
    "plank": YogaPose(
        name="Plank Pose", sanskrit="Kumbhakasana",
        description="High push-up position, body in a straight line.",
        steps=[
            "Begin on your hands and knees.",
            "Step both feet back so your body forms a straight line.",
            "Stack your wrists directly under your shoulders.",
            "Engage your core — draw your belly button toward your spine.",
            "Keep your hips level — not sagging or piking up.",
            "Press through your heels and gaze slightly forward.",
            "Hold for five breaths.",
        ],
    ),
    "low_lunge": YogaPose(
        name="Low Lunge", sanskrit="Anjaneyasana",
        description="Front knee bent at ninety degrees, back knee on the ground.",
        steps=[
            "Step your right foot forward between your hands.",
            "Lower your left knee to the mat.",
            "Sink your hips forward and down.",
            "Inhale and raise your arms overhead.",
            "Draw your shoulder blades down and open your chest.",
            "Hold for five breaths, then switch sides.",
        ],
    ),
    "high_lunge": YogaPose(
        name="High Lunge", sanskrit="Utthita Ashwa Sanchalanasana",
        description="Front knee bent, back leg straight, arms overhead.",
        steps=[
            "Step your right foot forward, keeping the back leg straight.",
            "Bend your right knee to ninety degrees.",
            "Keep your back heel lifted.",
            "Inhale and raise both arms overhead.",
            "Square your hips forward as much as possible.",
            "Hold for five breaths, then switch sides.",
        ],
    ),
    "cobra": YogaPose(
        name="Cobra Pose", sanskrit="Bhujangasana",
        description="Lying face down, chest lifted with elbows slightly bent.",
        steps=[
            "Lie face down with legs extended behind you.",
            "Place your palms flat under your shoulders.",
            "Press the tops of your feet into the mat.",
            "Inhale and slowly lift your chest.",
            "Keep your elbows slightly bent and close to your sides.",
            "Roll your shoulders back and open your chest.",
            "Hold for five breaths, then slowly lower down.",
        ],
    ),
    "upward_dog": YogaPose(
        name="Upward Dog", sanskrit="Urdhva Mukha Svanasana",
        description="Chest lifted high, arms straight, thighs off the mat.",
        steps=[
            "Lie face down with hands under your shoulders.",
            "Press firmly into your hands and straighten your arms.",
            "Lift your thighs and knees off the mat.",
            "Roll your shoulders back and lift your chest high.",
            "Press the tops of your feet into the mat.",
            "Hold for one to three breaths.",
        ],
    ),
    "child": YogaPose(
        name="Child's Pose", sanskrit="Balasana",
        description="Kneeling rest pose, forehead on the mat, arms extended.",
        steps=[
            "Kneel on the mat with your big toes touching.",
            "Sit back on your heels and open your knees hip-width apart.",
            "Exhale and fold your torso forward between your thighs.",
            "Extend your arms forward on the mat, palms facing down.",
            "Rest your forehead gently on the mat.",
            "Breathe into your back body.",
            "Rest here for five to ten breaths.",
        ],
    ),
    "seated_forward": YogaPose(
        name="Seated Forward Fold", sanskrit="Paschimottanasana",
        description="Seated, legs extended, folding forward over the legs.",
        steps=[
            "Sit with both legs extended straight in front of you.",
            "Flex your feet, pressing through your heels.",
            "Inhale and lengthen your spine, raising arms overhead.",
            "Exhale and hinge forward from your hips.",
            "Reach toward your shins, ankles, or feet.",
            "Keep your spine long rather than rounding your back.",
            "Hold for five to eight breaths.",
        ],
    ),
    "cat_cow": YogaPose(
        name="Cat-Cow", sanskrit="Marjaryasana-Bitilasana",
        description="On hands and knees, alternating spine arch and round.",
        steps=[
            "Come onto your hands and knees — wrists under shoulders, knees under hips.",
            "For Cow: inhale, drop your belly, lift your chest and tailbone.",
            "For Cat: exhale, round your spine up toward the ceiling.",
            "Tuck your chin to your chest and your tailbone under.",
            "Move slowly, linking each movement to your breath.",
            "Continue for five to ten cycles.",
        ],
    ),
    "unknown": YogaPose(
        name="Yoga Pose", sanskrit="",
        description="Pose in progress.",
        steps=["Follow the instructor's position, one joint at a time."],
    ),
}


# ── Constraint-based scoring ──────────────────────────────────────────────────

@dataclass
class Constraint:
    """A single angle range constraint for a joint."""
    joint:    str
    lo:       float     # minimum acceptable angle (degrees)
    hi:       float     # maximum acceptable angle (degrees)
    weight:   float     # importance (higher = disqualifies more strongly if violated)
    required: bool      # if True AND joint available, pose is rejected if violated


# Pose angle signatures — derived from reference yoga anatomy sources
# (Yoga Anatomy, Kaminoff & Matthews; MediaPipe landmark normalisation)
_POSE_CONSTRAINTS: dict[str, list[Constraint]] = {
    "mountain": [
        Constraint("left_knee",       155, 185, 2.0, required=True),
        Constraint("right_knee",      155, 185, 2.0, required=True),
        Constraint("left_hip",        150, 195, 1.5, required=True),
        Constraint("right_hip",       150, 195, 1.5, required=True),
        Constraint("left_elbow",      155, 185, 2.0, required=True),
        Constraint("right_elbow",     155, 185, 2.0, required=True),
        Constraint("left_shoulder",    10,  60, 1.0, required=False),
        Constraint("right_shoulder",   10,  60, 1.0, required=False),
    ],
    "forward_fold": [
        Constraint("left_hip",         20,  90, 2.0, required=True),
        Constraint("right_hip",        20,  90, 2.0, required=True),
        Constraint("left_knee",       130, 185, 1.5, required=False),
        Constraint("right_knee",      130, 185, 1.5, required=False),
        Constraint("left_shoulder",     5,  60, 1.0, required=False),
        Constraint("right_shoulder",    5,  60, 1.0, required=False),
    ],
    "halfway_lift": [
        Constraint("left_hip",         65, 115, 2.0, required=True),
        Constraint("right_hip",        65, 115, 2.0, required=True),
        Constraint("left_knee",       150, 185, 1.5, required=True),
        Constraint("right_knee",      150, 185, 1.5, required=True),
        Constraint("left_shoulder",    55, 100, 1.0, required=False),
        Constraint("right_shoulder",   55, 100, 1.0, required=False),
    ],
    "warrior_i": [
        Constraint("left_shoulder",   140, 210, 2.5, required=True),
        Constraint("right_shoulder",  140, 210, 2.5, required=True),
        Constraint("left_knee",        70, 120, 2.0, required=False),
        Constraint("right_knee",       70, 120, 2.0, required=False),
        Constraint("left_hip",        120, 175, 1.5, required=False),
        Constraint("right_hip",       120, 175, 1.5, required=False),
    ],
    "warrior_ii": [
        Constraint("left_shoulder",    70, 115, 3.0, required=True),
        Constraint("right_shoulder",   70, 115, 3.0, required=True),
        Constraint("left_knee",       145, 185, 2.5, required=False),
        Constraint("right_knee",      145, 185, 2.5, required=False),
        Constraint("left_knee",        70, 120, 1.5, required=False),
        Constraint("right_knee",       70, 120, 1.5, required=False),
        Constraint("left_hip",         90, 160, 1.0, required=False),
        Constraint("right_hip",        90, 160, 1.0, required=False),
    ],
    "warrior_iii": [
        Constraint("left_knee",       155, 185, 2.5, required=True),
        Constraint("left_hip",         75, 115, 2.0, required=True),
        Constraint("right_hip",        75, 115, 2.0, required=False),
    ],
    "chair": [
        Constraint("left_knee",        70, 120, 2.5, required=True),
        Constraint("right_knee",       70, 120, 2.5, required=True),
        Constraint("left_shoulder",   140, 210, 2.5, required=True),
        Constraint("right_shoulder",  140, 210, 2.5, required=True),
        Constraint("left_hip",         65, 125, 2.0, required=True),
        Constraint("right_hip",        65, 125, 2.0, required=True),
    ],
    "tree": [
        Constraint("left_knee",       155, 185, 2.5, required=True),
        Constraint("right_knee",       20,  80, 2.5, required=True),
        Constraint("right_hip",        30,  90, 2.0, required=True),
    ],
    "triangle": [
        Constraint("left_knee",       150, 185, 2.0, required=True),
        Constraint("right_knee",      150, 185, 2.0, required=True),
        Constraint("left_shoulder",   130, 200, 2.0, required=True),
        Constraint("right_shoulder",   40,  85, 2.0, required=True),
        Constraint("left_hip",         50, 110, 1.5, required=False),
    ],
    "downward_dog": [
        Constraint("left_knee",       140, 185, 2.0, required=True),
        Constraint("right_knee",      140, 185, 2.0, required=True),
        Constraint("left_hip",         45, 100, 2.5, required=True),
        Constraint("right_hip",        45, 100, 2.5, required=True),
        Constraint("left_shoulder",   150, 210, 2.0, required=True),
        Constraint("right_shoulder",  150, 210, 2.0, required=True),
    ],
    "plank": [
        Constraint("left_knee",       155, 185, 2.0, required=True),
        Constraint("right_knee",      155, 185, 2.0, required=True),
        Constraint("left_hip",        155, 195, 2.5, required=True),
        Constraint("right_hip",       155, 195, 2.5, required=True),
        Constraint("left_elbow",      155, 185, 1.5, required=False),
        Constraint("right_elbow",     155, 185, 1.5, required=False),
    ],
    "low_lunge": [
        Constraint("left_knee",        70, 115, 2.5, required=True),
        Constraint("right_knee",       70, 130, 2.0, required=True),
        Constraint("left_shoulder",   140, 210, 1.5, required=False),
        Constraint("right_shoulder",  140, 210, 1.5, required=False),
    ],
    "high_lunge": [
        Constraint("left_knee",        70, 115, 2.5, required=True),
        Constraint("right_knee",      145, 185, 2.0, required=True),
        Constraint("left_shoulder",   140, 210, 2.0, required=True),
        Constraint("right_shoulder",  140, 210, 2.0, required=True),
        Constraint("left_hip",         85, 148, 2.0, required=True),
        Constraint("right_hip",        85, 148, 2.0, required=True),
    ],
    "cobra": [
        Constraint("left_hip",        155, 200, 2.0, required=True),
        Constraint("right_hip",       155, 200, 2.0, required=True),
        Constraint("left_elbow",      100, 158, 2.5, required=True),
        Constraint("right_elbow",     100, 158, 2.5, required=True),
        Constraint("left_knee",       155, 185, 1.5, required=False),
        Constraint("right_knee",      155, 185, 1.5, required=False),
        Constraint("left_shoulder",    20,  75, 1.5, required=True),
        Constraint("right_shoulder",   20,  75, 1.5, required=True),
    ],
    "upward_dog": [
        Constraint("left_hip",        155, 200, 2.0, required=True),
        Constraint("right_hip",       155, 200, 2.0, required=True),
        Constraint("left_elbow",      155, 185, 2.5, required=True),
        Constraint("right_elbow",     155, 185, 2.5, required=True),
        Constraint("left_knee",       155, 185, 1.0, required=False),
    ],
    "child": [
        Constraint("left_hip",         20,  75, 2.5, required=True),
        Constraint("right_hip",        20,  75, 2.5, required=True),
        Constraint("left_knee",        20,  80, 2.0, required=True),
        Constraint("right_knee",       20,  80, 2.0, required=True),
    ],
    "seated_forward": [
        Constraint("left_hip",         20,  85, 2.5, required=True),
        Constraint("right_hip",        20,  85, 2.5, required=True),
        Constraint("left_knee",       150, 185, 2.0, required=True),
        Constraint("right_knee",      150, 185, 2.0, required=True),
    ],
    "cat_cow": [
        Constraint("left_hip",         80, 115, 2.5, required=True),
        Constraint("right_hip",        80, 115, 2.5, required=True),
        Constraint("left_knee",        78, 118, 2.5, required=True),
        Constraint("right_knee",       78, 118, 2.5, required=True),
        Constraint("left_elbow",      150, 185, 2.0, required=True),
        Constraint("right_elbow",     150, 185, 2.0, required=True),
        Constraint("left_shoulder",    65, 110, 2.0, required=True),
        Constraint("right_shoulder",   65, 110, 2.0, required=True),
    ],
}
# Minimum fraction of weighted constraints that must pass to accept a pose
_MIN_SCORE_THRESHOLD = 0.60

# ── ML model (auto-loaded if train_model.py has been run) ─────────────────────
_ML_MODEL     = None
_ML_ENCODER   = None
_ML_AVAILABLE = False

def _try_load_ml_model() -> None:
    global _ML_MODEL, _ML_ENCODER, _ML_AVAILABLE
    try:
        import joblib
        from pathlib import Path
        model_path = Path(__file__).parent / "model" / "pose_classifier.pkl"
        if model_path.exists():
            bundle = joblib.load(model_path)
            _ML_MODEL     = bundle["model"]
            _ML_ENCODER   = bundle["encoder"]
            _ML_AVAILABLE = True
            logger.info(
                "ML pose classifier loaded: %s  (acc=%.3f)",
                bundle.get("model_name", "?"),
                bundle.get("test_acc", 0.0),
            )
        else:
            logger.debug("No trained model found at %s — using rule-based classifier.", model_path)
    except Exception as e:
        logger.debug("ML model not loaded (%s) — using rule-based classifier.", e)

_try_load_ml_model()


def _make_feature_vector(angles: dict) -> "np.ndarray":
    import numpy as np
    ALL_JOINTS = [
        "left_shoulder", "right_shoulder", "left_elbow",  "right_elbow",
        "left_hip",      "right_hip",      "left_knee",   "right_knee",
        "left_ankle",    "right_ankle",    "left_wrist",  "right_wrist",
    ]
    base = [float(angles.get(j, 135.0)) for j in ALL_JOINTS]
    ls = angles.get("left_shoulder",  90); rs = angles.get("right_shoulder", 90)
    lh = angles.get("left_hip",       90); rh = angles.get("right_hip",      90)
    lk = angles.get("left_knee",      90); rk = angles.get("right_knee",     90)
    le = angles.get("left_elbow",     90); re = angles.get("right_elbow",    90)
    derived = [
        abs(ls - rs), abs(lh - rh), abs(lk - rk), abs(le - re),
        lk / max(lh, 1), rk / max(rh, 1),
        ls - lk, rs - rk,
        (lk + rk) / 2, (lh + rh) / 2, (ls + rs) / 2, (le + re) / 2,
    ]
    return np.array(base + derived, dtype=np.float32)


def classify_pose(angles: dict[str, Optional[float]]) -> YogaPose:
    """
    Classify a pose from joint angles.
    Uses the trained ML model if available, otherwise falls back to
    the rule-based constraint scorer.
    """
    # ── ML path ───────────────────────────────────────────────────────
    if _ML_AVAILABLE and _ML_MODEL is not None:
        try:
            clean = {k: float(v) for k, v in angles.items() if v is not None}
            fv = _make_feature_vector(clean).reshape(1, -1)
            pred_idx  = _ML_MODEL.predict(fv)[0]
            pred_label = _ML_ENCODER.inverse_transform([pred_idx])[0]
            # Get probability for confidence gate (reject if uncertain)
            if hasattr(_ML_MODEL, "predict_proba"):
                prob = _ML_MODEL.predict_proba(fv)[0].max()
                if prob < 0.45:
                    return POSES["unknown"]
            result = POSES.get(pred_label, POSES["unknown"])
            logger.debug("ML classified: %s", result.name)
            return result
        except Exception as e:
            logger.debug("ML classify failed, falling back: %s", e)

    # ── Rule-based fallback ───────────────────────────────────────────
    best_key   = "unknown"
    best_score = 0.0

    for pose_key, constraints in _POSE_CONSTRAINTS.items():
        total_weight    = 0.0
        satisfied_weight= 0.0
        disqualified    = False

        for c in constraints:
            val = angles.get(c.joint)
            if val is None:
                # Missing joint — skip (don't penalise)
                continue

            total_weight += c.weight
            in_range      = c.lo <= val <= c.hi

            if in_range:
                satisfied_weight += c.weight
            elif c.required:
                disqualified = True
                break   # hard fail — this pose is impossible

        if disqualified or total_weight == 0:
            continue

        score = satisfied_weight / total_weight
        if score > best_score:
            best_score = score
            best_key   = pose_key

    if best_score < _MIN_SCORE_THRESHOLD:
        return POSES["unknown"]

    result = POSES.get(best_key, POSES["unknown"])
    logger.debug("Classified: %s (score=%.2f)", result.name, best_score)
    return result
