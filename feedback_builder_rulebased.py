# feedback_builder.py — Natural, real-time yoga coaching voice
#
# Design:
#   • Corrections describe what the body IS DOING ("your knee is too straight")
#   • Pose-aware: Warrior II knee cues ≠ Child's Pose knee cues
#   • Severity-scaled: small diff → soft suggestion; large diff → clear direction
#   • Combined messages when two joints mismatch together
#   • Encouragement probability scales with how close the user is

from __future__ import annotations
from typing import Optional
import random
import config


_JOINT_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "left_elbow": {
        "increase": [
            "Straighten your left arm a little more.",
            "Let your left elbow open up — press the arm out.",
            "Left arm needs more length — extend through the elbow.",
            "Reach a bit further through your left hand.",
            "Your left elbow is slightly bent — try to lengthen it.",
            "Open your left elbow just a touch more.",
        ],
        "decrease": [
            "Soften your left elbow — let it bend naturally.",
            "Ease a little bend into your left arm.",
            "Your left arm is locking out — release the elbow slightly.",
            "Allow your left elbow to soften.",
            "Micro-bend your left elbow so it isn't fully locked.",
            "Relax the left arm — give it a gentle curve.",
        ],
    },
    "right_elbow": {
        "increase": [
            "Straighten your right arm a bit more.",
            "Extend through your right elbow — reach it long.",
            "Right arm needs more extension — press out through the hand.",
            "Let your right elbow open up a little more.",
            "Lengthen your right arm — there's more room to extend.",
            "Reach further through your right fingertips.",
        ],
        "decrease": [
            "Soften your right elbow — it doesn't need to lock.",
            "Ease a natural bend into your right arm.",
            "Release the right elbow — let it have a micro-bend.",
            "Your right arm is a bit rigid — allow a small softness.",
            "Relax the right elbow just slightly.",
            "Let your right arm breathe — a gentle bend there.",
        ],
    },
    "left_shoulder": {
        "increase": [
            "Lift your left arm higher — reach it up.",
            "Your left arm is dropping — bring it up.",
            "Left arm needs to rise — sweep it upward.",
            "Raise your left arm — there's more height to find.",
            "Let your left arm float up a little more.",
            "Extend your left arm higher — imagine a string pulling your wrist up.",
        ],
        "decrease": [
            "Lower your left arm — let it relax down.",
            "Bring your left arm in closer to your body.",
            "Your left arm is too high — ease it down gently.",
            "Release your left shoulder down.",
            "Let your left arm drop a little — allow gravity to help.",
            "Your left arm can come down — no need to hold it so high.",
        ],
    },
    "right_shoulder": {
        "increase": [
            "Raise your right arm — sweep it upward.",
            "Your right arm is too low — lift it up.",
            "Right arm needs more height — reach it up.",
            "Lift through your right arm — it can go higher.",
            "Float your right arm up a little more.",
            "Let your right arm rise — extend it skyward.",
        ],
        "decrease": [
            "Lower your right arm — let it come down naturally.",
            "Your right arm is too high — ease it down.",
            "Soften the right shoulder — let the arm drop a bit.",
            "Release your right arm downward, gently.",
            "Bring your right arm in — it can relax lower.",
            "Let your right shoulder soften and release down.",
        ],
    },
    "left_hip": {
        "increase": [
            "Open your left hip outward — rotate it gently.",
            "Let your left hip open up a little more.",
            "Your left hip is closing in — rotate it out.",
            "Turn your left hip open — give it more space.",
            "Open through the left hip — let that rotation come.",
            "Allow your left hip to externally rotate.",
        ],
        "decrease": [
            "Bring your left hip forward — square it up.",
            "Draw your left hip in slightly.",
            "Your left hip is too open — rotate it inward a touch.",
            "Close the left hip just a little.",
            "Stack your left hip over your knee.",
            "Let your left hip ease forward and down.",
        ],
    },
    "right_hip": {
        "increase": [
            "Open your right hip outward.",
            "Let your right hip rotate open a little more.",
            "Your right hip is too closed — rotate it out.",
            "Give your right hip more space — open it up.",
            "Turn your right hip outward gently.",
            "Allow the right hip to open and externally rotate.",
        ],
        "decrease": [
            "Draw your right hip forward.",
            "Square your right hip — bring it in.",
            "Your right hip is too open — ease it forward.",
            "Close the right hip slightly.",
            "Rotate your right hip inward just a touch.",
            "Let your right hip come forward and settle.",
        ],
    },
    "left_knee": {
        "increase": [
            "Straighten your left leg — press through the heel.",
            "Your left knee is too bent — lengthen that leg.",
            "Extend your left leg — find the straight line.",
            "Left leg needs more length — push the floor away.",
            "Micro-straighten your left knee.",
            "Engage your left thigh and let the knee open.",
        ],
        "decrease": [
            "Bend your left knee more — sink into it.",
            "Your left leg is too straight — add more bend.",
            "Soften your left knee — sit a little lower.",
            "Deepen the left knee bend.",
            "Let your left knee track forward — sit into it.",
            "Bring more bend into your left leg.",
        ],
    },
    "right_knee": {
        "increase": [
            "Straighten your right leg — press the heel down.",
            "Your right knee is too bent — lengthen the leg.",
            "Extend through your right leg.",
            "Right leg can be straighter — push through the floor.",
            "Open your right knee — let the leg lengthen.",
            "Engage the right thigh and press into a straight leg.",
        ],
        "decrease": [
            "Bend your right knee deeper.",
            "Sit into your right leg more — add that bend.",
            "Your right leg needs more bend — drop into it.",
            "Soften and bend your right knee.",
            "Let your right knee sink lower.",
            "Track your right knee forward and bend deeper.",
        ],
    },
}


_POSE_SPECIFIC: dict[str, dict[str, dict[str, list[str]]]] = {
    "Warrior II": {
        "left_knee": {
            "decrease": [
                "Warrior Two lives in the legs — sink deeper into that left knee, directly over your ankle.",
                "Bend your left knee to ninety degrees — it's the engine of Warrior Two.",
                "Let your left knee track over your pinky toe and go lower.",
            ],
            "increase": [
                "Left knee is past the ankle — draw it back so it stays stacked.",
                "Ease back with the left knee — keep it stacked over your foot.",
            ],
        },
        "right_knee": {
            "decrease": [
                "Sink into that right knee — Warrior Two wants depth.",
                "Bend deeper on the right — front knee directly over your ankle.",
                "Let your right knee travel forward and down.",
            ],
            "increase": [
                "Right knee is past the ankle — draw it back slightly.",
                "Keep your right knee stacked over your foot, not past it.",
            ],
        },
        "left_shoulder": {
            "decrease": [
                "Arms in Warrior Two are parallel to the floor — lower your left arm to match.",
                "Two wings spreading equally — let your left arm float down to shoulder height.",
            ],
            "increase": [
                "Left arm is dropping below shoulder height — extend it out level.",
                "Reach your left arm fully out to the side, parallel to the floor.",
            ],
        },
        "right_shoulder": {
            "decrease": [
                "Let your right arm float down to shoulder height — level with the floor.",
                "Right arm is too high for Warrior Two — ease it down to shoulder level.",
            ],
            "increase": [
                "Right arm is dropping — reach it out level with your shoulder.",
                "Extend your right arm out, parallel to the floor.",
            ],
        },
    },
    "Warrior I": {
        "left_shoulder": {
            "increase": [
                "In Warrior One both arms reach straight up — lift your left arm higher.",
                "Warrior One has arms overhead — let your left arm sweep upward.",
            ],
            "decrease": [
                "Release your left arm slightly — you're reaching back a little too far.",
            ],
        },
        "right_shoulder": {
            "increase": [
                "Both arms reach to the sky in Warrior One — raise your right arm up.",
                "Lift your right arm overhead — imagine pressing the ceiling up.",
            ],
        },
        "left_knee": {
            "decrease": [
                "Warrior One wants a deep front bend — sink your left knee toward ninety degrees.",
                "Bend deeper into that left knee — let the hips drop down.",
            ],
        },
        "right_knee": {
            "decrease": [
                "Bend deeper into your right knee — hips are dropping in Warrior One.",
                "Sink your right knee toward ninety degrees.",
            ],
        },
    },
    "Chair Pose": {
        "left_knee": {
            "decrease": [
                "Chair Pose asks you to sit as if on an invisible chair — bend deeper.",
                "Sit lower — imagine hovering just above a chair seat.",
                "Lower your hips and let those knees bend.",
            ],
            "increase": [
                "You've gone quite deep — rise up slightly, keeping knees behind toes.",
            ],
        },
        "right_knee": {
            "decrease": [
                "Both knees in Chair — sink lower on the right too.",
                "Right knee needs more bend — sit down into it.",
            ],
        },
        "left_shoulder": {
            "increase": [
                "Arms reach overhead in Chair — lift them higher, biceps by your ears.",
                "Sweep your arms up — reach tall through the fingertips.",
            ],
            "decrease": [
                "Ease your arms back slightly — keep them framing your head.",
            ],
        },
    },
    "Downward Dog": {
        "left_hip": {
            "increase": [
                "Send your hips higher — lift the sit bones to the ceiling.",
                "Downward Dog is an inverted V — drive the hips up and back.",
            ],
            "decrease": [
                "Ease your hips down slightly — you're over-tucking.",
            ],
        },
        "left_knee": {
            "increase": [
                "Straighten your left leg as much as your hamstrings allow — heel toward the floor.",
                "Left leg can lengthen more — micro-straighten the knee.",
            ],
            "decrease": [
                "Soften your left knee — a slight bend is fine if your hamstrings are tight.",
            ],
        },
        "right_knee": {
            "increase": [
                "Press your right heel toward the floor and straighten the leg.",
            ],
            "decrease": [
                "Soften your right knee — pedalling is great here.",
            ],
        },
    },
    "Tree Pose": {
        "left_shoulder": {
            "increase": [
                "Let your arms blossom overhead — lift them for Tree.",
                "Reach your arms up — like branches growing toward the light.",
            ],
        },
        "right_hip": {
            "increase": [
                "Let your right hip open outward — the bent knee points to the side.",
                "Open your right hip — let the knee fall away from the body.",
            ],
            "decrease": [
                "Your right hip is flaring out — draw the knee a little forward.",
            ],
        },
    },
    "Cobra Pose": {
        "left_elbow": {
            "decrease": [
                "Keep a gentle bend in your elbows for Cobra — don't lock out the arms.",
                "Cobra isn't straight arms — soften those elbows.",
            ],
            "increase": [
                "Extend through your arms a little more — lift that chest.",
            ],
        },
        "right_elbow": {
            "decrease": [
                "Soften your right elbow — Cobra keeps a bend in the arms.",
            ],
            "increase": [
                "Press through the right hand and extend — lift the chest more.",
            ],
        },
    },
    "Child's Pose": {
        "left_hip": {
            "decrease": [
                "Sink your hips back toward your heels — let gravity help.",
                "Melt your hips down — Child's Pose is all about releasing.",
            ],
        },
        "right_hip": {
            "decrease": [
                "Let the hips drop back and down toward the heels.",
            ],
        },
    },
    "Downward Dog": {
        "left_shoulder": {
            "increase": [
                "Press into the floor and let your shoulders externally rotate — open them up.",
            ],
            "decrease": [
                "Ease your shoulders away from your ears — there's tension there.",
            ],
        },
    },
}


_COMBINED: dict[frozenset, list[str]] = {
    frozenset({"left_knee:decrease", "right_knee:decrease"}): [
        "Sink into both knees equally — get lower.",
        "Both legs want more bend — lower your hips.",
        "Drop your weight down — bend both knees deeper.",
    ],
    frozenset({"left_shoulder:increase", "right_shoulder:increase"}): [
        "Sweep both arms up — reach them overhead.",
        "Both arms need to lift — raise them together.",
        "Float both arms up and extend through the fingertips.",
    ],
    frozenset({"left_shoulder:decrease", "right_shoulder:decrease"}): [
        "Lower both arms — let them relax down.",
        "Release both arms down — soften the shoulders.",
        "Both arms can come down — ease them toward your sides.",
    ],
    frozenset({"left_elbow:increase", "right_elbow:increase"}): [
        "Extend both arms — straighten through both elbows.",
        "Reach out through both arms — lengthen them.",
    ],
    frozenset({"left_knee:decrease", "right_knee:decrease", "left_shoulder:increase"}): [
        "Bend both knees and lift your arms — find the two movements at once.",
    ],
    frozenset({"left_hip:increase", "right_hip:decrease"}): [
        "Open the left hip while squaring the right — rotate and settle.",
    ],
    frozenset({"left_knee:increase", "right_knee:increase"}): [
        "Straighten both legs — press through both heels.",
        "Both legs can lengthen — extend through the knees.",
    ],
}


_ENCOURAGEMENT_SMALL = [
    "Almost there.",
    "Really close now.",
    "Nearly perfect.",
    "Just a hair off.",
    "Tiny adjustment.",
    "You're so close.",
]

_ENCOURAGEMENT_MEDIUM = [
    "You're doing well — keep adjusting.",
    "Good effort — stay with it.",
    "Nice work — keep breathing through it.",
    "You're finding it.",
]

_ENCOURAGEMENT_LARGE = [
    "Take your time — this one takes practice.",
    "Breathe and keep working through it.",
    "Every body is different — do what you can.",
    "Be patient with yourself — you're improving.",
]

_WELL_DONE = [
    "That's it! Now take a deep inhale, and hold it.",
    "Beautiful alignment. Exhale and settle into the shape.",
    "Perfect—you've found it! Breathe deeply.",
    "That's exactly right. Inhale to lengthen, exhale to hold.",
    "Great shape! Lock it in and focus on your breath.",
    "Spot on. Feel that alignment, take a slow breath in.",
    "You've nailed it. Deep breath in, slow breath out.",
    "Yes! That's the shape. Close your eyes and breathe.",
]

_SCORE_TEMPLATES = {
    "excellent": [
        "You're at {score} percent — excellent.",
        "{score} percent match. Beautiful.",
        "Ninety plus — your body is right where it needs to be.",
    ],
    "good": [
        "You're at {score} percent — good alignment, small adjustments left.",
        "{score} percent. Getting really close.",
        "Looking good at {score} — just a little more to refine.",
    ],
    "fair": [
        "{score} percent — you're making progress.",
        "At {score} percent — keep working through the corrections.",
        "{score} percent. Stay focused and keep breathing.",
    ],
    "needs work": [
        "{score} percent right now — let's work through this together.",
        "You're at {score} percent. No rush — keep adjusting.",
        "{score} percent — take it one joint at a time.",
    ],
}


def build_correction_message(
    comparison: dict[str, dict],
    pose_name: str = "",
    max_corrections: int = 2,
) -> Optional[str]:
    misaligned = [
        (j, info)
        for j, info in comparison.items()
        if info.get("misaligned") and info.get("direction") and info.get("diff") is not None
    ]
    if not misaligned:
        return None

    misaligned.sort(key=lambda x: x[1]["diff"], reverse=True)
    top = misaligned[:max_corrections]
    avg_diff = sum(i["diff"] for _, i in top) / len(top)

    # Try combined message
    if len(top) >= 2:
        key = frozenset(f"{j}:{i['direction']}" for j, i in top[:2])
        combined = _COMBINED.get(key)
        if combined:
            return _add_encouragement(random.choice(combined), avg_diff)

    # Per-joint
    parts = []
    for joint, info in top:
        direction = info["direction"]
        diff      = info["diff"]
        pose_overrides = _POSE_SPECIFIC.get(pose_name, {})
        joint_overrides = pose_overrides.get(joint, {}).get(direction, [])
        templates = joint_overrides if joint_overrides else _JOINT_TEMPLATES.get(joint, {}).get(direction, [])

        if templates:
            if diff > 35:
                parts.append(templates[0])
            elif diff > 20:
                parts.append(random.choice(templates[:4] if len(templates) >= 4 else templates))
            else:
                parts.append(random.choice(templates))
        else:
            friendly = config.JOINT_FRIENDLY_NAMES.get(joint, joint.replace("_", " "))
            verb = "open" if direction == "increase" else "close"
            parts.append(f"{verb.capitalize()} your {friendly} a little.")

    if not parts:
        return None
    return _add_encouragement(" ".join(parts), avg_diff)


def _add_encouragement(msg: str, avg_diff: float) -> str:
    r = random.random()
    if avg_diff < 15 and r < 0.70:
        return msg + " " + random.choice(_ENCOURAGEMENT_SMALL)
    elif avg_diff < 30 and r < 0.40:
        return msg + " " + random.choice(_ENCOURAGEMENT_MEDIUM)
    elif avg_diff >= 30 and r < 0.25:
        return msg + " " + random.choice(_ENCOURAGEMENT_LARGE)
    return msg


def build_score_message(score: float, label: str) -> str:
    templates = _SCORE_TEMPLATES.get(label, _SCORE_TEMPLATES["fair"])
    return random.choice(templates).format(score=int(score))


def build_well_done_message() -> str:
    return random.choice(_WELL_DONE)


def build_step_message(pose_name: str, step: str, step_num: int, total: int) -> str:
    intros = [f"Step {step_num} of {total}:", "Next,", "Now,", f"Step {step_num} —", ""]
    intro = random.choice(intros)
    return f"{intro} {step}".strip()


def build_pose_intro(pose_name: str, sanskrit: str, description: str) -> str:
    openers = [
        f"The instructor has moved into {pose_name}",
        f"We're going into {pose_name} now",
        f"Coming up is {pose_name}",
        f"Next pose is {pose_name}",
    ]
    opener = random.choice(openers)
    if sanskrit:
        return f"{opener}, {sanskrit}. {description}"
    return f"{opener}. {description}"
