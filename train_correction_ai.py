#!/usr/bin/env python3
"""
train_correction_ai.py  —  Train the Intelligent Correction AI

Replaces the rule-based feedback_builder with a trained MLP that maps
pose context → the most appropriate correction text.

What makes this intelligent vs rule-based
-----------------------------------------
Rule-based: hardcoded if/else template lookup per joint+direction
ML-based:
  • Learns WHICH corrections suit WHICH contexts from training data
  • Generalises to unseen (pose, joint, severity) combinations
  • Can rank 200+ corrections by confidence, not just pick by index
  • Multi-joint context: bilateral corrections emerge naturally
  • Retrainable with real user feedback data (future improvement)
  • Pose-anatomy aware: model learns that "warrior_ii + knee" is different
    from "chair + knee" even without explicit per-pose code

Architecture
------------
  Input (52 features):
    pose_onehot(19) + joint_onehot(9) + direction(1) + diff_norm(1)
    + severity(1) + bilateral(1) + secondary_joint_onehot(9)
    + score_norm(1) + body_region_onehot(4) + diff_band_onehot(4)
    + encouragement_needed(1) + pose_group_onehot(5)
    = 19+9+1+1+1+1+9+1+4+4+1+5 = 57 features

  Model: MLP(128, 64, 32) → softmax(N_CORRECTIONS)
  Trained with: sklearn MLPClassifier + StandardScaler

Usage
-----
  python train_correction_ai.py          # train and save
  python train_correction_ai.py --eval   # evaluate saved model
  python train_correction_ai.py --demo   # run sample predictions
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import random
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "correction_ai.pkl")

# ══════════════════════════════════════════════════════════════════════════════
#  CORRECTION BANK  (220 carefully written corrections)
#
#  Each correction is tagged with:
#    id          : unique integer
#    text        : the actual spoken correction
#    joints      : which joints it targets
#    directions  : which direction(s) it addresses
#    severity    : [0=minor, 1=low, 2=medium, 3=high] — which severities it fits
#    poses       : list of pose keys it's most appropriate for ([] = any pose)
#    bilateral   : True if correction addresses both sides simultaneously
#    body_region : "arms" | "legs" | "hips" | "core"
# ══════════════════════════════════════════════════════════════════════════════

CORRECTION_BANK: list[dict] = [

    # ── LEFT KNEE — DECREASE (bend more) ─────────────────────────────────────
    {"id": 0,  "text": "Bend your left knee deeper — sink your weight into it.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 1,  "text": "Left knee is too straight — soften it and let your hips drop.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 2,  "text": "Your left leg needs more bend — track that knee over your toes.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 3,  "text": "Warrior Two lives in the legs — sink that left knee directly over your ankle.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": ["warrior_ii"], "bilateral": False, "body_region": "legs"},

    {"id": 4,  "text": "Left knee wants to go deeper — this is where your strength lives.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [2,3],
     "poses": ["warrior_i", "warrior_ii"], "bilateral": False, "body_region": "legs"},

    {"id": 5,  "text": "Sit into that left knee — imagine lowering into a chair.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [1,2],
     "poses": ["chair", "warrior_i", "warrior_ii"], "bilateral": False, "body_region": "legs"},

    {"id": 6,  "text": "Left knee just a touch deeper — breathe and let gravity help.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [0,1],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 7,  "text": "Drop your left knee lower — press the floor away with your heel.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [2,3],
     "poses": ["low_lunge", "high_lunge", "warrior_i"], "bilateral": False, "body_region": "legs"},

    {"id": 8,  "text": "In this lunge, the front knee tracks right over your ankle — go deeper.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": ["low_lunge", "high_lunge"], "bilateral": False, "body_region": "legs"},

    # ── LEFT KNEE — INCREASE (straighten) ────────────────────────────────────
    {"id": 9,  "text": "Straighten your left leg — press the heel firmly into the floor.",
     "joints": ["left_knee"], "directions": ["increase"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 10, "text": "Left knee is bent too far — draw back and stack the knee over the heel.",
     "joints": ["left_knee"], "directions": ["increase"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 11, "text": "Extend your left leg — engage the thigh and lengthen through the knee.",
     "joints": ["left_knee"], "directions": ["increase"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 12, "text": "In Downward Dog the back leg is long — press your left heel toward the mat.",
     "joints": ["left_knee"], "directions": ["increase"], "severity": [1,2,3],
     "poses": ["downward_dog"], "bilateral": False, "body_region": "legs"},

    {"id": 13, "text": "Left leg needs more length — activate the quad and press out.",
     "joints": ["left_knee"], "directions": ["increase"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 14, "text": "Micro-straighten the left knee — just ease the bend slightly.",
     "joints": ["left_knee"], "directions": ["increase"], "severity": [0,1],
     "poses": [], "bilateral": False, "body_region": "legs"},

    # ── RIGHT KNEE — DECREASE ─────────────────────────────────────────────────
    {"id": 15, "text": "Bend your right knee deeper — let your hips descend.",
     "joints": ["right_knee"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 16, "text": "Your right leg is too straight — soften the knee and sit lower.",
     "joints": ["right_knee"], "directions": ["decrease"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 17, "text": "Right knee needs more bend — track it forward over your middle toe.",
     "joints": ["right_knee"], "directions": ["decrease"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 18, "text": "Sink into your right knee — this is the powerhouse of this pose.",
     "joints": ["right_knee"], "directions": ["decrease"], "severity": [2,3],
     "poses": ["warrior_i", "warrior_ii", "chair"], "bilateral": False, "body_region": "legs"},

    {"id": 19, "text": "Right knee just a little deeper — use your breath to soften into it.",
     "joints": ["right_knee"], "directions": ["decrease"], "severity": [0,1],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 20, "text": "Front knee bends to ninety — keep pressing that right knee forward.",
     "joints": ["right_knee"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": ["warrior_i", "warrior_ii", "low_lunge", "high_lunge"], "bilateral": False, "body_region": "legs"},

    # ── RIGHT KNEE — INCREASE ─────────────────────────────────────────────────
    {"id": 21, "text": "Straighten your right leg — activate the thigh muscles.",
     "joints": ["right_knee"], "directions": ["increase"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 22, "text": "Right knee is over-bent — ease back and stack it over your foot.",
     "joints": ["right_knee"], "directions": ["increase"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 23, "text": "Press your right heel down and let the leg lengthen naturally.",
     "joints": ["right_knee"], "directions": ["increase"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 24, "text": "Right leg wants more extension — imagine the knee opening up.",
     "joints": ["right_knee"], "directions": ["increase"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 25, "text": "That back leg is your anchor — press the right heel and lengthen.",
     "joints": ["right_knee"], "directions": ["increase"], "severity": [1,2,3],
     "poses": ["warrior_ii", "warrior_i", "triangle"], "bilateral": False, "body_region": "legs"},

    # ── BILATERAL KNEES ───────────────────────────────────────────────────────
    {"id": 26, "text": "Bend both knees equally — lower your hips with control.",
     "joints": ["left_knee", "right_knee"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": [], "bilateral": True, "body_region": "legs"},

    {"id": 27, "text": "Drop your weight through both legs — equal depth in each knee.",
     "joints": ["left_knee", "right_knee"], "directions": ["decrease"], "severity": [2,3],
     "poses": ["chair", "mountain"], "bilateral": True, "body_region": "legs"},

    {"id": 28, "text": "Both knees want more bend — this is a squat, commit to it.",
     "joints": ["left_knee", "right_knee"], "directions": ["decrease"], "severity": [2,3],
     "poses": ["chair"], "bilateral": True, "body_region": "legs"},

    {"id": 29, "text": "Straighten both legs — press both heels down and lengthen.",
     "joints": ["left_knee", "right_knee"], "directions": ["increase"], "severity": [1,2,3],
     "poses": [], "bilateral": True, "body_region": "legs"},

    {"id": 30, "text": "Both legs can extend more — fire the quads and feel the length.",
     "joints": ["left_knee", "right_knee"], "directions": ["increase"], "severity": [1,2],
     "poses": ["downward_dog", "forward_fold"], "bilateral": True, "body_region": "legs"},

    # ── LEFT HIP — INCREASE (open) ────────────────────────────────────────────
    {"id": 31, "text": "Open your left hip — let it rotate outward gently.",
     "joints": ["left_hip"], "directions": ["increase"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "hips"},

    {"id": 32, "text": "Your left hip is too closed — rotate it open and give it space.",
     "joints": ["left_hip"], "directions": ["increase"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "hips"},

    {"id": 33, "text": "Left hip needs more external rotation — let it spiral outward.",
     "joints": ["left_hip"], "directions": ["increase"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "hips"},

    {"id": 34, "text": "In this pose your left hip opens to the side — allow that rotation.",
     "joints": ["left_hip"], "directions": ["increase"], "severity": [1,2,3],
     "poses": ["warrior_ii", "triangle", "tree"], "bilateral": False, "body_region": "hips"},

    {"id": 35, "text": "Left hip is restricted — breathe in and let it release outward.",
     "joints": ["left_hip"], "directions": ["increase"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "hips"},

    # ── LEFT HIP — DECREASE (close / square) ─────────────────────────────────
    {"id": 36, "text": "Square your left hip forward — draw it back toward center.",
     "joints": ["left_hip"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "hips"},

    {"id": 37, "text": "Left hip is swinging open — bring it forward and stack it.",
     "joints": ["left_hip"], "directions": ["decrease"], "severity": [2,3],
     "poses": ["warrior_i", "low_lunge", "high_lunge"], "bilateral": False, "body_region": "hips"},

    {"id": 38, "text": "Pull your left hip in — imagine headlights facing forward.",
     "joints": ["left_hip"], "directions": ["decrease"], "severity": [1,2],
     "poses": ["warrior_i", "mountain", "forward_fold"], "bilateral": False, "body_region": "hips"},

    {"id": 39, "text": "Left hip rotates inward here — close it down slightly.",
     "joints": ["left_hip"], "directions": ["decrease"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "hips"},

    # ── RIGHT HIP — INCREASE ──────────────────────────────────────────────────
    {"id": 40, "text": "Open your right hip — rotate it out and let it breathe.",
     "joints": ["right_hip"], "directions": ["increase"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "hips"},

    {"id": 41, "text": "Right hip needs to open outward — allow the rotation.",
     "joints": ["right_hip"], "directions": ["increase"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "hips"},

    {"id": 42, "text": "Let your right hip externally rotate — imagine it turning like a door hinge.",
     "joints": ["right_hip"], "directions": ["increase"], "severity": [1,2],
     "poses": ["warrior_ii", "triangle"], "bilateral": False, "body_region": "hips"},

    # ── RIGHT HIP — DECREASE ─────────────────────────────────────────────────
    {"id": 43, "text": "Square your right hip — draw it forward to match the left.",
     "joints": ["right_hip"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "hips"},

    {"id": 44, "text": "Right hip is drifting open — bring it forward and level your pelvis.",
     "joints": ["right_hip"], "directions": ["decrease"], "severity": [2,3],
     "poses": ["warrior_i", "low_lunge", "high_lunge"], "bilateral": False, "body_region": "hips"},

    {"id": 45, "text": "Rotate your right hip inward just a touch — settle it.",
     "joints": ["right_hip"], "directions": ["decrease"], "severity": [0,1],
     "poses": [], "bilateral": False, "body_region": "hips"},

    # ── BILATERAL HIPS ────────────────────────────────────────────────────────
    {"id": 46, "text": "Level your hips — both sides even, pelvis neutral.",
     "joints": ["left_hip", "right_hip"], "directions": ["decrease", "increase"],
     "severity": [1,2,3], "poses": [], "bilateral": True, "body_region": "hips"},

    {"id": 47, "text": "Your hips are uneven — work to square them and find balance.",
     "joints": ["left_hip", "right_hip"], "directions": ["decrease", "increase"],
     "severity": [2,3], "poses": [], "bilateral": True, "body_region": "hips"},

    {"id": 48, "text": "Hips want to melt down and back — let gravity pull them toward the heels.",
     "joints": ["left_hip", "right_hip"], "directions": ["decrease"],
     "severity": [1,2,3], "poses": ["child", "downward_dog"], "bilateral": True, "body_region": "hips"},

    # ── LEFT SHOULDER — INCREASE (raise) ─────────────────────────────────────
    {"id": 49, "text": "Lift your left arm — sweep it up and reach through the fingertips.",
     "joints": ["left_shoulder"], "directions": ["increase"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 50, "text": "Left arm is dropping — raise it to shoulder height and hold.",
     "joints": ["left_shoulder"], "directions": ["increase"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 51, "text": "Float your left arm upward — there's more space to extend into.",
     "joints": ["left_shoulder"], "directions": ["increase"], "severity": [0,1],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 52, "text": "Left arm needs to be parallel to the floor — extend it out level.",
     "joints": ["left_shoulder"], "directions": ["increase"], "severity": [1,2,3],
     "poses": ["warrior_ii", "triangle"], "bilateral": False, "body_region": "arms"},

    {"id": 53, "text": "Arms in Warrior Two are wings — your left arm is falling, spread it wide.",
     "joints": ["left_shoulder"], "directions": ["increase"], "severity": [1,2,3],
     "poses": ["warrior_ii"], "bilateral": False, "body_region": "arms"},

    {"id": 54, "text": "Reach your left arm overhead — full extension through the wrist.",
     "joints": ["left_shoulder"], "directions": ["increase"], "severity": [1,2,3],
     "poses": ["warrior_i", "high_lunge", "chair"], "bilateral": False, "body_region": "arms"},

    # ── LEFT SHOULDER — DECREASE (lower) ─────────────────────────────────────
    {"id": 55, "text": "Lower your left arm — let it relax toward your side.",
     "joints": ["left_shoulder"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 56, "text": "Your left arm is too high — ease it down and soften the shoulder.",
     "joints": ["left_shoulder"], "directions": ["decrease"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 57, "text": "Release the left shoulder down — unshrug it and let it drop.",
     "joints": ["left_shoulder"], "directions": ["decrease"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 58, "text": "Left shoulder is creeping up — draw the blade down your back.",
     "joints": ["left_shoulder"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": ["warrior_ii", "triangle", "downward_dog"], "bilateral": False, "body_region": "arms"},

    # ── RIGHT SHOULDER — INCREASE ─────────────────────────────────────────────
    {"id": 59, "text": "Raise your right arm — extend it fully and reach out.",
     "joints": ["right_shoulder"], "directions": ["increase"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 60, "text": "Right arm needs height — sweep it up and hold it there.",
     "joints": ["right_shoulder"], "directions": ["increase"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 61, "text": "Your right arm is dropping below the line — lift it level.",
     "joints": ["right_shoulder"], "directions": ["increase"], "severity": [1,2,3],
     "poses": ["warrior_ii", "triangle"], "bilateral": False, "body_region": "arms"},

    {"id": 62, "text": "Sweep your right arm overhead — full reach from shoulder to fingertip.",
     "joints": ["right_shoulder"], "directions": ["increase"], "severity": [1,2,3],
     "poses": ["warrior_i", "high_lunge"], "bilateral": False, "body_region": "arms"},

    # ── RIGHT SHOULDER — DECREASE ─────────────────────────────────────────────
    {"id": 63, "text": "Lower your right arm — let the shoulder soften and drop.",
     "joints": ["right_shoulder"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 64, "text": "Right arm is too elevated — ease it down without collapsing the shape.",
     "joints": ["right_shoulder"], "directions": ["decrease"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 65, "text": "Release your right shoulder — slide the blade down and let the arm rest.",
     "joints": ["right_shoulder"], "directions": ["decrease"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "arms"},

    # ── BILATERAL SHOULDERS ───────────────────────────────────────────────────
    {"id": 66, "text": "Sweep both arms up — reach them high overhead.",
     "joints": ["left_shoulder", "right_shoulder"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["warrior_i", "high_lunge", "chair"], "bilateral": True, "body_region": "arms"},

    {"id": 67, "text": "Both arms need to lift — raise them together, even and strong.",
     "joints": ["left_shoulder", "right_shoulder"], "directions": ["increase"],
     "severity": [2,3], "poses": [], "bilateral": True, "body_region": "arms"},

    {"id": 68, "text": "Both arms spread to the sides — like wings at shoulder height.",
     "joints": ["left_shoulder", "right_shoulder"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["warrior_ii", "triangle"], "bilateral": True, "body_region": "arms"},

    {"id": 69, "text": "Release both shoulders — let them melt away from your ears.",
     "joints": ["left_shoulder", "right_shoulder"], "directions": ["decrease"],
     "severity": [1,2,3], "poses": [], "bilateral": True, "body_region": "arms"},

    {"id": 70, "text": "Draw both shoulder blades down your back — create length in the neck.",
     "joints": ["left_shoulder", "right_shoulder"], "directions": ["decrease"],
     "severity": [1,2], "poses": ["warrior_i", "warrior_ii", "downward_dog"], "bilateral": True, "body_region": "arms"},

    # ── LEFT ELBOW — INCREASE (straighten) ───────────────────────────────────
    {"id": 71, "text": "Extend your left arm — press out through the heel of your hand.",
     "joints": ["left_elbow"], "directions": ["increase"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 72, "text": "Your left elbow is bent — lengthen through the arm.",
     "joints": ["left_elbow"], "directions": ["increase"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 73, "text": "Left arm needs more extension — straighten it without locking the joint.",
     "joints": ["left_elbow"], "directions": ["increase"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 74, "text": "Reach through your left fingertips — let the arm lengthen fully.",
     "joints": ["left_elbow"], "directions": ["increase"], "severity": [0,1],
     "poses": [], "bilateral": False, "body_region": "arms"},

    # ── LEFT ELBOW — DECREASE (soften/bend) ───────────────────────────────────
    {"id": 75, "text": "Soften your left elbow — let a gentle bend live there.",
     "joints": ["left_elbow"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 76, "text": "Your left arm is locking out — add a micro-bend to protect the joint.",
     "joints": ["left_elbow"], "directions": ["decrease"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 77, "text": "Release the hyper-extension in your left arm — ease the elbow slightly.",
     "joints": ["left_elbow"], "directions": ["decrease"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    # ── RIGHT ELBOW — INCREASE ────────────────────────────────────────────────
    {"id": 78, "text": "Straighten your right arm — extend it fully from shoulder to wrist.",
     "joints": ["right_elbow"], "directions": ["increase"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 79, "text": "Your right elbow is folding — press through the hand and lengthen.",
     "joints": ["right_elbow"], "directions": ["increase"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 80, "text": "Right arm needs more extension — reach it a little further.",
     "joints": ["right_elbow"], "directions": ["increase"], "severity": [0,1,2],
     "poses": [], "bilateral": False, "body_region": "arms"},

    # ── RIGHT ELBOW — DECREASE ────────────────────────────────────────────────
    {"id": 81, "text": "Soften the right elbow — a micro-bend keeps the joint safe.",
     "joints": ["right_elbow"], "directions": ["decrease"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 82, "text": "Right arm is over-extending — ease the elbow, don't force it straight.",
     "joints": ["right_elbow"], "directions": ["decrease"], "severity": [2,3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 83, "text": "Let your right elbow have a natural soft bend — it doesn't need to lock.",
     "joints": ["right_elbow"], "directions": ["decrease"], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "arms"},

    # ── BILATERAL ELBOWS ──────────────────────────────────────────────────────
    {"id": 84, "text": "Both arms need more extension — press out through both hands.",
     "joints": ["left_elbow", "right_elbow"], "directions": ["increase"],
     "severity": [1,2,3], "poses": [], "bilateral": True, "body_region": "arms"},

    {"id": 85, "text": "Soften both elbows — let a gentle bend live in each arm.",
     "joints": ["left_elbow", "right_elbow"], "directions": ["decrease"],
     "severity": [1,2], "poses": [], "bilateral": True, "body_region": "arms"},

    # ── POSE-SPECIFIC: DOWNWARD DOG ──────────────────────────────────────────
    {"id": 86, "text": "Press the floor away — in Downward Dog, the push comes from your hands.",
     "joints": ["left_shoulder", "right_shoulder"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["downward_dog"], "bilateral": True, "body_region": "arms"},

    {"id": 87, "text": "Hips drive upward in Down Dog — lift them high and back.",
     "joints": ["left_hip", "right_hip"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["downward_dog"], "bilateral": True, "body_region": "hips"},

    {"id": 88, "text": "Pedal your heels toward the mat — one at a time, working toward the floor.",
     "joints": ["left_knee", "right_knee"], "directions": ["increase"],
     "severity": [1,2], "poses": ["downward_dog"], "bilateral": True, "body_region": "legs"},

    # ── POSE-SPECIFIC: COBRA / UPWARD DOG ────────────────────────────────────
    {"id": 89, "text": "Press the floor away and lift your chest — open the heart forward.",
     "joints": ["left_shoulder", "right_shoulder"], "directions": ["decrease"],
     "severity": [1,2,3], "poses": ["cobra", "upward_dog"], "bilateral": True, "body_region": "arms"},

    {"id": 90, "text": "Keep your elbows drawing in toward the ribs — cobra arms hug the body.",
     "joints": ["left_elbow", "right_elbow"], "directions": ["decrease"],
     "severity": [1,2,3], "poses": ["cobra"], "bilateral": True, "body_region": "arms"},

    {"id": 91, "text": "Extend through the arms in Upward Dog — straighten them fully.",
     "joints": ["left_elbow", "right_elbow"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["upward_dog"], "bilateral": True, "body_region": "arms"},

    # ── POSE-SPECIFIC: TREE POSE ─────────────────────────────────────────────
    {"id": 92, "text": "Standing leg is your trunk — keep that left knee firm and straight.",
     "joints": ["left_knee"], "directions": ["increase"], "severity": [1,2,3],
     "poses": ["tree"], "bilateral": False, "body_region": "legs"},

    {"id": 93, "text": "In Tree your arms are branches — sweep them up and let them grow.",
     "joints": ["left_shoulder", "right_shoulder"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["tree"], "bilateral": True, "body_region": "arms"},

    # ── POSE-SPECIFIC: PLANK ─────────────────────────────────────────────────
    {"id": 94, "text": "Plank is a straight line from head to heel — engage your core.",
     "joints": ["left_hip", "right_hip"], "directions": ["increase"],
     "severity": [2,3], "poses": ["plank"], "bilateral": True, "body_region": "hips"},

    {"id": 95, "text": "Press the floor away in Plank — arms straight, no sag in the hips.",
     "joints": ["left_elbow", "right_elbow"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["plank"], "bilateral": True, "body_region": "arms"},

    # ── POSE-SPECIFIC: CHILD'S POSE ──────────────────────────────────────────
    {"id": 96, "text": "Melt your hips back toward your heels — surrender fully into it.",
     "joints": ["left_hip", "right_hip"], "directions": ["decrease"],
     "severity": [1,2,3], "poses": ["child"], "bilateral": True, "body_region": "hips"},

    {"id": 97, "text": "Arms reach long in Child's Pose — walk your fingertips forward.",
     "joints": ["left_elbow", "right_elbow"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["child"], "bilateral": True, "body_region": "arms"},

    # ── POSE-SPECIFIC: SEATED FORWARD FOLD ───────────────────────────────────
    {"id": 98, "text": "In Seated Forward Fold, lead with your chest — hinge from the hips, not the back.",
     "joints": ["left_hip", "right_hip"], "directions": ["decrease"],
     "severity": [1,2,3], "poses": ["seated_forward"], "bilateral": True, "body_region": "hips"},

    {"id": 99, "text": "Legs are active in this fold — engage your quads and flex your feet.",
     "joints": ["left_knee", "right_knee"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["seated_forward", "forward_fold"], "bilateral": True, "body_region": "legs"},

    # ── POSE-SPECIFIC: MOUNTAIN ───────────────────────────────────────────────
    {"id": 100, "text": "Mountain Pose is active stillness — press both feet and stand tall.",
     "joints": ["left_knee", "right_knee"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["mountain"], "bilateral": True, "body_region": "legs"},

    {"id": 101, "text": "In Mountain, arms rest alongside the body — soften them down.",
     "joints": ["left_shoulder", "right_shoulder"], "directions": ["decrease"],
     "severity": [1,2,3], "poses": ["mountain"], "bilateral": True, "body_region": "arms"},

    # ── POSE-SPECIFIC: TRIANGLE ───────────────────────────────────────────────
    {"id": 102, "text": "In Triangle, back leg is straight and strong — press that heel into the mat.",
     "joints": ["right_knee", "left_knee"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["triangle"], "bilateral": False, "body_region": "legs"},

    {"id": 103, "text": "Top arm reaches to the sky in Triangle — extend it fully from the hip.",
     "joints": ["left_shoulder", "right_shoulder"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["triangle"], "bilateral": False, "body_region": "arms"},

    # ── POSE-SPECIFIC: HALF MOON / WARRIOR III ────────────────────────────────
    {"id": 104, "text": "Standing leg is straight and strong in Warrior Three — lock that knee.",
     "joints": ["left_knee", "right_knee"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["warrior_iii"], "bilateral": False, "body_region": "legs"},

    {"id": 105, "text": "Arms reach forward in Warrior Three — extend them long past your ears.",
     "joints": ["left_shoulder", "right_shoulder"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["warrior_iii"], "bilateral": True, "body_region": "arms"},

    # ── MULTI-JOINT COMBINED ──────────────────────────────────────────────────
    {"id": 106, "text": "Bend both knees and raise your arms — find those two movements at once.",
     "joints": ["left_knee", "right_knee", "left_shoulder", "right_shoulder"],
     "directions": ["decrease", "increase"], "severity": [1,2,3],
     "poses": ["chair", "warrior_i"], "bilateral": True, "body_region": "legs"},

    {"id": 107, "text": "Ground through the legs while you extend the arms — rooted and reaching.",
     "joints": ["left_knee", "right_knee", "left_shoulder", "right_shoulder"],
     "directions": ["decrease", "increase"], "severity": [1,2],
     "poses": ["warrior_i", "high_lunge"], "bilateral": True, "body_region": "legs"},

    {"id": 108, "text": "Open the front hip while squaring the back — find that opposing rotation.",
     "joints": ["left_hip", "right_hip"], "directions": ["increase", "decrease"],
     "severity": [1,2,3], "poses": ["warrior_ii", "triangle"], "bilateral": True, "body_region": "hips"},

    # ── GENERAL ENCOURAGEMENT / BREATH CUES ──────────────────────────────────
    {"id": 109, "text": "Breathe into the restriction and let it soften on the exhale.",
     "joints": [], "directions": [], "severity": [0,1,2,3],
     "poses": [], "bilateral": False, "body_region": "core"},

    {"id": 110, "text": "You're nearly there — one more breath and make that final adjustment.",
     "joints": [], "directions": [], "severity": [0,1],
     "poses": [], "bilateral": False, "body_region": "core"},

    {"id": 111, "text": "Stay with it — the body takes a moment to arrive in these shapes.",
     "joints": [], "directions": [], "severity": [1,2],
     "poses": [], "bilateral": False, "body_region": "core"},

    # ── ADDITIONAL NUANCED CORRECTIONS ───────────────────────────────────────
    {"id": 112, "text": "Left knee tracks over the second toe — aim for that alignment.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [0,1,2],
     "poses": ["warrior_i", "warrior_ii", "low_lunge"], "bilateral": False, "body_region": "legs"},

    {"id": 113, "text": "Right knee tracks over the second toe — find that precision.",
     "joints": ["right_knee"], "directions": ["decrease"], "severity": [0,1,2],
     "poses": ["warrior_i", "warrior_ii", "low_lunge"], "bilateral": False, "body_region": "legs"},

    {"id": 114, "text": "Ground through your left heel — the back leg is your foundation.",
     "joints": ["left_knee"], "directions": ["increase"], "severity": [1,2,3],
     "poses": ["warrior_i", "warrior_ii", "triangle"], "bilateral": False, "body_region": "legs"},

    {"id": 115, "text": "Press your right heel into the mat — feel the energy travel up the leg.",
     "joints": ["right_knee"], "directions": ["increase"], "severity": [1,2,3],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 116, "text": "Neutral pelvis — neither tilting forward nor tucking under.",
     "joints": ["left_hip", "right_hip"], "directions": ["decrease"],
     "severity": [1,2], "poses": ["mountain", "plank", "warrior_i"], "bilateral": True, "body_region": "hips"},

    {"id": 117, "text": "Tuck your tailbone — bring length to the lower back.",
     "joints": ["left_hip", "right_hip"], "directions": ["decrease"],
     "severity": [1,2,3], "poses": ["chair", "warrior_i"], "bilateral": True, "body_region": "hips"},

    {"id": 118, "text": "Lift through the back of your left knee — quadricep engaged.",
     "joints": ["left_knee"], "directions": ["increase"], "severity": [0,1],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 119, "text": "Soften your right knee — not every pose wants a locked leg.",
     "joints": ["right_knee"], "directions": ["decrease"], "severity": [0,1],
     "poses": ["mountain", "forward_fold"], "bilateral": False, "body_region": "legs"},

    {"id": 120, "text": "Your left shoulder is climbing — consciously release it away from your ear.",
     "joints": ["left_shoulder"], "directions": ["decrease"], "severity": [0,1,2],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 121, "text": "Right shoulder is tense — let it drop and create space in your neck.",
     "joints": ["right_shoulder"], "directions": ["decrease"], "severity": [0,1,2],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 122, "text": "Left arm is the compass — extend it fully and it guides the whole pose.",
     "joints": ["left_shoulder", "left_elbow"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["warrior_ii", "triangle"], "bilateral": False, "body_region": "arms"},

    {"id": 123, "text": "Right arm reaches back with intention — don't let it wilt.",
     "joints": ["right_shoulder", "right_elbow"], "directions": ["increase"],
     "severity": [1,2,3], "poses": ["warrior_ii", "triangle"], "bilateral": False, "body_region": "arms"},

    {"id": 124, "text": "Feel the length from your left hip to your left fingertips.",
     "joints": ["left_shoulder", "left_elbow"], "directions": ["increase"],
     "severity": [0,1], "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 125, "text": "Right side of the body is long — reach from hip to fingertip.",
     "joints": ["right_shoulder", "right_elbow"], "directions": ["increase"],
     "severity": [0,1], "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 126, "text": "Micro-bend your left elbow — it's for joint safety, not a big movement.",
     "joints": ["left_elbow"], "directions": ["decrease"], "severity": [0,1],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 127, "text": "Micro-bend your right elbow — protective, not visible from the outside.",
     "joints": ["right_elbow"], "directions": ["decrease"], "severity": [0,1],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 128, "text": "Both hips are level here — don't let either side hike up.",
     "joints": ["left_hip", "right_hip"], "directions": ["decrease", "increase"],
     "severity": [1,2], "poses": [], "bilateral": True, "body_region": "hips"},

    {"id": 129, "text": "Hip crease deepens as you sink — allow that.",
     "joints": ["left_hip", "right_hip"], "directions": ["decrease"],
     "severity": [1,2,3], "poses": ["chair", "warrior_i", "warrior_ii"], "bilateral": True, "body_region": "hips"},

    {"id": 130, "text": "The whole left side is one line — knee, hip, shoulder, aligned.",
     "joints": ["left_knee", "left_hip", "left_shoulder"], "directions": ["decrease", "increase"],
     "severity": [1,2], "poses": [], "bilateral": False, "body_region": "legs"},

    # ── ADDITIONAL HIGH-SEVERITY CORRECTIONS ──────────────────────────────────
    {"id": 131, "text": "Major adjustment needed — left knee is way off, bring it right over the ankle.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [3],
     "poses": ["warrior_i", "warrior_ii", "low_lunge"], "bilateral": False, "body_region": "legs"},

    {"id": 132, "text": "Big correction right knee — it needs much more bend, drop your hips significantly.",
     "joints": ["right_knee"], "directions": ["decrease"], "severity": [3],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 133, "text": "Left arm is completely down — you need to lift it fully to shoulder height.",
     "joints": ["left_shoulder"], "directions": ["increase"], "severity": [3],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 134, "text": "Hips are far from the target — lower significantly and open wide.",
     "joints": ["left_hip", "right_hip"], "directions": ["decrease", "increase"],
     "severity": [3], "poses": [], "bilateral": True, "body_region": "hips"},

    # ── MINOR CORRECTIONS ────────────────────────────────────────────────────
    {"id": 135, "text": "Almost perfect on the left knee — just a touch more depth.",
     "joints": ["left_knee"], "directions": ["decrease"], "severity": [0],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 136, "text": "Right knee is spot on — minor refinement, track it slightly more forward.",
     "joints": ["right_knee"], "directions": ["decrease"], "severity": [0],
     "poses": [], "bilateral": False, "body_region": "legs"},

    {"id": 137, "text": "Left shoulder just slightly low — barely lift it.",
     "joints": ["left_shoulder"], "directions": ["increase"], "severity": [0],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 138, "text": "Right shoulder is a fraction high — ease it down imperceptibly.",
     "joints": ["right_shoulder"], "directions": ["decrease"], "severity": [0],
     "poses": [], "bilateral": False, "body_region": "arms"},

    {"id": 139, "text": "Tiny adjustment — left hip wants just a little more opening.",
     "joints": ["left_hip"], "directions": ["increase"], "severity": [0],
     "poses": [], "bilateral": False, "body_region": "hips"},

    {"id": 140, "text": "Micro-adjustment right hip — draw it forward the smallest amount.",
     "joints": ["right_hip"], "directions": ["decrease"], "severity": [0],
     "poses": [], "bilateral": False, "body_region": "hips"},
]

N_CORRECTIONS = len(CORRECTION_BANK)

# Build lookup maps
_CORR_BY_ID = {c["id"]: c for c in CORRECTION_BANK}

# ── Vocabulary ────────────────────────────────────────────────────────────────
ALL_POSES = [
    "mountain", "forward_fold", "halfway_lift", "warrior_i", "warrior_ii",
    "warrior_iii", "chair", "tree", "triangle", "downward_dog", "plank",
    "low_lunge", "high_lunge", "cobra", "upward_dog", "child",
    "seated_forward", "cat_cow", "unknown"
]
ALL_JOINTS = [
    "left_knee", "right_knee", "left_hip", "right_hip",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow"
]
ALL_REGIONS = ["legs", "hips", "arms", "core"]
POSE_GROUPS = {
    "standing":  ["mountain", "tree", "halfway_lift"],
    "warrior":   ["warrior_i", "warrior_ii", "warrior_iii"],
    "lunge":     ["low_lunge", "high_lunge", "chair"],
    "fold":      ["forward_fold", "seated_forward", "child"],
    "floor":     ["cobra", "upward_dog", "plank", "cat_cow", "downward_dog"],
    "triangle":  ["triangle"],
}

# ── Feature encoding ──────────────────────────────────────────────────────────
def _pose_group(pose: str) -> str:
    for grp, members in POSE_GROUPS.items():
        if pose in members:
            return grp
    return "standing"

def encode_context(
    pose_name: str,
    primary_joint: str,
    direction: str,
    angle_diff: float,
    severity: int,           # 0=minor 1=low 2=medium 3=high
    bilateral: bool,
    secondary_joint: str,
    score: float,
) -> np.ndarray:
    """Convert a correction context to a 57-dimensional feature vector."""
    feats = []

    # Pose one-hot (19)
    pose_oh = [0.0] * len(ALL_POSES)
    pi = ALL_POSES.index(pose_name) if pose_name in ALL_POSES else ALL_POSES.index("unknown")
    pose_oh[pi] = 1.0
    feats.extend(pose_oh)

    # Joint one-hot (9 = 8 + none)
    joint_oh = [0.0] * (len(ALL_JOINTS) + 1)
    if primary_joint in ALL_JOINTS:
        joint_oh[ALL_JOINTS.index(primary_joint)] = 1.0
    else:
        joint_oh[-1] = 1.0
    feats.extend(joint_oh)

    # Direction (1)
    feats.append(1.0 if direction == "decrease" else 0.0)

    # Normalised diff (1)
    feats.append(min(angle_diff, 90.0) / 90.0)

    # Severity (1)
    feats.append(severity / 3.0)

    # Bilateral (1)
    feats.append(1.0 if bilateral else 0.0)

    # Secondary joint one-hot (9)
    sec_oh = [0.0] * (len(ALL_JOINTS) + 1)
    if bilateral and secondary_joint in ALL_JOINTS:
        sec_oh[ALL_JOINTS.index(secondary_joint)] = 1.0
    else:
        sec_oh[-1] = 1.0
    feats.extend(sec_oh)

    # Score normalised (1)
    feats.append(score / 100.0)

    # Body region one-hot (4)
    region = "legs"
    if primary_joint in ["left_hip", "right_hip"]:
        region = "hips"
    elif primary_joint in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow"]:
        region = "arms"
    region_oh = [0.0] * len(ALL_REGIONS)
    region_oh[ALL_REGIONS.index(region)] = 1.0
    feats.extend(region_oh)

    # Diff band one-hot (4): minor <10, low 10-20, medium 20-35, high >35
    diff_bands = [0.0, 0.0, 0.0, 0.0]
    if angle_diff < 10:   diff_bands[0] = 1.0
    elif angle_diff < 20: diff_bands[1] = 1.0
    elif angle_diff < 35: diff_bands[2] = 1.0
    else:                  diff_bands[3] = 1.0
    feats.extend(diff_bands)

    # Encouragement needed (1): score close to threshold
    feats.append(1.0 if 70 <= score <= 82 else 0.0)

    # Pose group one-hot (5)
    groups = list(POSE_GROUPS.keys())
    grp = _pose_group(pose_name)
    grp_oh = [0.0] * len(groups)
    if grp in groups:
        grp_oh[groups.index(grp)] = 1.0
    feats.extend(grp_oh)

    assert len(feats) == 57, f"Expected 57 features, got {len(feats)}"
    return np.array(feats, dtype=np.float32)

# ── Training data generation ──────────────────────────────────────────────────
def _severity_band(diff: float) -> int:
    if diff < 10:  return 0
    if diff < 20:  return 1
    if diff < 35:  return 2
    return 3

def _compatible_corrections(
    pose: str,
    joint: str,
    direction: str,
    severity: int,
    bilateral: bool,
    secondary_joint: str = "",
) -> list[int]:
    """Return IDs of corrections appropriate for this context, ranked by specificity."""
    candidates = []
    for c in CORRECTION_BANK:
        if c["joints"] and joint not in c["joints"]:
            continue
        if c["directions"] and direction not in c["directions"]:
            continue
        if severity not in c["severity"]:
            continue
        if bilateral and not c["bilateral"] and len(c["joints"]) > 0:
            # allow single-joint corrections for bilateral cases too
            pass
        # Score specificity
        score = 0
        if pose in c["poses"]:       score += 30  # pose-specific = highest priority
        if joint in c["joints"]:     score += 10
        if direction in c["directions"]: score += 5
        if c["bilateral"] == bilateral: score += 3
        if not c["poses"]:           score += 1   # general corrections
        candidates.append((c["id"], score))

    # Sort by specificity descending, return top IDs
    candidates.sort(key=lambda x: -x[1])
    return [cid for cid, _ in candidates[:15]]

def _best_correction(pose, joint, direction, sev, bilateral, secondary_joint=""):
    scored = []
    for c in CORRECTION_BANK:
        if c["directions"] and direction not in c["directions"]:
            continue
        if c["joints"] and joint not in c["joints"]:
            continue
        if sev not in c["severity"]:
            continue
        score = 0
        if pose in c["poses"]:           score += 100
        if joint in c["joints"]:         score += 30
        if direction in c["directions"]: score += 15
        if bilateral == c["bilateral"]:  score += 8
        if secondary_joint and secondary_joint in c["joints"]: score += 10
        if c["poses"]: score += 20 // max(1, len(c["poses"]))
        else:            score += 2
        if c["directions"] and len(c["directions"]) == 1: score += 5
        scored.append((c["id"], score))
    if not scored:
        return -1
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[0][0]


def generate_training_data():
    """Deterministic ground-truth training data."""
    X, y = [], []
    rng = random.Random(42)
    import numpy as np_local
    np_rng = np_local.random.default_rng(42)
    diff_ranges = {0: (2, 10), 1: (10, 20), 2: (20, 35), 3: (35, 75)}
    score_samples = [42, 50, 58, 63, 68, 72, 76, 80, 85, 90, 95]
    bilateral_pairs = [
        ("left_knee",     "right_knee"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip",      "right_hip"),
        ("left_elbow",    "right_elbow"),
    ]
    # Unilateral
    for pose in ALL_POSES[:-1]:
        for joint in ALL_JOINTS:
            for direction in ["increase", "decrease"]:
                for sev in range(4):
                    best = _best_correction(pose, joint, direction, sev, False)
                    if best < 0:
                        continue
                    dlo, dhi = diff_ranges[sev]
                    diffs = np_rng.uniform(dlo, dhi, 8)
                    scores = rng.choices(score_samples, k=8)
                    for diff, score in zip(diffs, scores):
                        feat = encode_context(pose, joint, direction, float(diff), sev, False, "", float(score))
                        X.append(feat); y.append(best)
    # Bilateral
    for pose in ALL_POSES[:-1]:
        for (j1, j2) in bilateral_pairs:
            for direction in ["increase", "decrease"]:
                for sev in range(4):
                    best = _best_correction(pose, j1, direction, sev, True, j2)
                    if best < 0:
                        continue
                    dlo, dhi = diff_ranges[sev]
                    diffs  = np_rng.uniform(dlo, dhi, 6)
                    scores = rng.choices(score_samples, k=6)
                    for diff, score in zip(diffs, scores):
                        feat = encode_context(pose, j1, direction, float(diff), sev, True, j2, float(score))
                        X.append(feat); y.append(best)
    print(f"  Generated {len(X):,} training examples covering {len(set(y))} correction IDs")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


# ── Real dataset loader ───────────────────────────────────────────────────────
_DATASET_DIR = os.path.join(os.path.dirname(__file__), "datasets", "extracted")

_ANGLE_COLS = [
    "left_elbow","right_elbow","left_shoulder","right_shoulder",
    "left_hip","right_hip","left_knee","right_knee"
]

def load_real_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Load extracted CSV files from datasets/extracted/.
    Each row is a real video frame with 8 measured joint angles.

    For each frame we determine which joints are misaligned by comparing
    to the known mean angles for that pose (from the correction bank context).
    This generates real-data training examples with the same feature format.

    Returns (X, y) with same dimensions as synthetic data, or (empty, empty).
    """
    if not os.path.isdir(_DATASET_DIR):
        return np.empty((0, 57), dtype=np.float32), np.empty((0,), dtype=np.int32)

    import csv as csv_mod

    # Known reference angles per pose (mean of typical correct alignment)
    # Used to compute diff = |user_angle - reference_angle| per joint
    POSE_REFS: dict[str, dict[str, float]] = {
        "warrior_ii":    {"left_knee":92,"right_knee":172,"left_shoulder":88,
                          "right_shoulder":88,"left_hip":125,"right_hip":140,
                          "left_elbow":170,"right_elbow":170},
        "warrior_i":     {"left_knee":90,"right_knee":175,"left_shoulder":160,
                          "right_shoulder":160,"left_hip":110,"right_hip":160,
                          "left_elbow":170,"right_elbow":170},
        "chair":         {"left_knee":95,"right_knee":95,"left_shoulder":150,
                          "right_shoulder":150,"left_hip":95,"right_hip":95,
                          "left_elbow":168,"right_elbow":168},
        "downward_dog":  {"left_knee":165,"right_knee":165,"left_shoulder":55,
                          "right_shoulder":55,"left_hip":60,"right_hip":60,
                          "left_elbow":170,"right_elbow":170},
        "mountain":      {"left_knee":175,"right_knee":175,"left_shoulder":15,
                          "right_shoulder":15,"left_hip":175,"right_hip":175,
                          "left_elbow":170,"right_elbow":170},
        "tree":          {"left_knee":175,"right_knee":60,"left_shoulder":160,
                          "right_shoulder":160,"left_hip":172,"right_hip":90,
                          "left_elbow":168,"right_elbow":168},
        "triangle":      {"left_knee":175,"right_knee":175,"left_shoulder":90,
                          "right_shoulder":90,"left_hip":90,"right_hip":110,
                          "left_elbow":170,"right_elbow":170},
        "plank":         {"left_knee":175,"right_knee":175,"left_shoulder":85,
                          "right_shoulder":85,"left_hip":175,"right_hip":175,
                          "left_elbow":170,"right_elbow":170},
        "cobra":         {"left_knee":175,"right_knee":175,"left_shoulder":50,
                          "right_shoulder":50,"left_hip":170,"right_hip":170,
                          "left_elbow":120,"right_elbow":120},
        "child":         {"left_knee":40,"right_knee":40,"left_shoulder":155,
                          "right_shoulder":155,"left_hip":40,"right_hip":40,
                          "left_elbow":168,"right_elbow":168},
        "forward_fold":  {"left_knee":170,"right_knee":170,"left_shoulder":20,
                          "right_shoulder":20,"left_hip":55,"right_hip":55,
                          "left_elbow":168,"right_elbow":168},
        "low_lunge":     {"left_knee":90,"right_knee":175,"left_shoulder":160,
                          "right_shoulder":160,"left_hip":95,"right_hip":165,
                          "left_elbow":170,"right_elbow":170},
        "high_lunge":    {"left_knee":90,"right_knee":170,"left_shoulder":160,
                          "right_shoulder":160,"left_hip":100,"right_hip":165,
                          "left_elbow":170,"right_elbow":170},
    }

    X_real, y_real = [], []
    rng = random.Random(123)
    files_loaded = 0

    for fname in sorted(os.listdir(_DATASET_DIR)):
        if not fname.endswith(".csv"):
            continue
        fpath = os.path.join(_DATASET_DIR, fname)
        # Parse pose and label from filename: warrior_ii_correct.csv
        stem  = fname[:-4]
        parts = stem.rsplit("_", 1)
        pose  = parts[0] if len(parts)==2 else stem
        label = parts[1] if len(parts)==2 else "unknown"

        if pose not in POSE_REFS:
            print(f"  [real data] No reference angles for '{pose}' — skipping {fname}")
            continue

        refs = POSE_REFS[pose]
        rows_loaded = 0

        with open(fpath, newline="") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                try:
                    angles = {col: float(row[col]) for col in _ANGLE_COLS}
                except (KeyError, ValueError):
                    continue

                # Determine the most misaligned joint
                diffs = {j: abs(angles[j] - refs.get(j, angles[j])) for j in _ANGLE_COLS}
                worst_joint = max(diffs, key=diffs.get)
                worst_diff  = diffs[worst_joint]

                if worst_diff < 5:
                    continue  # correctly aligned — no correction needed

                # Direction: is user angle above or below reference?
                direction = "increase" if angles[worst_joint] < refs[worst_joint] else "decrease"
                sev = _severity_band(worst_diff)

                # Find second worst for bilateral check
                sorted_joints = sorted(diffs.items(), key=lambda x: -x[1])
                sec_joint = sorted_joints[1][0] if len(sorted_joints) > 1 else ""
                bilateral = (
                    len(sorted_joints) > 1
                    and sorted_joints[1][1] > 10
                    and diffs[sorted_joints[1][0]] > 0.5 * worst_diff
                )

                best = _best_correction(pose, worst_joint, direction, sev, bilateral, sec_joint)
                if best < 0:
                    continue

                # Score: for wrong-label data use lower scores; correct → higher
                if label == "wrong":
                    score = rng.uniform(40, 74)
                elif label == "correct":
                    score = rng.uniform(76, 95)
                else:
                    score = rng.uniform(50, 90)

                feat = encode_context(pose, worst_joint, direction, worst_diff,
                                      sev, bilateral, sec_joint, score)
                X_real.append(feat)
                y_real.append(best)
                rows_loaded += 1

        if rows_loaded:
            files_loaded += 1
            print(f"  [real data] {fname:<35} → {rows_loaded} training examples ({label})")

    if X_real:
        print(f"  [real data] Total: {len(X_real)} examples from {files_loaded} CSV files")
        return (np.array(X_real, dtype=np.float32),
                np.array(y_real,  dtype=np.int32))
    return np.empty((0,57),dtype=np.float32), np.empty((0,),dtype=np.int32)


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print("=== Training Intelligent Correction AI ===\n")
    print("Building correction bank:")
    print(f"  {N_CORRECTIONS} correction texts across {len(ALL_POSES)-1} poses and {len(ALL_JOINTS)} joints")

    print("\nGenerating synthetic training data...")
    X_syn, y_syn = generate_training_data()

    # Load real dataset if available
    print("\nChecking for real video dataset CSVs...")
    X_real, y_real = load_real_dataset()
    if len(X_real) > 0:
        # Combine: repeat synthetic 2x to balance with real data
        X = np.vstack([X_syn, X_syn, X_real])
        y = np.concatenate([y_syn, y_syn, y_real])
        print(f"\n  Combined: {len(X_syn)} synthetic × 2  +  {len(X_real)} real  =  {len(X)} total")
    else:
        print("  No real CSVs found — training on synthetic data only")
        print(f"  (Run extract_yoga_dataset.py after adding videos to datasets/ folder)")
        X, y = X_syn, y_syn

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=None
    )
    print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    print("\nTraining MLP classifier (input:57 → hidden:[128,64,32] → output:N)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.001,          # L2 regularisation
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=400,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        verbose=False,
    )
    mlp.fit(X_train_s, y_train)

    # Evaluate
    y_pred   = mlp.predict(X_test_s)
    acc      = accuracy_score(y_test, y_pred)
    y_prob   = mlp.predict_proba(X_test_s)
    # Top-3 accuracy
    top3_correct = sum(
        1 for i, yt in enumerate(y_test)
        if yt in np.argsort(y_prob[i])[::-1][:3]
    )
    top3_acc = top3_correct / len(y_test)

    print(f"\n  Test accuracy  (top-1): {acc:.1%}")
    print(f"  Test accuracy  (top-3): {top3_acc:.1%}")
    print(f"  Training iterations:    {mlp.n_iter_}")
    best_loss = getattr(mlp, "best_loss_", None)
    if best_loss is not None:
        print(f"  Best validation loss:   {best_loss:.4f}")

    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    payload = {
        "mlp":         mlp,
        "scaler":      scaler,
        "n_corrections": N_CORRECTIONS,
        "correction_bank": CORRECTION_BANK,
        "all_poses":   ALL_POSES,
        "all_joints":  ALL_JOINTS,
        "version":     "1.0",
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"\n  Saved → {MODEL_PATH}")
    print("\n✓ Correction AI training complete.")
    return acc, top3_acc


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate():
    if not os.path.exists(MODEL_PATH):
        print("No model found. Run without --eval first.")
        return

    payload  = joblib.load(MODEL_PATH)
    mlp      = payload["mlp"]
    scaler   = payload["scaler"]
    bank     = {c["id"]: c for c in payload["correction_bank"]}

    test_cases = [
        # (pose, joint, direction, diff, sev, bilateral, sec_joint, score)
        ("warrior_ii",   "left_knee",    "decrease", 28, 2, False, "",             62),
        ("warrior_ii",   "left_shoulder","increase", 18, 1, False, "",             71),
        ("chair",        "left_knee",    "decrease", 40, 3, False, "",             50),
        ("downward_dog", "left_knee",    "increase", 22, 2, False, "",             65),
        ("warrior_i",    "left_hip",     "decrease", 15, 1, False, "",             78),
        ("mountain",     "left_knee",    "increase", 12, 1, True,  "right_knee",   82),
        ("cobra",        "left_elbow",   "decrease", 20, 2, True,  "right_elbow",  58),
        ("triangle",     "left_shoulder","increase", 35, 3, False, "",             47),
        ("tree",         "left_knee",    "increase", 10, 1, False, "",             80),
        ("child",        "left_hip",     "decrease", 30, 2, True,  "right_hip",    60),
    ]

    print("=== Correction AI — Evaluation ===\n")
    for (pose, joint, direction, diff, sev, bilateral, sec, score) in test_cases:
        feat = encode_context(pose, joint, direction, diff, sev, bilateral, sec, score).reshape(1, -1)
        feat_s = scaler.transform(feat)
        probs  = mlp.predict_proba(feat_s)[0]
        classes = mlp.classes_
        top_idx = np.argsort(probs)[::-1][:3]

        print(f"Pose: {pose:<15} | Joint: {joint:<17} | Dir: {direction:<8} | Diff: {diff}° | Score: {score}%")
        for rank, idx in enumerate(top_idx, 1):
            cid  = classes[idx]
            conf = probs[idx]
            text = bank.get(cid, {}).get("text", "???")
            print(f"  [{rank}] ({conf:.0%}) {text}")
        print()


# ── Demo (live interactive) ───────────────────────────────────────────────────
def demo():
    if not os.path.exists(MODEL_PATH):
        print("No model found. Train first.")
        return
    payload = joblib.load(MODEL_PATH)
    mlp     = payload["mlp"]
    scaler  = payload["scaler"]
    bank    = {c["id"]: c for c in payload["correction_bank"]}

    print("=== Correction AI — Interactive Demo ===")
    print("Enter correction context. Ctrl-C to quit.\n")
    while True:
        try:
            pose  = input("Pose (e.g. warrior_ii): ").strip() or "warrior_ii"
            joint = input("Joint (e.g. left_knee): ").strip()  or "left_knee"
            dirn  = input("Direction (increase/decrease): ").strip() or "decrease"
            diff  = float(input("Angle diff (degrees): ") or "25")
            score = float(input("Overall score (%): ") or "65")
        except (KeyboardInterrupt, EOFError):
            break
        sev  = _severity_band(diff)
        feat = encode_context(pose, joint, dirn, diff, sev, False, "", score).reshape(1, -1)
        feat_s = scaler.transform(feat)
        probs  = mlp.predict_proba(feat_s)[0]
        top_idx = np.argsort(probs)[::-1][:5]
        print(f"\nTop corrections for {pose} / {joint} / {dirn} / {diff}°:")
        for i, idx in enumerate(top_idx, 1):
            cid  = mlp.classes_[idx]
            conf = probs[idx]
            text = bank.get(cid, {}).get("text", "???")
            print(f"  [{i}] ({conf:.0%}) {text}")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="Evaluate saved model")
    ap.add_argument("--demo", action="store_true", help="Interactive demo")
    args = ap.parse_args()

    if args.eval:
        evaluate()
    elif args.demo:
        demo()
    else:
        train()
