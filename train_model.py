#!/usr/bin/env python3
# train_model.py — Train a custom pose classifier using MediaPipe landmark features
#
# PIPELINE:
#   1. Generate a synthetic landmark dataset from anatomically-accurate angle ranges
#      (same ranges as the rule-based classifier, but sampled into thousands of examples
#       with realistic noise)
#   2. If you have REAL data (CSV of MediaPipe landmarks), it will be loaded and merged
#   3. Extract features: joint angles + normalised landmark positions
#   4. Train a Random Forest + a MLP, pick the better one
#   5. Save to model/pose_classifier.pkl
#   6. The saved model is auto-loaded by pose_classifier.py at startup if it exists
#
# REAL DATASETS:
#   Place any of these CSVs in a folder called 'datasets/' and they will be auto-loaded:
#   • Yoga-82 (Kaggle: shrutisaxena/yoga-pose-image-dataset)
#     After running MediaPipe on the images, export landmarks as CSV.
#   • Yoga Pose Dataset (Kaggle: ujjwalchowdhury/yoga-pose-classification)
#     Contains pre-extracted keypoints.
#   • Custom: record yourself doing poses and save via --record flag.
#
# USAGE:
#   python train_model.py                    # train on synthetic + any CSVs in datasets/
#   python train_model.py --record warrior_ii  # record live landmarks for a pose label
#   python train_model.py --epochs 200       # more training iterations
#   python train_model.py --eval             # evaluate saved model on test split

from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network    import MLPClassifier
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.model_selection   import train_test_split, cross_val_score
from sklearn.metrics           import classification_report, confusion_matrix
from sklearn.pipeline          import Pipeline
import joblib

warnings.filterwarnings("ignore")

MODEL_DIR  = Path(__file__).parent / "model"
MODEL_PATH = MODEL_DIR / "pose_classifier.pkl"
DATA_DIR   = Path(__file__).parent / "datasets"

# ── Pose angle definitions (mean ± std for each joint per pose) ───────────────
# Format: { pose_label: { joint: (mean_degrees, std_degrees) } }
# These are derived from yoga anatomy references + the rule-based classifier ranges.
_POSE_ANGLE_DEFS: dict[str, dict[str, tuple[float, float]]] = {
    "mountain": {
        "left_knee":      (178, 3),   "right_knee":      (178, 3),
        "left_hip":       (175, 5),   "right_hip":       (175, 5),
        "left_shoulder":  (30,  8),   "right_shoulder":  (30,  8),
        "left_elbow":     (175, 5),   "right_elbow":     (175, 5),
        "left_ankle":     (90,  5),   "right_ankle":     (90,  5),
    },
    "forward_fold": {
        "left_knee":      (162, 12),  "right_knee":      (162, 12),
        "left_hip":       (55,  18),  "right_hip":       (55,  18),
        "left_shoulder":  (25,  12),  "right_shoulder":  (25,  12),
        "left_elbow":     (160, 15),  "right_elbow":     (160, 15),
        "left_ankle":     (90,  8),   "right_ankle":     (90,  8),
    },
    "halfway_lift": {
        "left_knee":      (170, 8),   "right_knee":      (170, 8),
        "left_hip":       (90,  12),  "right_hip":       (90,  12),
        "left_shoulder":  (72,  10),  "right_shoulder":  (72,  10),
        "left_elbow":     (165, 10),  "right_elbow":     (165, 10),
    },
    "warrior_i": {
        "left_knee":      (92,  12),  "right_knee":      (170, 8),
        "left_hip":       (145, 12),  "right_hip":       (155, 10),
        "left_shoulder":  (175, 12),  "right_shoulder":  (175, 12),
        "left_elbow":     (175, 5),   "right_elbow":     (175, 5),
        "left_ankle":     (90,  8),   "right_ankle":     (72,  8),
    },
    "warrior_ii": {
        "left_knee":      (92,  12),  "right_knee":      (172, 8),
        "left_hip":       (125, 12),  "right_hip":       (155, 10),
        "left_shoulder":  (90,  8),   "right_shoulder":  (90,  8),
        "left_elbow":     (175, 5),   "right_elbow":     (175, 5),
        "left_ankle":     (90,  8),   "right_ankle":     (72,  8),
    },
    "warrior_iii": {
        "left_knee":      (175, 5),   "right_knee":      (170, 8),
        "left_hip":       (92,  8),   "right_hip":       (90,  10),
        "left_shoulder":  (175, 8),   "right_shoulder":  (175, 8),
        "left_elbow":     (175, 5),   "right_elbow":     (175, 5),
    },
    "chair": {
        "left_knee":      (95,  12),  "right_knee":      (95,  12),
        "left_hip":       (95,  12),  "right_hip":       (95,  12),
        "left_shoulder":  (168, 12),  "right_shoulder":  (168, 12),
        "left_elbow":     (175, 5),   "right_elbow":     (175, 5),
    },
    "tree": {
        "left_knee":      (178, 3),   "right_knee":      (52,  15),
        "left_hip":       (175, 5),   "right_hip":       (62,  15),
        "left_shoulder":  (165, 20),  "right_shoulder":  (165, 20),
        "left_elbow":     (155, 20),  "right_elbow":     (155, 20),
    },
    "triangle": {
        "left_knee":      (175, 5),   "right_knee":      (175, 5),
        "left_hip":       (75,  12),  "right_hip":       (120, 12),
        "left_shoulder":  (162, 12),  "right_shoulder":  (62,  12),
        "left_elbow":     (175, 5),   "right_elbow":     (175, 5),
    },
    "downward_dog": {
        "left_knee":      (168, 10),  "right_knee":      (168, 10),
        "left_hip":       (68,  10),  "right_hip":       (68,  10),
        "left_shoulder":  (182, 8),   "right_shoulder":  (182, 8),
        "left_elbow":     (175, 5),   "right_elbow":     (175, 5),
        "left_ankle":     (72,  8),   "right_ankle":     (72,  8),
    },
    "plank": {
        "left_knee":      (178, 3),   "right_knee":      (178, 3),
        "left_hip":       (178, 5),   "right_hip":       (178, 5),
        "left_shoulder":  (82,  8),   "right_shoulder":  (82,  8),
        "left_elbow":     (178, 5),   "right_elbow":     (178, 5),
        "left_ankle":     (80,  8),   "right_ankle":     (80,  8),
    },
    "low_lunge": {
        "left_knee":      (92,  10),  "right_knee":      (100, 12),
        "left_hip":       (105, 12),  "right_hip":       (115, 12),
        "left_shoulder":  (170, 12),  "right_shoulder":  (170, 12),
        "left_elbow":     (175, 5),   "right_elbow":     (175, 5),
    },
    "high_lunge": {
        "left_knee":      (92,  10),  "right_knee":      (168, 8),
        "left_hip":       (112, 10),  "right_hip":       (118, 10),
        "left_shoulder":  (172, 10),  "right_shoulder":  (172, 10),
        "left_elbow":     (175, 5),   "right_elbow":     (175, 5),
    },
    "cobra": {
        "left_knee":      (176, 5),   "right_knee":      (176, 5),
        "left_hip":       (175, 5),   "right_hip":       (175, 5),
        "left_shoulder":  (42,  12),  "right_shoulder":  (42,  12),
        "left_elbow":     (128, 15),  "right_elbow":     (128, 15),
        "left_ankle":     (135, 10),  "right_ankle":     (135, 10),
    },
    "upward_dog": {
        "left_knee":      (176, 5),   "right_knee":      (176, 5),
        "left_hip":       (175, 5),   "right_hip":       (175, 5),
        "left_shoulder":  (38,  10),  "right_shoulder":  (38,  10),
        "left_elbow":     (175, 5),   "right_elbow":     (175, 5),
        "left_ankle":     (130, 8),   "right_ankle":     (130, 8),
    },
    "child": {
        "left_knee":      (48,  15),  "right_knee":      (48,  15),
        "left_hip":       (45,  15),  "right_hip":       (45,  15),
        "left_shoulder":  (155, 20),  "right_shoulder":  (155, 20),
        "left_elbow":     (155, 20),  "right_elbow":     (155, 20),
        "left_ankle":     (130, 10),  "right_ankle":     (130, 10),
    },
    "seated_forward": {
        "left_knee":      (172, 8),   "right_knee":      (172, 8),
        "left_hip":       (52,  18),  "right_hip":       (52,  18),
        "left_shoulder":  (148, 18),  "right_shoulder":  (148, 18),
        "left_elbow":     (145, 20),  "right_elbow":     (145, 20),
        "left_ankle":     (80,  8),   "right_ankle":     (80,  8),
    },
    "cat_cow": {
        "left_knee":      (95,  8),   "right_knee":      (95,  8),
        "left_hip":       (95,  8),   "right_hip":       (95,  8),
        "left_shoulder":  (85,  8),   "right_shoulder":  (85,  8),
        "left_elbow":     (170, 8),   "right_elbow":     (170, 8),
        "left_ankle":     (110, 8),   "right_ankle":     (110, 8),
    },
}

_ALL_JOINTS = [
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_hip",      "right_hip",
    "left_knee",     "right_knee",
    "left_ankle",    "right_ankle",
    "left_wrist",    "right_wrist",
]

_DERIVED_FEATURES = [
    # Symmetry features
    "shoulder_diff",   # |left_shoulder - right_shoulder|
    "hip_diff",        # |left_hip - right_hip|
    "knee_diff",       # |left_knee - right_knee|
    "elbow_diff",      # |left_elbow - right_elbow|
    # Ratios
    "knee_hip_ratio_l",  # left_knee / left_hip
    "knee_hip_ratio_r",
    "shoulder_knee_diff_l",  # left_shoulder - left_knee (arm vs leg relative)
    "shoulder_knee_diff_r",
    # Mean angles (pooled bilateral)
    "mean_knee",
    "mean_hip",
    "mean_shoulder",
    "mean_elbow",
]


def _make_feature_vector(angles: dict[str, float]) -> np.ndarray:
    """Convert angle dict to a fixed-length feature vector."""
    base = [angles.get(j, 135.0) for j in _ALL_JOINTS]  # 135° = anatomical neutral guess

    derived = [
        abs(angles.get("left_shoulder", 90) - angles.get("right_shoulder", 90)),
        abs(angles.get("left_hip",       90) - angles.get("right_hip",       90)),
        abs(angles.get("left_knee",      90) - angles.get("right_knee",      90)),
        abs(angles.get("left_elbow",     90) - angles.get("right_elbow",     90)),
        angles.get("left_knee",  90) / max(angles.get("left_hip",  1), 1),
        angles.get("right_knee", 90) / max(angles.get("right_hip", 1), 1),
        angles.get("left_shoulder",  90) - angles.get("left_knee",  90),
        angles.get("right_shoulder", 90) - angles.get("right_knee", 90),
        (angles.get("left_knee",  90) + angles.get("right_knee",  90)) / 2,
        (angles.get("left_hip",   90) + angles.get("right_hip",   90)) / 2,
        (angles.get("left_shoulder", 90) + angles.get("right_shoulder", 90)) / 2,
        (angles.get("left_elbow",  90) + angles.get("right_elbow",  90)) / 2,
    ]

    return np.array(base + derived, dtype=np.float32)


def generate_synthetic_dataset(
    n_per_pose: int = 600,
    noise_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training examples by sampling from angle distributions."""
    X, y = [], []
    rng = np.random.default_rng(42)

    for pose_label, joint_defs in _POSE_ANGLE_DEFS.items():
        for _ in range(n_per_pose):
            angles: dict[str, float] = {}
            for joint, (mean, std) in joint_defs.items():
                val = rng.normal(mean, std * noise_scale)
                val = float(np.clip(val, 0, 360))
                angles[joint] = val
            # Fill missing joints with neutral
            for j in _ALL_JOINTS:
                if j not in angles:
                    angles[j] = float(rng.normal(135, 20))

            X.append(_make_feature_vector(angles))
            y.append(pose_label)

    return np.array(X, dtype=np.float32), np.array(y)


def load_real_datasets() -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load real landmark CSVs from datasets/ directory.

    Supported formats:
      • Columns: label, left_knee, right_knee, left_hip, ... (angle CSV)
      • Columns: label, x0,y0,z0, x1,y1,z1, ... (raw MediaPipe keypoints)
    """
    if not DATA_DIR.exists():
        return None

    csvs = list(DATA_DIR.glob("*.csv"))
    if not csvs:
        return None

    X_all, y_all = [], []
    for csv_path in csvs:
        print(f"  Loading {csv_path.name}...")
        try:
            df = pd.read_csv(csv_path)

            # Detect format
            angle_cols = [c for c in df.columns if c in _ALL_JOINTS]
            if "label" not in df.columns and "class" in df.columns:
                df = df.rename(columns={"class": "label"})
            if "label" not in df.columns:
                print(f"    Skipping — no 'label' column found.")
                continue

            if angle_cols:
                # Angle CSV format
                for _, row in df.iterrows():
                    angles = {j: float(row[j]) for j in angle_cols if j in row}
                    for j in _ALL_JOINTS:
                        if j not in angles:
                            angles[j] = 135.0
                    X_all.append(_make_feature_vector(angles))
                    y_all.append(str(row["label"]).lower().replace(" ", "_"))
            else:
                # Raw keypoint format (x0,y0,z0,...) — compute pseudo-angles
                kp_cols = [c for c in df.columns if c.startswith("x") or c.startswith("y")]
                if len(kp_cols) >= 6:
                    for _, row in df.iterrows():
                        angles = _angles_from_keypoints(row)
                        X_all.append(_make_feature_vector(angles))
                        y_all.append(str(row["label"]).lower().replace(" ", "_"))

            print(f"    Loaded {len(df)} rows.")
        except Exception as e:
            print(f"    Error: {e}")

    if not X_all:
        return None
    return np.array(X_all, dtype=np.float32), np.array(y_all)


def _angles_from_keypoints(row: pd.Series) -> dict[str, float]:
    """
    Compute approximate joint angles from raw MediaPipe landmark columns.
    Expects x0,y0,z0 ... x32,y32,z32 columns (MediaPipe 33-point format).
    """
    def get(idx):
        try:
            return np.array([float(row[f"x{idx}"]), float(row[f"y{idx}"]), float(row.get(f"z{idx}", 0))])
        except Exception:
            return np.array([0.0, 0.0, 0.0])

    def angle(a, b, c):
        ba = a - b; bc = c - b
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
        return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

    # MediaPipe landmark indices
    # 11=L_shoulder, 12=R_shoulder, 13=L_elbow, 14=R_elbow
    # 15=L_wrist, 16=R_wrist, 23=L_hip, 24=R_hip
    # 25=L_knee, 26=R_knee, 27=L_ankle, 28=R_ankle
    return {
        "left_shoulder":  angle(get(13), get(11), get(23)),
        "right_shoulder": angle(get(14), get(12), get(24)),
        "left_elbow":     angle(get(11), get(13), get(15)),
        "right_elbow":    angle(get(12), get(14), get(16)),
        "left_hip":       angle(get(11), get(23), get(25)),
        "right_hip":      angle(get(12), get(24), get(26)),
        "left_knee":      angle(get(23), get(25), get(27)),
        "right_knee":     angle(get(24), get(26), get(28)),
        "left_ankle":     angle(get(25), get(27), get(29)) if True else 90.0,
        "right_ankle":    angle(get(26), get(28), get(30)) if True else 90.0,
    }


def train(n_per_pose: int = 600, epochs: int = 300) -> None:
    MODEL_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  Yoga Pose Classifier — Training Pipeline")
    print("=" * 60)

    # ── Generate synthetic data ───────────────────────────────────────
    print(f"\n[1/5] Generating synthetic dataset ({n_per_pose} samples/pose)...")
    X_syn, y_syn = generate_synthetic_dataset(n_per_pose=n_per_pose)
    print(f"      Synthetic: {len(X_syn)} samples, {len(set(y_syn))} classes")

    # ── Load real data ────────────────────────────────────────────────
    print("\n[2/5] Loading real datasets from datasets/...")
    real = load_real_datasets()
    if real:
        X_real, y_real = real
        # Filter to known classes only
        known = set(_POSE_ANGLE_DEFS.keys())
        mask = np.array([l in known for l in y_real])
        X_real, y_real = X_real[mask], y_real[mask]
        print(f"      Real data: {len(X_real)} samples")
        X = np.vstack([X_syn, X_real])
        y = np.concatenate([y_syn, y_real])
    else:
        print("      No real datasets found — using synthetic only.")
        print("      Tip: place landmark CSVs in datasets/ for better accuracy.")
        X, y = X_syn, y_syn

    print(f"      Total: {len(X)} samples across {len(set(y))} poses")

    # ── Encode labels ─────────────────────────────────────────────────
    print("\n[3/5] Encoding and splitting...")
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_enc, test_size=0.15, stratify=y_enc, random_state=42)
    print(f"      Train: {len(X_tr)}   Test: {len(X_te)}")

    # ── Train models ──────────────────────────────────────────────────
    print("\n[4/5] Training models...")

    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42,
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=min(epochs, 300),
                max_depth=5,
                learning_rate=0.08,
                random_state=42,
            )),
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                activation="relu",
                max_iter=epochs,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=20,
            )),
        ]),
    }

    best_name, best_model, best_acc = None, None, 0.0

    for name, pipeline in models.items():
        t0 = time.time()
        pipeline.fit(X_tr, y_tr)
        acc = pipeline.score(X_te, y_te)
        elapsed = time.time() - t0
        print(f"      {name:20s}  test_acc={acc:.3f}  ({elapsed:.1f}s)")
        if acc > best_acc:
            best_acc  = acc
            best_name = name
            best_model = pipeline

    print(f"\n      Best model: {best_name}  (acc={best_acc:.3f})")

    # ── Cross-validation on best model ───────────────────────────────
    print("\n[5/5] Cross-validating best model...")
    cv_scores = cross_val_score(best_model, X, y_enc, cv=5, n_jobs=-1)
    print(f"      CV scores: {cv_scores.round(3)}  mean={cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── Classification report ─────────────────────────────────────────
    y_pred = best_model.predict(X_te)
    print("\n  Per-class report:")
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0))

    # ── Save ─────────────────────────────────────────────────────────
    bundle = {
        "model":   best_model,
        "encoder": le,
        "features": _ALL_JOINTS + _DERIVED_FEATURES,
        "model_name": best_name,
        "cv_mean": float(cv_scores.mean()),
        "test_acc": float(best_acc),
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\n  ✅ Model saved → {MODEL_PATH}")
    print(f"     It will be auto-loaded by pose_classifier.py on next run.")


def evaluate() -> None:
    """Quick evaluation of a saved model."""
    if not MODEL_PATH.exists():
        print("No saved model found. Run: python train_model.py")
        return

    bundle  = joblib.load(MODEL_PATH)
    model   = bundle["model"]
    le      = bundle["encoder"]
    print(f"Model: {bundle['model_name']}  test_acc={bundle['test_acc']:.3f}  cv={bundle['cv_mean']:.3f}")

    print("\nGenerating evaluation dataset...")
    X, y = generate_synthetic_dataset(n_per_pose=200)
    y_enc = le.transform([l for l in y if l in le.classes_])
    mask  = np.array([l in le.classes_ for l in y])
    acc   = model.score(X[mask], y_enc)
    print(f"Accuracy on fresh synthetic data: {acc:.3f}")


def record_live(pose_label: str, n_samples: int = 200) -> None:
    """
    Live recording mode: opens the webcam, runs MediaPipe, and records
    landmark angles to datasets/{pose_label}.csv for training.
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("cv2 and mediapipe required. pip install opencv-python mediapipe")
        return

    mp_pose    = mp.solutions.pose
    cap        = cv2.VideoCapture(0)
    pose_model = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / f"{pose_label.lower().replace(' ','_')}.csv"
    rows     = []

    print(f"Recording '{pose_label}'. Press SPACE to capture a sample. ESC to finish.")
    print(f"Target: {n_samples} samples. Saved to: {out_path}")

    while len(rows) < n_samples:
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose_model.process(rgb)

        overlay = frame.copy()
        cv2.putText(overlay, f"{pose_label}  [{len(rows)}/{n_samples}]",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(overlay, "SPACE = capture   ESC = done",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        if result.pose_landmarks:
            lm = result.pose_landmarks.landmark

            def pt(i): return np.array([lm[i].x, lm[i].y, lm[i].z])
            def ang(a,b,c):
                ba=pt(a)-pt(b); bc=pt(c)-pt(b)
                cos=np.dot(ba,bc)/(np.linalg.norm(ba)*np.linalg.norm(bc)+1e-9)
                return float(np.degrees(np.arccos(np.clip(cos,-1,1))))

            row = {
                "label": pose_label,
                "left_shoulder":  ang(13,11,23), "right_shoulder": ang(14,12,24),
                "left_elbow":     ang(11,13,15), "right_elbow":    ang(12,14,16),
                "left_hip":       ang(11,23,25), "right_hip":      ang(12,24,26),
                "left_knee":      ang(23,25,27), "right_knee":     ang(24,26,28),
                "left_ankle":     ang(25,27,29), "right_ankle":    ang(26,28,30),
            }

            mp.solutions.drawing_utils.draw_landmarks(
                overlay, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            cv2.imshow("Record Pose", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            elif key == 32:
                rows.append(row)
                print(f"  Captured sample {len(rows)}")
        else:
            cv2.imshow("Record Pose", overlay)
            cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    pose_model.close()

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(rows)} samples to {out_path}")
        print("Run 'python train_model.py' to retrain with this data.")
    else:
        print("No samples recorded.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Yoga pose classifier training pipeline")
    parser.add_argument("--n",       type=int, default=600,  help="Synthetic samples per pose")
    parser.add_argument("--epochs",  type=int, default=300,  help="Max MLP/GB epochs")
    parser.add_argument("--eval",    action="store_true",    help="Evaluate saved model")
    parser.add_argument("--record",  type=str, default=None, help="Record live data for a pose label")
    parser.add_argument("--samples", type=int, default=200,  help="Samples to record in --record mode")
    args = parser.parse_args()

    if args.eval:
        evaluate()
    elif args.record:
        record_live(args.record, n_samples=args.samples)
    else:
        train(n_per_pose=args.n, epochs=args.epochs)
