#!/usr/bin/env python3
"""
extract_yoga_dataset.py  —  Video Dataset → MediaPipe Angles → CSV Pipeline

Processes yoga video files (MP4, AVI, MOV) through MediaPipe Pose and
extracts the same 8 joint angles that the live system uses.

Supports three dataset layouts:

  Layout A  —  Mendeley "Yoga for All" (labelled correct/wrong folders)
  ─────────────────────────────────────────────────────────────────────
  datasets/
    yoga_for_all/
      warrior_ii/
        correct/  video1.mp4  video2.mp4 ...
        wrong/    video1.mp4  video2.mp4 ...

  Layout B  —  Flat folder per pose (no correct/wrong split)
  ──────────────────────────────────────────────────────────
  datasets/
    my_dataset/
      warrior_ii/  video1.mp4 video2.mp4 ...
      chair/       video1.mp4 ...

  Layout C  —  Single folder, pose name in filename
  ──────────────────────────────────────────────────
  datasets/
    raw/
      warrior_ii_clip1.mp4
      chair_student2.mp4

Usage
─────
  # Extract everything in datasets/ folder
  python extract_yoga_dataset.py

  # Extract a specific folder
  python extract_yoga_dataset.py --input datasets/yoga_for_all

  # Extract + immediately retrain correction AI
  python extract_yoga_dataset.py --retrain

  # Show stats on extracted CSVs
  python extract_yoga_dataset.py --stats

Output
──────
  datasets/extracted/
    warrior_ii_correct.csv
    warrior_ii_wrong.csv
    chair_correct.csv
    ...

CSV columns: pose, label, frame, left_elbow, right_elbow,
             left_shoulder, right_shoulder, left_hip, right_hip,
             left_knee, right_knee, visibility_ok
"""

from __future__ import annotations
import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Pose name normalisation ───────────────────────────────────────────────────
# Maps common alternate names → internal pose key
POSE_ALIASES: dict[str, str] = {
    # Sanskrit names → English keys
    "tadasana":        "mountain",
    "mountain":        "mountain",
    "uttanasana":      "forward_fold",
    "forward_fold":    "forward_fold",
    "forwardfold":     "forward_fold",
    "ardha_uttanasana":"halfway_lift",
    "halfway_lift":    "halfway_lift",
    "virabhadrasana_i":"warrior_i",
    "warrior_i":       "warrior_i",
    "warrior1":        "warrior_i",
    "warrior_1":       "warrior_i",
    "virabhadrasana_ii":"warrior_ii",
    "warrior_ii":      "warrior_ii",
    "warrior2":        "warrior_ii",
    "warrior_2":       "warrior_ii",
    "virabhadrasana_iii":"warrior_iii",
    "warrior_iii":     "warrior_iii",
    "warrior3":        "warrior_iii",
    "utkatasana":      "chair",
    "chair":           "chair",
    "vrksasana":       "tree",
    "vrikshasana":     "tree",
    "tree":            "tree",
    "trikonasana":     "triangle",
    "triangle":        "triangle",
    "adho_mukha":      "downward_dog",
    "downdog":         "downward_dog",
    "downward_dog":    "downward_dog",
    "phalakasana":     "plank",
    "plank":           "plank",
    "anjaneyasana":    "low_lunge",
    "low_lunge":       "low_lunge",
    "high_lunge":      "high_lunge",
    "bhujangasana":    "cobra",
    "cobra":           "cobra",
    "urdhva_mukha":    "upward_dog",
    "upward_dog":      "upward_dog",
    "balasana":        "child",
    "childs_pose":     "child",
    "child":           "child",
    "paschimottanasana":"seated_forward",
    "seated_forward":  "seated_forward",
    "padmasana":       "cat_cow",   # Mendeley uses padmasana
    "cat_cow":         "cat_cow",
}

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# MediaPipe landmark indices (same as the live system)
_LM = {
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
}
MIN_VIS = 0.50   # same as config.MIN_LANDMARK_VISIBILITY


# ── Angle math ────────────────────────────────────────────────────────────────
def _angle(a, b, c) -> Optional[float]:
    """Angle at point b formed by vectors b→a and b→c."""
    ax, ay = a.x - b.x, a.y - b.y
    cx_, cy = c.x - b.x, c.y - b.y
    dot  = ax*cx_ + ay*cy
    mag  = math.sqrt(ax**2+ay**2) * math.sqrt(cx_**2+cy**2)
    if mag < 1e-8:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/mag))))


def _extract_angles(landmarks) -> Optional[dict[str, float]]:
    """Extract 8 joint angles from a MediaPipe landmark list."""
    lm = landmarks.landmark

    def lm_vis(key):
        return lm[_LM[key]].visibility

    # Check all required landmarks are visible
    required = ["left_shoulder","right_shoulder","left_elbow","right_elbow",
                "left_wrist","right_wrist","left_hip","right_hip",
                "left_knee","right_knee","left_ankle","right_ankle"]
    if any(lm_vis(k) < MIN_VIS for k in required):
        return None

    def pt(key):
        return lm[_LM[key]]

    joints = {
        "left_elbow":     _angle(pt("left_shoulder"),  pt("left_elbow"),  pt("left_wrist")),
        "right_elbow":    _angle(pt("right_shoulder"), pt("right_elbow"), pt("right_wrist")),
        "left_shoulder":  _angle(pt("left_elbow"),     pt("left_shoulder"), pt("left_hip")),
        "right_shoulder": _angle(pt("right_elbow"),    pt("right_shoulder"), pt("right_hip")),
        "left_hip":       _angle(pt("left_shoulder"),  pt("left_hip"),    pt("left_knee")),
        "right_hip":      _angle(pt("right_shoulder"), pt("right_hip"),   pt("right_knee")),
        "left_knee":      _angle(pt("left_hip"),       pt("left_knee"),   pt("left_ankle")),
        "right_knee":     _angle(pt("right_hip"),      pt("right_knee"),  pt("right_ankle")),
    }
    if any(v is None for v in joints.values()):
        return None
    return joints


# ── Core extraction ───────────────────────────────────────────────────────────
def extract_video(
    video_path: Path,
    pose_name: str,
    label: str,          # "correct" | "wrong" | "unknown"
    sample_every_n: int = 3,   # extract every Nth frame (reduces redundancy)
) -> list[dict]:
    """
    Run MediaPipe on a video and return angle rows.
    sample_every_n=3 → ~10 FPS from 30 FPS video (balanced speed vs coverage)
    """
    try:
        import cv2
        import mediapipe as mp
    except ImportError:
        print("ERROR: pip install opencv-python mediapipe")
        sys.exit(1)

    mp_pose = mp.solutions.pose
    rows = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Cannot open {video_path.name}")
        return []

    total_frames   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps            = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx      = 0
    extracted      = 0
    visibility_skipped = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % sample_every_n != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if not result.pose_landmarks:
                continue

            angles = _extract_angles(result.pose_landmarks)
            if angles is None:
                visibility_skipped += 1
                continue

            rows.append({
                "pose":  pose_name,
                "label": label,
                "frame": frame_idx,
                **angles,
                "visibility_ok": 1,
            })
            extracted += 1

    cap.release()

    duration_s = total_frames / fps
    print(f"  ✓ {video_path.name:40s} → {extracted:4d} frames "
          f"({duration_s:.0f}s video, {visibility_skipped} low-vis skipped)")
    return rows


# ── Dataset layout detection ──────────────────────────────────────────────────
def _normalise_pose(name: str) -> Optional[str]:
    """Try to map a folder/filename fragment to an internal pose key."""
    clean = name.lower().replace("-","_").replace(" ","_").strip()
    if clean in POSE_ALIASES:
        return POSE_ALIASES[clean]
    # Substring match
    for alias, key in POSE_ALIASES.items():
        if alias in clean:
            return key
    return None


def _detect_label(folder_name: str) -> str:
    f = folder_name.lower()
    if any(x in f for x in ["correct","right","good","proper"]):
        return "correct"
    if any(x in f for x in ["wrong","incorrect","bad","improper","error"]):
        return "wrong"
    return "unknown"


def discover_videos(root: Path) -> list[tuple[Path, str, str]]:
    """
    Walk the root folder and return (video_path, pose_key, label) tuples.
    Handles layouts A, B, and C automatically.
    """
    found = []

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        # Collect all parent folder names relative to root
        rel_parts = path.relative_to(root).parts[:-1]  # exclude filename

        # Try to find pose from parent folder names (deepest first)
        pose_key = None
        for part in reversed(rel_parts):
            key = _normalise_pose(part)
            if key:
                pose_key = key
                break

        # If not in folders, try filename
        if pose_key is None:
            stem_parts = path.stem.replace("-","_").split("_")
            for n in range(len(stem_parts), 0, -1):
                candidate = "_".join(stem_parts[:n])
                key = _normalise_pose(candidate)
                if key:
                    pose_key = key
                    break

        if pose_key is None:
            print(f"  ⚠ Cannot determine pose for {path.relative_to(root)} — skipping")
            continue

        # Detect label from folder structure
        label = "unknown"
        for part in rel_parts:
            l = _detect_label(part)
            if l != "unknown":
                label = l
                break

        found.append((path, pose_key, label))

    return found


# ── CSV writer ────────────────────────────────────────────────────────────────
CSV_FIELDNAMES = [
    "pose", "label", "frame",
    "left_elbow","right_elbow","left_shoulder","right_shoulder",
    "left_hip","right_hip","left_knee","right_knee",
    "visibility_ok"
]

def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if output_path.exists():
        with open(output_path, newline="") as f:
            reader = csv.DictReader(f)
            existing = list(reader)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        for row in existing + rows:
            writer.writerow({k: row.get(k,"") for k in CSV_FIELDNAMES})


# ── Main entry ────────────────────────────────────────────────────────────────
def run_extraction(input_dir: Path, output_dir: Path, sample_every: int = 3) -> dict:
    """Extract all videos and write CSVs. Returns summary stats."""
    videos = discover_videos(input_dir)
    if not videos:
        print(f"No videos found in {input_dir}")
        return {}

    print(f"\nFound {len(videos)} videos across {len(set(p for _,p,_ in videos))} poses\n")

    # Group by (pose, label) → one CSV per group
    groups: dict[tuple, list] = {}
    for vpath, pose, label in videos:
        groups.setdefault((pose, label), []).append(vpath)

    stats = {}
    t0 = time.time()

    for (pose, label), vpaths in sorted(groups.items()):
        print(f"\n── {pose}  [{label}]  ({len(vpaths)} videos)")
        all_rows = []
        for vp in vpaths:
            rows = extract_video(vp, pose, label, sample_every_n=sample_every)
            all_rows.extend(rows)

        if all_rows:
            csv_name = f"{pose}_{label}.csv"
            out_path = output_dir / csv_name
            write_csv(all_rows, out_path)
            stats[(pose, label)] = len(all_rows)
            print(f"  → Saved {len(all_rows)} rows to {csv_name}")

    elapsed = time.time() - t0
    print(f"\n✓ Extraction complete in {elapsed:.1f}s")
    print(f"  Total frames extracted: {sum(stats.values())}")
    return stats


def show_stats(output_dir: Path) -> None:
    """Print summary of extracted CSVs."""
    csvs = sorted(output_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {output_dir}")
        return
    print(f"\n{'Pose':<20} {'Label':<10} {'Frames':>8}")
    print("─" * 42)
    total = 0
    for csv_path in csvs:
        with open(csv_path) as f:
            rows = sum(1 for _ in f) - 1  # subtract header
        parts = csv_path.stem.rsplit("_", 1)
        pose  = parts[0] if len(parts)==2 else csv_path.stem
        label = parts[1] if len(parts)==2 else "unknown"
        print(f"{pose:<20} {label:<10} {rows:>8}")
        total += rows
    print("─" * 42)
    print(f"{'TOTAL':<30} {total:>8} frames")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    BASE = Path(__file__).parent
    DEFAULT_INPUT  = BASE / "datasets"
    DEFAULT_OUTPUT = BASE / "datasets" / "extracted"

    ap = argparse.ArgumentParser(
        description="Extract MediaPipe joint angles from yoga videos"
    )
    ap.add_argument("--input",    type=Path, default=DEFAULT_INPUT,
                    help="Root folder containing yoga videos")
    ap.add_argument("--output",   type=Path, default=DEFAULT_OUTPUT,
                    help="Output folder for CSVs")
    ap.add_argument("--sample",   type=int, default=3,
                    help="Extract every Nth frame (default 3 = ~10fps from 30fps)")
    ap.add_argument("--retrain",  action="store_true",
                    help="Retrain correction AI after extraction")
    ap.add_argument("--stats",    action="store_true",
                    help="Show stats on already-extracted CSVs and exit")
    args = ap.parse_args()

    if args.stats:
        show_stats(args.output)
        sys.exit(0)

    if not args.input.exists():
        print(f"Input folder not found: {args.input}")
        print(f"\nCreate the folder and add your yoga videos:")
        print(f"  {args.input}/")
        print(f"    warrior_ii/correct/video1.mp4")
        print(f"    warrior_ii/wrong/video1.mp4")
        print(f"    chair/correct/video1.mp4")
        sys.exit(1)

    stats = run_extraction(args.input, args.output, sample_every=args.sample)

    if args.retrain and stats:
        print("\n── Retraining Correction AI with real data ──")
        import subprocess
        result = subprocess.run(
            [sys.executable, str(BASE / "train_correction_ai.py")],
            cwd=str(BASE)
        )
        sys.exit(result.returncode)
