# 🧘 AI-Based Background Yoga Posture Correction System

A real-time, voice-guided yoga posture correction tool that runs silently in
the background while you follow any online yoga instructor (YouTube, Zoom, etc.).
It compares your body's joint angles with the instructor's using computer vision
and speaks corrections through your speakers — no extra hardware needed.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     YogaCorrector (orchestrator)                │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────────┐   │
│  │ WebcamCapture│   │ScreenCapture │   │  AudioFeedback    │   │
│  │  (user pose) │   │(instructor   │   │  (pyttsx3 TTS)    │   │
│  └──────┬───────┘   │  pose)       │   └────────▲──────────┘   │
│         │           └──────┬───────┘            │              │
│         ▼                  ▼                    │              │
│  ┌──────────────┐   ┌──────────────┐            │              │
│  │PoseEstimator │   │PoseEstimator │            │              │
│  │  (MediaPipe) │   │  (MediaPipe) │            │              │
│  └──────┬───────┘   └──────┬───────┘            │              │
│         │                  │                    │              │
│         ▼                  │                    │              │
│  ┌──────────────┐           │                    │              │
│  │CameraChecker │           │                    │              │
│  │(alignment)   │           │                    │              │
│  └──────┬───────┘           │                    │              │
│         │                  │                    │              │
│         ▼                  ▼                    │              │
│  ┌─────────────────────────────────┐            │              │
│  │       AngleCalculator           │            │              │
│  │  compute_joint_angles (×2)      │            │              │
│  │  compare_angles(user, ref)      │            │              │
│  └──────────────┬──────────────────┘            │              │
│                 │                               │              │
│                 ▼                               │              │
│  ┌──────────────────────────────────────────────┤              │
│  │          FeedbackBuilder                     │──────────────┘
│  │  build_correction_message / well_done        │
│  └──────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────┘
```

### Key Behaviours
| Situation | System Response |
|-----------|----------------|
| User too close / far | "Please move closer / farther from the camera." |
| Body partially out of frame | "Move to the left / right so your full body is visible." |
| Turned sideways | "Please face the camera directly." |
| Joint misaligned | e.g. "Straighten your left arm." / "Bend your right knee more." |
| Pose correct for ~5 s | "Great work! Your pose looks good." (max once per 30 s) |
| Instructor not detected | Silent — waits for next frame |

---

## Project Structure

```
yoga_corrector/
├── main.py             ← Entry point (CLI args, logging)
├── config.py           ← All tunable constants
├── orchestrator.py     ← Main processing loop
├── pose_estimator.py   ← MediaPipe wrapper → PoseResult
├── angle_calculator.py ← Joint angle maths + comparison
├── camera_checker.py   ← Webcam alignment validation
├── audio_feedback.py   ← Threaded TTS engine (pyttsx3)
├── feedback_builder.py ← Natural-language correction messages
├── screen_capture.py   ← mss-based screen grab
├── webcam_capture.py   ← OpenCV webcam wrapper
└── requirements.txt
```

---

## Installation

### Prerequisites
- Python 3.9 – 3.11 (MediaPipe requires < 3.12 on some platforms)
- A webcam
- Speakers / headphones

### Steps

```bash
# 1. Clone / download the project
cd yoga_corrector

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

> **macOS note:** `pyaudio` requires PortAudio. Install with Homebrew first:
> `brew install portaudio`
>
> **Linux note:** TTS may require `espeak`:
> `sudo apt-get install espeak`

---

## Running

```bash
# Default — captures monitor 1, webcam 0
python main.py

# Show verbose debug output
python main.py --debug

# Use a second monitor and a different webcam
python main.py --monitor 2 --webcam 1

# Looser angle tolerance (more forgiving corrections)
python main.py --tolerance 30

# Reduce feedback frequency (longer cooldown between prompts)
python main.py --cooldown 8
```

Press **Ctrl + C** to stop.

---

## Configuration (`config.py`)

| Constant | Default | Description |
|----------|---------|-------------|
| `ANGLE_TOLERANCE_DEGREES` | `20` | Max allowed joint angle difference (°) |
| `MIN_MISALIGNED_JOINTS` | `2` | Joints that must fail before feedback fires |
| `FEEDBACK_COOLDOWN_SECONDS` | `5` | Min seconds between spoken prompts |
| `MIN_BODY_FRAME_RATIO` | `0.30` | Body must occupy ≥ 30 % of frame height |
| `MAX_BODY_FRAME_RATIO` | `0.90` | Body must not exceed 90 % of frame height |
| `PROCESSING_FPS` | `5` | Pose comparison cycles per second |
| `SCREEN_MONITOR_INDEX` | `1` | Monitor to screen-capture (1 = primary) |
| `WEBCAM_INDEX` | `0` | Webcam device index |

---

## Joints Analysed

| Joint | Landmarks used |
|-------|---------------|
| Left / Right Elbow | Shoulder → Elbow → Wrist |
| Left / Right Shoulder | Elbow → Shoulder → Hip |
| Left / Right Hip | Shoulder → Hip → Knee |
| Left / Right Knee | Hip → Knee → Ankle |

---

## Known Limitations

- **Screen capture detects instructor pose** — works best when the instructor
  occupies most of the screen and is facing the camera (front-facing classes).
- **3-D accuracy** — MediaPipe provides estimated depth; very deep poses or
  unusual camera angles may reduce accuracy.
- **Multi-person screen** — if the video has multiple people, MediaPipe
  selects the most prominent one, which may not always be the instructor.
- **Performance** — processing at 5 FPS requires ~2–4 GB RAM and a modern CPU.
  Reduce `PROCESSING_FPS` on slower machines.

---

## License
MIT — free for personal and educational use.
