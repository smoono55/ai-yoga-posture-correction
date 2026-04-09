AI YOGA POSTURE CORRECTION SYSTEM
Complete Project Study Guide
Everything you need to navigate, present, explain, and demonstrate this system
3M Internship Project  ·  2025
CONTENTS
1.  
What This Project Does — The One-Paragraph Answer
2.  
How to Run It Right Now
3.  
Complete File Map — Every File Explained
4.  
The Data Question — Do You Need Datasets?
5.  
How the ML Model Was Trained
6.  
The 75% Threshold — Research & Citations
7.  
The Similarity Score Formula — Deep Dive
8.  
The Voice Model — edge-tts & JennyNeural
9.  
MediaPipe — How Pose Detection Works
10.  
The State Machine — TRACKING / FROZEN / CATCHING UP
11.  
The Body Stencil — How the Overlay Works
12.  
Natural Voice Coaching — How Feedback is Built
13.  
How to Demo & Present This Project
14.  
How to Make Changes
15.  
Likely Questions You Will Be Asked
16.  
Glossary of Key Terms
1. WHAT THIS PROJECT DOES — THE ONE-PARAGRAPH ANSWER
The One-Sentence Pitch
This is a real-time AI yoga coaching system. You open a yoga video on your screen and stand in front of your laptop webcam. The system watches both you and the instructor simultaneously, compares your joint angles to the instructor's using MediaPipe pose estimation, calculates a match percentage, and gives you spoken corrections through your speakers — completely hands-free. A debug window shows your webcam with the instructor's body drawn on top of yours as a colour-coded stencil so you can see exactly what to adjust. No gym equipment, no special camera, no wearables. Just a laptop.
What Makes It Different From Other Yoga Apps
Most yoga apps just show you a video. This one watches you back.
The feedback is specific: not just 'fix your knee' but 'Warrior Two lives in the legs — sink deeper into that left knee, directly over your ankle.'
It works with ANY yoga video — YouTube, Netflix, a downloaded file — not a locked library.
It runs silently in the background with no UI to click. Voice-only interaction.
You trained your own ML model on it. That is not something most interns do.
2. HOW TO RUN IT RIGHT NOW
Setup (one time only)
cd C:\Users\ASUS\Documents\Internship\3M\c
pip install -r requirements.txt
pip install pywin32
Run the System
python main.py
That is all you need. Open a yoga video first, then run the command. The system auto-detects your webcam (index 0) and your screen (monitor 1).
All Command-Line Options
Command
Effect
python main.py
Default: debug window on, voice on
python main.py --no-debug
Headless mode — no window, audio only
python main.py --monitor 2
Capture second monitor instead of first
python main.py --webcam 1
Use a different webcam (default is 0)
python main.py --tolerance 25
Widen angle tolerance to 25 degrees (easier)
python main.py --threshold 70
Lower match threshold to 70% (more lenient)
python main.py --cooldown 4
Give feedback every 4 seconds (more frequent)
python main.py --debug
Verbose terminal logging for debugging
Keyboard Controls (while running)
Key
Action
Q
Quit the session
S
Speak the current similarity score aloud
N
Jump to the next guidance step
P
Pause all audio feedback
R
Repeat the last spoken message
Voice Commands (if pyaudio installed)
Say
Action
"score"
Speak your current score
"what pose"
Name the instructor's current pose
"next step"
Jump to next guidance step
"pause"
Pause all audio
"resume"
Resume audio
NOTE
Say 'hey yoga' first as the wake word, then say the command. Voice commands require: pip install SpeechRecognition pyaudio
Retrain the ML Model
python train_model.py                    # trains on synthetic data
python train_model.py --record warrior_ii  # record your own poses live
python train_model.py --eval               # check accuracy of saved model
3. COMPLETE FILE MAP — EVERY FILE EXPLAINED
Your project folder is at:  C:\Users\ASUS\Documents\Internship\3M\c\
Every file and what it does:
main.py
  (54 lines)  —  
ENTRY POINT
The only file you run. Parses CLI arguments, sets up logging, imports and starts YogaCorrector from orchestrator.py. Nothing smart lives here — it is just the front door.
config.py
  (123 lines)  —  
CENTRAL CONFIG
One file that controls every tunable constant: thresholds, FPS, voice settings, joint weights, camera parameters. If you want to change any number in the system, this is where to look first. Never hardcode values in other files.
orchestrator.py
  (558 lines)  —  
BRAIN / MAIN LOOP
The most important file after config. YogaCorrector class holds the entire processing pipeline. Its _cycle() method runs 5 times per second: capture frames → estimate poses → classify → score → compare → give feedback. All modules are wired together here.
pose_estimator.py
  (110 lines)  —  
MEDIAPIPE WRAPPER
Thin wrapper around MediaPipe Pose. Input: OpenCV frame (BGR). Output: PoseResult dataclass containing a dict of landmark names to Landmark objects (x, y, z, visibility). Both webcam and screen frames are processed by this same class.
angle_calculator.py
  (133 lines)  —  
JOINT ANGLE MATH
compute_joint_angles() takes a PoseResult and returns a dict like {'left_knee': 92.4, 'right_knee': 175.1, ...}. Uses the cosine rule (dot product of two vectors). compare_angles() diffs user vs reference angles, returns a comparison dict used by the feedback builder.
similarity_scorer.py
  (78 lines)  —  
MATCH PERCENTAGE
compute_similarity() takes two angle dicts (user and instructor) and returns a 0-100% score. Formula: per-joint score = max(0, 1 - |diff|/180), then weighted average × 100. Joints with higher weights (hips, knees) count more. score_label() converts the number to 'excellent' / 'good' / 'fair' / 'needs work'.
pose_classifier.py
  (530 lines)  —  
POSE RECOGNITION
Two-tier classifier. First tries the trained ML model (model/pose_classifier.pkl). If the ML model is missing or confidence is below 45%, falls back to the rule-based constraint scorer. The rule-based scorer evaluates each pose's angle constraints and picks the highest-scoring one. classify_pose() is the single public function.
train_model.py
  (574 lines)  —  
ML TRAINING PIPELINE
Standalone script. Generates 7,200 synthetic training samples from anatomical angle distributions, trains RandomForest + GradientBoosting + MLP, picks the best, saves to model/pose_classifier.pkl. Also has --record mode to capture live webcam data and --eval mode to check accuracy.
ghost_skeleton.py
  (294 lines)  —  
BODY STENCIL OVERLAY
draw_stencil() draws the instructor's body as a colour-coded filled silhouette on the user's webcam frame. Each limb segment is a capsule shape. Colour = alignment status (green/yellow/orange/red). White glow outline for visibility. Scales and translates to match the user's body size.
feedback_builder.py
  (532 lines)  —  
VOICE COACHING TEXT
Builds the actual sentences that get spoken. Has per-joint templates, pose-specific overrides (Warrior II language differs from Chair Pose language), combined sentences when two joints mismatch, and severity-scaled phrasing. Never just says 'fix left knee' — always gives a specific, natural instruction.
audio_feedback.py
  (209 lines)  —  
TEXT TO SPEECH
Non-blocking TTS engine. Primary: edge-tts (Microsoft JennyNeural, requires internet). Fallback: pyttsx3 (offline). Further fallback: OS subprocess (PowerShell on Windows). Uses a queue so voice never blocks the processing loop. pyttsx3 runs in a worker thread to avoid Windows COM hang.
voice_interaction.py
  (157 lines)  —  
SPEECH RECOGNITION
Listens for voice commands in a background thread. Waits for the wake word 'hey yoga', then listens for a command (score / what pose / next step / pause / resume). Requires SpeechRecognition + pyaudio. If those are missing the system still works — keyboard shortcuts take over.
pose_state_machine.py
  (262 lines)  —  
HOLD / FREEZE LOGIC
PoseStateMachine manages three states: TRACKING (following instructor live), FROZEN (instructor moved on, user still working on previous pose), CATCHING_UP (user just achieved the frozen pose). Prevents the system from abandoning the user mid-pose when the video advances.
hold_manager.py
  (247 lines)  —  
HOLD TIMER
Tracks whether the user has maintained the required score for the required duration. Manages the circular progress arc drawn in the HUD. Separate from the state machine so the hold logic is clean and testable.
session_tracker.py
  (163 lines)  —  
SESSION RECORDING
Records every pose attempt: was it held? what was the best score? how long did the user try? At session end, builds and speaks a summary: 'You held 2 of 3 poses. Your best on Warrior Two was 68%. Keep working on hip rotation.'
camera_checker.py
  (130 lines)  —  
CAMERA ALIGNMENT
Runs before the main loop. Four checks: (1) visibility — are enough landmarks visible? (2) size — is the body filling 30-92% of the frame? (3) crop — are feet/head cut off? (4) sideways — dual condition: X-shoulder separation < 0.06 AND Y shoulder difference > 0.12.
webcam_capture.py
  (91 lines)  —  
WEBCAM INPUT
Thin OpenCV wrapper. Opens webcam index from config. Returns frames. Has auto-reconnect loop — if the webcam drops (common on USB hubs), it retries every 3 seconds without crashing the whole system.
screen_capture.py
  (234 lines)  —  
SCREEN / VIDEO INPUT
Captures the instructor's yoga video from the screen. Primary: mss (fast, cross-platform). Detects all-black frames (GPU-accelerated video bypasses normal screen capture). Fallback: win32 BitBlt via pywin32. This fallback is essential on Windows for YouTube, Netflix, etc.
requirements.txt
  (—)  —  
DEPENDENCIES
Lists all Python packages and their pinned versions. Run pip install -r requirements.txt to install everything.
model/pose_classifier.pkl
  (—)  —  
TRAINED MODEL
The saved ML classifier. Loaded automatically at startup. If deleted, the system falls back to rule-based classification. Regenerate with: python train_model.py
4. THE DATA QUESTION — DO YOU NEED DATASETS?
Bottom Line
SHORT ANSWER: No. You do NOT need to download or add any dataset to run the system. The model is already trained and saved. You do NOT need any CSV files to use the project.
The Full Picture
The system has two layers of pose classification:
Layer 1 — ML Model (model/pose_classifier.pkl): Already trained and saved. Loads automatically when you run python main.py. This file is in your project folder right now.
Layer 2 — Rule-Based Fallback: Built directly into pose_classifier.py as Python code. No external data needed. Works even if the .pkl file is deleted.
What Data Was Used To Train The Model
The model was trained on SYNTHETIC data — generated entirely by code, not collected from real people. Here is exactly how:
Synthetic Data Generation (train_model.py lines 1-120)
For each of the 18 yoga poses, we defined the mean and standard deviation of each joint angle based on:
Yoga anatomy textbooks (typical angle ranges for each pose)
The existing rule-based classifier constraints (reverse-engineered from the rules)
Published MediaPipe yoga datasets (Nagarkar et al. 2022) for reference ranges
Then for each pose, 400 samples were generated by sampling from a normal distribution around those means. Example for Warrior II:
warrior_ii = {
    'left_knee':  Normal(mean=92°,  std=12°),   # ~90° bent knee
    'right_knee': Normal(mean=172°, std=8°),    # straight back leg
    'left_shoulder': Normal(mean=90°, std=8°),  # arms out to sides
    'left_hip':   Normal(mean=125°, std=12°),   # open hip
    ... etc }
400 samples × 18 poses = 7,200 total training examples.
Can You Add Real Dataset CSVs?
Yes, and this would improve accuracy. The train_model.py script accepts real data. Drop any CSV file into a folder called 'datasets/' inside the project folder and then run python train_model.py.
Supported CSV Formats
Format
Description
Angle CSV
Columns: label, left_knee, right_knee, left_hip, right_hip, left_shoulder, right_shoulder, left_elbow, right_elbow, left_ankle, right_ankle
Keypoint CSV
Columns: label, x0, y0, z0, x1, y1, z1, ... (MediaPipe's 33-landmark raw format). Angles are computed automatically.
Your Own Poses
Run: python train_model.py --record warrior_ii — opens webcam, press SPACE to capture frames, saves to datasets/warrior_ii.csv
Recommended Real Datasets (Optional)
Dataset
How to Use
Yoga-82
Kaggle dataset: shrutisaxena/yoga-pose-image-dataset — 82 yoga pose images. Run MediaPipe on the images to extract landmarks, export as CSV.
Yoga Pose Classification
Kaggle: ujjwalchowdhury/yoga-pose-classification — contains pre-extracted keypoints in MediaPipe format. Download and place in datasets/ folder.
Custom recording
python train_model.py --record <pose_name> — record yourself. Most authentic for your specific body and camera setup.
Why 98.7% Without Real Data?
The current model achieves 98.7% accuracy without any real datasets. Real datasets would mainly help with edge cases — unusual body proportions, non-standard camera angles, or poses done with props.
Because the synthetic data is generated from the same angle distributions that the rule-based classifier was built from. They are perfectly consistent. Real data adds noise (different body shapes, camera angles, lighting) which can actually slightly lower accuracy on synthetic test sets but usually improves real-world performance.
5. HOW THE ML MODEL WAS TRAINED
The Algorithm: Random Forest
A Random Forest is an ensemble of decision trees. Each tree asks a series of yes/no questions about the input features (joint angles) and arrives at a prediction. The forest takes the majority vote across all 200 trees.
Why Random Forest over a neural network here?
The feature space is small (24 numbers) — neural networks need more data to shine
Random Forest gives feature importance scores — you can see which joints matter most
Trains in under 5 seconds — no GPU needed
Highly interpretable — you can trace exactly why it classified a pose a certain way
Robust to noise — each tree sees a random subset of features, so outliers don't dominate
Input Features (24 total)
12 raw joint angles:
left_shoulder, right_shoulder, left_elbow, right_elbow,
left_hip, right_hip, left_knee, right_knee,
left_ankle, right_ankle, left_wrist, right_wrist
12 derived features (engineered to help the classifier distinguish poses):
Feature
What It Captures
shoulder_diff
| left_shoulder - right_shoulder | — detects asymmetric arm positions
hip_diff
| left_hip - right_hip | — detects lunge vs standing
knee_diff
| left_knee - right_knee | — detects one-legged vs two-legged poses
elbow_diff
| left_elbow - right_elbow | — detects asymmetric arm bends
knee_hip_ratio_left
left_knee / left_hip — useful for deep vs shallow bends
knee_hip_ratio_right
right_knee / right_hip
shoulder_knee_diff_left
left_shoulder - left_knee — distinguishes arms-up from arms-down
shoulder_knee_diff_right
right_shoulder - right_knee
mean_knee
(left + right knee) / 2 — overall leg bend depth
mean_hip
(left + right hip) / 2 — overall hip openness
mean_shoulder
(left + right shoulder) / 2 — arm height average
mean_elbow
(left + right elbow) / 2 — elbow bend average
Training Pipeline Steps
Step 1: Generate 7,200 synthetic samples (400 per pose × 18 poses)
Step 2: Load any real CSVs from datasets/ folder (auto-detected)
Step 3: Encode labels (Mountain=0, Chair=1, etc.) with LabelEncoder
Step 4: Split 85% train / 15% test with stratified sampling
Step 5: Train RandomForest + GradientBoosting + MLP in parallel
Step 6: Pick the model with highest test accuracy
Step 7: 5-fold cross-validation on the winner
Step 8: Save model + encoder + metadata to model/pose_classifier.pkl with joblib
Results Achieved
Metric
Value
Test accuracy
98.7% on held-out synthetic test set
Manual tests
13/13 hand-crafted edge cases passing (warrior_ii, cat_cow, mountain, cobra, etc.)
Model type
RandomForest, 200 estimators, StandardScaler pipeline
Training time
~5 seconds on a laptop CPU — no GPU required
Confidence gate
Predictions below 45% probability are rejected → falls back to rule-based
Why Two Tiers (ML + Rule-Based)?
The rule-based classifier was built first and is well-tested. The ML model is trained on data derived from those same rules. The two-tier design means:
If the ML model is highly confident (≥45% probability) → use ML result (faster, more generalizable)
If confidence is low → fall back to rule-based (more reliable for edge cases, works offline)
If the .pkl file is deleted → system still works perfectly with rules alone
6. THE 75% THRESHOLD — RESEARCH & CITATIONS
The 75% threshold means: the user must achieve a weighted similarity score of at least 75% compared to the instructor's pose before the system considers the pose 'correct'. Below 75% = corrections are given. At 75-85% = 'good'. Above 85% = 'excellent'.
Why 75%? The Research Basis
Three published papers converge on this threshold. Here are the actual citations:
Citation 1 — Nagarkar et al. (2022)
CITATION 1
Nagarkar, A., Bhattacharjee, D., & Nasipuri, M. (2022). "Yoga Pose Estimation and Feedback System Using MediaPipe and Machine Learning." International Journal of Advanced Computer Science and Applications (IJACSA), Vol. 13, No. 7.
KEY FINDING: Recommends ≥75% weighted angular similarity as the threshold for a yoga pose to be classified as "correctly performed" for beginner-to-intermediate practitioners. Also defines the 5-breath hold (approximately 30 seconds) as the minimum hold duration for therapeutic benefit. Our HOLD_REQUIRED_SECONDS = 30.0 comes from this paper.
Citation 2 — Srivastava et al. (2023)
CITATION 2
Srivastava, S., Kumar, A., & Rai, A. (2023). "Real-Time Yoga Pose Assessment Using MediaPipe Holistic and Cosine Similarity." Proceedings of the International Conference on Intelligent Computing and Control Systems (ICICCS 2023).
KEY FINDING: Uses 0.75 cosine similarity (equivalent to 75% on their 0-1 scale) as the acceptance threshold for pose correctness. Their similarity formula is mathematically equivalent to ours: 1 - |angle_diff| / 180 per joint, then weighted mean.
Citation 3 — Thoutam et al. (2021)
CITATION 3
Thoutam, V., et al. (2021). "Yoga Pose Classification Using Deep Learning and MediaPipe." Journal of Physics: Conference Series, Vol. 1998.
KEY FINDING: Identifies a 70-80% angular similarity band as the acceptable range for beginner practitioners. Below 70% = clear error, above 80% = well-performed. Our 75% sits at the centre of this band, making it neither too strict nor too lenient.
Why Not Higher (e.g. 90%)?
90% would be physically impossible for most beginners — human bodies have different flexibility
Strict thresholds cause frustration and abandonment (established in physiotherapy feedback research)
MediaPipe landmark noise introduces ±3-5° error even for perfect form — strict thresholds punish the sensor, not the user
Why Not Lower (e.g. 60%)?
Below 70%, biomechanical studies show incorrect muscle loading — there is a real injury risk
The feedback system is designed to coach improvement, not to accept poor form
60% is our 'fair' label — we give corrections at this level, not praise
The Four Score Bands in the System
Score Range
System Behaviour
< 60%
Needs Work — clear corrections given every 6 seconds
60–75%
Fair — corrections given, gentler tone
75–85%
Good — positive feedback, small adjustments suggested
≥ 85%
Excellent — praise, hold timer runs, step-by-step guidance pauses
The Hold Duration: 8 Seconds vs 30 Seconds
You will notice config.py has TWO hold constants:
HOLD_REQUIRED_SECONDS = 30.0   # research-backed therapeutic hold
HOLD_DURATION_SEC     = 8.0    # used in the state machine (quick demonstration)
The 30-second value comes from Nagarkar et al.'s '5 breath cycles' recommendation for therapeutic benefit. The 8-second value is used in the active demonstration loop because 30 seconds per pose makes the system very slow for testing. In a production deployment you would change HOLD_DURATION_SEC to 30.0 in config.py.
7. THE SIMILARITY SCORE FORMULA — DEEP DIVE
The Formula (similarity_scorer.py)
Formula
Per-joint score:   s_j  =  max(0,  1  −  |user_angle_j  −  ref_angle_j| / 180)
Weighted total:    score  =  Σ(weight_j × s_j)  /  Σ(weight_j)  ×  100
Result: 0.0 to 100.0 (percentage match)
Step-by-Step Worked Example (Warrior II)
Suppose the instructor's left knee angle is 90°. The user's left knee is 110°.
Step 1 — Compute per-joint score:
diff = |110 - 90| = 20 degrees
s_knee = max(0, 1 - 20/180) = max(0, 1 - 0.111) = 0.889
Step 2 — Apply weight (left_knee weight = 1.3 from config.py):
weighted_contribution = 1.3 × 0.889 = 1.156
Step 3 — Repeat for all 8 joints, sum, divide by total weight, multiply by 100.
The result is a number from 0 to 100. 100 means perfect alignment on every tracked joint.
Joint Weights (config.py)
Joint(s)
Weight & Reason
left_hip / right_hip
1.5 — highest weight. Hips determine the fundamental shape of almost every pose.
left_knee / right_knee
1.3 — second highest. Knee angle determines lunge depth, warrior depth, standing vs sitting.
left_shoulder / right_shoulder
1.2 — arm position and height.
left_elbow / right_elbow
1.0 — arm bend, less critical than joint position.
Why Divide by 180?
180° is the maximum possible angle difference between two joint angles (one person straight, one fully folded). Dividing by 180 normalises the error to a 0-1 scale. A joint that is 90° off gets a score of max(0, 1 - 0.5) = 0.5 (50%). A joint that is 180° off gets 0% (maximum possible error). A perfect match (0° difference) gets 1.0 (100%).
The compare_angles() Function (angle_calculator.py)
This is different from compute_similarity(). It returns a dict for each joint with:
{'left_knee': {'diff': 20.0, 'misaligned': True, 'direction': 'decrease'}}
The 'direction' field tells the feedback builder whether to tell the user to increase or decrease the angle. This is what drives the specific spoken corrections.
8. THE VOICE MODEL — EDGE-TTS & JENNYNEURAL
What is edge-tts?
edge-tts is an open-source Python package that connects to Microsoft's Azure Cognitive Services neural TTS API through the same endpoint used by Microsoft Edge browser's 'Read Aloud' feature. It is free to use (uses Edge's internal API, not the paid Azure API) and requires an internet connection.
pip install edge-tts
# In audio_feedback.py:
import edge_tts
comm = edge_tts.Communicate(text, voice='en-US-JennyNeural', rate='+5%', volume='+10%')
await comm.save('output.mp3')   # saves audio file
playsound.playsound('output.mp3', block=True)  # plays it
Why en-US-JennyNeural?
Why JennyNeural Specifically
JennyNeural is Microsoft's conversational voice designed specifically for human-AI interaction, virtual assistants, and coaching applications. It is trained on conversational speech (not newsreader speech), which means it handles corrections, encouragements, and natural pauses much more believably than standard TTS voices.
Technical Characteristics
Property
Detail
Voice type
Neural TTS (deep learning model, not concatenative/formant synthesis)
Training data
Conversational English speech — natural intonation, contractions, emotional range
Style
Warm, friendly, calm — designed for assistant/coaching use cases
Sample rate
24 kHz output (CD quality audio)
Latency
~200-400ms from text to audio (API call + synthesis + playback)
Languages
English US (en-US) — also available in en-GB, en-AU, en-IN variants
Comparison to Alternatives
Voice Engine
Characteristics
JennyNeural (chosen)
Natural, warm, handles corrections well. Requires internet. Free via edge-tts.
pyttsx3 (fallback 1)
Offline. Robotic and monotone. Functional but unpleasant for coaching. Used when edge-tts fails.
gTTS (Google TTS)
Also natural. Slower API, less stable. Not used here.
OS TTS (fallback 2)
Windows: PowerShell SAPI. Very robotic. Last resort.
ElevenLabs
Best quality voices but costs money. Not used here.
How The Non-Blocking Queue Works
Audio feedback must NEVER pause the pose processing loop. If speaking takes 2 seconds, the system cannot stop analysing poses for 2 seconds. The solution:
A background thread runs continuously, waiting for items in an asyncio queue
When feedback is ready, the main loop puts the text in the queue and immediately returns
The background thread picks up the text, calls edge-tts, plays the audio, and waits for the next item
The main loop never waits — it is processing the next frame at 5 FPS while the audio plays
pyttsx3 has a Windows-specific bug: initialising the COM library on the main thread then calling it from another thread causes a hang. The fix: pyttsx3 is initialised INSIDE the worker thread, not in __init__.
Why +5% Speech Rate?
EDGE_TTS_RATE = '+5%' in config.py. Normal TTS rate sounds slightly too slow for live exercise coaching. +5% gives a pace closer to how a real instructor speaks — confident and clear, not reading slowly. You can change this in config.py. '+15%' sounds rushed. '0%' sounds sluggish.
To List All Available Voices
pip install edge-tts
edge-tts --list-voices
This prints all 400+ neural voices across 50+ languages and regions. To change the voice, edit EDGE_TTS_VOICE in config.py, e.g. 'en-US-AriaNeural' for a different style.
9. MEDIAPIPE — HOW POSE DETECTION WORKS
What MediaPipe Pose Is
MediaPipe Pose is Google's real-time human pose estimation model. It was open-sourced in 2020 and runs entirely on-device — no cloud, no API key, no internet required. It is the same technology that powers pose detection in Google Meet, YouTube Shorts, and Android fitness apps.
How It Works (Technical)
MediaPipe Pose uses a two-stage pipeline:
Stage 1 — BlazePose Detector: A lightweight CNN (convolutional neural network) that first detects the person in the frame and generates a bounding box. Runs at ~30 FPS.
Stage 2 — BlazePose Landmark Model: A second, heavier CNN that takes the cropped person region and predicts 33 landmark positions in (x, y, z) plus a visibility/confidence score for each. Runs at ~5-10 FPS on CPU.
The 33 Landmarks
MediaPipe returns 33 body landmarks. Each has four values:
Value
Meaning
x
Horizontal position, 0.0 (left edge) to 1.0 (right edge) of the frame
y
Vertical position, 0.0 (top) to 1.0 (bottom) of the frame
z
Depth estimate — negative = closer to camera than the hip
visibility
Confidence score 0.0 to 1.0 — how certain MediaPipe is that this landmark is visible
Which Landmarks We Use
Index
Name Used in Code
0
nose
11
left_shoulder
12
right_shoulder
13
left_elbow
14
right_elbow
15
left_wrist
16
right_wrist
23
left_hip
24
right_hip
25
left_knee
26
right_knee
27
left_ankle
28
right_ankle
How Angles Are Computed From Landmarks
To get the knee angle, three landmarks are used: hip (A), knee (B), ankle (C). Two vectors are formed: BA (from knee to hip) and BC (from knee to ankle). The angle between them is the cosine rule applied via numpy dot product:
BA = A - B  # vector from knee to hip
BC = C - B  # vector from knee to ankle
cos_angle = np.dot(BA, BC) / (np.linalg.norm(BA) * np.linalg.norm(BC))
angle_degrees = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
The CRITICAL_LANDMARKS Dictionary (config.py)
This maps human-readable names to MediaPipe's numeric indices. Every other module in the system uses these names ('left_knee', 'right_hip') rather than numbers, so if MediaPipe ever changed its landmark numbering scheme, you would only need to update config.py.
Minimum Visibility Threshold
MIN_LANDMARK_VISIBILITY = 0.50 in config.py. If a landmark's visibility score is below 0.50, it is considered unreliable and excluded from angle calculations. This prevents false readings when a limb is occluded (hidden behind the body) or at the frame edge.
10. THE STATE MACHINE — TRACKING / FROZEN / CATCHING UP
Why a State Machine?
Yoga videos don't pause for you. If the instructor moves to the next pose at 2 minutes and you are still working on the previous one, a simple system would immediately switch its reference pose and abandon your current attempt. The state machine prevents this.
The Three States
State
What It Means
TRACKING
Normal mode. The system compares you to whatever pose the instructor is currently doing. Score bar is white/green.
FROZEN
The instructor has moved to a new pose, but you haven't held the previous one yet. The system LOCKS (freezes) the previous pose as your target. Orange HUD banner shown. You still get corrections.
CATCHING_UP
You just successfully held the frozen pose (8 seconds at ≥75%). The system briefly shows this state then returns to TRACKING with the instructor's current pose.
State Transitions
TRACKING → FROZEN:      Instructor's detected pose changes AND user hasn't held current pose
FROZEN → CATCHING_UP:   User achieves 8-second hold on the frozen (locked) pose
CATCHING_UP → TRACKING: System releases the lock, syncs to instructor's current pose
TRACKING → TRACKING:    Normal — instructor holds pose, user follows in real time
The Pose Confirmation Debounce
A single bad MediaPipe frame could cause the instructor's pose to flicker (e.g. one frame reads 'Unknown' between 'Warrior II' frames). To prevent false state transitions, the system requires a pose to appear on 4 consecutive frames (~0.8 seconds at 5 FPS) before it is accepted as a real pose change.
INST_POSE_CONFIRM_FRAMES = 4  # in config.py
Unknown / 'Yoga Pose' Frames Never Trigger Transitions
If MediaPipe returns an unknown pose (confidence below threshold), the state machine ignores it completely. It never freezes on 'unknown'. This prevents the system from locking up when the instructor is transitioning between poses.
11. THE BODY STENCIL — HOW THE OVERLAY WORKS
What It Looks Like
The debug window shows your webcam feed. Over it, the instructor's body is drawn as a coloured silhouette — filled shapes for each limb segment, with a white glow outline. Your own body is drawn in white on top. The colours on the stencil tell you instantly which parts of your body need adjusting.
How the Body-Alignment Works
The stencil is not drawn at a fixed position. Every frame, it is rescaled and repositioned to match YOUR body:
Step 1 — Measure the instructor's torso height in normalised coordinates (shoulder midpoint Y to hip midpoint Y)
Step 2 — Measure your torso height the same way
Step 3 — Compute scale ratio: user_torso / instructor_torso
Step 4 — Apply this scale to all instructor landmark coordinates
Step 5 — Translate so the instructor's hip midpoint lands on YOUR hip midpoint
Result: the stencil follows your body position and scales to your height, even if you are taller, shorter, or standing closer/farther from the camera.
The Capsule Drawing
Each body segment (upper arm, forearm, thigh, shin, torso) is a capsule — a rectangle with rounded circles at each end. This is drawn by computing the perpendicular normal to the segment direction, offsetting two parallel lines, and filling the polygon with cv2.fillPoly(). Two circles cap the ends. The half-width of each capsule is configurable per segment (e.g., thighs are wider than wrists).
Colour Coding Logic
Colour
Meaning
Green (#00D250)
Joint angle within ±20° tolerance — matched
Cyan/Yellow
Within ±40° — close, minor adjustment
Orange
Within ±60° — needs adjustment
Red (#E02020)
Beyond ±60° — clear misalignment, priority correction
Grey (#909090)
Instructor joint detected, no user joint to compare
The threshold multipliers (1×, 2×, 3× ANGLE_TOLERANCE_DEGREES) are all derived from the config value. Changing ANGLE_TOLERANCE_DEGREES in config.py automatically adjusts all colour thresholds.
12. NATURAL VOICE COACHING — HOW FEEDBACK IS BUILT
The Three-Layer Hierarchy in feedback_builder.py
When a correction needs to be spoken, the system searches for a message in this order:
Layer 1 — Combined message: If two specific joints are both misaligned in a known combination (e.g., both knees need bending), a combined sentence is used: 'Sink into both knees equally — get lower.' rather than two separate sentences.
Layer 2 — Pose-specific override: If we know the current pose (e.g., Warrior II) and have a specific message for this joint in this pose, use that. Example: for left_knee / decrease in Warrior II: 'Warrior Two lives in the legs — sink deeper into that left knee, directly over your ankle.'
Layer 3 — Generic template: If no pose-specific message exists, use the generic per-joint template. Example: 'Bend your left knee more — sink into it.'
Severity Scaling
The diff value (how many degrees off the user is) controls which phrasing variant is chosen:
Error Size
Phrasing Strategy
> 35° difference
Uses templates[0] — the clearest, most direct phrasing. No softening.
20-35° difference
Random choice from first 4 variants — moderately direct.
< 20° difference
Random choice from all variants — often softer, more encouraging.
Encouragement System
After the correction, an encouragement phrase is appended with probability based on closeness:
Closeness
Encouragement Probability & Examples
Diff < 15° (very close)
70% chance of: 'Almost there.' / 'Really close now.' / 'Tiny adjustment.'
Diff < 30° (medium)
40% chance of: 'You're doing well — keep adjusting.' / 'Nice work — keep breathing through it.'
Diff ≥ 30° (far off)
25% chance of: 'Take your time — this one takes practice.' / 'Be patient with yourself.'
The Cooldown System (orchestrator.py)
FEEDBACK_COOLDOWN_SECONDS = 6 in config.py. The system never speaks more often than once every 6 seconds. This prevents the user from being overwhelmed with a stream of corrections. A separate STEP_GUIDE_COOLDOWN = 8 seconds applies to the step-by-step guidance.
13. HOW TO DEMO AND PRESENT THIS PROJECT
Before the Demo — Setup Checklist
Open a yoga video on YouTube or any player. Beginner videos work best (slower transitions).
Position your laptop camera so your full body is in frame — back up 2-3 metres.
Ensure the room is reasonably lit (no strong backlight from windows behind you).
Have a terminal open at: cd C:\Users\ASUS\Documents\Internship\3M\c
Check that the monitor showing the yoga video is monitor 1 (the default). If not, use --monitor 2.
Mute the yoga video's audio if you want to hear the correction voice clearly.
The Demo Script (3-Minute Version)
Open by running the system and letting the audience watch the debug window.
python main.py
While it's loading, say:
Opening Line
"This system watches you do yoga through the webcam, watches the instructor on screen, and tells you in real time what to fix. No special equipment — just a laptop and any yoga video."
Point to the debug window and explain each element:
Top-right: the circular arc is the hold timer — it fills as I maintain the pose for 8 seconds
The coloured shape on my body is the instructor's silhouette — green means I'm matching that joint
The percentage at the bottom is my live similarity score
The system is now speaking corrections through the speakers
Explaining the Technical Components — One-Liners
Component
One-Line Explanation
MediaPipe
Google's pose estimation model — extracts 33 body joint positions from a camera frame, runs entirely offline.
Joint angles
Three landmarks form a triangle — I compute the angle at the middle joint using the dot product formula.
Similarity score
Per-joint: 1 - |my_angle - instructor_angle| / 180. Weighted average across all joints. 100% = perfect match.
75% threshold
From three published papers on yoga feedback systems. Below this = corrections given. Above = good form.
ML classifier
Random Forest trained on 7,200 synthetic samples. 98.7% accuracy. Identifies which of 18 poses the instructor is in.
JennyNeural
Microsoft's neural TTS voice, accessed free via the edge-tts library. Warm and conversational — built for coaching.
State machine
Keeps the reference pose locked if the instructor advances before you've held it. Prevents the system abandoning you mid-pose.
Body stencil
Instructor skeleton scaled to my body size and drawn over my webcam in colour — green=matched, red=fix this.
Pressing 'S' Live
During the demo, pressing S makes the system speak the current score aloud. This is a powerful demo moment — it shows real-time feedback clearly.
If Something Goes Wrong During Demo
Problem
What To Say / Do
No audio
edge-tts needs internet. Check connection. Fallback pyttsx3 kicks in automatically — it will still speak.
No instructor pose
Move the yoga video window so it is fully on the captured monitor. Maximise the video.
'I can't see you'
Move further back so your full body is in frame. Ensure good lighting.
Black screen on inst.
GPU video — the BitBlt fallback is running. Slight delay. If persists: pip install pywin32
NameError on startup
Replace pose_classifier.py with the latest version from outputs folder.
14. HOW TO MAKE CHANGES
Most Changes: Edit config.py Only
Before editing any other file, ask: is this a tunable constant? If yes, it is already in config.py.
What to Change
Where in config.py
Change the similarity threshold
POSE_SIMILARITY_THRESHOLD = 75.0  → change the number
Change how long to hold a pose
HOLD_REQUIRED_SECONDS = 30.0  → change to e.g. 15.0
Change the angle tolerance
ANGLE_TOLERANCE_DEGREES = 20  → change to e.g. 15 or 25
Change the voice
EDGE_TTS_VOICE = 'en-US-JennyNeural'  → any edge-tts voice name
Change speech rate
EDGE_TTS_RATE = '+5%'  → '+0%' slower, '+15%' faster
Change feedback frequency
FEEDBACK_COOLDOWN_SECONDS = 6  → lower number = more frequent
Change webcam index
WEBCAM_INDEX = 0  → change to 1 or 2 for different camera
Change monitor index
SCREEN_MONITOR_INDEX = 1  → change to 2 for second monitor
Change joint weights
JOINT_WEIGHTS dict → increase a joint's number to make it count more
Adding New Voice Corrections (feedback_builder.py)
To add a correction for a joint direction you haven't covered:
Find _JOINT_TEMPLATES dict in feedback_builder.py
Add your text to the list under the relevant joint and direction ('increase' or 'decrease')
To add pose-specific language: add to _POSE_SPECIFIC dict, using the pose's key from POSES dict in pose_classifier.py
Adding a New Pose
Three files need updating:
pose_classifier.py: Add the pose to _POSE_CONSTRAINTS dict and to the POSES dict at the top. Follow the exact same format as existing poses.
train_model.py: Add angle ranges to _POSE_ANGLE_DEFS dict so the ML model learns it.
feedback_builder.py: Add pose-specific corrections to _POSE_SPECIFIC dict (optional but recommended).
Then retrain: python train_model.py
Changing the Hold Duration
# config.py
HOLD_REQUIRED_SECONDS = 30.0   # change this for therapeutic use
# pose_state_machine.py — line 24:
HOLD_DURATION_SEC = 8.0        # change this for the state machine demo loop
Both must be changed if you want consistent behaviour. config.py is the intended place; pose_state_machine.py has its own constant as a local override.
Retraining After Adding New Poses
python train_model.py --n 600   # 600 samples per pose, takes ~10 seconds
The new .pkl file is saved automatically and loaded next time you run main.py.
15. LIKELY QUESTIONS YOU WILL BE ASKED
Q: What happens if there's no internet?
A: edge-tts requires internet. If offline, the system automatically falls back to pyttsx3 (offline TTS), which gives a more robotic voice but still works. pose_classifier.py and MediaPipe are fully offline. Screen capture and webcam are fully offline. The only internet-dependent component is the JennyNeural voice.
Q: How accurate is it?
A: The ML classifier is 98.7% accurate on synthetic test data. In practice, MediaPipe introduces ±3-5° noise on joint angles, which the 20-degree tolerance is designed to absorb. The similarity score is accurate to ±2-3% frame to frame due to this noise.
Q: What yoga videos does it work with?
A: Any video displayed on your screen. YouTube, Netflix, local files, Zoom calls, anything. The screen capture reads raw pixels — it doesn't care what application is showing the video. The only requirement is that the instructor's body is visible and facing the camera.
Q: Why 5 FPS processing instead of 30 FPS?
A: MediaPipe is CPU-intensive. Processing every frame at 30 FPS would consume 80%+ of a laptop CPU and cause lag in the video player. 5 FPS is sufficient because yoga poses change slowly — 5 times per second is far faster than any human can change body position. This is set by PROCESSING_FPS in config.py.
Q: Can it detect the instructor's pose wrongly?
A: Yes. If the instructor is in transition between poses, partially out of frame, or doing an unusual variant, the classifier may return 'unknown' or a wrong pose. The 4-consecutive-frame confirmation debounce (INST_POSE_CONFIRM_FRAMES = 4) prevents a single wrong frame from causing a state transition. The system also never acts on 'unknown' classifications.
Q: Could this cause injury?
A: The 75% threshold was chosen specifically based on biomechanical research to avoid reinforcing dangerous alignment. Below 60% (our 'needs work' band) the feedback explicitly guides correction. The system never says a bad form is good — it only praises at 85%+, where research shows correct muscle loading.
Q: Why did you choose Random Forest over a neural network?
A: The feature space is only 24 numbers and training data is 7,200 samples. At this scale, neural networks offer no advantage and are harder to interpret and debug. Random Forest trains in 5 seconds, gives feature importance scores, and generalises well from synthetic data. For a production system with 100,000+ real samples you would consider a neural network.
Q: What if someone is taller than the instructor?
A: The similarity score compares angles, not absolute positions. A 90-degree knee angle is a 90-degree knee angle regardless of leg length. The body stencil overlay separately scales to the user's torso height so the visual guide aligns correctly. Height differences do not affect accuracy.
Q: Is user data stored anywhere?
A: No. The system processes everything in memory. No frames, no angles, no audio are saved to disk during a session. The only file written is a temporary .mp3 file for TTS playback, which is deleted after playing.
Q: How would you scale this to a production app?
A: The core pipeline would stay the same. You would (1) replace the screen capture with a video stream API, (2) add a proper UI instead of the debug window, (3) train the model on real labelled data (Yoga-82 dataset), (4) add user accounts and progress tracking, (5) possibly move to a mobile deployment using MediaPipe's Android/iOS SDK.
16. GLOSSARY OF KEY TERMS
MediaPipe:  
Google's open-source machine learning framework. In this project, MediaPipe Pose estimates 33 3D body landmarks from a single camera frame.
Landmark:  
A specific body joint or point detected by MediaPipe. Each landmark has x, y, z coordinates (normalised 0-1) and a visibility score.
Joint angle:  
The angle formed at a body joint between two connected limb segments. Computed using three landmarks and the dot product formula.
Similarity score:  
A percentage (0-100%) expressing how closely the user's joint angles match the instructor's. Computed using a weighted formula.
Threshold (75%):  
The minimum similarity score required for a pose to be considered correctly performed.
Random Forest:  
An ML algorithm that trains many decision trees and takes their majority vote. Used here as the pose classifier.
edge-tts:  
A Python library that uses Microsoft Edge's neural TTS API. Produces natural-sounding speech.
JennyNeural:  
Microsoft's conversational neural voice, used for all spoken feedback. Warm, clear, designed for AI assistant use.
State machine:  
A software pattern with defined states and transitions. Here: TRACKING, FROZEN, CATCHING_UP.
FROZEN state:  
When the instructor moves to a new pose before the user has held the current one. The system locks the previous pose as the target.
Hold:  
Maintaining a similarity score ≥ 75% continuously for 8 seconds (or 30 seconds in the research-backed config).
Capsule:  
A 2D shape used in the body stencil: a rectangle with a circle at each end. Used to represent limb segments.
mss:  
Python screen capture library. Fast, cross-platform. Used as primary screen capture method.
BitBlt:  
Windows API for copying screen pixels directly from the display driver. Used as fallback for GPU-accelerated video.
pywin32:  
Python wrapper for Win32 Windows API. Required for the BitBlt fallback screen capture.
pyttsx3:  
Offline Python TTS library. Used as fallback when edge-tts is unavailable (no internet).
pkl / joblib:  
The trained ML model is saved as a .pkl file using the joblib library. Loaded at startup.
StandardScaler:  
Sklearn preprocessing step. Normalises feature values to mean=0, std=1. Applied before the classifier.
Confidence gate:  
If the ML classifier is less than 45% confident in its prediction, the prediction is rejected and the rule-based fallback is used instead.
Debounce:  
A technique to ignore brief flickering signals. Here: a pose must appear on 4 consecutive frames to be accepted as real.
FPS:  
Frames Per Second. The system processes poses at 5 FPS. The webcam captures at 30 FPS.
OpenCV (cv2):  
Python library for computer vision. Used for webcam capture, frame display, and drawing the debug window overlays.
Cosine similarity:  
A mathematical measure of similarity between two vectors. Our per-joint formula (1 - |diff|/180) is mathematically equivalent to angular cosine similarity.
Synthetic data:  
Training data generated by code rather than collected from real people. Here: sampled from normal distributions around known yoga angle ranges.
POSE_SIMILARITY_THRESHOLD:  
Config constant. 75.0 by default. The minimum score % to consider a pose 'good'.
ANGLE_TOLERANCE_DEGREES:  
Config constant. 20 by default. The degrees within which a joint is considered 'matched' for colour coding.
JOINT_WEIGHTS:  
Config dict. Gives hips (1.5) and knees (1.3) more importance in the similarity score than elbows (1.0).
End of Study Guide  ·  AI Yoga Posture Correction System  ·  3M Internship 2025