---
marp: true
theme: default
paginate: true
header: "AI Yoga Posture Correction System"
footer: "IEEE Research Presentation"
style: |
  section {
    justify-content: flex-start;
  }
  h1 {
    color: #0056b3;
    margin-top: 50px;
  }
  h2 {
    color: #333;
  }
---

<!-- class: lead -->
# Real-Time Multilingual AI Yoga Posture 
# Correction System 
## Featuring Dynamic Ghost Skeleton Overlay
**Author:** [Your Name / Team Name]  
**Organization:** [Your Organization/Institution]

---

## 1. The Problem Space

- **Rise of Digital Fitness:** Online yoga is booming, but lacks two-way feedback.
- **Injury Risk:** Incorrect postures repeated over time lead to joint strain and physical injury.
- **Limitations of Current Solutions:**
  - Wearable sensors (IMUs) are intrusive and expensive.
  - Existing AI checks require high-end GPUs.
  - Visual-only feedback forces users to constantly look at screens, breaking their balance.

---

## 2. Our Proposed Solution

An automated, hardware-agnostic AI system running locally on the user's desktop to provide:

1. **Dual-Modality Architecture:**
   - Visual: **Dynamic Ghost Skeleton Overlay**
   - Auditory: **Multilingual Audio Corrections**
2. **Zero Calibration:** Automatically scales the instructor’s body to match the user's proportions in real-time.
3. **Accessibility:** Operates natively alongside Zoom/YouTube without needing a discrete graphics card.

---

## 3. System Architecture

The pipeline consists of decoupled parallel modules:

- **Reference Layer:** Parses the screen to find the instructor using `mss`.
- **Practitioner Layer:** Parses the webcam using OpenCV.
- **MediaPipe Engine:** Extracts 33 3D biomechanical joint landmarks using BlazePose topology.
- **Alignment Matrix:** Synchronizes the spatial frames dynamically.

---

## 4. The "Ghost Skeleton" Algorithm

How do we compare a 6-foot user with a 5-foot instructor on a screen?

- **Dimensional Anchoring:** Core torso breadth $D = \| L_{Shoulder} - R_{Shoulder} \|$ is used to derive an affine $X, Y$ scale.
- **Projection:** The instructor’s joints are projected directly onto the user's torso.
- **Color-Coded Feedback Matrix:**
  - 🟩 **Green:** Perfect alignment
  - 🟦 **Cyan:** Warning
  - 🟧 **Orange:** Action Required
  - 🟥 **Red:** Critical Misalignment 

---

## 5. Multilingual Audio Engine

- Prevents the user from having to break their neck to look at the screen!
- **Rule-Based Engine:** Translates joint angle deltas $\Delta \theta$ into semantic cues.
  *(e.g. "Straighten your left knee")*
- **Debounce Logic:** Feedback requires the error to persist for 3+ frames to prevent jumpy audio.
- **Offline / Edge NLP:** Uses asynchronous python text-to-speech tools to support translation across 14 languages instantly.

---

## 6. Real-Time Hardware Optimization

- **CPU-First Execution:** Throttles visual inference to 5 FPS.
- Yoga is a slow, sustained practice; 60 FPS is computationally wasteful.
- Adaptive sleep pipelines free up over 80% CPU overhead, preventing thermal throttling.
- Consumes ~28% CPU on a standard mid-range integrated laptop.
- End-to-End latency stays under ~65ms.

---

## 7. Experimental Results

Evaluated on a cohort of 15 amateur practitioners tracking 21,200 total frames:

* **Correction Accuracy:** 92.1% overall localization match against manual expert annotation.
* **Exceptional Performance on Standing Poses:** 
   - Warrior II (94.2%) 
   - Tree Pose (96.5%)
* **Qualitative (SUS Feedback):**
   - Users rated the Audio Feedback as heavily preferred to strict visual checking (**4.8 / 5.0**).

---

## 8. Limitations & Fail-Safes

- **Z-Depth Ambiguity:** Poses pointing directly at the camera distort MediaPipe’s pseudo-depth.
   *(Solution: Visual render halts if confidence drops below 60%)*
- **Baggy Clothing:** Drapes degrade the precision of finding the precise ulna/radius joint.
- **Multi-Person Error:** The system currently latches onto the most prominent body, struggling in chaotic group classes.

---

## 9. Future Work

1. **Temporal Action Localization:** Use LSTMs or Transformers to grade *transitions* (e.g. flow into Chaturanga) instead of just static poses.
2. **YOLO-Based Tracking:** Pre-process the screen capture to lock the bounding box specifically onto the main instructor.
3. **Custom Biometrics:** Allow users to log historical injuries, adjusting the sensitivity tolerance dynamically.

---

<!-- class: lead -->
# Thank You
## Any Questions?
