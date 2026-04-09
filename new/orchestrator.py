# orchestrator.py — Main loop integrating state machine + session tracker

from __future__ import annotations
import logging
import time
import threading
import cv2
import numpy as np
from typing import Optional

import config
from pose_estimator      import PoseEstimator, PoseResult
from angle_calculator    import compute_joint_angles, compare_angles
from camera_checker      import CameraChecker
from audio_feedback      import AudioFeedback
from similarity_scorer   import compute_similarity, score_label, score_colour
from pose_classifier     import classify_pose, YogaPose, POSES
from pose_state_machine  import PoseStateMachine, State, HOLD_DURATION_SEC
from session_tracker     import SessionTracker
from voice_strings    import t, t_summary
from feedback_builder    import (
    build_correction_message, build_score_message,
    build_well_done_message,  build_step_message,
    build_pose_intro,
)
from voice_interaction   import VoiceInteraction
from ghost_skeleton      import draw_stencil

logger = logging.getLogger(__name__)

_WHITE  = (240, 240, 240)
_DARK   = (20,  20,  20)
_GREEN  = (0,  210,  80)
_YELLOW = (0,  200, 255)
_RED    = (30,  60, 220)
_CYAN   = (255, 210,  0)
_ORANGE = (0,  160, 255)

_BONES = [
    ("left_shoulder","right_shoulder"),
    ("left_shoulder","left_elbow"),   ("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
    ("left_shoulder","left_hip"),     ("right_shoulder","right_hip"),
    ("left_hip","right_hip"),
    ("left_hip","left_knee"),         ("left_knee","left_ankle"),
    ("right_hip","right_knee"),       ("right_knee","right_ankle"),
    ("nose","left_shoulder"),         ("nose","right_shoulder"),
]


def _draw_skeleton(frame, result: PoseResult, colour=_GREEN) -> None:
    if not result.detected:
        return
    h, w = frame.shape[:2]
    pts  = {
        n: (int(lm.x * w), int(lm.y * h))
        for n, lm in result.landmarks.items()
        if lm.visibility >= config.MIN_LANDMARK_VISIBILITY
    }
    for a, b in _BONES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], colour, 3, cv2.LINE_AA)
    for pt in pts.values():
        cv2.circle(frame, pt, 6, colour,  -1, cv2.LINE_AA)
        cv2.circle(frame, pt, 6, _DARK,    1, cv2.LINE_AA)


def _draw_score_bar(frame, score: float, y: int) -> None:
    h, w  = frame.shape[:2]
    bar_w = int(w * 0.58)
    bar_x = (w - bar_w) // 2
    bar_h = 13
    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), (55,55,55), -1)
    fill  = int(bar_w * score / 100)
    cv2.rectangle(frame, (bar_x, y), (bar_x + fill, y + bar_h), score_colour(score), -1)
    cv2.rectangle(frame, (bar_x, y), (bar_x + bar_w, y + bar_h), _WHITE, 1)
    # Threshold marker line
    tx = bar_x + int(bar_w * config.POSE_SIMILARITY_THRESHOLD / 100)
    cv2.line(frame, (tx, y - 2), (tx, y + bar_h + 2), (0, 180, 255), 2)


def _draw_hold_arc(frame, hold_sec: float) -> None:
    """Draw a circular hold-progress indicator in the top-right corner."""
    h, w   = frame.shape[:2]
    cx, cy = w - 42, 42
    radius = 28
    fraction = min(hold_sec / HOLD_DURATION_SEC, 1.0)
    angle    = int(360 * fraction)

    cv2.circle(frame, (cx, cy), radius, (60,60,60), 3)
    if angle > 0:
        col = _GREEN if fraction >= 1.0 else _ORANGE
        cv2.ellipse(frame, (cx, cy), (radius, radius),
                    -90, 0, angle, col, 3, cv2.LINE_AA)

    pct_txt = f"{int(fraction*100)}%"
    cv2.putText(frame, pct_txt, (cx - 14, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, _WHITE, 1)

    hold_lbl = "Hold"
    cv2.putText(frame, hold_lbl, (cx - 13, cy + radius + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, _WHITE, 1)


def _draw_hud(frame, pose: YogaPose, score: float, state: State,
              cam_ok: bool, inst_ok: bool, last_msg: str,
              step_num: int, total_steps: int, hold_sec: float,
              raw_inst_pose: str = "") -> None:
    h, w   = frame.shape[:2]
    panel  = 125
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - panel), (w, h), _DARK, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    # State badge
    state_labels = {
        State.TRACKING:    ("LIVE",         _GREEN),
        State.FROZEN:      ("FROZEN",       _ORANGE),
        State.CATCHING_UP: ("CATCHING UP",  _CYAN),
    }
    slabel, scol = state_labels.get(state, ("", _WHITE))
    cv2.putText(frame, slabel, (w - 110, h - panel + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, scol, 2)

    # Pose name (target — what user should be doing)
    pose_txt = pose.name
    if pose.sanskrit:
        pose_txt += f"  ({pose.sanskrit})"
    cv2.putText(frame, pose_txt, (10, h - panel + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, _CYAN, 2)

    # Instructor's currently detected pose (live, may differ during freeze)
    if raw_inst_pose and raw_inst_pose not in ("Yoga Pose", "unknown", ""):
        detected_lbl = f"On screen: {raw_inst_pose}"
        detected_col = _GREEN if raw_inst_pose == pose.name else _YELLOW
        cv2.putText(frame, detected_lbl, (10, h - panel + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, detected_col, 1)

    # Score
    cv2.putText(frame, f"Match: {score:.0f}%  [{score_label(score)}]",
                (10, h - panel + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, score_colour(score), 2)
    _draw_score_bar(frame, score, h - panel + 52)

    # Hold timer text
    hold_txt = f"Hold: {hold_sec:.1f}s / {HOLD_DURATION_SEC:.0f}s"
    cv2.putText(frame, hold_txt, (10, h - panel + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, _WHITE, 1)

    # Step counter
    if total_steps:
        cv2.putText(frame, f"Step {step_num}/{total_steps}",
                    (w // 2 - 35, h - panel + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, _WHITE, 1)

    # Cam / instructor status
    cv2.putText(frame, f"Cam: {'OK' if cam_ok else 'CHECK'}",
                (w - 120, h - panel + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                _GREEN if cam_ok else _RED, 1)

    # Last spoken message
    if last_msg:
        max_c   = (w - 20) // 9
        display = last_msg[:max_c] + ("…" if len(last_msg) > max_c else "")
        cv2.putText(frame, display, (10, h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    # Title bar
    cv2.putText(frame,
                "Yoga Corrector  [Q=quit | S=score | N=next step | P=pause | R=repeat]",
                (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.40, _WHITE, 1)

    # Frozen banner
    if state == State.FROZEN:
        banner = f"Instructor moved on — hold {pose.name} for {HOLD_DURATION_SEC:.0f}s to continue"
        (bw, _bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 2)
        cv2.putText(frame, banner,
                    (max(6, w // 2 - bw // 2), h - panel - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, _ORANGE, 2)


# ── Main controller ───────────────────────────────────────────────────────────
class YogaCorrector:
    def __init__(self, show_debug: bool = True) -> None:
        logger.info("Initialising Yoga Posture Correction System…")
        self.show_debug = show_debug

        from webcam_capture import WebcamCapture
        from screen_capture import ScreenCapture
        self.webcam       = WebcamCapture()
        self.screen       = ScreenCapture()
        self.est_user     = PoseEstimator()
        self.est_inst     = PoseEstimator()
        self.cam_checker  = CameraChecker()
        self.audio        = AudioFeedback()
        self.state_machine= PoseStateMachine()
        self.session      = SessionTracker()

        self._cmd_queue:   list[str] = []
        self._cmd_lock     = threading.Lock()
        self.voice_cmd     = VoiceInteraction(
            on_command       = self._enqueue_command,
            energy_threshold = config.MIC_ENERGY_THRESHOLD,
        )

        self._stop_event     = threading.Event()
        self._paused         = False
        self._frame_interval = 1.0 / config.PROCESSING_FPS

        # Pose & step state
        self._current_step:   int   = 0
        self._step_spoken_at: float = 0.0

        # Score / feedback state
        self._score:          float = 0.0
        self._cam_ok:         bool  = False
        self._inst_detected:  bool  = False
        self._last_msg:       str   = ""
        self._correction_at:  float = 0.0
        self._well_done_at:   float = 0.0
        self._well_done_streak: int = 0

        self._raw_pose_name: str = ''
        self._inst_pose:    PoseResult = PoseResult(detected=False)
        self._comparison:   dict = {}
        # Track last announced hold to avoid repeated "hold achieved" messages
        self._last_hold_announced: str = ""

        # Instructor pose debounce — require same pose N consecutive frames
        # before treating it as a real change (prevents flicker false-triggering state machine)
        self._inst_pose_candidate: str = ""
        self._inst_pose_streak:    int = 0
        self._INST_POSE_CONFIRM    = 4   # frames (~0.8 s at 5 FPS)
        self._confirmed_inst_pose: str = ""

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        print("\n" + "="*64, flush=True)
        print("  🧘  Yoga Posture Correction System", flush=True)
        print("="*64, flush=True)
        print("  Q = quit    S = score    N = next step", flush=True)
        print("  P = pause   R = repeat", flush=True)
        print("  Voice: 'score' | 'what pose' | 'next step' | 'pause' | 'resume'", flush=True)
        print("="*64 + "\n", flush=True)

        self.voice_cmd.start()
        try:
            while not self._stop_event.is_set():
                t0 = time.monotonic()
                self._handle_commands()
                if not self._paused:
                    self._cycle()
                time.sleep(max(0.0, self._frame_interval - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            pass
        finally:
            if self.show_debug:
                cv2.destroyAllWindows()
            self._end_session()

    def stop(self) -> None:
        self._stop_event.set()

    # ------------------------------------------------------------------ #
    # Voice / keyboard commands                                            #
    # ------------------------------------------------------------------ #
    def _enqueue_command(self, cmd: str) -> None:
        with self._cmd_lock:
            self._cmd_queue.append(cmd)

    def _handle_commands(self) -> None:
        with self._cmd_lock:
            cmds, self._cmd_queue = self._cmd_queue[:], []
        for cmd in cmds:
            self._execute_command(cmd)

    def _execute_command(self, cmd: str) -> None:
        pose = self.state_machine.target_pose
        if cmd == "score":
            self._say(build_score_message(self._score, score_label(self._score)), force=True)
        elif cmd == "what_pose":
            if pose.name not in ("Yoga Pose", "unknown"):
                self._say(t("cmd_what_pose_known", pose_name=pose.name, sanskrit=pose.sanskrit), force=True)
            else:
                self._say(t("cmd_what_pose_unknown"), force=True)
        elif cmd == "next_step":
            self._speak_next_step(force=True)
        elif cmd == "repeat":
            self._say(self._last_msg or t("cmd_repeat_nothing"), force=True)
        elif cmd == "pause":
            self._paused = True
            self._say(t("cmd_pause"), force=True)
        elif cmd == "resume":
            self._paused = False
            self._say(t("cmd_resume"), force=True)
        elif cmd == "start_over":
            self._current_step   = 0
            self._step_spoken_at = 0.0
            self._say(t("cmd_restart", pose_name=pose.name), force=True)
            self._speak_next_step(force=True)
        elif cmd == "help":
            self._say(t("cmd_help"), force=True)

    # ------------------------------------------------------------------ #
    # Main cycle                                                           #
    # ------------------------------------------------------------------ #
    def _cycle(self) -> None:
        user_frame = self.webcam.read()
        inst_frame = self.screen.grab()
        if user_frame is None:
            return

        user_pose = self.est_user.estimate(user_frame)
        cam_check = self.cam_checker.check(user_pose)
        self._cam_ok = cam_check.ok

        if not cam_check.ok:
            self._say(cam_check.message)
            self._render(user_frame, user_pose)
            return

        # Instructor pose
        self._inst_detected = False
        inst_pose = PoseResult(detected=False)
        if inst_frame is not None:
            inst_pose = self.est_inst.estimate(inst_frame)
            self._inst_detected = inst_pose.detected

        if not inst_pose.detected:
            self._render(user_frame, user_pose)
            return

        # Compute angles
        user_angles = compute_joint_angles(user_pose)
        inst_angles = compute_joint_angles(inst_pose)

        # Classify instructor's pose with debounce to prevent flicker
        raw_pose = classify_pose(inst_angles)

        # Only count a pose as "confirmed" once we've seen it N frames in a row.
        # "Yoga Pose" (unknown) is never promoted — it just resets the streak.
        if raw_pose.name == "Yoga Pose":
            self._inst_pose_streak = 0
            # Keep the last confirmed pose so the state machine doesn't see a flicker
        else:
            if raw_pose.name == self._inst_pose_candidate:
                self._inst_pose_streak += 1
            else:
                self._inst_pose_candidate = raw_pose.name
                self._inst_pose_streak    = 1

        # Promote to confirmed after N consecutive identical frames
        if (self._inst_pose_streak >= self._INST_POSE_CONFIRM
                and self._inst_pose_candidate != ""):
            self._confirmed_inst_pose = self._inst_pose_candidate

        # Use confirmed pose for state machine (falls back to raw if never confirmed)
        instructor_pose = (
            POSES.get(
                next((k for k, v in POSES.items()
                      if v.name == self._confirmed_inst_pose), None),
                raw_pose,
            )
            if self._confirmed_inst_pose else raw_pose
        )

        # Compute similarity
        self._score, _ = compute_similarity(user_angles, inst_angles)

        # ── Diagnostic: print instructor angles every 25 frames ───────
        if not hasattr(self, '_diag_counter'):
            self._diag_counter = 0
        self._diag_counter += 1
        if self._diag_counter % 25 == 0:
            angle_str = "  ".join(
                f"{k[:6]}={v:.0f}" for k, v in inst_angles.items() if v is not None
            )
            print(f"[ANGLES] raw={raw_pose.name:<20s}  confirmed={self._confirmed_inst_pose or 'none':<20s}  {angle_str}", flush=True)

        # ── State machine update ──────────────────────────────────────
        event = self.state_machine.update(instructor_pose, self._score)
        target_pose = self.state_machine.target_pose

        # ── React to state machine events ─────────────────────────────
        if event == "new_pose" and target_pose.name not in ("Yoga Pose", "unknown"):
            intro = build_pose_intro(
                target_pose.name, target_pose.sanskrit, target_pose.description
            )
            self._say(intro, force=True)
            self._current_step   = 0
            self._step_spoken_at = 0.0   # trigger first step soon

        elif event == "hold_achieved":
            self._last_hold_announced = target_pose.name
            if self.state_machine.state == State.CATCHING_UP:
                pass   # catching_up event will fire next cycle with message
            else:
                self._say(t("event_hold_achieved", pose_name=target_pose.name), force=True)

        elif event == "catching_up":
            new_target = self.state_machine.target_pose
            self._current_step   = 0
            self._step_spoken_at = 0.0
            self._say(t("event_catching_up", new_pose=new_target.name), force=True)

        elif event == "frozen":
            self._say(t("event_frozen", pose_name=target_pose.name, threshold=int(config.POSE_SIMILARITY_THRESHOLD), hold_sec=int(HOLD_DURATION_SEC)), force=True)

        # ── Always compute comparison (needed for ghost skeleton + feedback) ──
        comparison = compare_angles(user_angles, inst_angles)
        misaligned = sum(1 for v in comparison.values() if v.get("misaligned"))
        self._comparison = comparison
        self._inst_pose  = inst_pose

        # ── Feedback (corrections / steps) ────────────────────────────
        now      = time.monotonic()
        above    = self._score >= config.POSE_SIMILARITY_THRESHOLD

        if above:
            self._well_done_streak += 1
            if (self._well_done_streak >= config.PROCESSING_FPS * 4
                    and now - self._well_done_at > 30.0):
                self._say(build_well_done_message())
                self._well_done_at     = now
                self._well_done_streak = 0
        else:
            self._well_done_streak = 0
            # Step-by-step guide
            if now - self._step_spoken_at > config.STEP_GUIDE_COOLDOWN:
                self._speak_next_step()
            # Joint-level correction (always try even when steps are due,
            # but respect cooldown so they don't overlap)
            if (misaligned >= config.MIN_MISALIGNED_JOINTS
                    and now - self._correction_at > config.FEEDBACK_COOLDOWN_SECONDS):
                msg = build_correction_message(comparison, target_pose.name)
                if msg:
                    self._say(msg)
                    self._correction_at = now

        # ── Terminal log ──────────────────────────────────────────────
        print(
            f"[POSE] {target_pose.name:22s}  "
            f"score={self._score:5.1f}%  "
            f"hold={self.state_machine.hold_seconds:4.1f}s  "
            f"misaligned={misaligned}  "
            f"state={self.state_machine.state.name}",
            flush=True,
        )

        self._raw_pose_name = raw_pose.name
        self._render(user_frame, user_pose)

    # ------------------------------------------------------------------ #
    # Step guidance                                                        #
    # ------------------------------------------------------------------ #
    def _speak_next_step(self, force: bool = False) -> None:
        pose  = self.state_machine.target_pose
        steps = pose.steps
        if not steps:
            return
        idx      = self._current_step % len(steps)
        pose_key = getattr(pose, "key", pose.name.lower().replace(" ","_"))
        msg = build_step_message(
            pose.name, steps[idx], idx + 1, len(steps),
            pose_key=pose_key, step_index=idx
        )
        self._say(msg, force=force)
        self._step_spoken_at = time.monotonic()
        self._current_step  += 1

    # ------------------------------------------------------------------ #
    # Render                                                               #
    # ------------------------------------------------------------------ #
    def _render(self, frame, user_pose: PoseResult) -> None:
        if not self.show_debug:
            return
        vis = frame.copy()

        # 1. Ghost instructor skeleton (behind user skeleton)
        if self._inst_detected and self._inst_pose.detected:
            draw_stencil(
                frame      = vis,
                inst_pose  = self._inst_pose,
                user_pose  = user_pose,
                comparison = self._comparison,
                alpha      = 0.42,
            )

        # 2. User skeleton (on top, solid)
        _draw_skeleton(vis, user_pose, colour=(255, 255, 255))

        # 3. HUD overlays
        _draw_hold_arc(vis, self.state_machine.hold_seconds)
        _draw_hud(
            vis,
            pose         = self.state_machine.target_pose,
            score        = self._score,
            state        = self.state_machine.state,
            cam_ok       = self._cam_ok,
            inst_ok      = self._inst_detected,
            last_msg     = self._last_msg,
            step_num     = min(self._current_step, len(self.state_machine.target_pose.steps)),
            total_steps  = len(self.state_machine.target_pose.steps),
            hold_sec     = self.state_machine.hold_seconds,
            raw_inst_pose= getattr(self, "_raw_pose_name", ""),
        )
        cv2.imshow("Yoga Corrector", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            self.stop()
        elif key == ord("s"): self._enqueue_command("score")
        elif key == ord("n"): self._enqueue_command("next_step")
        elif key == ord("p"): self._enqueue_command("pause" if not self._paused else "resume")
        elif key == ord("r"): self._enqueue_command("repeat")

    # ------------------------------------------------------------------ #
    # Session end                                                          #
    # ------------------------------------------------------------------ #
    def _end_session(self) -> None:
        logger.info("Ending session…")
        self.state_machine.finish_session()
        attempts = self.state_machine.completed_attempts

        # Print report to terminal
        self.session.print_report(attempts)

        # Spoken summary (fully translated)
        summary = t_summary(attempts, self.session._session_start)
        self.audio.speak(summary, force=True)

        time.sleep(len(summary.split()) * 0.45 + 2)  # rough TTS duration estimate
        self.audio.stop()
        self.voice_cmd.stop()
        self.webcam.release()
        self.screen.close()
        self.est_user.close()
        self.est_inst.close()

    # ------------------------------------------------------------------ #
    def _say(self, msg: Optional[str], force: bool = False) -> None:
        if msg:
            self._last_msg = msg
            self.audio.speak(msg, force=force)
