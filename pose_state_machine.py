# pose_state_machine.py — Manages freeze/catch-up logic when instructor
# advances before the user has successfully held the current pose.
#
# States:
#   TRACKING  — user is following the instructor's current pose in real time
#   FROZEN    — instructor moved on; user is still working on the previous pose
#   CATCHING_UP — user just held the frozen pose; advancing to instructor's current pose
#
# Hold criteria (research-backed, per user preference):
#   Score ≥ POSE_SIMILARITY_THRESHOLD (75%) continuously for HOLD_DURATION_SEC (8 s)

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import config
from pose_classifier import YogaPose, POSES

logger = logging.getLogger(__name__)

HOLD_DURATION_SEC = 8.0     # seconds score must stay above threshold


class State(Enum):
    TRACKING     = auto()   # following instructor live
    FROZEN       = auto()   # locked on previous pose; instructor is ahead
    CATCHING_UP  = auto()   # held frozen pose; transitioning to instructor pose


@dataclass
class PoseAttempt:
    """Record of a single pose attempt within the session."""
    pose_name:   str
    sanskrit:    str
    started_at:  float          = field(default_factory=time.monotonic)
    held:        bool           = False
    best_score:  float          = 0.0
    hold_started_at: float      = 0.0   # when the current hold streak began
    hold_achieved_at: float     = 0.0   # when 8-second hold was first completed


class PoseStateMachine:
    """
    Tracks whether the user has successfully held a pose, and manages
    freezing the target pose when the instructor advances early.

    Usage
    -----
    Each processing cycle, call:
        event = machine.update(instructor_pose, user_score)
    Then act on the returned PoseEvent.
    """

    def __init__(self) -> None:
        self._state:            State       = State.TRACKING
        self._target_pose:      YogaPose    = POSES["unknown"]
        self._instructor_pose:  YogaPose    = POSES["unknown"]
        self._attempt:          Optional[PoseAttempt] = None

        # Hold tracking
        self._hold_start:       float = 0.0
        self._holding:          bool  = False
        self._hold_seconds:     float = 0.0   # running tally for HUD display

        # Completed attempts (for session summary)
        self._attempts: list[PoseAttempt] = []

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def target_pose(self) -> YogaPose:
        """The pose the user should currently be working on."""
        return self._target_pose

    @property
    def state(self) -> State:
        return self._state

    @property
    def hold_seconds(self) -> float:
        """How many continuous seconds the user has been above threshold."""
        return self._hold_seconds

    @property
    def is_frozen(self) -> bool:
        return self._state == State.FROZEN

    @property
    def completed_attempts(self) -> list[PoseAttempt]:
        return list(self._attempts)

    # ------------------------------------------------------------------ #
    # Core update                                                          #
    # ------------------------------------------------------------------ #

    def update(
        self,
        instructor_pose: YogaPose,
        user_score:      float,
    ) -> Optional[str]:
        """
        Call every processing cycle.

        Parameters
        ----------
        instructor_pose : YogaPose
            Pose currently detected on screen.
        user_score : float
            Current pose similarity score (0–100).

        Returns
        -------
        Optional[str]
            An event string (or None) that the orchestrator should act on:
              "new_pose"         — instructor changed pose (or first pose seen)
              "hold_achieved"    — user successfully held the frozen/target pose
              "pose_missed"      — (not used in freeze mode, kept for future)
              "catching_up"      — target just advanced to instructor's current pose
        """
        now   = time.monotonic()
        above = user_score >= config.POSE_SIMILARITY_THRESHOLD

        # Update hold timer
        if above:
            if not self._holding:
                self._holding    = True
                self._hold_start = now
            self._hold_seconds = now - self._hold_start
        else:
            self._holding      = False
            self._hold_seconds = 0.0

        # Update best score for current attempt
        if self._attempt:
            self._attempt.best_score = max(self._attempt.best_score, user_score)

        # ── State machine ─────────────────────────────────────────────
        if self._state == State.TRACKING:
            return self._handle_tracking(instructor_pose, now)

        elif self._state == State.FROZEN:
            return self._handle_frozen(instructor_pose, now)

        elif self._state == State.CATCHING_UP:
            return self._handle_catching_up(instructor_pose, now)

        return None

    # ------------------------------------------------------------------ #
    # State handlers                                                       #
    # ------------------------------------------------------------------ #

    def _handle_tracking(self, instructor_pose: YogaPose, now: float) -> Optional[str]:
        """TRACKING: following instructor in real time."""

        # Ignore unclassified / unknown frames — never treat as a real pose change
        if instructor_pose.name in ("Yoga Pose", "unknown", ""):
            return None

        # Instructor changed pose
        if instructor_pose.name != self._instructor_pose.name:
            old_inst = self._instructor_pose
            self._instructor_pose = instructor_pose

            # First pose of the session — just start tracking it
            if old_inst.name in ("Yoga Pose", "unknown", ""):
                self._start_attempt(instructor_pose, now)
                self._target_pose = instructor_pose
                return "new_pose"

            # Instructor moved on — check if user already held it
            if self._attempt and self._attempt.held:
                # Already held — follow instructor immediately
                self._finish_attempt()
                self._start_attempt(instructor_pose, now)
                self._target_pose = instructor_pose
                return "new_pose"
            else:
                # NOT held yet → FREEZE on current target
                logger.info(
                    "Instructor advanced to %s but user hasn't held %s — freezing.",
                    instructor_pose.name, self._target_pose.name,
                )
                self._state = State.FROZEN
                return "frozen"

        # Same pose — check for hold completion
        if self._holding and self._hold_seconds >= HOLD_DURATION_SEC:
            if self._attempt and not self._attempt.held:
                self._attempt.held            = True
                self._attempt.hold_achieved_at= now
                logger.info("Hold achieved: %s", self._target_pose.name)
                return "hold_achieved"

        return None

    def _handle_frozen(self, instructor_pose: YogaPose, now: float) -> Optional[str]:
        """FROZEN: waiting for user to hold the previous pose."""
        # Ignore unknown frames
        if instructor_pose.name in ("Yoga Pose", "unknown", ""):
            return None
        # Keep tracking instructor changes (they might move several poses ahead)
        if instructor_pose.name != self._instructor_pose.name:
            self._instructor_pose = instructor_pose
            logger.info("Instructor now on %s (still frozen).", instructor_pose.name)

        # Check for hold completion on the frozen pose
        if self._holding and self._hold_seconds >= HOLD_DURATION_SEC:
            if self._attempt and not self._attempt.held:
                self._attempt.held            = True
                self._attempt.hold_achieved_at= now
                logger.info(
                    "Hold achieved on frozen pose %s — catching up to %s.",
                    self._target_pose.name, self._instructor_pose.name,
                )
                self._state = State.CATCHING_UP
                return "hold_achieved"

        return None

    def _handle_catching_up(self, instructor_pose: YogaPose, now: float) -> Optional[str]:
        """CATCHING_UP: user just held frozen pose; advance to instructor's pose."""
        # Update instructor tracking
        if instructor_pose.name != self._instructor_pose.name:
            self._instructor_pose = instructor_pose

        # Immediately advance to the instructor's current pose
        self._finish_attempt()
        self._start_attempt(self._instructor_pose, now)
        self._target_pose = self._instructor_pose
        self._state       = State.TRACKING
        self._holding     = False
        self._hold_seconds= 0.0
        return "catching_up"

    # ------------------------------------------------------------------ #
    # Attempt management                                                   #
    # ------------------------------------------------------------------ #

    def _start_attempt(self, pose: YogaPose, now: float) -> None:
        self._attempt = PoseAttempt(
            pose_name  = pose.name,
            sanskrit   = pose.sanskrit,
            started_at = now,
        )
        self._holding      = False
        self._hold_seconds = 0.0
        self._hold_start   = 0.0

    def _finish_attempt(self) -> None:
        if self._attempt:
            self._attempts.append(self._attempt)
            self._attempt = None

    def finish_session(self) -> None:
        """Call at end of session to close any open attempt."""
        self._finish_attempt()
