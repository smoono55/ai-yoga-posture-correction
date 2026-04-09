# hold_manager.py — Manages pose hold tracking, grace periods, and freeze logic
#
# Handles the scenario where the on-screen instructor moves to the next pose
# before the user has successfully achieved and held the current one.
#
# State machine:
#
#   FOLLOWING  ──instructor changes pose──►  GRACE_PERIOD
#      ▲                                          │
#      │                                    user reaches score
#      │                                    threshold in time
#      │                                          │
#      │                                          ▼
#      │                                     HOLDING
#      │                                          │
#      │                                    hold long enough
#      │                                          │
#      ◄──────────────────────────────────────────┘  (success)
#      ◄── grace period expires without success ──┘  (timeout, gentle)
#
# Reference angles are FROZEN when the instructor changes pose so the system
# keeps comparing against what the instructor *was* doing, not whatever they
# moved on to.

from __future__ import annotations
import time
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import config

try:
    from voice_strings import t as _t
except ImportError:
    def _t(k, **kw): return k

logger = logging.getLogger(__name__)


class HoldState(Enum):
    FOLLOWING    = auto()   # actively tracking instructor live
    GRACE_PERIOD = auto()   # instructor moved on; user given extra time
    HOLDING      = auto()   # user reached threshold; counting hold duration
    SUCCESS      = auto()   # hold completed — briefly shown before reset


@dataclass
class HoldStatus:
    """Snapshot of current hold state for UI / audio feedback."""
    state:            HoldState
    pose_name:        str
    score:            float
    hold_elapsed:     float   # seconds held above threshold (HOLDING state)
    hold_required:    float   # seconds needed for success
    grace_elapsed:    float   # seconds since grace period started
    grace_allowed:    float   # total grace period seconds
    frozen:           bool    # True when using frozen reference angles


@dataclass
class HoldManager:
    """
    Tracks whether the user has achieved and held a yoga pose.

    Parameters
    ----------
    hold_required_seconds : float
        How long the user must hold the pose above the similarity threshold.
        Default: 5 breaths ≈ 30 s (research-backed; Nagarkar et al. 2022
        recommends 5 breath cycles as the minimum hold for a beneficial pose).
    grace_period_seconds : float
        How long to wait after the instructor moves on before giving up on
        the current pose and releasing the frozen reference.
    """

    hold_required_seconds: float = 30.0
    grace_period_seconds:  float = 45.0

    # Internal state
    _state:             HoldState               = field(default=HoldState.FOLLOWING, init=False)
    _pose_name:         str                     = field(default="",   init=False)
    _frozen_angles:     Optional[dict]          = field(default=None, init=False)
    _hold_started_at:   float                   = field(default=0.0,  init=False)
    _grace_started_at:  float                   = field(default=0.0,  init=False)
    _score:             float                   = field(default=0.0,  init=False)
    _success_shown_at:  float                   = field(default=0.0,  init=False)
    _SUCCESS_DISPLAY_S: float                   = field(default=3.0,  init=False)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def on_instructor_pose_changed(
        self,
        new_pose_name:    str,
        last_inst_angles: dict,
    ) -> Optional[str]:
        """
        Called when the instructor's detected pose name changes.

        Returns an optional spoken message to announce to the user, or None.
        """
        if self._state == HoldState.FOLLOWING:
            # Instructor moved on — freeze reference and start grace period
            self._frozen_angles    = last_inst_angles.copy()
            self._grace_started_at = time.monotonic()
            self._state            = HoldState.GRACE_PERIOD
            self._pose_name        = self._pose_name   # keep old name

            remaining = int(self.grace_period_seconds)
            msg = _t("hold_grace_period", pose_name=self._pose_name, breaths=remaining // 5)
            logger.info("Grace period started for '%s'.", self._pose_name)
            return msg

        elif self._state in (HoldState.GRACE_PERIOD, HoldState.HOLDING):
            # Instructor changed again before user succeeded — release gently
            msg = self._release(
_t("hold_release", pose_name=self._pose_name)
            )
            self._set_following(new_pose_name)
            return msg

        # SUCCESS or fresh start
        self._set_following(new_pose_name)
        return None

    def on_pose_same(
        self,
        current_pose_name: str,
        inst_angles:       dict,
    ) -> None:
        """
        Called every cycle when the instructor's pose has NOT changed.
        Updates the frozen reference if we're still FOLLOWING.
        """
        if self._state == HoldState.FOLLOWING:
            self._pose_name     = current_pose_name
            self._frozen_angles = inst_angles  # live reference

    def update_user_score(self, score: float) -> Optional[str]:
        """
        Called every cycle with the user's current similarity score.
        Drives state transitions; returns a spoken message or None.
        """
        self._score = score
        now         = time.monotonic()

        if self._state == HoldState.FOLLOWING:
            return None  # normal feedback handled by orchestrator

        if self._state == HoldState.GRACE_PERIOD:
            grace_elapsed = now - self._grace_started_at
            if score >= config.POSE_SIMILARITY_THRESHOLD:
                # User reached the pose! Start hold countdown.
                self._state          = HoldState.HOLDING
                self._hold_started_at= now
                secs = int(self.hold_required_seconds)
                return (
                    f"Great — you're in the pose! "
                    f"Now hold it steady for {secs} seconds. Keep breathing."
                )
            elif grace_elapsed >= self.grace_period_seconds:
                # Grace period expired
                msg = self._release(
                    f"That's okay — {self._pose_name} is tricky. "
                    f"Let's keep practising and we'll come back to it."
                )
                self._set_following(self._pose_name)
                return msg
            else:
                return None  # still in grace; corrections handled elsewhere

        if self._state == HoldState.HOLDING:
            hold_elapsed = now - self._hold_started_at
            if score < config.POSE_SIMILARITY_THRESHOLD - 5:
                # Dropped out of pose — restart hold countdown
                self._state = HoldState.GRACE_PERIOD
                self._grace_started_at = now   # reset grace timer
                return (
                    "You slipped out of the pose — that's okay! "
                    "Get back into position and hold again."
                )
            elif hold_elapsed >= self.hold_required_seconds:
                # 🎉 Success!
                self._state           = HoldState.SUCCESS
                self._success_shown_at= now
                return (
                    f"Wonderful! You held {self._pose_name} beautifully. "
                    f"That's a perfect hold — well done!"
                )
            else:
                # Mid-hold — occasional encouragement
                remaining = int(self.hold_required_seconds - hold_elapsed)
                if remaining % 10 == 0 and remaining > 0:
                    return f"Keep holding — {remaining} more seconds. You're doing great!"
            return None

        if self._state == HoldState.SUCCESS:
            if now - self._success_shown_at > self._SUCCESS_DISPLAY_S:
                self._set_following(self._pose_name)
            return None

        return None

    @property
    def reference_angles(self) -> Optional[dict]:
        """
        The angles to use for comparison.
        Returns frozen angles during GRACE_PERIOD / HOLDING / SUCCESS,
        otherwise None (orchestrator uses live instructor angles).
        """
        if self._state in (HoldState.GRACE_PERIOD, HoldState.HOLDING, HoldState.SUCCESS):
            return self._frozen_angles
        return None

    def status(self) -> HoldStatus:
        now           = time.monotonic()
        hold_elapsed  = (now - self._hold_started_at)  if self._state == HoldState.HOLDING    else 0.0
        grace_elapsed = (now - self._grace_started_at) if self._state == HoldState.GRACE_PERIOD else 0.0
        return HoldStatus(
            state         = self._state,
            pose_name     = self._pose_name,
            score         = self._score,
            hold_elapsed  = hold_elapsed,
            hold_required = self.hold_required_seconds,
            grace_elapsed = grace_elapsed,
            grace_allowed = self.grace_period_seconds,
            frozen        = self._state in (
                HoldState.GRACE_PERIOD, HoldState.HOLDING, HoldState.SUCCESS
            ),
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _set_following(self, pose_name: str) -> None:
        self._state          = HoldState.FOLLOWING
        self._pose_name      = pose_name
        self._frozen_angles  = None
        self._hold_started_at= 0.0

    def _release(self, message: str) -> str:
        logger.info("Hold released for '%s'.", self._pose_name)
        return message
