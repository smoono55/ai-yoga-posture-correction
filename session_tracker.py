# session_tracker.py — Records pose attempts and generates end-of-session summary
#
# Tracks:
#   - Which poses were attempted
#   - Which were successfully held (score ≥ 75% for ≥ 8 s)
#   - Best score per pose
#   - Time spent on each pose
#
# At session end: builds a spoken summary + prints a text report.

from __future__ import annotations
import time
import logging
from typing import TYPE_CHECKING

try:
    from voice_strings import t_summary as _t_summary
except ImportError:
    _t_summary = None

if TYPE_CHECKING:
    from pose_state_machine import PoseAttempt

logger = logging.getLogger(__name__)


class SessionTracker:
    """Consumes completed PoseAttempt records and builds session summary."""

    def __init__(self) -> None:
        self._session_start = time.monotonic()

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def build_spoken_summary(self, attempts: list["PoseAttempt"]) -> str:
        """Build spoken end-of-session summary (fully translated via voice_strings)."""
        if _t_summary is not None:
            return _t_summary(attempts, self._session_start)
        # Fallback to English if voice_strings unavailable
        if not attempts:
            return (
                "Your session is complete. Great effort anyway!"
            )
        import random
        held = [a for a in attempts if a.held]
        return f"Session complete. You held {len(held)} of {len(attempts)} poses. Well done!"


    def print_report(self, attempts: list["PoseAttempt"]) -> None:
        """Print a formatted text report to the terminal."""
        session_sec = time.monotonic() - self._session_start
        print("\n" + "="*62, flush=True)
        print("  📋  SESSION REPORT", flush=True)
        print("="*62, flush=True)
        print(f"  Duration : {int(session_sec // 60)}m {int(session_sec % 60)}s", flush=True)
        print(f"  Poses    : {len(attempts)} attempt(s)", flush=True)
        print("-"*62, flush=True)

        if not attempts:
            print("  No pose attempts recorded.", flush=True)
        else:
            # De-dup
            best: dict[str, "PoseAttempt"] = {}
            for a in attempts:
                if a.pose_name not in best or a.best_score > best[a.pose_name].best_score:
                    best[a.pose_name] = a

            print(f"  {'POSE':<24} {'HELD':>5}  {'BEST SCORE':>10}  {'TIME(s)':>8}", flush=True)
            print("-"*62, flush=True)
            for a in best.values():
                duration = (
                    (a.hold_achieved_at or time.monotonic()) - a.started_at
                )
                held_str  = "✅ YES" if a.held else "❌  NO"
                print(
                    f"  {a.pose_name:<24} {held_str:>5}  "
                    f"{a.best_score:>9.1f}%  {int(duration):>7}s",
                    flush=True,
                )

        print("="*62 + "\n", flush=True)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _natural_list(items: list[str]) -> str:
        """['A','B','C'] → 'A, B, and C'"""
        if not items:
            return ""
        if len(items) == 1:
            return items[0]
        if len(items) == 2:
            return f"{items[0]} and {items[1]}"
        return ", ".join(items[:-1]) + f", and {items[-1]}"
