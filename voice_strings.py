"""
voice_strings.py  —  Central translation lookup for all spoken text

Every file that produces voice output imports t() from here.

Usage
─────
  from voice_strings import t

  t("cmd_pause")
  t("event_hold_achieved", pose_name="Warrior II")
  t("correction_3")
  t("step_warrior_ii_0")
"""

from __future__ import annotations
import json
import os
import random
from pathlib import Path
from typing import Any

_BASE = Path(__file__).parent
_LANG_DIR = _BASE / "languages"
_SETTING  = _BASE / "language_setting.json"

# ── State (no cache — always read fresh so language changes take effect) ──────
_current_lang: str = "en"
_strings:      dict[str, str] = {}
_last_read_lang: str = ""   # track which lang is currently loaded


def _load() -> None:
    """Read language_setting.json and load translations. Called on every access."""
    global _current_lang, _strings, _last_read_lang

    # Read current language from file (fast — tiny JSON)
    try:
        _current_lang = json.loads(_SETTING.read_text(encoding="utf-8")).get("lang", "en")
    except Exception:
        _current_lang = "en"

    # Only reload translation strings if the language actually changed
    if _current_lang == _last_read_lang:
        return

    _strings = {}
    if _current_lang != "en":
        path = _LANG_DIR / f"translations_{_current_lang}.json"
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                _strings = data.get("strings", {})
            except Exception as e:
                print(f"[voice_strings] Warning: could not load {path}: {e}")

    _last_read_lang = _current_lang


def reload() -> None:
    """Force reload (resets language tracking)."""
    global _last_read_lang
    _last_read_lang = ""
    _load()


def current_lang() -> str:
    _load()
    return _current_lang


def t(key: str, **kwargs: Any) -> str:
    """
    Return translated string for key, with optional format substitutions.

    Falls back to English if translation missing.
    Falls back to key itself if English also missing.
    """
    _load()
    from generate_translations import STRINGS as _ENGLISH

    # Get translation or English fallback
    text = _strings.get(key) or _ENGLISH.get(key) or key

    # Apply format substitutions if any
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, ValueError):
            pass  # return unformatted if template mismatch

    return text


def t_pick(keys: list[str], **kwargs: Any) -> str:
    """Pick a random key from the list and return translated text."""
    return t(random.choice(keys), **kwargs)


# ── Grouped helpers ────────────────────────────────────────────────────────────

def t_score(score: float, label: str) -> str:
    """Return a random translated score message for the given label."""
    key_map = {
        "excellent":  ["score_excellent_1", "score_excellent_2", "score_excellent_3"],
        "good":       ["score_good_1",       "score_good_2",      "score_good_3"],
        "fair":       ["score_fair_1",        "score_fair_2",      "score_fair_3"],
        "needs work": ["score_needs_work_1",  "score_needs_work_2","score_needs_work_3"],
    }
    keys = key_map.get(label, key_map["fair"])
    return t(random.choice(keys), score=int(score))


def t_well_done() -> str:
    keys = [f"well_done_{i}" for i in range(1, 9)]
    return t(random.choice(keys))


def t_encouragement(avg_diff: float) -> str:
    r = random.random()
    if avg_diff < 15 and r < 0.70:
        return t_pick([f"enc_small_{i}" for i in range(1, 6)])
    elif avg_diff < 30 and r < 0.40:
        return t_pick([f"enc_medium_{i}" for i in range(1, 4)])
    elif avg_diff >= 30 and r < 0.25:
        return t_pick([f"enc_large_{i}" for i in range(1, 4)])
    return ""


def t_correction(correction_id: int, english_text: str) -> str:
    """Return translated correction text for the given ID."""
    _load()
    return _strings.get(f"correction_{correction_id}") or english_text


def t_pose_step(pose_key: str, step_index: int, english_text: str) -> str:
    """Return translated pose step text."""
    _load()
    return _strings.get(f"step_{pose_key}_{step_index}") or english_text


def t_pose_intro(pose_key: str, pose_name: str, sanskrit: str, description: str) -> str:
    """Build a full translated pose introduction."""
    prefixes = [
        t("pose_new_intro_prefix2", pose_name=pose_name),
        t("pose_new_intro_prefix3", pose_name=pose_name),
        t("pose_new_intro_prefix4", pose_name=pose_name),
    ]
    opener = random.choice(prefixes)
    if sanskrit:
        return f"{opener}, {sanskrit}. {description}"
    return f"{opener}. {description}"


def t_summary(attempts: list, session_start_monotonic: float) -> str:
    """Build translated end-of-session spoken summary."""
    import time

    if not attempts:
        return t("summary_no_attempts")

    # De-duplicate by pose name
    best: dict[str, Any] = {}
    for a in attempts:
        if a.pose_name not in best or a.best_score > best[a.pose_name].best_score:
            best[a.pose_name] = a

    held   = [a for a in best.values() if a.held]
    missed = [a for a in best.values() if not a.held]

    total       = len(best)
    held_count  = len(held)
    session_min = int((time.monotonic() - session_start_monotonic) / 60)

    parts: list[str] = []

    # Opening
    if session_min >= 1:
        minute_word = t("summary_minute") if session_min == 1 else t("summary_minutes")
        parts.append(t("summary_opening_with_time", minutes=session_min, minute_word=minute_word))
    else:
        parts.append(t("summary_opening_short"))

    # Held poses
    if held:
        names = _natural_list([a.pose_name for a in held])
        if held_count == total:
            parts.append(t("summary_all_held", names=names))
        else:
            parts.append(t("summary_some_held", held=held_count, total=total, names=names))

    # Missed poses
    if missed:
        missed_names = _natural_list([a.pose_name for a in missed])
        scores_str   = ", ".join(
            f"{a.pose_name} at {int(a.best_score)} percent" for a in missed
        )
        key = "summary_missed_singular" if len(missed) == 1 else "summary_missed_plural"
        parts.append(t(key, names=missed_names, scores=scores_str))

    # Closer
    parts.append(t_pick([f"summary_closer_{i}" for i in range(1, 5)]))

    return " ".join(parts)


def _natural_list(items: list[str]) -> str:
    if not items:      return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
