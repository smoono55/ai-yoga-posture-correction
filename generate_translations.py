#!/usr/bin/env python3
"""
generate_translations.py  —  Translate ALL yoga system strings to target language(s)

Translates every spoken string in the system:
  • 141 correction bank texts
  • All orchestrator voice strings (state changes, commands, events)
  • Session summary strings
  • Camera checker messages
  • Score / well-done / encouragement messages
  • Pose names and step instructions

Saves to:  languages/translations_{lang}.json

Usage
─────
  # Pre-built files for hi/ta/te/kn are included — no internet needed.
  # For other languages, install deep-translator first:
  #   pip install deep-translator

  python generate_translations.py --lang hi          # Hindi
  python generate_translations.py --lang ta          # Tamil
  python generate_translations.py --lang te          # Telugu
  python generate_translations.py --lang kn          # Kannada
  python generate_translations.py --lang all         # all 15 languages
  python generate_translations.py --lang hi --force  # re-translate even if file exists
  python generate_translations.py --list             # show progress
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

BASE = Path(__file__).parent
OUT  = BASE / "languages"

LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese (Simplified)",
    "ar": "Arabic",
    "pt": "Portuguese",
    "ru": "Russian",
    "ko": "Korean",
    "it": "Italian",
}

# ══════════════════════════════════════════════════════════════════════════════
#  MASTER STRING BANK  — every spoken string in the system, keyed
# ══════════════════════════════════════════════════════════════════════════════

STRINGS: dict[str, str] = {

    # ── Startup ───────────────────────────────────────────────────────────────
    "startup_greeting":
        "Yoga posture correction system is starting.",

    # ── Voice command responses (orchestrator.py) ─────────────────────────────
    "cmd_score":              # build_score_message fills {score} and {label} separately
        "{score} percent — {label}.",
    "cmd_what_pose_known":
        "You're working on {pose_name}, or {sanskrit}.",
    "cmd_what_pose_unknown":
        "I haven't identified the pose yet — give me a moment.",
    "cmd_repeat_nothing":
        "Nothing to repeat yet.",
    "cmd_pause":
        "Corrections paused. Say resume whenever you're ready.",
    "cmd_resume":
        "Resuming. Let's go!",
    "cmd_restart":
        "Restarting the guide for {pose_name} from the top.",
    "cmd_help":
        "You can say: score, what pose, next step, repeat, pause, resume, or restart.",

    # ── Camera / alignment checks (camera_checker.py) ─────────────────────────
    "cam_too_close":
        "Please step back — you're too close to the camera.",
    "cam_too_far":
        "Please move closer — you're too far from the camera.",
    "cam_not_centred":
        "Move toward the centre of the frame.",
    "cam_sideways":
        "Please face the camera directly — turn to face forward.",
    "cam_cropped_left":
        "Please shift a little to the right so your full body is visible.",
    "cam_cropped_right":
        "Please shift a little to the left so your full body is visible.",
    "cam_cropped_top":
        "Your head is cut off — lower the camera or move down slightly.",
    "cam_cropped_bottom":
        "Your feet are cut off — raise the camera or move up slightly.",
    "cam_ok":
        "Camera alignment looks good.",
    "cam_low_visibility":
        "Some body parts aren't visible — check your lighting and make sure your full body is in frame.",

    # ── Pose events (orchestrator.py) ─────────────────────────────────────────
    "pose_new_intro_prefix":
        "The instructor has moved into",
    "pose_new_intro_prefix2":
        "We're going into {pose_name} now",
    "pose_new_intro_prefix3":
        "Coming up is {pose_name}",
    "pose_new_intro_prefix4":
        "Next pose is {pose_name}",

    "event_hold_achieved":
        "Excellent! You've held {pose_name} successfully. Keep breathing and stay in the pose.",
    "event_catching_up":
        "Nicely done holding that pose! Now let's catch up — the instructor is in {new_pose}. I'll guide you in.",
    "event_frozen":
        "The instructor has moved on, but that's okay — let's focus on holding {pose_name} first. "
        "Get your score above {threshold} percent and hold it for {hold_sec} seconds.",

    # ── Step guide ────────────────────────────────────────────────────────────
    "step_prefix_numbered":
        "Step {n} of {total}:",
    "step_prefix_next":
        "Next,",
    "step_prefix_now":
        "Now,",

    # ── Score messages ────────────────────────────────────────────────────────
    "score_excellent_1":  "You're at {score} percent — excellent.",
    "score_excellent_2":  "{score} percent match. Beautiful.",
    "score_excellent_3":  "Ninety plus — your body is right where it needs to be.",
    "score_good_1":       "You're at {score} percent — good alignment, small adjustments left.",
    "score_good_2":       "{score} percent. Getting really close.",
    "score_good_3":       "Looking good at {score} — just a little more to refine.",
    "score_fair_1":       "{score} percent — you're making progress.",
    "score_fair_2":       "At {score} percent — keep working through the corrections.",
    "score_fair_3":       "{score} percent. Stay focused and keep breathing.",
    "score_needs_work_1": "{score} percent right now — let's work through this together.",
    "score_needs_work_2": "You're at {score} percent. No rush — keep adjusting.",
    "score_needs_work_3": "{score} percent — take it one joint at a time.",

    # ── Well done messages ────────────────────────────────────────────────────
    "well_done_1": "That's it — hold that and breathe.",
    "well_done_2": "Yes! That's the shape. Stay with it.",
    "well_done_3": "Beautiful alignment. Keep breathing.",
    "well_done_4": "Perfect — you've found it. Don't move.",
    "well_done_5": "That's exactly right. Nice work.",
    "well_done_6": "Great shape. Lock it in.",
    "well_done_7": "You're matching the instructor well. Maintain it.",
    "well_done_8": "Spot on. Feel that alignment.",

    # ── Encouragement ─────────────────────────────────────────────────────────
    "enc_small_1": "Almost there.",
    "enc_small_2": "Really close now.",
    "enc_small_3": "Nearly perfect.",
    "enc_small_4": "Tiny adjustment.",
    "enc_small_5": "You're so close.",
    "enc_medium_1": "You're doing well — keep adjusting.",
    "enc_medium_2": "Good effort — stay with it.",
    "enc_medium_3": "Nice work — keep breathing through it.",
    "enc_large_1": "Take your time — this one takes practice.",
    "enc_large_2": "Breathe and keep working through it.",
    "enc_large_3": "Every body is different — do what you can.",

    # ── Session summary ───────────────────────────────────────────────────────
    "summary_no_attempts":
        "Your session is complete. I didn't record any full pose attempts this time "
        "— that might mean the session was very short or the instructor wasn't detected. "
        "Great effort anyway!",
    "summary_opening_with_time":
        "Great session! You practised for about {minutes} {minute_word}.",
    "summary_opening_short":
        "Great effort on your session!",
    "summary_minute":   "minute",
    "summary_minutes":  "minutes",
    "summary_all_held":
        "You successfully held every pose — {names}. That's a fantastic result!",
    "summary_some_held":
        "You successfully held {held} out of {total} poses: {names}.",
    "summary_missed_singular":
        "The pose to keep working on is {names}. Your best score was: {scores}. "
        "You'll get there with a little more practice!",
    "summary_missed_plural":
        "The poses to keep working on are {names}. Your best scores were: {scores}. "
        "You'll get there with a little more practice!",
    "summary_closer_1": "Remember, every practice counts — see you on the mat again soon!",
    "summary_closer_2": "Consistency is the key to progress. Keep showing up!",
    "summary_closer_3": "Each session makes you stronger. Well done today!",
    "summary_closer_4": "Take a moment to breathe and celebrate your effort today.",

    # ── Pose names (spoken, not classified) ───────────────────────────────────
    "pose_mountain":       "Mountain Pose",
    "pose_forward_fold":   "Forward Fold",
    "pose_halfway_lift":   "Halfway Lift",
    "pose_warrior_i":      "Warrior One",
    "pose_warrior_ii":     "Warrior Two",
    "pose_warrior_iii":    "Warrior Three",
    "pose_chair":          "Chair Pose",
    "pose_tree":           "Tree Pose",
    "pose_triangle":       "Triangle Pose",
    "pose_downward_dog":   "Downward Facing Dog",
    "pose_plank":          "Plank Pose",
    "pose_low_lunge":      "Low Lunge",
    "pose_high_lunge":     "High Lunge",
    "pose_cobra":          "Cobra Pose",
    "pose_upward_dog":     "Upward Facing Dog",
    "pose_child":          "Child's Pose",
    "pose_seated_forward": "Seated Forward Fold",
    "pose_cat_cow":        "Cat Cow",
    "pose_unknown":        "Unknown Pose",

    # ── Camera checker ────────────────────────────────────────────────────────────
    "cam_cant_see":
        "I can't see you. Please stand in front of the camera in a well-lit area.",
    "cam_parts_missing":
        "Parts of your body aren't visible. Please step back so your full body is in frame.",

    # ── Hold manager ──────────────────────────────────────────────────────────────
    "hold_grace_period":
        "The instructor has moved on, but don't worry — take your time to get into {pose_name}. "
        "You have about {breaths} more breath cycles.",
    "hold_release":
        "No problem — let's move on with the instructor. We'll practise {pose_name} again next time.",
}

# ── Correction bank strings (from train_correction_ai.py) ────────────────────
def _get_corrections() -> dict[str, str]:
    sys.path.insert(0, str(BASE))
    try:
        from train_correction_ai import CORRECTION_BANK
        return {f"correction_{c['id']}": c["text"] for c in CORRECTION_BANK}
    except Exception as e:
        print(f"  Warning: Could not import CORRECTION_BANK: {e}")
        return {}

# ── Pose steps and descriptions (from pose_classifier.py) ─────────────────────────────────────
def _get_pose_steps_and_descriptions() -> dict[str, str]:
    sys.path.insert(0, str(BASE))
    try:
        from pose_classifier import POSES
        result = {}
        for key, pose in POSES.items():
            if key == "unknown":
                continue
            result[f"desc_{key}"] = pose.description
            for i, step in enumerate(pose.steps):
                result[f"step_{key}_{i}"] = step
        return result
    except Exception as e:
        print(f"  Warning: Could not import pose steps or descriptions: {e}")
        return {}

# ── Translator ────────────────────────────────────────────────────────────────
def _translate_all(texts: dict[str, str], target_lang: str) -> dict[str, str]:
  
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("\nERROR: pip install deep-translator\n")
        sys.exit(1)

    result = {}
    keys   = list(texts.keys())
    values = list(texts.values())
    total  = len(keys)

    print(f"  Translating {total} strings to {LANGUAGES[target_lang]}...")
    print(f"  (translating one by one — takes 2-4 minutes, saved permanently after)")

    translator = GoogleTranslator(source="auto", target=target_lang)
    failed = 0

    for i, (key, text) in enumerate(zip(keys, values)):
        # Skip empty or placeholder-only strings
        if not text.strip():
            result[key] = text
            continue

        translated = None
        for attempt in range(4):
            try:
                translated = translator.translate(text)
                if translated:
                    break
            except Exception as e:
                if attempt == 3:
                    print(f"\n  Warning: failed '{key[:30]}': {e}")
                    failed += 1
                else:
                    time.sleep(1 + attempt)

        result[key] = translated if translated else text  # fallback to English

        # Progress every 10 strings
        if (i + 1) % 10 == 0 or i == total - 1:
            pct = int((i+1) / total * 100)
            print(f"  [{i+1:4d}/{total}] {pct:3d}%  (failed: {failed})", end="\r")

        # Small delay between requests to avoid rate limiting
        time.sleep(0.15)

    print(f"  [{total:4d}/{total}] 100% — done  (failed: {failed})          ")
    return result


def generate(lang: str, force: bool = False) -> None:
    if lang == "en":
        print("English needs no translation.")
        return

    if lang not in LANGUAGES:
        print(f"Unknown language: {lang}. Use --list to see options.")
        sys.exit(1)

    OUT.mkdir(exist_ok=True)
    out_path = OUT / f"translations_{lang}.json"

    if out_path.exists() and not force:
        print(f"  ✓ {lang} already done at {out_path}  (use --force to redo)")
        return

    print(f"\n=== Generating {LANGUAGES[lang]} ({lang}) translations ===")

    # Collect all strings
    all_strings = dict(STRINGS)
    all_strings.update(_get_corrections())
    all_strings.update(_get_pose_steps_and_descriptions())

    print(f"  Total strings: {len(all_strings)}")

    translated = _translate_all(all_strings, lang)

    payload = {
        "lang":        lang,
        "lang_name":   LANGUAGES[lang],
        "total":       len(translated),
        "strings":     translated,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ Saved {len(translated)} strings to {out_path}")


def show_list() -> None:
    OUT.mkdir(exist_ok=True)
    print(f"\n{'Lang':<6} {'Name':<22} {'Status'}")
    print("─" * 44)
    for code, name in LANGUAGES.items():
        if code == "en":
            status = "✓ built-in"
        elif (OUT / f"translations_{code}.json").exists():
            n = json.loads((OUT / f"translations_{code}.json").read_text())["total"]
            status = f"✓ ready ({n} strings)"
        else:
            status = "✗ not generated"
        print(f"  {code:<6} {name:<22} {status}")
    print(f"\nGenerate: python generate_translations.py --lang hi")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang",  default="",       help="Language code or 'all'")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--list",  action="store_true")
    args = ap.parse_args()

    if args.list:
        show_list()
    elif args.lang == "all":
        for code in LANGUAGES:
            if code != "en":
                generate(code, force=args.force)
    elif args.lang:
        generate(args.lang, force=args.force)
    else:
        ap.print_help()
