#!/usr/bin/env python3
"""
language_manager.py  —  Multilingual Yoga Correction System

Translates the correction bank into the target language and saves a
ready-to-use JSON file.  At runtime, feedback_builder.py loads this
JSON so every correction is spoken in the user's language.

Supported Languages
───────────────────
  en  English       (default — no translation needed)
  hi  Hindi
  ta  Tamil
  te  Telugu
  kn  Kannada
  es  Spanish
  fr  French
  de  German
  ja  Japanese
  zh  Chinese (Simplified)
  ar  Arabic
  pt  Portuguese
  ru  Russian
  ko  Korean
  it  Italian

Usage
─────
  # Set up Hindi
  python language_manager.py --setup --lang hi

  # List all available languages
  python language_manager.py --list

  # Show current language setting
  python language_manager.py --current

  # Reset to English
  python language_manager.py --setup --lang en

  # Test speak a correction in the current language
  python language_manager.py --test

Requirements
────────────
  pip install deep-translator
  (edge-tts is already required by the main system)
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time

# ── Language registry ──────────────────────────────────────────────────────────
LANGUAGES: dict[str, dict] = {
    "en": {
        "name":        "English",
        "native":      "English",
        "tts_voice":   "en-US-JennyNeural",
        "tts_rate":    "+5%",
        "greeting":    "Yoga posture correction system is starting.",
        "rtl":         False,
    },
    "hi": {
        "name":        "Hindi",
        "native":      "हिन्दी",
        "tts_voice":   "hi-IN-SwaraNeural",
        "tts_rate":    "+0%",
        "greeting":    "योग आसन सुधार प्रणाली शुरू हो रही है।",
        "rtl":         False,
    },
    "ta": {
        "name":        "Tamil",
        "native":      "தமிழ்",
        "tts_voice":   "ta-IN-PallaviNeural",
        "tts_rate":    "+0%",
        "greeting":    "யோகா தோரணை திருத்த அமைப்பு தொடங்குகிறது.",
        "rtl":         False,
    },
    "te": {
        "name":        "Telugu",
        "native":      "తెలుగు",
        "tts_voice":   "te-IN-ShrutiNeural",
        "tts_rate":    "+0%",
        "greeting":    "యోగా భంగిమ సరిదిద్దే వ్యవస్థ ప్రారంభమవుతోంది.",
        "rtl":         False,
    },
    "kn": {
        "name":        "Kannada",
        "native":      "ಕನ್ನಡ",
        "tts_voice":   "kn-IN-GeethaNeural",
        "tts_rate":    "+0%",
        "greeting":    "ಯೋಗ ಭಂಗಿ ತಿದ್ದುಪಡಿ ವ್ಯವಸ್ಥೆ ಪ್ರಾರಂಭವಾಗುತ್ತಿದೆ.",
        "rtl":         False,
    },
    "es": {
        "name":        "Spanish",
        "native":      "Español",
        "tts_voice":   "es-ES-ElviraNeural",
        "tts_rate":    "+5%",
        "greeting":    "El sistema de corrección de postura de yoga está iniciando.",
        "rtl":         False,
    },
    "fr": {
        "name":        "French",
        "native":      "Français",
        "tts_voice":   "fr-FR-DeniseNeural",
        "tts_rate":    "+5%",
        "greeting":    "Le système de correction de posture yoga démarre.",
        "rtl":         False,
    },
    "de": {
        "name":        "German",
        "native":      "Deutsch",
        "tts_voice":   "de-DE-KatjaNeural",
        "tts_rate":    "+5%",
        "greeting":    "Das Yoga-Haltungskorrektur-System startet.",
        "rtl":         False,
    },
    "ja": {
        "name":        "Japanese",
        "native":      "日本語",
        "tts_voice":   "ja-JP-NanamiNeural",
        "tts_rate":    "+0%",
        "greeting":    "ヨガ姿勢矯正システムを起動します。",
        "rtl":         False,
    },
    "zh": {
        "name":        "Chinese (Simplified)",
        "native":      "中文",
        "tts_voice":   "zh-CN-XiaoxiaoNeural",
        "tts_rate":    "+0%",
        "greeting":    "瑜伽姿势纠正系统正在启动。",
        "rtl":         False,
    },
    "ar": {
        "name":        "Arabic",
        "native":      "العربية",
        "tts_voice":   "ar-EG-SalmaNeural",
        "tts_rate":    "+0%",
        "greeting":    "نظام تصحيح وضعية اليوغا يبدأ.",
        "rtl":         True,
    },
    "pt": {
        "name":        "Portuguese",
        "native":      "Português",
        "tts_voice":   "pt-BR-FranciscaNeural",
        "tts_rate":    "+5%",
        "greeting":    "O sistema de correção de postura de yoga está iniciando.",
        "rtl":         False,
    },
    "ru": {
        "name":        "Russian",
        "native":      "Русский",
        "tts_voice":   "ru-RU-SvetlanaNeural",
        "tts_rate":    "+5%",
        "greeting":    "Система коррекции позы йоги запускается.",
        "rtl":         False,
    },
    "ko": {
        "name":        "Korean",
        "native":      "한국어",
        "tts_voice":   "ko-KR-SunHiNeural",
        "tts_rate":    "+0%",
        "greeting":    "요가 자세 교정 시스템이 시작됩니다.",
        "rtl":         False,
    },
    "it": {
        "name":        "Italian",
        "native":      "Italiano",
        "tts_voice":   "it-IT-ElsaNeural",
        "tts_rate":    "+5%",
        "greeting":    "Il sistema di correzione della postura yoga si avvia.",
        "rtl":         False,
    },
}

# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE         = os.path.dirname(os.path.abspath(__file__))
_LANG_DIR     = os.path.join(_BASE, "languages")
_SETTING_FILE = os.path.join(_BASE, "language_setting.json")


def get_current_lang() -> str:
    """Return the currently configured language code (default 'en')."""
    if os.path.exists(_SETTING_FILE):
        try:
            with open(_SETTING_FILE) as f:
                data = json.load(f)
            return data.get("lang", "en")
        except Exception:
            pass
    return "en"


def get_lang_info(lang_code: str) -> dict:
    return LANGUAGES.get(lang_code, LANGUAGES["en"])


def _save_setting(lang_code: str) -> None:
    with open(_SETTING_FILE, "w") as f:
        json.dump({"lang": lang_code}, f)


def _translations_path(lang_code: str) -> str:
    return os.path.join(_LANG_DIR, f"corrections_{lang_code}.json")


def _translations_exist(lang_code: str) -> bool:
    return os.path.exists(_translations_path(lang_code))


# ── Translation ────────────────────────────────────────────────────────────────
def _translate_batch(texts: list[str], target_lang: str) -> list[str]:
    """
    Translate a list of texts using deep-translator (Google Translate backend).
    Falls back to the original text on any error.
    """
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        print("ERROR: pip install deep-translator")
        sys.exit(1)

    translator = GoogleTranslator(source="en", target=target_lang)
    results = []
    for i, text in enumerate(texts):
        try:
            translated = translator.translate(text)
            results.append(translated or text)
            # Small delay to avoid rate limiting
            if (i + 1) % 10 == 0:
                time.sleep(0.5)
                print(f"  Translated {i+1}/{len(texts)}...", end="\r")
        except Exception as e:
            print(f"\n  Warning: translation failed for '{text[:40]}': {e}")
            results.append(text)
    return results


def _get_correction_bank() -> list[dict]:
    """Import correction bank from train_correction_ai.py."""
    sys.path.insert(0, _BASE)
    try:
        from train_correction_ai import CORRECTION_BANK
        return CORRECTION_BANK
    except ImportError as e:
        print(f"ERROR: Cannot import CORRECTION_BANK: {e}")
        sys.exit(1)


def setup_language(lang_code: str, force: bool = False) -> None:
    """
    Translate the correction bank into lang_code and save to languages/ folder.
    Also updates config.py with the correct TTS voice.
    """
    if lang_code not in LANGUAGES:
        print(f"Unknown language '{lang_code}'. Run --list to see options.")
        sys.exit(1)

    lang_info = LANGUAGES[lang_code]
    os.makedirs(_LANG_DIR, exist_ok=True)

    print(f"\n=== Setting up language: {lang_info['name']} ({lang_code}) ===")

    # ── English: no translation needed ───────────────────────────────────────
    if lang_code == "en":
        _save_setting("en")
        _update_config_voice("en-US-JennyNeural", "+5%", "+10%")
        print("✓ Language set to English (no translation needed)")
        return

    # ── Check if already translated ───────────────────────────────────────────
    trans_path = _translations_path(lang_code)
    if os.path.exists(trans_path) and not force:
        print(f"✓ Translations already exist at {trans_path}")
        print("  (Use --force to re-translate)")
        _save_setting(lang_code)
        _update_config_voice(lang_info["tts_voice"], lang_info["tts_rate"], "+10%")
        print(f"✓ Language set to {lang_info['name']}")
        return

    # ── Translate correction bank ─────────────────────────────────────────────
    bank = _get_correction_bank()
    print(f"  Translating {len(bank)} corrections to {lang_info['name']}...")
    print("  (This may take 30-60 seconds due to rate limiting)\n")

    texts     = [c["text"] for c in bank]
    ids       = [c["id"]   for c in bank]
    translated = _translate_batch(texts, lang_code)

    # Also translate static messages
    static_texts = {
        "well_done": [
            "That's it — hold that and breathe.",
            "Yes! That's the shape. Stay with it.",
            "Beautiful alignment. Keep breathing.",
            "Perfect — you've found it. Don't move.",
            "Great shape. Lock it in.",
            "You're matching the instructor well. Maintain it.",
        ],
        "encouragement_small": [
            "Almost there.", "Really close now.", "Nearly perfect.",
            "Tiny adjustment.", "You're so close.",
        ],
        "encouragement_medium": [
            "You're doing well — keep adjusting.",
            "Good effort — stay with it.",
            "Nice work — keep breathing through it.",
        ],
        "encouragement_large": [
            "Take your time — this one takes practice.",
            "Breathe and keep working through it.",
            "Every body is different — do what you can.",
        ],
        "score_excellent": [
            "You're at {score} percent — excellent.",
            "{score} percent match. Beautiful.",
        ],
        "score_good": [
            "You're at {score} percent — good alignment, small adjustments left.",
            "{score} percent. Getting really close.",
        ],
        "score_fair": [
            "{score} percent — you're making progress.",
            "At {score} percent — keep working through the corrections.",
        ],
        "score_needs_work": [
            "{score} percent right now — let's work through this together.",
            "{score} percent — take it one joint at a time.",
        ],
    }

    print("\n  Translating UI messages...")
    translated_static = {}
    for key, msgs in static_texts.items():
        # Replace {score} placeholder before translating
        placeholder = "SCORENUMBER"
        msgs_safe = [m.replace("{score}", placeholder) for m in msgs]
        trans = _translate_batch(msgs_safe, lang_code)
        # Restore placeholder
        translated_static[key] = [t.replace(placeholder, "{score}") for t in trans]

    # Save
    payload = {
        "lang":        lang_code,
        "lang_name":   lang_info["name"],
        "corrections": {str(cid): text for cid, text in zip(ids, translated)},
        "static":      translated_static,
        "voice":       lang_info["tts_voice"],
    }
    with open(trans_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Saved {len(translated)} corrections to {trans_path}")

    # Update config
    _save_setting(lang_code)
    _update_config_voice(lang_info["tts_voice"], lang_info["tts_rate"], "+10%")
    print(f"✓ TTS voice set to {lang_info['tts_voice']}")
    print(f"✓ Language setup complete — restart main.py to apply")


def _update_config_voice(voice: str, rate: str, volume: str) -> None:
    """Update EDGE_TTS_VOICE and EDGE_TTS_RATE in config.py."""
    config_path = os.path.join(_BASE, "config.py")
    if not os.path.exists(config_path):
        return
    with open(config_path) as f:
        content = f.read()

    import re
    content = re.sub(
        r'^EDGE_TTS_VOICE\s*=.*$',
        f'EDGE_TTS_VOICE = "{voice}"',
        content, flags=re.MULTILINE
    )
    content = re.sub(
        r'^EDGE_TTS_RATE\s*=.*$',
        f'EDGE_TTS_RATE  = "{rate}"',
        content, flags=re.MULTILINE
    )
    with open(config_path, "w") as f:
        f.write(content)


# ── Runtime loader (called by feedback_builder.py) ────────────────────────────
_translation_cache: dict | None = None

def load_translations() -> dict | None:
    """
    Load the translation JSON for the current language.
    Returns None if language is English or translations don't exist.
    Cached after first load.
    """
    global _translation_cache
    if _translation_cache is not None:
        return _translation_cache

    lang = get_current_lang()
    if lang == "en":
        return None

    path = _translations_path(lang)
    if not os.path.exists(path):
        return None

    try:
        with open(path, encoding="utf-8") as f:
            _translation_cache = json.load(f)
        return _translation_cache
    except Exception as e:
        print(f"Warning: Could not load translations: {e}")
        return None


def get_correction_text(correction_id: int, english_text: str) -> str:
    """Return the translated correction text for the given ID, or English fallback."""
    trans = load_translations()
    if trans is None:
        return english_text
    return trans.get("corrections", {}).get(str(correction_id), english_text)


def get_static(key: str, english_options: list[str]) -> list[str]:
    """Return translated static message list, or English fallback."""
    trans = load_translations()
    if trans is None:
        return english_options
    translated = trans.get("static", {}).get(key)
    return translated if translated else english_options


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Yoga correction multilingual setup")
    ap.add_argument("--setup",   action="store_true", help="Set up a language")
    ap.add_argument("--lang",    default="",          help="Language code (e.g. hi, ta, es)")
    ap.add_argument("--force",   action="store_true", help="Re-translate even if file exists")
    ap.add_argument("--list",    action="store_true", help="List available languages")
    ap.add_argument("--current", action="store_true", help="Show current language")
    ap.add_argument("--test",    action="store_true", help="Speak a test correction")
    args = ap.parse_args()

    if args.list:
        print("\nAvailable languages:\n")
        cur = get_current_lang()
        print(f"  {'Code':<6} {'Name':<22} {'Native':<18} {'TTS Voice'}")
        print("  " + "─"*72)
        for code, info in LANGUAGES.items():
            marker = " ← current" if code == cur else ""
            setup  = " [ready]" if (code=="en" or _translations_exist(code)) else ""
            print(f"  {code:<6} {info['name']:<22} {info['native']:<18} {info['tts_voice']}{setup}{marker}")
        print(f"\nSetup: python language_manager.py --setup --lang hi")

    elif args.current:
        lang = get_current_lang()
        info = get_lang_info(lang)
        print(f"Current language: {info['name']} ({lang})")
        print(f"TTS Voice:        {info['tts_voice']}")
        if _translations_exist(lang):
            print(f"Translations:     Ready")
        elif lang == "en":
            print(f"Translations:     Not needed (English)")
        else:
            print(f"Translations:     Not yet set up — run: python language_manager.py --setup --lang {lang}")

    elif args.setup:
        if not args.lang:
            print("Specify a language with --lang. Example: --setup --lang hi")
            sys.exit(1)
        setup_language(args.lang, force=args.force)

    elif args.test:
        import asyncio, random
        lang = get_current_lang()
        info = get_lang_info(lang)
        trans = load_translations()
        if trans and trans.get("corrections"):
            text = random.choice(list(trans["corrections"].values()))
        else:
            text = "Bend your left knee deeper — sink your weight into it."

        print(f"Language: {info['name']}")
        print(f"Voice:    {info['tts_voice']}")
        print(f"Text:     {text}")

        async def _speak():
            try:
                import edge_tts, tempfile, subprocess, platform
                comm = edge_tts.Communicate(text, voice=info["tts_voice"], rate=info["tts_rate"])
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                    tmp = f.name
                await comm.save(tmp)
                if platform.system() == "Windows":
                    try:
                        import playsound; playsound.playsound(tmp, block=True)
                    except ImportError:
                        subprocess.run(["powershell","-c",
                            f'$m=New-Object System.Windows.Media.MediaPlayer; '
                            f'Add-Type -A presentationCore; '
                            f'$m.Open([Uri]("{tmp}")); $m.Play(); Start-Sleep 4'],
                            check=False)
                elif platform.system() == "Darwin":
                    subprocess.run(["afplay", tmp])
                os.unlink(tmp)
            except Exception as e:
                print(f"TTS error: {e}")

        asyncio.run(_speak())

    else:
        ap.print_help()
