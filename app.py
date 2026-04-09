#!/usr/bin/env python3
"""
app.py  —  Yoga Posture Correction UI Server

Usage:
  python app.py              → opens browser at http://localhost:5000
  python app.py --port 8080
  python app.py --no-browser
"""

from __future__ import annotations
import argparse
import os
import re
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory, Response

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MAIN_PY  = BASE_DIR / "main.py"
LM_PY    = BASE_DIR / "language_manager.py"
CONFIG_PY = BASE_DIR / "config.py"

app = Flask(__name__, static_folder=str(BASE_DIR))
LOG_FILE = BASE_DIR / "yoga_session.log"   # captures main.py stdout+stderr

# ── Process state ──────────────────────────────────────────────────────────────
_yoga_instance = None
_yoga_thread = None
_yoga_lock    = threading.Lock()
_setup_status = {"running": False, "message": "", "done": False, "error": False}

# ── Windows subprocess constants (defined as fallbacks for safety) ────────────
_CREATE_NO_WINDOW      = getattr(subprocess, "CREATE_NO_WINDOW",      0x08000000)
_STARTF_USESHOWWINDOW = getattr(subprocess, "STARTF_USESHOWWINDOW",   0x00000001)
_SW_HIDE               = getattr(subprocess, "SW_HIDE",                0)

# ── Language registry (mirrors language_manager.py — no import needed) ─────────
LANGUAGES = {
    "en": {"name": "English",            "native": "🇺🇸", "tts_voice": "en-US-JennyNeural",    "tts_rate": "+5%"},
    "hi": {"name": "Hindi",              "native": "🇮🇳", "tts_voice": "hi-IN-SwaraNeural",     "tts_rate": "+0%"},
    "ta": {"name": "Tamil",              "native": "🇮🇳", "tts_voice": "ta-IN-PallaviNeural",   "tts_rate": "+0%"},
    "te": {"name": "Telugu",             "native": "🇮🇳", "tts_voice": "te-IN-ShrutiNeural",    "tts_rate": "+0%"},
    "kn": {"name": "Kannada",            "native": "🇮🇳", "tts_voice": "kn-IN-SapnaNeural",    "tts_rate": "+0%"},
    "es": {"name": "Spanish",            "native": "🇪🇸", "tts_voice": "es-ES-ElviraNeural",    "tts_rate": "+5%"},
    "fr": {"name": "French",             "native": "🇫🇷", "tts_voice": "fr-FR-DeniseNeural",    "tts_rate": "+5%"},
    "de": {"name": "German",             "native": "🇩🇪", "tts_voice": "de-DE-KatjaNeural",     "tts_rate": "+5%"},
    "ja": {"name": "Japanese",           "native": "🇯🇵", "tts_voice": "ja-JP-NanamiNeural",    "tts_rate": "+0%"},
    "zh": {"name": "Chinese",            "native": "🇨🇳", "tts_voice": "zh-CN-XiaoxiaoNeural",  "tts_rate": "+0%"},
    "ar": {"name": "Arabic",             "native": "🇸🇦", "tts_voice": "ar-EG-SalmaNeural",     "tts_rate": "+0%"},
    "pt": {"name": "Portuguese",         "native": "🇧🇷", "tts_voice": "pt-BR-FranciscaNeural", "tts_rate": "+5%"},
    "ru": {"name": "Russian",            "native": "🇷🇺", "tts_voice": "ru-RU-SvetlanaNeural",  "tts_rate": "+5%"},
    "ko": {"name": "Korean",             "native": "🇰🇷", "tts_voice": "ko-KR-SunHiNeural",     "tts_rate": "+0%"},
    "it": {"name": "Italian",            "native": "🇮🇹", "tts_voice": "it-IT-ElsaNeural",      "tts_rate": "+5%"},
}

LANG_SETTING_FILE = BASE_DIR / "language_setting.json"
LANG_DIR          = BASE_DIR / "languages"


def _get_current_lang() -> str:
    import json
    try:
        return json.loads(LANG_SETTING_FILE.read_text()).get("lang", "en")
    except Exception:
        return "en"


def _translations_exist(code: str) -> bool:
    return (
        (LANG_DIR / f"translations_{code}.json").exists() or
        (LANG_DIR / f"corrections_{code}.json").exists()
    )


def _apply_voice_to_config(lang_code: str) -> None:
    """
    Directly patch EDGE_TTS_VOICE and EDGE_TTS_RATE in config.py.
    This is the fast path — always works, no internet required.
    """
    if not CONFIG_PY.exists():
        return
    info  = LANGUAGES.get(lang_code, LANGUAGES["en"])
    voice = info["tts_voice"]
    rate  = info["tts_rate"]
    text  = CONFIG_PY.read_text(encoding="utf-8")
    text  = re.sub(r'^EDGE_TTS_VOICE\s*=.*$', f'EDGE_TTS_VOICE = "{voice}"', text, flags=re.MULTILINE)
    text  = re.sub(r'^EDGE_TTS_RATE\s*=.*$',  f'EDGE_TTS_RATE  = "{rate}"',  text, flags=re.MULTILINE)
    CONFIG_PY.write_text(text, encoding="utf-8")


def _save_lang_setting(lang_code: str) -> None:
    import json
    LANG_SETTING_FILE.write_text(json.dumps({"lang": lang_code}))


def _hidden_popen(cmd: list, cwd: str) -> subprocess.Popen:
    """
    Launch a process with NO terminal window. Output goes to yoga_session.log.
    """
    log_handle = open(str(LOG_FILE), "w", buffering=1, encoding="utf-8", errors="replace")
    # Force UTF-8 so emoji in orchestrator.py print() calls don't crash on Windows cp1252
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"]       = "1"       # Python 3.7+ UTF-8 mode
    kwargs: dict = {
        "cwd":    cwd,
        "stdout": log_handle,
        "stderr": log_handle,
        "env":    env,
    }
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags    |= _STARTF_USESHOWWINDOW
        si.wShowWindow = _SW_HIDE
        kwargs["startupinfo"]  = si
        kwargs["creationflags"] = _CREATE_NO_WINDOW
    return subprocess.Popen(cmd, **kwargs)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/practice")
def practice():
    """Browser-based yoga coach — works on any device including mobile phones."""
    return send_from_directory(str(BASE_DIR), "practice.html")


@app.route("/api/languages")
def get_languages():
    current = _get_current_lang()
    result  = []
    for code, info in LANGUAGES.items():
        result.append({
            "code":    code,
            "name":    info["name"],
            "native":  info["native"],
            "voice":   info["tts_voice"],
            # "ready" = voice always works; translations are a bonus
            "ready":   True,
            "current": code == current,
            "translated": code == "en" or _translations_exist(code),
        })
    return jsonify({"languages": result, "current": current})


@app.route("/api/status")
def get_status():
    global _yoga_instance
    with _yoga_lock:
        running = _yoga_instance is not None and not getattr(_yoga_instance, "_stop_event", threading.Event()).is_set()
        returncode = 0 if running else None
    return jsonify({"running": running, "returncode": returncode, "setup": _setup_status})


@app.route("/api/start", methods=["POST"])
def start_yoga():
    global _yoga_instance
    data = request.get_json() or {}
    lang = data.get("lang", "en")

    with _yoga_lock:
        if _yoga_instance and not getattr(_yoga_instance, "_stop_event", threading.Event()).is_set():
            return jsonify({"error": "Yoga system is already running"}), 400

    def _run():
        global _yoga_process
        _setup_status.update({"running": True, "message": "Applying language settings...",
                               "done": False, "error": False})

        # ── Step 1: Apply voice (always fast, no internet) ─────────────────────
        try:
            _apply_voice_to_config(lang)
            _save_lang_setting(lang)
        except Exception as e:
            # Non-fatal — continue with whatever voice config.py currently has
            print(f"[app] Voice config warning: {e}")

        # ── Step 2: Translate ALL strings via generate_translations.py ──────
        GT_PY = BASE_DIR / "generate_translations.py"
        if lang != "en" and GT_PY.exists():
            lang_name = LANGUAGES.get(lang, {}).get("name", lang)
            if not _translations_exist(lang):
                _setup_status["message"] = f"Translating to {lang_name} — first time only (~30s)..."
                try:
                    result = subprocess.run(
                        [sys.executable, str(GT_PY), "--lang", lang],
                        cwd=str(BASE_DIR),
                        capture_output=True, text=True, timeout=300,
                    )
                    if result.returncode != 0:
                        print(f"[app] Translation failed: {result.stderr[:300]}")
                        _setup_status["message"] = (
                            f"Voice set to {lang_name}. "
                            f"Full translation unavailable "
                            f"(pip install googletrans==4.0.0rc1). Starting..."
                        )
                    else:
                        _setup_status["message"] = f"{lang_name} fully ready. Starting..."
                except Exception as e:
                    print(f"[app] Translation error (non-fatal): {e}")
                    _setup_status["message"] = f"Voice set to {lang_name}. Starting..."
            else:
                _setup_status["message"] = f"{lang_name} translations loaded. Starting..."
        else:
            _setup_status["message"] = "Starting yoga system..."

        # ── Step 3: Launch YogaCorrector directly — no terminal window ─────────────────
        try:
            global _yoga_thread

            def _yoga_thread_runner():
                global _yoga_instance
                try:
                    from orchestrator import YogaCorrector
                    # Instantiating the engine with show_debug=True enables the always-on-top native camera mirror
                    proc = YogaCorrector(show_debug=True)
                    with _yoga_lock:
                        _yoga_instance = proc
                        _setup_status.update({
                            "running": True, "done": True, "error": False,
                            "message": f"Running securely — your 'Yoga Mirror' window is active. You may now start your video.",
                        })
                    proc.run()
                except Exception as ex:
                    print(f"Yoga Runner Error: {ex}")
                    with _yoga_lock:
                         _setup_status.update({
                             "running": False, "done": True, "error": True,
                             "message": f"System Crash: {ex}"
                         })

            # Hand off immediately to unblock the browser POST request
            _setup_status.update({
                "running": True, "done": False, "error": False,
                "message": f"Initializing Machine Learning engine...",
            })
            _yoga_thread = threading.Thread(target=_yoga_thread_runner, daemon=True)
            _yoga_thread.start()

        except Exception as e:
            _setup_status.update({
                "running": False, "done": True, "error": True,
                "message": f"Failed to start system: {e}"
            })

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "starting", "lang": lang})


@app.route("/api/log")
def get_log():
    """Return last 30 lines of yoga_session.log for debugging."""
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        return jsonify({"lines": lines[-30:], "total": len(lines)})
    except FileNotFoundError:
        return jsonify({"lines": [], "total": 0})


@app.route("/api/stop", methods=["POST"])
def stop_yoga():
    global _yoga_instance
    with _yoga_lock:
        if _yoga_instance:
            _yoga_instance.stop()
            _yoga_instance = None
            _setup_status.update({"running": False, "message": "Session stopped.", "done": True})
            return jsonify({"status": "stopped"})
    return jsonify({"status": "not_running"})

@app.route("/video_feed")
def video_feed():
    def generate_frames():
        import cv2
        import time
        while True:
            if _yoga_instance and hasattr(_yoga_instance, 'latest_frame'):
                frame = _yoga_instance.latest_frame
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.04)
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port",       type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    url = f"http://localhost:{args.port}"
    print(f"\n{'-'*50}")
    print(f"  Yoga Posture Correction UI  ->  {url}")
    print(f"  Browser Practice Mode       ->  {url}/practice")
    print(f"{'-'*50}\n")

    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)
