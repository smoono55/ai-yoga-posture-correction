# audio_feedback.py — Neural TTS (edge-tts) with subprocess fallback
#
# Primary: Microsoft Edge Neural TTS via edge-tts (requires internet)
#   Voice: en-US-JennyNeural — warm, conversational, human-sounding
# Fallback 1: pyttsx3 (offline, robotic but functional)
# Fallback 2: OS-native (say / espeak / PowerShell)
# Fallback 3: Console print

from __future__ import annotations
import asyncio
import threading
import subprocess
import platform
import tempfile
import time
import queue
import logging
import os
from typing import Optional

import config

# ── Language / voice selection ─────────────────────────────────────────────────
def _get_voice_settings():
    """Read language fresh at worker-thread startup."""
    try:
        from voice_strings import t as _t, reload as _reload
        from language_manager import get_lang_info, get_current_lang
        _reload()
        info = get_lang_info(get_current_lang())
        print(get_current_lang())
        return info["tts_voice"], info["tts_rate"], "+10%", _t("startup_greeting")
    except Exception:
        return (config.EDGE_TTS_VOICE, config.EDGE_TTS_RATE,
                getattr(config, "EDGE_TTS_VOLUME", "+10%"),
                "Yoga posture correction system is starting.")

logger = logging.getLogger(__name__)
_IS_WINDOWS = platform.system() == "Windows"


# ── edge-tts async helper ─────────────────────────────────────────────────────
async def _edge_speak_async(text: str, voice=None, rate=None, volume=None) -> bool:
    """Speak text using edge-tts. Returns True on success."""
    try:
        import edge_tts
        comm = edge_tts.Communicate(
            text,
            voice=voice or config.EDGE_TTS_VOICE,
            rate=rate or config.EDGE_TTS_RATE,
            volume=volume or getattr(config, "EDGE_TTS_VOLUME", "+10%"),
        )
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name

        await comm.save(tmp)

        # Play the audio file
        try:
            if _IS_WINDOWS:
                try:
                    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
                    import pygame
                    pygame.mixer.init()
                    pygame.mixer.music.load(tmp)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    pygame.mixer.quit()
                except Exception:
                    try:
                        import playsound
                        playsound.playsound(tmp, block=True)
                    except Exception:
                        ps_cmd = (
                            f'Add-Type -AssemblyName presentationCore; '
                            f'$mp=New-Object System.Windows.Media.MediaPlayer; '
                            f'$mp.Open([System.Uri]("{tmp}")); '
                            f'$mp.Play(); Start-Sleep 4'
                        )
                        subprocess.run(["powershell", "-c", ps_cmd], timeout=10, check=False)
            elif platform.system() == "Darwin":
                subprocess.run(["afplay", tmp], timeout=15, check=False)
            else:
                subprocess.run(
                    ["ffplay", "-nodisp", "-autoexit", tmp],
                    timeout=15, check=False,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return True
    except Exception as exc:
        logger.debug("edge-tts failed: %s", exc)
        return False


def _edge_speak(text: str, voice=None, rate=None, volume=None) -> bool:
    """Synchronous wrapper around edge-tts async call."""
    try:
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(_edge_speak_async(text, voice, rate, volume))
        loop.close()
        return result
    except Exception as exc:
        logger.debug("edge-tts sync wrapper error: %s", exc)
        return False


# ── pyttsx3 helper ────────────────────────────────────────────────────────────
def _try_init_pyttsx3():
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 155)
        engine.setProperty("volume", 0.95)
        return engine
    except Exception as exc:
        logger.warning("pyttsx3 init failed: %s", exc)
        return None


def _os_speak(text: str) -> None:
    os_name = platform.system()
    try:
        if os_name == "Darwin":
            subprocess.run(["say", text],
                           timeout=15, check=False)
        elif os_name == "Linux":
            subprocess.run(["espeak", text],
                           timeout=15, check=False)
        elif os_name == "Windows":
            ps = (f'Add-Type -AssemblyName System.Speech; '
                  f'$s=New-Object System.Speech.Synthesis.SpeechSynthesizer; '
                  f'$s.Rate=2; $s.Speak("{text}")')
            subprocess.run(["powershell", "-Command", ps],
                           timeout=15, check=False)
    except Exception:
        print(f"[VOICE] {text}", flush=True)


# ── Main class ────────────────────────────────────────────────────────────────
class AudioFeedback:
    """
    Asynchronous TTS engine.
    Tries: edge-tts (neural) → pyttsx3 → OS TTS → console print.
    All speech runs on a dedicated background thread.
    """

    def __init__(self) -> None:
        self._queue: queue.Queue[Optional[str]] = queue.Queue(maxsize=4)
        self._last_spoken_time: float = 0.0
        self._last_message:     str   = ""
        self._running = True
        self._thread  = threading.Thread(
            target=self._worker, daemon=True, name="TTS-Worker"
        )
        self._thread.start()

    # ------------------------------------------------------------------ #
    def speak(self, message: str, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self._last_spoken_time) < config.FEEDBACK_COOLDOWN_SECONDS:
            return
        if message == self._last_message and not force:
            return
        try:
            self._queue.put_nowait(message)
            self._last_spoken_time = now
            self._last_message     = message
            print(f"\n[VOICE] {message}", flush=True)
        except queue.Full:
            logger.debug("TTS queue full — dropped.")

    def stop(self) -> None:
        self._running = False
        try: self._queue.put_nowait(None)
        except queue.Full: pass
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------ #
    def _worker(self) -> None:
        self._v, self._r, self._vol, _greeting = _get_voice_settings()
        pyttsx3_engine = _try_init_pyttsx3()
        # Probe edge-tts once at startup
        edge_ok = _edge_speak(_greeting, self._v, self._r, self._vol)
        if edge_ok:
            logger.info("Using edge-tts (neural voice).")
        else:
            logger.info("edge-tts unavailable — using pyttsx3 / OS TTS.")
            if pyttsx3_engine:
                try:
                    pyttsx3_engine.say(_greeting)
                    pyttsx3_engine.runAndWait()
                except Exception:
                    pass
            else:
                _os_speak(_greeting)

        while self._running:
            try:
                message = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if message is None:
                break
            self._say(message, pyttsx3_engine, edge_ok)

        if pyttsx3_engine:
            try: pyttsx3_engine.stop()
            except Exception: pass

    def _say(self, message: str, pyttsx3_engine, edge_ok: bool) -> None:
        if edge_ok:
            if _edge_speak(message, getattr(self,"_v",None), getattr(self,"_r",None), getattr(self,"_vol",None)):
                return
        if pyttsx3_engine:
            try:
                pyttsx3_engine.say(message)
                pyttsx3_engine.runAndWait()
                return
            except Exception as exc:
                logger.debug("pyttsx3 say error: %s", exc)
        _os_speak(message)
