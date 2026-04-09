# voice_interaction.py — Background speech recognition for voice commands
#
# User is far from the device, so we listen continuously for commands.
# Supported commands (spoken naturally):
#   "score"          → announce current similarity score
#   "what pose"      → announce the detected yoga pose name
#   "next step"      → read the next setup step
#   "repeat"         → repeat the last spoken message
#   "pause"          → pause corrections for 60 seconds
#   "resume"         → resume corrections
#   "help"           → list available commands
#   "start over"     → reset step-by-step guidance to step 1

from __future__ import annotations
import threading
import logging
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class VoiceInteraction:
    """
    Listens in the background for user voice commands.
    When a command is recognised, the registered callback is called with
    the command string.

    Parameters
    ----------
    on_command : Callable[[str], None]
        Called on the main-loop-friendly queue whenever a command fires.
    energy_threshold : int
        Microphone sensitivity (higher = less sensitive).
    """

    COMMANDS = {
        "score":       ["score", "what's my score", "how am i doing", "percentage", "points", "status", "record"],
        "what_pose":   ["what pose", "which pose", "pose name", "what's this pose", "current pose", "what posture"],
        "next_step":   ["next step", "next", "what's next", "next instruction", "next phase", "forward", "continue step"],
        "repeat":      ["repeat", "say again", "what did you say", "come again", "pardon", "again"],
        "pause":       ["pause", "stop corrections", "quiet", "silence", "halt", "hold on", "stop"],
        "resume":      ["resume", "continue", "start corrections", "go", "play", "unpause"],
        "start_over":  ["start over", "from the beginning", "reset steps", "restart", "begin again", "reset"],
        "help":        ["help", "commands", "what can i say", "options", "menu", "what are the commands"],
    }

    def __init__(
        self,
        on_command: Callable[[str], None],
        energy_threshold: int = 300,
    ) -> None:
        self._on_command      = on_command
        self._energy_threshold= energy_threshold
        self._running         = False
        self._thread: Optional[threading.Thread] = None
        self._sr_available    = False
        self._recogniser      = None
        self._mic             = None
        self._init_sr()

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if not self._sr_available:
            logger.warning(
                "SpeechRecognition not available. "
                "Install: pip install SpeechRecognition pyaudio"
            )
            print(
                "[VOICE CMD] Speech recognition unavailable.\n"
                "            To enable: pip install SpeechRecognition pyaudio\n"
                "            Keyboard shortcuts active instead (see below).\n",
                flush=True,
            )
            return

        self._running = True
        self._thread  = threading.Thread(
            target=self._listen_loop, daemon=True, name="VoiceCmd"
        )
        self._thread.start()
        print(
            "\n[VOICE CMD] Listening for commands. Say any of:\n"
            "  'score'      — hear your current pose score\n"
            "  'what pose'  — hear the pose name\n"
            "  'next step'  — get the next setup instruction\n"
            "  'repeat'     — repeat the last message\n"
            "  'pause'      — pause corrections\n"
            "  'resume'     — resume corrections\n"
            "  'start over' — restart step-by-step guidance\n"
            "  'help'       — list commands\n",
            flush=True,
        )

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _init_sr(self) -> None:
        try:
            import speech_recognition as sr
            self._recogniser = sr.Recognizer()
            self._recogniser.energy_threshold        = self._energy_threshold
            self._recogniser.dynamic_energy_threshold= True
            self._recogniser.pause_threshold         = 0.6
            self._mic         = sr.Microphone()
            self._sr_available= True
            logger.info("Speech recognition initialised.")
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("Speech recognition init error: %s", exc)

    def _listen_loop(self) -> None:
        import speech_recognition as sr
        logger.info("Voice command listener started.")

        # Calibrate for ambient noise
        try:
            with self._mic as source:
                self._recogniser.adjust_for_ambient_noise(source, duration=1.5)
        except Exception:
            pass

        while self._running:
            try:
                with self._mic as source:
                    audio = self._recogniser.listen(source, timeout=5, phrase_time_limit=5)
                # Get current language and locale for STT
                from language_manager import get_current_lang, get_lang_info
                lang_code = get_current_lang()
                info = get_lang_info(lang_code)
                # tts_voice looks like "hi-IN-SwaraNeural" or "en-US-JennyNeural"
                # Extract the locale part like "hi-IN" for Google Speech Recognition
                locale = info["tts_voice"].rsplit("-", 1)[0] if "-" in info["tts_voice"] else "en-US"
                
                text = self._recogniser.recognize_google(audio, language=locale).lower().strip()
                logger.debug("Heard (%s): %s", lang_code, text)
                print(f"[VOICE CMD] Heard: '{text}'", flush=True)
                
                if lang_code != "en":
                    try:
                        from deep_translator import GoogleTranslator
                        # Translate the recognized native text back to English to match COMMANDS keys
                        text = GoogleTranslator(source=lang_code, target='en').translate(text).lower().strip()
                        logger.debug("Translated to English: %s", text)
                        print(f"[VOICE CMD] Native command translated to: '{text}'", flush=True)
                    except Exception as e:
                        logger.debug("Translation of voice command failed: %s", e)

                cmd = self._match_command(text)
                if cmd:
                    print(f"[VOICE CMD] Command: {cmd}", flush=True)
                    self._on_command(cmd)
            except sr.WaitTimeoutError:
                pass   # nothing heard — normal
            except sr.UnknownValueError:
                pass   # could not understand
            except sr.RequestError as exc:
                logger.warning("Speech API error: %s — check internet connection.", exc)
                time.sleep(3)
            except Exception as exc:
                logger.debug("Listen loop error: %s", exc)

    def _match_command(self, text: str) -> Optional[str]:
        # 1. Try exact or partial strict match first (fastest)
        for cmd, phrases in self.COMMANDS.items():
            for phrase in phrases:
                if phrase in text:
                    return cmd
                    
        # 2. Try fuzzy matching (forgiving for translation nuances)
        try:
            from thefuzz import process, fuzz
            phrase_to_cmd = {phrase: cmd for cmd, phrases in self.COMMANDS.items() for phrase in phrases}
            best_match, score = process.extractOne(text, list(phrase_to_cmd.keys()), scorer=fuzz.partial_ratio)
            
            if score >= 75:  # High confidence threshold
                logger.debug("Fuzzy matched '%s' to '%s' (score: %s)", text, best_match, score)
                return phrase_to_cmd[best_match]
        except ImportError:
            logger.warning("thefuzz library not installed, fuzzy voice matching skipped.")
            
        return None
