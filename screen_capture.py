# screen_capture.py — Windows-robust screen capture with DPI fix + debug window
#
# Problem 1: Windows DPI scaling → mss reports wrong coordinates
# Fix: Set DPI awareness before grabbing
#
# Problem 2: Hardware-accelerated video (YouTube/Chrome/Edge) renders via GPU
#            so mss (which reads the GDI framebuffer) gets a black frame.
# Fix: Primary method uses mss with DPI fix. If the captured frame is
#      mostly black we automatically fall back to a win32 BitBlt grab
#      which reads the composited screen including GPU layers.

from __future__ import annotations
import platform
import logging
import time
import numpy as np
import cv2
from typing import Optional

import config

logger = logging.getLogger(__name__)

_IS_WINDOWS = platform.system() == "Windows"


# ── DPI awareness (must be called before any window/screen API) ───────────
def _set_dpi_aware() -> None:
    if not _IS_WINDOWS:
        return
    try:
        import ctypes
        # Per-monitor DPI awareness (Windows 8.1+)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


_set_dpi_aware()


# ── BitBlt fallback (captures GPU-accelerated / DRM-free content) ─────────
def _grab_bitblt(monitor: dict) -> Optional[np.ndarray]:
    """
    Use win32 GDI BitBlt to capture a monitor region.
    Works with hardware-accelerated windows that mss cannot see.
    Requires pywin32: pip install pywin32
    """
    try:
        import win32gui, win32ui, win32con
        x, y   = monitor["left"], monitor["top"]
        w, h   = monitor["width"], monitor["height"]

        hdesktop = win32gui.GetDesktopWindow()
        desktop_dc = win32gui.GetWindowDC(hdesktop)
        img_dc     = win32ui.CreateDCFromHandle(desktop_dc)
        mem_dc     = img_dc.CreateCompatibleDC()

        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(img_dc, w, h)
        mem_dc.SelectObject(bmp)
        mem_dc.BitBlt((0, 0), (w, h), img_dc, (x, y), win32con.SRCCOPY)

        bmp_info = bmp.GetInfo()
        raw      = bmp.GetBitmapBits(True)
        frame    = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
        frame    = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        mem_dc.DeleteDC()
        img_dc.DeleteDC()
        win32gui.ReleaseDC(hdesktop, desktop_dc)
        win32gui.DeleteObject(bmp.GetHandle())

        return frame
    except ImportError:
        logger.debug("pywin32 not available — BitBlt fallback skipped.")
        return None
    except Exception as exc:
        logger.debug("BitBlt grab error: %s", exc)
        return None


def _is_mostly_black(frame: np.ndarray, threshold: float = 0.97) -> bool:
    """Return True if >threshold fraction of pixels are near-black."""
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    black  = np.sum(gray < 15)
    return (black / gray.size) > threshold


class ScreenCapture:
    """
    Captures a monitor region and returns BGR frames for pose estimation.

    Capture strategy (Windows):
      1. mss  (fast, CPU framebuffer)
      2. If frame is mostly black → BitBlt (captures GPU/HW-accelerated video)
      3. If pywin32 unavailable → warn user

    On macOS / Linux: mss only (no hardware-acceleration issue).
    """

    def __init__(self, monitor_index: int = config.SCREEN_MONITOR_INDEX) -> None:
        self._monitor_index = monitor_index
        self._sct           = None
        self._monitor_def:  Optional[dict] = None
        self._use_bitblt    = False          # auto-detected after first black frame
        self._black_streak  = 0
        self._init_mss()

        # Disable the live preview window (per user request to run hands-free with mirroring)
        self._show_preview  = False
        self._preview_name  = "Instructor Screen Feed [debug]"

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def grab(self) -> Optional[np.ndarray]:
        raw = self._grab_raw()
        if raw is None:
            return None

        # Auto-detect black frame → switch to BitBlt
        if _IS_WINDOWS and not self._use_bitblt and _is_mostly_black(raw):
            self._black_streak += 1
            if self._black_streak >= 3:
                logger.warning(
                    "Screen capture returning black frames — "
                    "switching to BitBlt (hardware-accelerated capture). "
                    "Make sure pywin32 is installed: pip install pywin32"
                )
                self._use_bitblt = True
        else:
            self._black_streak = 0

        if self._use_bitblt and self._monitor_def:
            bitblt_frame = _grab_bitblt(self._monitor_def)
            if bitblt_frame is not None:
                raw = bitblt_frame
            else:
                # pywin32 not installed — print helpful message once
                if not hasattr(self, "_bitblt_warned"):
                    self._bitblt_warned = True
                    print(
                        "\n[SCREEN] ⚠  YouTube/Chrome uses GPU rendering — "
                        "mss sees a black screen.\n"
                        "         Fix: pip install pywin32\n"
                        "         Then re-run the program.\n",
                        flush=True,
                    )

        # Resize for cheaper inference
        frame = cv2.resize(
            raw,
            config.SCREEN_CAPTURE_RESIZE,
            interpolation=cv2.INTER_AREA,
        )

        # Live preview window
        if self._show_preview:
            self._draw_preview(frame)

        return frame

    def close(self) -> None:
        if self._sct:
            self._sct.close()
            self._sct = None
        if self._show_preview:
            try:
                cv2.destroyWindow(self._preview_name)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _grab_raw(self) -> Optional[np.ndarray]:
        if self._sct is None or self._monitor_def is None:
            return None
        try:
            shot  = self._sct.grab(self._monitor_def)
            frame = np.array(shot)
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        except Exception as exc:
            logger.error("mss grab error: %s", exc)
            # Try to reinitialise
            self._init_mss()
            return None

    def _init_mss(self) -> None:
        try:
            import mss
            if self._sct:
                self._sct.close()
            self._sct = mss.mss()
            monitors  = self._sct.monitors
            idx       = self._monitor_index
            if idx >= len(monitors):
                logger.warning("Monitor %d not found; using monitor 1.", idx)
                idx = 1
            self._monitor_def = monitors[idx]
            logger.info(
                "Screen capture ready — monitor %d (%dx%d)",
                idx,
                self._monitor_def["width"],
                self._monitor_def["height"],
            )
            print(
                f"[SCREEN] Capturing monitor {idx}  "
                f"({self._monitor_def['width']}x{self._monitor_def['height']})",
                flush=True,
            )
        except ImportError:
            logger.error("mss not installed. Run: pip install mss")
        except Exception as exc:
            logger.error("Screen capture init error: %s", exc)

    def _draw_preview(self, frame: np.ndarray) -> None:
        """Show a small preview of the captured screen (so user can verify)."""
        try:
            h, w  = frame.shape[:2]
            thumb = cv2.resize(frame, (w // 2, h // 2))
            label = "Screen capture preview  (close = instructor not visible)"
            cv2.putText(thumb, label, (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.imshow(self._preview_name, thumb)
            cv2.waitKey(1)
        except Exception:
            pass
