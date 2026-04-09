# webcam_capture.py — Reads frames from the laptop webcam

from __future__ import annotations
import cv2
import numpy as np
import logging
from typing import Optional

import config

logger = logging.getLogger(__name__)


class WebcamCapture:
    """
    Thin wrapper around ``cv2.VideoCapture`` with automatic reconnection.

    Parameters
    ----------
    index : int
        Camera device index (default from config).
    """

    def __init__(self, index: int = config.WEBCAM_INDEX) -> None:
        self._index = index
        self._cap: Optional[cv2.VideoCapture] = None
        self._open()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def read(self) -> Optional[np.ndarray]:
        """
        Read one frame from the webcam.

        Returns
        -------
        np.ndarray or None
            BGR image, or None if capture failed.
        """
        if not self.is_open:
            logger.warning("Webcam not open; attempting reconnect…")
            self._open()
            if not self.is_open:
                return None

        ret, frame = self._cap.read()
        if not ret or frame is None:
            logger.warning("Webcam read failed; attempting reconnect…")
            self._open()
            return None

        return frame

    def release(self) -> None:
        if self._cap:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open(self) -> None:
        if self._cap:
            self._cap.release()

        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            logger.error("Cannot open webcam (index %d).", self._index)
            self._cap = None
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.WEBCAM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.WEBCAM_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          config.WEBCAM_FPS)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # minimal buffer → latest frame

        self._cap = cap
        logger.info(
            "Webcam opened (index=%d, %dx%d @ %d fps)",
            self._index,
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(cap.get(cv2.CAP_PROP_FPS)),
        )
