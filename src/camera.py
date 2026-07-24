"""Webcam capture module with thread-safe frame access and FPS tracking."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from src.config import CameraSettings

Frame = NDArray[np.uint8]


class CameraError(Exception):
    """Raised when a camera cannot be opened or frames cannot be read."""


@dataclass(frozen=True)
class CameraInfo:
    """Describes the active camera stream properties."""

    index: int
    width: int
    height: int
    fps_target: int
    backend: str


class CameraCapture:
    """
    Thread-safe webcam capture using OpenCV.

    A background thread continuously reads frames so the main loop always
    receives the most recent image without blocking on camera I/O.
    """

    def __init__(self, settings: CameraSettings, logger: logging.Logger | None = None) -> None:
        """
        Initialize the camera capture manager.

        Args:
            settings: Camera configuration from application settings.
            logger: Optional logger instance. Uses module logger if not provided.
        """
        self._settings = settings
        self._logger = logger or logging.getLogger("hand_gesture_mouse.camera")

        self._capture: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()

        self._latest_frame: Frame | None = None
        self._frame_count = 0
        self._fps = 0.0
        self._fps_lock = threading.Lock()
        self._fps_start_time = 0.0
        self._fps_frame_count = 0

        self._is_running = False
        self._camera_info: CameraInfo | None = None

    @property
    def is_running(self) -> bool:
        """Return True if the capture thread is active."""
        return self._is_running

    @property
    def fps(self) -> float:
        """Return the current measured frames-per-second."""
        with self._fps_lock:
            return self._fps

    @property
    def camera_info(self) -> CameraInfo | None:
        """Return metadata about the opened camera stream."""
        return self._camera_info

    def start(self) -> None:
        """
        Open the webcam and begin background frame capture.

        Raises:
            CameraError: If the camera cannot be opened or configured.
        """
        if self._is_running:
            self._logger.warning("Camera is already running.")
            return

        self._logger.info(
            "Opening camera index %s at %sx%s",
            self._settings.index,
            self._settings.width,
            self._settings.height,
        )

        capture = cv2.VideoCapture(self._settings.index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            raise CameraError(
                f"Unable to open camera at index {self._settings.index}. "
                "Check that a webcam is connected and not in use by another app."
            )

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._settings.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._settings.height)
        capture.set(cv2.CAP_PROP_FPS, self._settings.fps_target)

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if actual_width <= 0 or actual_height <= 0:
            capture.release()
            raise CameraError("Camera opened but returned invalid frame dimensions.")

        self._capture = capture
        self._camera_info = CameraInfo(
            index=self._settings.index,
            width=actual_width,
            height=actual_height,
            fps_target=self._settings.fps_target,
            backend="DirectShow",
        )

        self._stop_event.clear()
        self._frame_count = 0
        self._fps = 0.0
        self._fps_start_time = time.perf_counter()
        self._fps_frame_count = 0

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureThread",
            daemon=True,
        )
        self._thread.start()
        self._is_running = True

        self._logger.info(
            "Camera started: %sx%s (target FPS: %s)",
            actual_width,
            actual_height,
            self._settings.fps_target,
        )

    def stop(self) -> None:
        """Stop the capture thread and release the camera resource."""
        if not self._is_running:
            return

        self._logger.info("Stopping camera...")
        self._stop_event.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        with self._frame_lock:
            self._latest_frame = None

        self._thread = None
        self._is_running = False
        self._camera_info = None
        self._logger.info("Camera stopped.")

    def read_frame(self) -> Frame | None:
        """
        Return a copy of the most recent frame.

        Returns:
            Latest BGR frame as a NumPy array, or None if no frame is available.
        """
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def _capture_loop(self) -> None:
        """Continuously read frames from the camera in a background thread."""
        assert self._capture is not None

        while not self._stop_event.is_set():
            success, frame = self._capture.read()

            if not success or frame is None:
                self._logger.warning("Failed to read frame from camera.")
                time.sleep(0.01)
                continue

            with self._frame_lock:
                self._latest_frame = frame

            self._frame_count += 1
            self._update_fps()

    def _update_fps(self) -> None:
        """Calculate rolling FPS based on frames captured in the last second."""
        self._fps_frame_count += 1
        elapsed = time.perf_counter() - self._fps_start_time

        if elapsed >= 1.0:
            measured_fps = self._fps_frame_count / elapsed
            with self._fps_lock:
                self._fps = measured_fps
            self._fps_frame_count = 0
            self._fps_start_time = time.perf_counter()

    def __enter__(self) -> CameraCapture:
        """Enter context manager and start capture."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and stop capture."""
        self.stop()
