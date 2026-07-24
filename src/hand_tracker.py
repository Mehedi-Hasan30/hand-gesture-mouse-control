"""Hand tracking module using MediaPipe Hands."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as base_options_module
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils
from mediapipe.tasks.python.vision.core import vision_task_running_mode as running_mode_module
from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
from numpy.typing import NDArray

from src.config import HandTrackingSettings
from src.utils import ensure_hand_landmarker_model

Frame = NDArray[np.uint8]


class HandLandmarkIndex(IntEnum):
    """MediaPipe hand landmark indices (21 points per hand)."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass(frozen=True)
class NormalizedLandmark:
    """A single hand landmark in normalized image coordinates."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class HandData:
    """Detected hand with normalized and pixel landmark coordinates."""

    landmarks: tuple[NormalizedLandmark, ...]
    pixel_landmarks: tuple[tuple[int, int], ...]
    handedness: str

    @property
    def index_finger_tip(self) -> NormalizedLandmark:
        """Return the normalized index finger tip landmark."""
        return self.landmarks[HandLandmarkIndex.INDEX_FINGER_TIP]

    @property
    def wrist(self) -> NormalizedLandmark:
        """Return the normalized wrist landmark."""
        return self.landmarks[HandLandmarkIndex.WRIST]


@dataclass(frozen=True)
class HandTrackingResult:
    """Result of hand detection on a single frame."""

    hands: tuple[HandData, ...]

    @property
    def primary_hand(self) -> HandData | None:
        """Return the first detected hand, used for mouse control in later phases."""
        if not self.hands:
            return None
        return self.hands[0]

    @property
    def hand_count(self) -> int:
        """Return the number of detected hands."""
        return len(self.hands)


class HandTracker:
    """
    Detect and track hand landmarks using the MediaPipe Hand Landmarker task.

    Uses the MediaPipe Tasks API (compatible with mediapipe >= 0.10.14).
    Converts BGR frames to RGB, returns normalized landmark coordinates,
    and provides drawing utilities for the preview overlay.
    """

    def __init__(
        self,
        settings: HandTrackingSettings,
        project_root: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the MediaPipe hand tracker.

        Args:
            settings: Hand tracking configuration from application settings.
            project_root: Project root path used to locate or download the model.
            logger: Optional logger instance. Uses module logger if not provided.

        Raises:
            RuntimeError: If the hand landmarker model cannot be loaded.
        """
        self._settings = settings
        self._logger = logger or logging.getLogger("hand_gesture_mouse.hand_tracker")
        self._timestamp_ms = 0
        self._last_frame_time = time.perf_counter()

        model_path = ensure_hand_landmarker_model(project_root, logger=self._logger)
        options = vision.HandLandmarkerOptions(
            base_options=base_options_module.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode_module.VisionTaskRunningMode.VIDEO,
            num_hands=settings.max_hands,
            min_hand_detection_confidence=settings.min_detection_confidence,
            min_hand_presence_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

        self._logger.info(
            "Hand tracker initialized (max_hands=%s, detection=%.2f, tracking=%.2f)",
            settings.max_hands,
            settings.min_detection_confidence,
            settings.min_tracking_confidence,
        )

    def process(self, frame: Frame) -> HandTrackingResult:
        """
        Detect hands and extract landmark coordinates from a BGR frame.

        Args:
            frame: BGR image from the webcam.

        Returns:
            HandTrackingResult containing zero or more detected hands.
        """
        height, width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self._timestamp_ms = self._next_timestamp_ms()
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not results.hand_landmarks:
            return HandTrackingResult(hands=tuple())

        hands: list[HandData] = []
        for index, hand_landmarks in enumerate(results.hand_landmarks):
            normalized, pixel = self._convert_landmarks(hand_landmarks, width, height)
            label = self._extract_handedness_label(results, index)
            hands.append(
                HandData(
                    landmarks=normalized,
                    pixel_landmarks=pixel,
                    handedness=label,
                )
            )

        return HandTrackingResult(hands=tuple(hands))

    def draw(self, frame: Frame, result: HandTrackingResult) -> None:
        """
        Draw hand landmarks and connections on the frame.

        Args:
            frame: BGR image to annotate in place.
            result: Hand tracking result from process().
        """
        for hand in result.hands:
            landmark_list = [
                mp.tasks.components.containers.NormalizedLandmark(
                    x=point.x,
                    y=point.y,
                    z=point.z,
                )
                for point in hand.landmarks
            ]
            drawing_utils.draw_landmarks(
                frame,
                landmark_list,
                HandLandmarksConnections.HAND_CONNECTIONS,
                drawing_styles.get_default_hand_landmarks_style(),
                drawing_styles.get_default_hand_connections_style(),
            )
            self._draw_handedness_label(frame, hand)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()
        self._logger.info("Hand tracker closed.")

    def _next_timestamp_ms(self) -> int:
        """
        Calculate monotonically increasing timestamp for video mode detection.

        Returns:
            Timestamp in milliseconds for MediaPipe video processing.
        """
        now = time.perf_counter()
        elapsed_ms = int((now - self._last_frame_time) * 1000)
        self._last_frame_time = now
        self._timestamp_ms += max(elapsed_ms, 1)
        return self._timestamp_ms

    @staticmethod
    def _extract_handedness_label(results, index: int) -> str:
        """
        Extract the left/right label for a detected hand.

        Args:
            results: Raw MediaPipe hand landmarker result.
            index: Index of the hand in the result list.

        Returns:
            Handedness label string, or "Unknown" if unavailable.
        """
        if not results.handedness or index >= len(results.handedness):
            return "Unknown"
        categories = results.handedness[index]
        if not categories:
            return "Unknown"
        return categories[0].category_name

    @staticmethod
    def _convert_landmarks(
        hand_landmarks,
        width: int,
        height: int,
    ) -> tuple[tuple[NormalizedLandmark, ...], tuple[tuple[int, int], ...]]:
        """
        Convert MediaPipe landmarks to normalized and pixel coordinates.

        Args:
            hand_landmarks: MediaPipe normalized landmark list.
            width: Frame width in pixels.
            height: Frame height in pixels.

        Returns:
            Tuple of normalized landmarks and pixel coordinate tuples.
        """
        normalized: list[NormalizedLandmark] = []
        pixel: list[tuple[int, int]] = []

        for landmark in hand_landmarks:
            normalized.append(
                NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            )
            pixel.append((int(landmark.x * width), int(landmark.y * height)))

        return tuple(normalized), tuple(pixel)

    @staticmethod
    def _draw_handedness_label(frame: Frame, hand: HandData) -> None:
        """
        Draw the hand label (Left/Right) near the wrist landmark.

        Args:
            frame: BGR image to annotate in place.
            hand: Detected hand data.
        """
        if not hand.pixel_landmarks:
            return

        wrist_x, wrist_y = hand.pixel_landmarks[HandLandmarkIndex.WRIST]
        label_position = (wrist_x - 30, wrist_y - 20)
        cv2.putText(
            frame,
            hand.handedness,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def __enter__(self) -> HandTracker:
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and release resources."""
        self.close()
