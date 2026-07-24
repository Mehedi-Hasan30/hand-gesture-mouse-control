"""Hand gesture detection from landmark data."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum

from src.config import GestureSettings
from src.hand_tracker import HandData, HandLandmarkIndex, NormalizedLandmark


class GestureType(str, Enum):
    """Supported hand gestures that map to mouse actions."""

    NONE = "none"
    LEFT_PINCH = "left_pinch"
    RIGHT_PINCH = "right_pinch"
    DRAG = "drag"
    SCROLL = "scroll"


@dataclass(frozen=True)
class GestureDetectionResult:
    """Output of gesture detection for a single frame."""

    gesture: GestureType
    index_pinch_active: bool
    middle_pinch_active: bool
    index_pinch_pressed: bool
    index_pinch_released: bool
    middle_pinch_pressed: bool
    scroll_delta_y: float
    scroll_amount: int
    double_click_triggered: bool


class GestureDetector:
    """
    Detect mouse-control gestures from hand landmark positions.

    Uses finger pinch distances and finger extension states to recognize
    left click, right click, double click, drag, and scroll gestures.
    """

    def __init__(self, settings: GestureSettings) -> None:
        """
        Initialize the gesture detector.

        Args:
            settings: Gesture detection thresholds and timing configuration.
        """
        self._settings = settings
        self._index_pinch_active = False
        self._middle_pinch_active = False
        self._pinch_start_time: float | None = None
        self._last_click_time: float | None = None
        self._dragging = False
        self._scroll_mode = False
        self._last_scroll_y: float | None = None
        self._accumulated_scroll_delta = 0.0

    @property
    def is_dragging(self) -> bool:
        """Return True if a drag operation is in progress."""
        return self._dragging

    def reset(self) -> None:
        """Reset all internal gesture tracking state."""
        self._index_pinch_active = False
        self._middle_pinch_active = False
        self._pinch_start_time = None
        self._last_click_time = None
        self._dragging = False
        self._scroll_mode = False
        self._last_scroll_y = None
        self._accumulated_scroll_delta = 0.0

    def update_settings(self, settings: GestureSettings) -> None:
        """
        Apply updated gesture detection settings at runtime.

        Args:
            settings: New gesture configuration values.
        """
        self._settings = settings

    def detect(self, hand: HandData | None) -> GestureDetectionResult:
        """
        Analyze hand landmarks and return gesture events for the current frame.

        Args:
            hand: Detected hand data, or None if no hand is visible.

        Returns:
            GestureDetectionResult with active gesture and edge-triggered events.
        """
        if hand is None:
            release = self._index_pinch_active and self._dragging
            self.reset()
            return GestureDetectionResult(
                gesture=GestureType.NONE,
                index_pinch_active=False,
                middle_pinch_active=False,
                index_pinch_pressed=False,
                index_pinch_released=release,
                middle_pinch_pressed=False,
                scroll_delta_y=0.0,
                scroll_amount=0,
                double_click_triggered=False,
            )

        index_pinch = self._is_pinching(
            hand.landmarks[HandLandmarkIndex.INDEX_FINGER_TIP],
            hand.landmarks[HandLandmarkIndex.THUMB_TIP],
        )
        middle_pinch = self._is_pinching(
            hand.landmarks[HandLandmarkIndex.MIDDLE_FINGER_TIP],
            hand.landmarks[HandLandmarkIndex.THUMB_TIP],
        )
        scroll_mode = self._is_scroll_pose(hand)

        index_pressed = index_pinch and not self._index_pinch_active
        index_released = (not index_pinch) and self._index_pinch_active
        middle_pressed = middle_pinch and not self._middle_pinch_active and not index_pinch

        double_click_triggered = False
        if index_pressed:
            self._pinch_start_time = time.perf_counter()

        if index_released:
            held_ms = self._pinch_duration_ms()
            if not self._dragging and held_ms < self._settings.drag_hold_ms:
                if self._last_click_time is not None and self._is_within_double_click_window():
                    double_click_triggered = True
                    self._last_click_time = None
                else:
                    self._last_click_time = time.perf_counter()
            if self._dragging:
                self._dragging = False
            self._pinch_start_time = None

        if index_pinch and not self._dragging and not double_click_triggered:
            if self._pinch_duration_ms() >= self._settings.drag_hold_ms:
                self._dragging = True
                self._last_click_time = None

        scroll_delta_y = 0.0
        scroll_amount = 0
        if scroll_mode and not index_pinch and not middle_pinch:
            scroll_delta_y, scroll_amount = self._calculate_scroll(hand)
            active_gesture = GestureType.SCROLL
        elif self._dragging and index_pinch:
            active_gesture = GestureType.DRAG
        elif index_pinch:
            active_gesture = GestureType.LEFT_PINCH
        elif middle_pinch:
            active_gesture = GestureType.RIGHT_PINCH
        else:
            active_gesture = GestureType.NONE

        self._index_pinch_active = index_pinch
        self._middle_pinch_active = middle_pinch
        self._scroll_mode = scroll_mode

        return GestureDetectionResult(
            gesture=active_gesture,
            index_pinch_active=index_pinch,
            middle_pinch_active=middle_pinch,
            index_pinch_pressed=index_pressed,
            index_pinch_released=index_released,
            middle_pinch_pressed=middle_pressed,
            scroll_delta_y=scroll_delta_y,
            scroll_amount=scroll_amount,
            double_click_triggered=double_click_triggered,
        )

    def _is_pinching(self, tip: NormalizedLandmark, thumb_tip: NormalizedLandmark) -> bool:
        """
        Determine whether a finger tip is pinching the thumb.

        Args:
            tip: Normalized finger tip landmark.
            thumb_tip: Normalized thumb tip landmark.

        Returns:
            True if fingertip and thumb are close enough to count as a pinch.
        """
        distance = self._landmark_distance(tip, thumb_tip)
        return distance < self._settings.pinch_threshold

    def _is_scroll_pose(self, hand: HandData) -> bool:
        """
        Detect the two-finger scroll pose (index and middle extended, others folded).

        Args:
            hand: Detected hand data.

        Returns:
            True if the hand is in scroll-ready pose.
        """
        index_up = self._is_finger_extended(
            hand,
            HandLandmarkIndex.INDEX_FINGER_TIP,
            HandLandmarkIndex.INDEX_FINGER_PIP,
        )
        middle_up = self._is_finger_extended(
            hand,
            HandLandmarkIndex.MIDDLE_FINGER_TIP,
            HandLandmarkIndex.MIDDLE_FINGER_PIP,
        )
        ring_down = not self._is_finger_extended(
            hand,
            HandLandmarkIndex.RING_FINGER_TIP,
            HandLandmarkIndex.RING_FINGER_PIP,
        )
        pinky_down = not self._is_finger_extended(
            hand,
            HandLandmarkIndex.PINKY_TIP,
            HandLandmarkIndex.PINKY_PIP,
        )
        return index_up and middle_up and ring_down and pinky_down

    def _calculate_scroll(self, hand: HandData) -> tuple[float, int]:
        """
        Calculate vertical scroll delta from index finger movement.

        Args:
            hand: Detected hand data in scroll pose.

        Returns:
            Tuple of normalized Y delta and discrete scroll amount.
        """
        current_y = hand.index_finger_tip.y
        if self._last_scroll_y is None:
            self._last_scroll_y = current_y
            return 0.0, 0

        delta_y = current_y - self._last_scroll_y
        self._last_scroll_y = current_y

        if abs(delta_y) < self._settings.scroll_dead_zone:
            return delta_y, 0

        self._accumulated_scroll_delta += delta_y
        scroll_threshold = 1.0 / self._settings.scroll_sensitivity
        scroll_amount = 0

        while self._accumulated_scroll_delta <= -scroll_threshold:
            scroll_amount += 1
            self._accumulated_scroll_delta += scroll_threshold
        while self._accumulated_scroll_delta >= scroll_threshold:
            scroll_amount -= 1
            self._accumulated_scroll_delta -= scroll_threshold

        return delta_y, scroll_amount

    def _pinch_duration_ms(self) -> float:
        """
        Return how long the current index pinch has been held.

        Returns:
            Duration in milliseconds, or zero if not pinching.
        """
        if self._pinch_start_time is None:
            return 0.0
        return (time.perf_counter() - self._pinch_start_time) * 1000.0

    def _is_within_double_click_window(self) -> bool:
        """
        Check whether the last click falls within the double-click interval.

        Returns:
            True if a double click should be triggered.
        """
        if self._last_click_time is None:
            return False
        elapsed_ms = (time.perf_counter() - self._last_click_time) * 1000.0
        return elapsed_ms <= self._settings.double_click_interval_ms

    @staticmethod
    def _is_finger_extended(
        hand: HandData,
        tip_index: HandLandmarkIndex,
        pip_index: HandLandmarkIndex,
    ) -> bool:
        """
        Determine whether a finger is extended based on tip-to-wrist distance.

        Args:
            hand: Detected hand data.
            tip_index: Landmark index of the finger tip.
            pip_index: Landmark index of the finger PIP joint.

        Returns:
            True if the finger appears extended.
        """
        tip = hand.landmarks[tip_index]
        pip = hand.landmarks[pip_index]
        wrist = hand.wrist
        tip_distance = GestureDetector._landmark_distance(tip, wrist)
        pip_distance = GestureDetector._landmark_distance(pip, wrist)
        return tip_distance > pip_distance * 1.05

    @staticmethod
    def _landmark_distance(
        first: NormalizedLandmark,
        second: NormalizedLandmark,
    ) -> float:
        """
        Compute Euclidean distance between two normalized landmarks.

        Args:
            first: First landmark.
            second: Second landmark.

        Returns:
            Distance in normalized coordinate space.
        """
        delta_x = first.x - second.x
        delta_y = first.y - second.y
        return math.sqrt((delta_x ** 2) + (delta_y ** 2))
