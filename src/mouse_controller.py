"""Mouse cursor control using hand position input."""

from __future__ import annotations

import logging

import pyautogui

from src.config import MouseSettings
from src.hand_tracker import HandData, NormalizedLandmark
from src.smoothing import ScreenPoint, SmoothingFilter

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0


class MouseController:
    """
    Maps hand landmarks to screen cursor movement.

    Uses the index finger tip as the control point, applies exponential
    smoothing, and moves the system cursor via PyAutoGUI.
    """

    def __init__(
        self,
        settings: MouseSettings,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the mouse controller.

        Args:
            settings: Mouse movement configuration.
            logger: Optional logger instance. Uses module logger if not provided.
        """
        self._settings = settings
        self._logger = logger or logging.getLogger("hand_gesture_mouse.mouse")
        self._smoothing_filter = SmoothingFilter(settings)
        self._enabled = False
        self._screen_width, self._screen_height = pyautogui.size()
        self._last_screen_position: tuple[int, int] | None = None

        self._logger.info(
            "Mouse controller initialized (screen: %sx%s, sensitivity=%.2f)",
            self._screen_width,
            self._screen_height,
            settings.sensitivity,
        )

    @property
    def is_enabled(self) -> bool:
        """Return True if cursor movement is active."""
        return self._enabled

    @property
    def screen_size(self) -> tuple[int, int]:
        """Return the current screen width and height."""
        return self._screen_width, self._screen_height

    @property
    def last_screen_position(self) -> tuple[int, int] | None:
        """Return the last cursor position sent to the screen."""
        return self._last_screen_position

    def enable(self) -> None:
        """Enable cursor movement."""
        if not self._enabled:
            self._logger.info("Mouse control enabled.")
        self._enabled = True

    def disable(self) -> None:
        """Disable cursor movement and reset smoothing state."""
        if self._enabled:
            self._logger.info("Mouse control disabled.")
        self._enabled = False
        self._smoothing_filter.reset()
        self._last_screen_position = None

    def toggle(self) -> bool:
        """
        Toggle cursor movement on or off.

        Returns:
            New enabled state after toggling.
        """
        if self._enabled:
            self.disable()
        else:
            self.enable()
        return self._enabled

    def update_from_hand(self, hand: HandData | None) -> tuple[int, int] | None:
        """
        Update cursor position based on detected hand landmarks.

        Args:
            hand: Detected hand data, or None if no hand is visible.

        Returns:
            Screen coordinates (x, y) if the cursor moved, otherwise None.
        """
        if not self._enabled:
            return None

        if hand is None:
            self._smoothing_filter.reset()
            self._last_screen_position = None
            return None

        target = self._landmark_to_normalized_target(hand.index_finger_tip)
        smoothed = self._smoothing_filter.update(target)
        screen_x, screen_y = self._normalized_to_screen(smoothed)

        if self._last_screen_position == (screen_x, screen_y):
            return self._last_screen_position

        pyautogui.moveTo(screen_x, screen_y)
        self._last_screen_position = (screen_x, screen_y)
        return self._last_screen_position

    def update_settings(self, settings: MouseSettings) -> None:
        """
        Apply updated mouse settings at runtime.

        Args:
            settings: New mouse movement configuration.
        """
        self._settings = settings
        self._smoothing_filter.update_settings(settings)

    def refresh_screen_size(self) -> None:
        """Refresh cached screen dimensions (useful for multi-monitor setups later)."""
        self._screen_width, self._screen_height = pyautogui.size()
        self._logger.info(
            "Screen size refreshed: %sx%s",
            self._screen_width,
            self._screen_height,
        )

    def _landmark_to_normalized_target(self, landmark: NormalizedLandmark) -> ScreenPoint:
        """
        Convert a normalized landmark to a mirrored, sensitivity-adjusted target.

        The x-axis is mirrored so movement feels natural (like a mirror).
        Sensitivity scales movement relative to the frame center.

        Args:
            landmark: Normalized index finger tip landmark.

        Returns:
            Target point in normalized 0-1 coordinate space.
        """
        mirrored_x = 1.0 - landmark.x
        adjusted_x = self._apply_sensitivity(mirrored_x)
        adjusted_y = self._apply_sensitivity(landmark.y)
        return ScreenPoint(
            x=self._clamp(adjusted_x, 0.0, 1.0),
            y=self._clamp(adjusted_y, 0.0, 1.0),
        )

    def _apply_sensitivity(self, value: float) -> float:
        """
        Scale a normalized coordinate based on sensitivity setting.

        Args:
            value: Normalized coordinate centered around 0.5.

        Returns:
            Sensitivity-adjusted coordinate.
        """
        sensitivity = max(self._settings.sensitivity, 0.1)
        return 0.5 + ((value - 0.5) * sensitivity)

    def _normalized_to_screen(self, point: ScreenPoint) -> tuple[int, int]:
        """
        Map normalized coordinates to pixel screen coordinates.

        Args:
            point: Normalized point in range 0-1.

        Returns:
            Screen pixel coordinates clamped to screen bounds.
        """
        screen_x = int(point.x * (self._screen_width - 1))
        screen_y = int(point.y * (self._screen_height - 1))
        screen_x = max(0, min(screen_x, self._screen_width - 1))
        screen_y = max(0, min(screen_y, self._screen_height - 1))
        return screen_x, screen_y

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        """
        Clamp a value between minimum and maximum bounds.

        Args:
            value: Input value.
            minimum: Lower bound.
            maximum: Upper bound.

        Returns:
            Clamped value.
        """
        return max(minimum, min(value, maximum))
