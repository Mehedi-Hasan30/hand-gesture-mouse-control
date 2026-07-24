"""Cursor position smoothing utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass

from src.config import MouseSettings


@dataclass
class ScreenPoint:
    """A point in normalized (0-1) or screen coordinate space."""

    x: float
    y: float


class SmoothingFilter:
    """
    Exponential moving average filter for 2D cursor coordinates.

    Reduces jitter from hand tracking by blending new positions with
    the previous smoothed value. A dead zone ignores micro-movements.
    """

    def __init__(self, settings: MouseSettings) -> None:
        """
        Initialize the smoothing filter.

        Args:
            settings: Mouse settings containing smoothing factor and dead zone.
        """
        self._smoothing_factor = self._clamp(settings.smoothing_factor, 0.05, 1.0)
        self._dead_zone = max(settings.dead_zone, 0.0)
        self._current: ScreenPoint | None = None

    @property
    def is_initialized(self) -> bool:
        """Return True if at least one point has been processed."""
        return self._current is not None

    def reset(self) -> None:
        """Clear smoothed state, used when hand tracking is lost."""
        self._current = None

    def update(self, target: ScreenPoint) -> ScreenPoint:
        """
        Apply smoothing to a new target point.

        Args:
            target: New normalized cursor target (x and y in range 0-1).

        Returns:
            Smoothed normalized cursor point.
        """
        if self._current is None:
            self._current = ScreenPoint(x=target.x, y=target.y)
            return ScreenPoint(x=self._current.x, y=self._current.y)

        if self._is_within_dead_zone(target):
            return ScreenPoint(x=self._current.x, y=self._current.y)

        alpha = self._smoothing_factor
        smoothed_x = (alpha * target.x) + ((1.0 - alpha) * self._current.x)
        smoothed_y = (alpha * target.y) + ((1.0 - alpha) * self._current.y)

        self._current = ScreenPoint(x=smoothed_x, y=smoothed_y)
        return ScreenPoint(x=smoothed_x, y=smoothed_y)

    def update_settings(self, settings: MouseSettings) -> None:
        """
        Apply updated smoothing parameters at runtime.

        Args:
            settings: New mouse settings containing smoothing values.
        """
        self._smoothing_factor = self._clamp(settings.smoothing_factor, 0.05, 1.0)
        self._dead_zone = max(settings.dead_zone, 0.0)

    def _is_within_dead_zone(self, target: ScreenPoint) -> bool:
        """
        Check whether movement is smaller than the configured dead zone.

        Args:
            target: New target point in normalized coordinates.

        Returns:
            True if movement should be ignored.
        """
        if self._current is None or self._dead_zone <= 0.0:
            return False

        delta_x = target.x - self._current.x
        delta_y = target.y - self._current.y
        distance = math.sqrt((delta_x ** 2) + (delta_y ** 2))
        return distance < self._dead_zone

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
