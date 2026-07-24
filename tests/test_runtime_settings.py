"""Unit tests for runtime settings updates."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from src.config import GestureSettings, MouseSettings
from src.gesture_detector import GestureDetector
from src.mouse_controller import MouseController
from src.smoothing import SmoothingFilter


@pytest.fixture
def logger() -> logging.Logger:
    """Return a simple logger for tests."""
    return logging.getLogger("test.runtime_settings")


def test_smoothing_filter_update_settings() -> None:
    """Smoothing filter should accept updated parameters."""
    initial = MouseSettings(smoothing_factor=0.35, sensitivity=1.0, dead_zone=0.02)
    smoother = SmoothingFilter(initial)
    smoother.update_settings(MouseSettings(smoothing_factor=0.8, sensitivity=1.0, dead_zone=0.05))

    from src.smoothing import ScreenPoint

    smoother.update(ScreenPoint(x=0.0, y=0.0))
    result = smoother.update(ScreenPoint(x=1.0, y=1.0))
    assert result.x == pytest.approx(0.8)
    assert result.y == pytest.approx(0.8)


def test_gesture_detector_update_settings() -> None:
    """Gesture detector should use updated pinch threshold."""
    detector = GestureDetector(GestureSettings(pinch_threshold=0.05))
    detector.update_settings(GestureSettings(pinch_threshold=0.2))
    assert detector._settings.pinch_threshold == 0.2


@patch("src.mouse_controller.pyautogui.size", return_value=(1920, 1080))
def test_mouse_controller_update_settings(mock_size, logger) -> None:
    """Mouse controller should accept updated mouse settings."""
    controller = MouseController(
        MouseSettings(smoothing_factor=0.35, sensitivity=1.0, dead_zone=0.02),
        logger=logger,
    )
    controller.update_settings(
        MouseSettings(smoothing_factor=0.9, sensitivity=1.5, dead_zone=0.03)
    )
    assert controller._settings.sensitivity == 1.5
