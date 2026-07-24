"""Unit tests for mouse controller."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from src.config import MouseSettings
from src.hand_tracker import HandData, NormalizedLandmark
from src.mouse_controller import MouseController


@pytest.fixture
def mouse_settings() -> MouseSettings:
    """Return default mouse settings for tests."""
    return MouseSettings(smoothing_factor=1.0, sensitivity=1.0, dead_zone=0.0)


@pytest.fixture
def logger() -> logging.Logger:
    """Return a simple logger for tests."""
    return logging.getLogger("test.mouse_controller")


def _make_hand(x: float, y: float) -> HandData:
    """Create hand data with index finger tip at the given normalized position."""
    landmarks = tuple(NormalizedLandmark(x=x, y=y, z=0.0) for _ in range(21))
    pixel = tuple((int(x * 640), int(y * 480)) for _ in range(21))
    return HandData(landmarks=landmarks, pixel_landmarks=pixel, handedness="Right")


@patch("src.mouse_controller.pyautogui.size", return_value=(1920, 1080))
@patch("src.mouse_controller.pyautogui.moveTo")
def test_update_moves_cursor_when_enabled(
    mock_move_to,
    mock_size,
    mouse_settings,
    logger,
) -> None:
    """Cursor should move when control is enabled and a hand is detected."""
    controller = MouseController(mouse_settings, logger=logger)
    controller.enable()

    hand = _make_hand(x=0.25, y=0.5)
    position = controller.update_from_hand(hand)

    assert position is not None
    mock_move_to.assert_called_once()
    called_x, called_y = mock_move_to.call_args[0]
    assert called_x == int((1.0 - 0.25) * 1919)
    assert called_y == int(0.5 * 1079)


@patch("src.mouse_controller.pyautogui.size", return_value=(1920, 1080))
@patch("src.mouse_controller.pyautogui.moveTo")
def test_update_does_nothing_when_disabled(
    mock_move_to,
    mock_size,
    mouse_settings,
    logger,
) -> None:
    """Cursor should not move when control is disabled."""
    controller = MouseController(mouse_settings, logger=logger)
    hand = _make_hand(x=0.5, y=0.5)

    result = controller.update_from_hand(hand)

    assert result is None
    mock_move_to.assert_not_called()


@patch("src.mouse_controller.pyautogui.size", return_value=(1920, 1080))
@patch("src.mouse_controller.pyautogui.moveTo")
def test_toggle_enables_and_disables_control(
    mock_move_to,
    mock_size,
    mouse_settings,
    logger,
) -> None:
    """Toggle should switch enabled state."""
    controller = MouseController(mouse_settings, logger=logger)

    assert controller.toggle() is True
    assert controller.is_enabled is True

    assert controller.toggle() is False
    assert controller.is_enabled is False


@patch("src.mouse_controller.pyautogui.size", return_value=(1920, 1080))
@patch("src.mouse_controller.pyautogui.moveTo")
def test_hand_loss_resets_smoothing(
    mock_move_to,
    mock_size,
    mouse_settings,
    logger,
) -> None:
    """Passing None should reset controller state without moving the cursor."""
    controller = MouseController(mouse_settings, logger=logger)
    controller.enable()
    controller.update_from_hand(_make_hand(x=0.5, y=0.5))

    result = controller.update_from_hand(None)

    assert result is None
    assert controller.last_screen_position is None
