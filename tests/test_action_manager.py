"""Unit tests for gesture action execution."""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from src.action_manager import ActionManager
from src.gesture_detector import GestureDetectionResult, GestureType


@pytest.fixture
def logger() -> logging.Logger:
    """Return a simple logger for tests."""
    return logging.getLogger("test.action_manager")


def _detection(**kwargs) -> GestureDetectionResult:
    """Build a gesture detection result with defaults."""
    defaults = {
        "gesture": GestureType.NONE,
        "index_pinch_active": False,
        "middle_pinch_active": False,
        "index_pinch_pressed": False,
        "index_pinch_released": False,
        "middle_pinch_pressed": False,
        "scroll_delta_y": 0.0,
        "scroll_amount": 0,
        "double_click_triggered": False,
    }
    defaults.update(kwargs)
    return GestureDetectionResult(**defaults)


@patch("src.action_manager.pyautogui.click")
def test_left_click_on_pinch_release(mock_click, logger) -> None:
    """Releasing an index pinch should trigger a left click."""
    manager = ActionManager(logger=logger)
    manager.process(_detection(index_pinch_released=True), mouse_enabled=True)
    mock_click.assert_called_once_with(button="left")


@patch("src.action_manager.pyautogui.click")
def test_right_click_on_middle_pinch(mock_click, logger) -> None:
    """Middle pinch press should trigger a right click."""
    manager = ActionManager(logger=logger)
    manager.process(_detection(middle_pinch_pressed=True), mouse_enabled=True)
    mock_click.assert_called_once_with(button="right")


@patch("src.action_manager.pyautogui.doubleClick")
def test_double_click_trigger(mock_double_click, logger) -> None:
    """Double click flag should trigger pyautogui.doubleClick."""
    manager = ActionManager(logger=logger)
    manager.process(_detection(double_click_triggered=True), mouse_enabled=True)
    mock_double_click.assert_called_once_with(button="left")


@patch("src.action_manager.pyautogui.mouseUp")
@patch("src.action_manager.pyautogui.mouseDown")
def test_drag_lifecycle(mock_down, mock_up, logger) -> None:
    """Drag gesture should press on start and release on pinch end."""
    manager = ActionManager(logger=logger)
    manager.process(_detection(gesture=GestureType.DRAG), mouse_enabled=True)
    mock_down.assert_called_once_with(button="left")

    manager.process(_detection(index_pinch_released=True), mouse_enabled=True)
    mock_up.assert_called_once_with(button="left")


@patch("src.action_manager.pyautogui.scroll")
def test_scroll_action(mock_scroll, logger) -> None:
    """Non-zero scroll amount should call pyautogui.scroll."""
    manager = ActionManager(logger=logger)
    manager.process(_detection(gesture=GestureType.SCROLL, scroll_amount=2), mouse_enabled=True)
    mock_scroll.assert_called_once_with(2)


@patch("src.action_manager.pyautogui.click")
def test_no_actions_when_mouse_disabled(mock_click, logger) -> None:
    """Gestures should be ignored when mouse control is disabled."""
    manager = ActionManager(logger=logger)
    manager.process(_detection(index_pinch_released=True), mouse_enabled=False)
    mock_click.assert_not_called()
