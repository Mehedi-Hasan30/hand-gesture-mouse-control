"""Execute mouse actions triggered by hand gestures."""

from __future__ import annotations

import logging

import pyautogui

from src.gesture_detector import GestureDetectionResult, GestureType

pyautogui.PAUSE = 0


class ActionManager:
    """
    Translate detected gestures into PyAutoGUI mouse actions.

    Handles left/right click, double click, drag, and scroll while
    coordinating with cursor movement from MouseController.
    """

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """
        Initialize the action manager.

        Args:
            logger: Optional logger instance. Uses module logger if not provided.
        """
        self._logger = logger or logging.getLogger("hand_gesture_mouse.actions")
        self._drag_active = False
        self._last_gesture = GestureType.NONE

    @property
    def is_dragging(self) -> bool:
        """Return True if the left mouse button is held for dragging."""
        return self._drag_active

    @property
    def last_gesture(self) -> GestureType:
        """Return the most recently processed gesture type."""
        return self._last_gesture

    def reset(self) -> None:
        """Release any held mouse buttons and reset internal state."""
        if self._drag_active:
            pyautogui.mouseUp(button="left")
            self._logger.info("Drag cancelled on reset.")
        self._drag_active = False
        self._last_gesture = GestureType.NONE

    def process(
        self,
        detection: GestureDetectionResult,
        mouse_enabled: bool,
    ) -> None:
        """
        Process gesture detection results and execute matching mouse actions.

        Args:
            detection: Gesture detection output for the current frame.
            mouse_enabled: Whether mouse control is currently enabled.
        """
        if not mouse_enabled:
            if self._drag_active:
                self.reset()
            self._last_gesture = GestureType.NONE
            return

        self._last_gesture = detection.gesture

        if detection.double_click_triggered:
            self._perform_double_click()
            return

        if detection.index_pinch_pressed and not detection.double_click_triggered:
            self._logger.debug("Index pinch started.")

        if detection.middle_pinch_pressed:
            self._perform_right_click()
            return

        if detection.gesture == GestureType.DRAG:
            if not self._drag_active:
                pyautogui.mouseDown(button="left")
                self._drag_active = True
                self._logger.info("Drag started.")
            return

        if self._drag_active and detection.index_pinch_released:
            pyautogui.mouseUp(button="left")
            self._drag_active = False
            self._logger.info("Drag ended.")
            return

        if detection.index_pinch_released and not self._drag_active:
            self._perform_left_click()
            return

        if detection.scroll_amount != 0:
            pyautogui.scroll(detection.scroll_amount)
            direction = "up" if detection.scroll_amount > 0 else "down"
            self._logger.debug("Scroll %s by %s.", direction, abs(detection.scroll_amount))

    def _perform_left_click(self) -> None:
        """Execute a single left mouse click."""
        pyautogui.click(button="left")
        self._logger.info("Left click.")

    def _perform_right_click(self) -> None:
        """Execute a single right mouse click."""
        pyautogui.click(button="right")
        self._logger.info("Right click.")

    def _perform_double_click(self) -> None:
        """Execute a double left mouse click."""
        pyautogui.doubleClick(button="left")
        self._logger.info("Double click.")
