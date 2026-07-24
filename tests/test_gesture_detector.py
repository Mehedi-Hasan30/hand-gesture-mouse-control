"""Unit tests for gesture detection."""

from __future__ import annotations

import pytest

from src.config import GestureSettings
from src.gesture_detector import GestureDetector, GestureType
from src.hand_tracker import HandData, NormalizedLandmark


@pytest.fixture
def gesture_settings() -> GestureSettings:
    """Return default gesture settings for tests."""
    return GestureSettings(
        pinch_threshold=0.05,
        drag_hold_ms=250,
        double_click_interval_ms=400,
        scroll_sensitivity=800.0,
        scroll_dead_zone=0.01,
    )


def _make_hand(
    index_tip: tuple[float, float] = (0.5, 0.5),
    thumb_tip: tuple[float, float] = (0.52, 0.52),
    middle_tip: tuple[float, float] = (0.6, 0.3),
) -> HandData:
    """Build hand data with configurable fingertip positions."""
    landmarks: list[NormalizedLandmark] = []
    for index in range(21):
        if index == 4:
            x, y = thumb_tip
        elif index == 8:
            x, y = index_tip
        elif index == 12:
            x, y = middle_tip
        elif index in (6, 10, 14, 18):
            x, y = 0.5, 0.55
        elif index in (5, 9, 13, 17):
            x, y = 0.5, 0.6
        elif index == 0:
            x, y = 0.5, 0.8
        else:
            x, y = 0.5, 0.5
        landmarks.append(NormalizedLandmark(x=x, y=y, z=0.0))

    pixel = tuple((int(x * 640), int(y * 480)) for x, y in ((lm.x, lm.y) for lm in landmarks))
    return HandData(landmarks=tuple(landmarks), pixel_landmarks=pixel, handedness="Right")


def test_no_hand_returns_none_gesture(gesture_settings) -> None:
    """Missing hand should reset gesture state."""
    detector = GestureDetector(gesture_settings)
    detector.detect(_make_hand())
    result = detector.detect(None)

    assert result.gesture == GestureType.NONE
    assert result.index_pinch_active is False


def test_index_pinch_detected(gesture_settings) -> None:
    """Index and thumb close together should register as left pinch."""
    detector = GestureDetector(gesture_settings)
    hand = _make_hand(index_tip=(0.50, 0.50), thumb_tip=(0.51, 0.51))
    result = detector.detect(hand)

    assert result.index_pinch_active is True
    assert result.gesture == GestureType.LEFT_PINCH


def test_middle_pinch_detected(gesture_settings) -> None:
    """Middle and thumb close together should register as right pinch."""
    detector = GestureDetector(gesture_settings)
    hand = _make_hand(
        index_tip=(0.3, 0.3),
        thumb_tip=(0.52, 0.52),
        middle_tip=(0.51, 0.51),
    )
    result = detector.detect(hand)

    assert result.middle_pinch_active is True
    assert result.gesture == GestureType.RIGHT_PINCH


def test_pinch_press_and_release_edges(gesture_settings) -> None:
    """Pinch start and release should produce edge events."""
    detector = GestureDetector(gesture_settings)
    open_hand = _make_hand(index_tip=(0.3, 0.3), thumb_tip=(0.7, 0.7))
    pinch_hand = _make_hand(index_tip=(0.5, 0.5), thumb_tip=(0.51, 0.51))

    idle = detector.detect(open_hand)
    assert idle.index_pinch_pressed is False

    pressed = detector.detect(pinch_hand)
    assert pressed.index_pinch_pressed is True

    released = detector.detect(open_hand)
    assert released.index_pinch_released is True


def test_scroll_pose_detection(gesture_settings) -> None:
    """Index and middle extended with ring/pinky down should enable scroll mode."""
    landmarks: list[NormalizedLandmark] = []
    for index in range(21):
        if index in (8, 12):
            landmarks.append(NormalizedLandmark(x=0.5, y=0.2, z=0.0))
        elif index in (6, 10):
            landmarks.append(NormalizedLandmark(x=0.5, y=0.35, z=0.0))
        elif index in (5, 9):
            landmarks.append(NormalizedLandmark(x=0.5, y=0.45, z=0.0))
        elif index in (16, 20):
            landmarks.append(NormalizedLandmark(x=0.5, y=0.55, z=0.0))
        elif index in (15, 19):
            landmarks.append(NormalizedLandmark(x=0.5, y=0.5, z=0.0))
        elif index == 0:
            landmarks.append(NormalizedLandmark(x=0.5, y=0.8, z=0.0))
        else:
            landmarks.append(NormalizedLandmark(x=0.5, y=0.5, z=0.0))

    pixel = tuple((320, 240) for _ in landmarks)
    hand = HandData(landmarks=tuple(landmarks), pixel_landmarks=pixel, handedness="Right")

    detector = GestureDetector(gesture_settings)
    first = detector.detect(hand)
    hand_moved = HandData(
        landmarks=tuple(
            NormalizedLandmark(x=lm.x, y=lm.y + 0.05, z=lm.z) for lm in hand.landmarks
        ),
        pixel_landmarks=hand.pixel_landmarks,
        handedness="Right",
    )
    second = detector.detect(hand_moved)

    assert first.gesture == GestureType.SCROLL
    assert second.gesture == GestureType.SCROLL
