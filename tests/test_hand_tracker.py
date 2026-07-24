"""Unit tests for the hand tracking module."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.config import HandTrackingSettings
from src.hand_tracker import (
    HandData,
    HandLandmarkIndex,
    HandTracker,
    HandTrackingResult,
    NormalizedLandmark,
)


@pytest.fixture
def hand_settings() -> HandTrackingSettings:
    """Return default hand tracking settings for tests."""
    return HandTrackingSettings(
        max_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )


@pytest.fixture
def logger() -> logging.Logger:
    """Return a simple logger for tests."""
    return logging.getLogger("test.hand_tracker")


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Return a temporary project root with a fake model file."""
    model_dir = tmp_path / "assets" / "models"
    model_dir.mkdir(parents=True)
    (model_dir / "hand_landmarker.task").write_bytes(b"fake-model")
    return tmp_path


def test_hand_landmark_index_values() -> None:
    """Landmark indices should match MediaPipe's 21-point hand model."""
    assert HandLandmarkIndex.WRIST == 0
    assert HandLandmarkIndex.INDEX_FINGER_TIP == 8
    assert HandLandmarkIndex.PINKY_TIP == 20
    assert len(HandLandmarkIndex) == 21


def test_hand_tracking_result_primary_hand() -> None:
    """primary_hand should return the first detected hand or None."""
    landmarks = tuple(
        NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(21)
    )
    pixel = tuple((320, 240) for _ in range(21))
    hand = HandData(landmarks=landmarks, pixel_landmarks=pixel, handedness="Right")

    result_with_hand = HandTrackingResult(hands=(hand,))
    assert result_with_hand.primary_hand is hand
    assert result_with_hand.hand_count == 1

    result_empty = HandTrackingResult(hands=tuple())
    assert result_empty.primary_hand is None
    assert result_empty.hand_count == 0


def test_hand_data_index_finger_tip_property() -> None:
    """HandData should expose the index finger tip via a convenience property."""
    landmarks = tuple(
        NormalizedLandmark(x=i / 20, y=0.5, z=0.0) for i in range(21)
    )
    pixel = tuple((i * 10, 240) for i in range(21))
    hand = HandData(landmarks=landmarks, pixel_landmarks=pixel, handedness="Left")

    assert hand.index_finger_tip.x == pytest.approx(8 / 20)
    assert hand.wrist.x == 0.0


@patch("src.hand_tracker.vision.HandLandmarker.create_from_options")
@patch("src.hand_tracker.ensure_hand_landmarker_model")
def test_process_returns_empty_when_no_hands(
    mock_ensure_model,
    mock_create_landmarker,
    hand_settings,
    project_root,
    logger,
) -> None:
    """process() should return an empty result when no hands are detected."""
    mock_ensure_model.return_value = project_root / "assets" / "models" / "hand_landmarker.task"

    mock_landmarker = MagicMock()
    mock_results = MagicMock()
    mock_results.hand_landmarks = []
    mock_landmarker.detect_for_video.return_value = mock_results
    mock_create_landmarker.return_value = mock_landmarker

    tracker = HandTracker(hand_settings, project_root=project_root, logger=logger)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = tracker.process(frame)
    tracker.close()

    assert result.hand_count == 0
    assert result.primary_hand is None


@patch("src.hand_tracker.vision.HandLandmarker.create_from_options")
@patch("src.hand_tracker.ensure_hand_landmarker_model")
def test_process_returns_hand_data_when_detected(
    mock_ensure_model,
    mock_create_landmarker,
    hand_settings,
    project_root,
    logger,
) -> None:
    """process() should parse landmarks when MediaPipe detects a hand."""
    mock_ensure_model.return_value = project_root / "assets" / "models" / "hand_landmarker.task"

    mock_landmark = MagicMock()
    mock_landmark.x = 0.5
    mock_landmark.y = 0.5
    mock_landmark.z = 0.0

    mock_category = MagicMock()
    mock_category.category_name = "Right"

    mock_results = MagicMock()
    mock_results.hand_landmarks = [[mock_landmark] * 21]
    mock_results.handedness = [[mock_category]]

    mock_landmarker = MagicMock()
    mock_landmarker.detect_for_video.return_value = mock_results
    mock_create_landmarker.return_value = mock_landmarker

    tracker = HandTracker(hand_settings, project_root=project_root, logger=logger)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = tracker.process(frame)
    tracker.close()

    assert result.hand_count == 1
    assert result.primary_hand is not None
    assert result.primary_hand.handedness == "Right"
    assert len(result.primary_hand.landmarks) == 21
    assert result.primary_hand.pixel_landmarks[0] == (320, 240)
