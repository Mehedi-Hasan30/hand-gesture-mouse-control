"""Unit tests for the webcam capture module."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.camera import CameraCapture, CameraError, CameraInfo
from src.config import CameraSettings


@pytest.fixture
def camera_settings() -> CameraSettings:
    """Return default camera settings for tests."""
    return CameraSettings(index=0, width=640, height=480, fps_target=30)


@pytest.fixture
def logger() -> logging.Logger:
    """Return a simple logger for tests."""
    return logging.getLogger("test.camera")


def test_camera_info_dataclass() -> None:
    """CameraInfo should store stream metadata."""
    info = CameraInfo(index=0, width=640, height=480, fps_target=30, backend="DirectShow")
    assert info.width == 640
    assert info.backend == "DirectShow"


@patch("src.camera.cv2.VideoCapture")
def test_start_success(mock_video_capture, camera_settings, logger) -> None:
    """Camera should start when OpenCV opens the device successfully."""
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.get.side_effect = lambda prop: {3: 640, 4: 480}.get(prop, 0)
    mock_capture.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    mock_video_capture.return_value = mock_capture

    camera = CameraCapture(camera_settings, logger=logger)
    camera.start()

    assert camera.is_running is True
    assert camera.camera_info is not None
    assert camera.camera_info.width == 640

    camera.stop()
    mock_capture.release.assert_called()


@patch("src.camera.cv2.VideoCapture")
def test_start_failure_when_camera_not_opened(
    mock_video_capture, camera_settings, logger
) -> None:
    """CameraError should be raised when the device cannot be opened."""
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = False
    mock_video_capture.return_value = mock_capture

    camera = CameraCapture(camera_settings, logger=logger)

    with pytest.raises(CameraError, match="Unable to open camera"):
        camera.start()


def test_read_frame_returns_none_before_start(camera_settings, logger) -> None:
    """read_frame should return None when capture has not started."""
    camera = CameraCapture(camera_settings, logger=logger)
    assert camera.read_frame() is None


@patch("src.camera.cv2.VideoCapture")
def test_read_frame_returns_copy(mock_video_capture, camera_settings, logger) -> None:
    """read_frame should return a copy of the latest captured frame."""
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = True
    mock_capture.get.side_effect = lambda prop: {3: 640, 4: 480}.get(prop, 0)

    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_capture.read.return_value = (True, test_frame)
    mock_video_capture.return_value = mock_capture

    camera = CameraCapture(camera_settings, logger=logger)
    camera.start()

    import time

    time.sleep(0.05)

    frame = camera.read_frame()
    camera.stop()

    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert frame is not test_frame
