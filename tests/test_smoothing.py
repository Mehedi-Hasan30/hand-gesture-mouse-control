"""Unit tests for cursor smoothing."""

from __future__ import annotations

import pytest

from src.config import MouseSettings
from src.smoothing import ScreenPoint, SmoothingFilter


@pytest.fixture
def mouse_settings() -> MouseSettings:
    """Return default mouse settings for tests."""
    return MouseSettings(smoothing_factor=0.5, sensitivity=1.0, dead_zone=0.02)


def test_first_update_returns_target_directly(mouse_settings) -> None:
    """The first update should initialize smoothing with the target point."""
    smoother = SmoothingFilter(mouse_settings)
    result = smoother.update(ScreenPoint(x=0.25, y=0.75))

    assert result.x == pytest.approx(0.25)
    assert result.y == pytest.approx(0.75)
    assert smoother.is_initialized is True


def test_exponential_smoothing_blends_values(mouse_settings) -> None:
    """Subsequent updates should blend toward the new target."""
    smoother = SmoothingFilter(mouse_settings)
    smoother.update(ScreenPoint(x=0.0, y=0.0))
    result = smoother.update(ScreenPoint(x=1.0, y=1.0))

    assert result.x == pytest.approx(0.5)
    assert result.y == pytest.approx(0.5)


def test_dead_zone_ignores_small_movements() -> None:
    """Movements below the dead zone should not update the smoothed point."""
    settings = MouseSettings(smoothing_factor=1.0, sensitivity=1.0, dead_zone=0.1)
    smoother = SmoothingFilter(settings)
    smoother.update(ScreenPoint(x=0.5, y=0.5))
    result = smoother.update(ScreenPoint(x=0.51, y=0.51))

    assert result.x == pytest.approx(0.5)
    assert result.y == pytest.approx(0.5)


def test_reset_clears_state(mouse_settings) -> None:
    """Reset should clear smoothed state."""
    smoother = SmoothingFilter(mouse_settings)
    smoother.update(ScreenPoint(x=0.2, y=0.3))
    smoother.reset()

    assert smoother.is_initialized is False
