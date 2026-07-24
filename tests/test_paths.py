"""Unit tests for application path resolution."""

from __future__ import annotations

from pathlib import Path

from src.paths import get_bundle_root, get_runtime_root, is_frozen


def test_is_frozen_false_in_development() -> None:
    """Development runs should not report a frozen executable."""
    assert is_frozen() is False


def test_bundle_root_points_to_project() -> None:
    """Bundle root should resolve to the project directory in development."""
    bundle_root = get_bundle_root()
    assert (bundle_root / "config" / "settings.json").exists()


def test_runtime_root_points_to_project() -> None:
    """Runtime root should resolve to the project directory in development."""
    runtime_root = get_runtime_root()
    assert runtime_root == get_bundle_root()
    assert isinstance(runtime_root, Path)
