"""Unit tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ConfigManager, Settings


@pytest.fixture
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def test_load_settings_success(project_root: Path) -> None:
    """Settings should load from the default JSON file."""
    manager = ConfigManager(bundle_root=project_root, runtime_root=project_root)
    settings = manager.load()

    assert isinstance(settings, Settings)
    assert settings.app.name == "AI Hand Gesture Mouse Control"
    assert settings.camera.index == 0
    assert settings.gui.theme == "dark"


def test_save_local_settings(project_root: Path, tmp_path: Path) -> None:
    """Local settings should be written to the runtime config directory."""
    from dataclasses import replace

    runtime_root = tmp_path / "runtime"
    manager = ConfigManager(bundle_root=project_root, runtime_root=runtime_root)
    settings = manager.load()
    updated = replace(settings, mouse=replace(settings.mouse, sensitivity=1.25))
    manager.save_local_settings(updated)

    local_path = runtime_root / "config" / "settings.local.json"
    assert local_path.exists()

    reloaded = ConfigManager(bundle_root=project_root, runtime_root=runtime_root)
    merged = reloaded.load()
    assert merged.mouse.sensitivity == 1.25


def test_settings_file_missing(tmp_path: Path) -> None:
    """Loading should fail when settings.json is absent."""
    manager = ConfigManager(bundle_root=tmp_path, runtime_root=tmp_path)

    with pytest.raises(FileNotFoundError):
        manager.load()
