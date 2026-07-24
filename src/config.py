"""Configuration management for the Hand Gesture Mouse Control application."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.paths import get_bundle_root, get_runtime_root


@dataclass
class AppSettings:
    """Application-level settings."""

    name: str = "AI Hand Gesture Mouse Control"
    version: str = "0.2.0"
    debug: bool = False


@dataclass
class CameraSettings:
    """Webcam capture settings."""

    index: int = 0
    width: int = 640
    height: int = 480
    fps_target: int = 30


@dataclass
class HandTrackingSettings:
    """MediaPipe hand tracking parameters."""

    max_hands: int = 1
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.7


@dataclass
class MouseSettings:
    """Mouse movement and gesture sensitivity settings."""

    smoothing_factor: float = 0.35
    sensitivity: float = 1.0
    dead_zone: float = 0.02


@dataclass
class GestureSettings:
    """Hand gesture detection thresholds and timing."""

    pinch_threshold: float = 0.05
    drag_hold_ms: int = 250
    double_click_interval_ms: int = 400
    scroll_sensitivity: float = 800.0
    scroll_dead_zone: float = 0.01


@dataclass
class GuiSettings:
    """Graphical user interface preferences."""

    theme: str = "dark"
    window_width: int = 900
    window_height: int = 600


@dataclass
class LoggingSettings:
    """Logging configuration."""

    level: str = "INFO"
    log_to_file: bool = True
    log_file: str = "logs/app.log"


@dataclass
class Settings:
    """Root settings container loaded from JSON configuration."""

    app: AppSettings = field(default_factory=AppSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    hand_tracking: HandTrackingSettings = field(default_factory=HandTrackingSettings)
    mouse: MouseSettings = field(default_factory=MouseSettings)
    gestures: GestureSettings = field(default_factory=GestureSettings)
    gui: GuiSettings = field(default_factory=GuiSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)


class ConfigManager:
    """Loads, validates, and provides access to application settings."""

    def __init__(
        self,
        bundle_root: Path | None = None,
        runtime_root: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        """
        Initialize the configuration manager.

        Args:
            bundle_root: Read-only resource root. Defaults to bundle path helper.
            runtime_root: Writable runtime root. Defaults to runtime path helper.
            project_root: Deprecated alias for runtime_root.
        """
        self._bundle_root = bundle_root or get_bundle_root()
        self._runtime_root = runtime_root or project_root or get_runtime_root()
        self._settings_path = self._bundle_root / "config" / "settings.json"
        self._local_settings_path = self._runtime_root / "config" / "settings.local.json"
        self._settings: Settings | None = None

    @staticmethod
    def _detect_project_root() -> Path:
        """
        Detect the writable project root directory.

        Returns:
            Path to the runtime root folder.
        """
        return get_runtime_root()

    @property
    def project_root(self) -> Path:
        """Return the writable runtime root directory."""
        return self._runtime_root

    @property
    def bundle_root(self) -> Path:
        """Return the bundled read-only resource root."""
        return self._bundle_root

    @property
    def settings_path(self) -> Path:
        """Return the path to the primary settings file."""
        return self._settings_path

    def load(self) -> Settings:
        """
        Load settings from JSON, merging local overrides when present.

        Returns:
            Parsed Settings instance.

        Raises:
            FileNotFoundError: If the primary settings file is missing.
            json.JSONDecodeError: If settings JSON is malformed.
            ValueError: If required configuration sections are missing.
        """
        if not self._settings_path.exists():
            raise FileNotFoundError(f"Settings file not found: {self._settings_path}")

        with self._settings_path.open(encoding="utf-8") as file:
            data: dict[str, Any] = json.load(file)

        if self._local_settings_path.exists():
            with self._local_settings_path.open(encoding="utf-8") as file:
                local_data: dict[str, Any] = json.load(file)
            data = self._deep_merge(data, local_data)

        self._settings = self._parse_settings(data)
        return self._settings

    def save_local_settings(self, settings: Settings) -> None:
        """
        Persist user-adjusted settings to the local override file.

        Args:
            settings: Settings instance to save.

        Raises:
            OSError: If the settings file cannot be written.
        """
        payload = asdict(settings)
        local_config_dir = self._runtime_root / "config"
        local_config_dir.mkdir(parents=True, exist_ok=True)
        with self._local_settings_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
        self._settings = settings

    def get_settings(self) -> Settings:
        """
        Return cached settings, loading them if necessary.

        Returns:
            Current Settings instance.
        """
        if self._settings is None:
            return self.load()
        return self._settings

    def _parse_settings(self, data: dict[str, Any]) -> Settings:
        """
        Parse a raw dictionary into structured Settings objects.

        Args:
            data: Raw configuration dictionary.

        Returns:
            Validated Settings instance.

        Raises:
            ValueError: If a required section is absent.
        """
        required_sections = (
            "app",
            "camera",
            "hand_tracking",
            "mouse",
            "gestures",
            "gui",
            "logging",
        )
        missing = [section for section in required_sections if section not in data]
        if missing:
            raise ValueError(f"Missing configuration sections: {', '.join(missing)}")

        return Settings(
            app=AppSettings(**data["app"]),
            camera=CameraSettings(**data["camera"]),
            hand_tracking=HandTrackingSettings(**data["hand_tracking"]),
            mouse=MouseSettings(**data["mouse"]),
            gestures=GestureSettings(**data["gestures"]),
            gui=GuiSettings(**data["gui"]),
            logging=LoggingSettings(**data["logging"]),
        )

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively merge override values into base configuration.

        Args:
            base: Base configuration dictionary.
            override: Override values to apply.

        Returns:
            Merged configuration dictionary.
        """
        merged = base.copy()
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager._deep_merge(merged[key], value)
            else:
                merged[key] = value
        return merged
