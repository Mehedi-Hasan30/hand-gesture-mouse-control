"""Shared utilities for logging, paths, and common helpers."""

from __future__ import annotations

import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path

from src.config import ConfigManager, LoggingSettings, Settings
from src.paths import get_bundle_root, is_frozen

HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_FILENAME = "hand_landmarker.task"


def get_logs_directory(project_root: Path) -> Path:
    """
    Return the logs directory, creating it if needed.

    Args:
        project_root: Root path of the project.

    Returns:
        Path to the logs directory.
    """
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


def get_models_directory(project_root: Path) -> Path:
    """
    Return the models directory, creating it if needed.

    Args:
        project_root: Root path of the project.

    Returns:
        Path to the models directory.
    """
    models_dir = project_root / "assets" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def ensure_hand_landmarker_model(
    runtime_root: Path,
    logger: logging.Logger | None = None,
) -> Path:
    """
    Ensure the MediaPipe hand landmarker model file is available locally.

    Checks bundled resources first, then the runtime directory, and downloads
    the model on first run if it is not already present.

    Args:
        runtime_root: Writable root path for runtime assets.
        logger: Optional logger for download status messages.

    Returns:
        Path to the local hand landmarker model file.

    Raises:
        RuntimeError: If the model cannot be downloaded or saved.
    """
    log = logger or logging.getLogger("hand_gesture_mouse.utils")
    bundled_path = get_bundle_root() / "assets" / "models" / HAND_LANDMARKER_MODEL_FILENAME
    if bundled_path.exists():
        log.info("Hand landmarker model found in bundle: %s", bundled_path)
        return bundled_path

    model_path = get_models_directory(runtime_root) / HAND_LANDMARKER_MODEL_FILENAME

    if model_path.exists():
        log.info("Hand landmarker model found at: %s", model_path)
        return model_path

    log.info("Hand landmarker model not found. Downloading from MediaPipe...")
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_MODEL_URL, model_path)
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Failed to download hand landmarker model. "
            "Check your internet connection and try again."
        ) from error

    if not model_path.exists() or model_path.stat().st_size == 0:
        raise RuntimeError("Downloaded hand landmarker model file is missing or empty.")

    log.info("Hand landmarker model downloaded to: %s", model_path)
    return model_path


def setup_logging(settings: LoggingSettings, project_root: Path) -> logging.Logger:
    """
    Configure application-wide logging based on settings.

    Args:
        settings: Logging configuration section.
        project_root: Root path of the project.

    Returns:
        Configured root logger for the application.
    """
    logger = logging.getLogger("hand_gesture_mouse")
    logger.handlers.clear()
    logger.setLevel(getattr(logging, settings.level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if settings.log_to_file:
        logs_dir = get_logs_directory(project_root)
        log_file_path = logs_dir / Path(settings.log_file).name
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def validate_environment(
    settings: Settings,
    logger: logging.Logger,
    bundle_root: Path,
    runtime_root: Path,
) -> bool:
    """
    Perform basic startup validation checks.

    Args:
        settings: Loaded application settings.
        logger: Logger instance for reporting validation results.
        bundle_root: Read-only bundled resource root.
        runtime_root: Writable runtime root.

    Returns:
        True if all checks pass, False otherwise.
    """
    checks_passed = True

    logger.info("Running environment validation...")
    logger.info("Application: %s v%s", settings.app.name, settings.app.version)

    if is_frozen():
        logger.info("Running as packaged executable.")
        logger.info("Runtime directory: %s", runtime_root)
    else:
        logger.info("Python version: %s", sys.version.split()[0])
        if sys.version_info < (3, 11):
            logger.error("Python 3.11 or higher is required.")
            checks_passed = False
        else:
            logger.info("Python version check passed.")

    settings_path = bundle_root / "config" / "settings.json"
    if not settings_path.exists():
        logger.error("Settings file missing at: %s", settings_path)
        checks_passed = False
    else:
        logger.info("Settings file found at: %s", settings_path)

    runtime_root.mkdir(parents=True, exist_ok=True)
    logger.info("Runtime directory ready at: %s", runtime_root)

    if checks_passed:
        logger.info("Environment validation completed successfully.")
    else:
        logger.error("Environment validation failed.")

    return checks_passed
