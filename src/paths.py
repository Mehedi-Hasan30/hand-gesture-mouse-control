"""Application path resolution for development and PyInstaller builds."""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """
    Return True when the application is running as a packaged executable.

    Returns:
        True if launched from a PyInstaller bundle, otherwise False.
    """
    return getattr(sys, "frozen", False)


def get_bundle_root() -> Path:
    """
    Return the read-only root containing bundled application resources.

    In development this is the project root. In a PyInstaller build this is
    the temporary ``_MEIPASS`` extraction directory.

    Returns:
        Path to bundled resource root.
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent.parent


def get_runtime_root() -> Path:
    """
    Return the writable root used for logs, local settings, and models.

    In development this is the project root. In a PyInstaller build this is
    the directory containing the executable.

    Returns:
        Path to writable runtime root.
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent
