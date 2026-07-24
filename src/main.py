"""Entry point for the AI Hand Gesture Mouse Control application."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script.
if not getattr(sys, "frozen", False):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ConfigManager  # noqa: E402
from src.gui import launch_gui  # noqa: E402
from src.paths import get_bundle_root, get_runtime_root  # noqa: E402
from src.utils import setup_logging, validate_environment  # noqa: E402


def main() -> int:
    """
    Bootstrap the application and launch the GUI.

    Returns:
        Process exit code. Zero indicates success.
    """
    bundle_root = get_bundle_root()
    runtime_root = get_runtime_root()
    config_manager = ConfigManager(
        bundle_root=bundle_root,
        runtime_root=runtime_root,
    )

    try:
        settings = config_manager.load()
    except FileNotFoundError as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    except ValueError as error:
        print(f"[ERROR] Invalid configuration: {error}", file=sys.stderr)
        return 1

    logger = setup_logging(settings.logging, runtime_root)
    logger.info("Starting %s", settings.app.name)

    if not validate_environment(settings, logger, bundle_root, runtime_root):
        logger.error("Startup aborted due to validation failures.")
        return 1

    logger.info("Launching GUI.")
    launch_gui(config_manager, logger)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
