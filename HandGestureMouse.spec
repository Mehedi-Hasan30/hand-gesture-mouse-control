# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for AI Hand Gesture Mouse Control."""

from __future__ import annotations

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

project_root = Path(SPECPATH)
model_path = project_root / "assets" / "models" / "hand_landmarker.task"

datas = [(str(project_root / "config" / "settings.json"), "config")]
if model_path.exists():
    datas.append((str(model_path), "assets/models"))

binaries: list[tuple[str, str]] = []
hiddenimports: list[str] = [
    "src",
    "src.action_manager",
    "src.camera",
    "src.config",
    "src.gesture_detector",
    "src.gui",
    "src.hand_tracker",
    "src.mouse_controller",
    "src.paths",
    "src.smoothing",
    "src.utils",
    "PIL._tkinter_finder",
]

for package_name in ("mediapipe", "customtkinter", "cv2"):
    pkg_datas, pkg_binaries, pkg_hiddenimports = collect_all(package_name)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hiddenimports

hiddenimports += collect_submodules("mediapipe.tasks.python")

a = Analysis(
    [str(project_root / "src" / "main.py")],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="HandGestureMouse",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
