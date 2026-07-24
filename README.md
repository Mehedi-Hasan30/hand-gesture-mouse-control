# AI Hand Gesture Mouse Control



Control your Windows mouse using hand gestures detected from a webcam.



## Status



**Phase 7 — Standalone EXE Packaging** (current)



The application can be packaged into a single Windows executable with PyInstaller. The GUI, gesture control, and settings panel are included in the build.



## Requirements



- Windows 10/11

- Python 3.11+ (development only)

- Webcam



## Quick Start (Development)



```powershell

cd HandGestureMouse

python -m venv .venv

.venv\Scripts\Activate.ps1

pip install -r requirements.txt

python src/main.py

pytest tests/ -v

```



## Build Standalone EXE



```powershell

cd HandGestureMouse

powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1

```



Output:



```

dist/HandGestureMouse.exe

```



### Build Notes



- The build script runs tests, downloads the MediaPipe model, then invokes PyInstaller.

- The executable bundles default settings and the hand landmarker model when available.

- Saved settings, logs, and downloaded models are stored **next to the `.exe`**.

- First launch may take a few seconds while PyInstaller extracts bundled files.

- Internet is only required on first run if the model was not bundled during build.



## Project Structure



```

HandGestureMouse/

├── src/

│   ├── main.py              # Application entry point

│   ├── gui.py               # CustomTkinter GUI + app runner

│   ├── paths.py             # Dev vs frozen path resolution

│   ├── camera.py

│   ├── hand_tracker.py

│   ├── mouse_controller.py

│   ├── gesture_detector.py

│   ├── action_manager.py

│   ├── smoothing.py

│   ├── config.py

│   └── utils.py

├── config/settings.json

├── scripts/build_exe.ps1

├── HandGestureMouse.spec

├── requirements.txt

└── README.md

```



## Gestures



| Gesture | Action |

|---------|--------|

| Index + thumb pinch (quick) | Left click |

| Two quick pinches | Double click |

| Index + thumb hold | Drag |

| Middle + thumb pinch | Right click |

| Index + middle up, move hand | Scroll |



Press **M** in the camera preview or use **Enable Mouse** in the GUI.



## Development Phases



| Phase | Focus |

|-------|-------|

| 1–5 | Foundation through gesture actions |

| 6 | GUI and settings panel |

| 7 | Standalone EXE packaging (current) |



## License



MIT License — see [LICENSE](LICENSE).

