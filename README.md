# AI Hand Gesture Mouse Control 🖐️🖱️

A computer vision-based application that allows users to control a Windows mouse using real-time hand gestures through a webcam.

## Overview

AI Hand Gesture Mouse Control is an AI-powered human-computer interaction (HCI) project that uses computer vision and hand tracking to translate natural hand gestures into mouse actions.

The application detects hand landmarks from webcam input and converts specific gestures into computer control commands.

## Features

- 🖱️ Move mouse using hand movement
- 👆 Left click using pinch gesture
- ✌️ Double click gesture
- 🤏 Drag and drop control
- 👉 Right click gesture
- 📜 Scroll control
- ⚙️ Custom GUI settings panel
- 📦 Windows standalone executable support

## How It Works

1. Webcam captures real-time video input
2. Hand landmarks are detected using computer vision models
3. Gesture recognition logic identifies user actions
4. Detected gestures are converted into mouse commands

## Technology Stack

### Programming Language
- Python

### Computer Vision & AI
- OpenCV
- MediaPipe

### GUI Framework
- CustomTkinter

### Testing & Packaging
- PyTest
- PyInstaller

## Supported Gestures

| Gesture | Action |
|---------|--------|
| Index + thumb pinch (quick) | Left click |
| Two quick pinches | Double click |
| Index + thumb hold | Drag |
| Middle + thumb pinch | Right click |
| Index + middle finger up | Scroll |

## Requirements

- Windows 10/11
- Webcam
- Python 3.11+ (development only)

## Installation

Clone the repository:

```powershell
git clone <repository-url>
```

Navigate to the project:

```powershell
cd HandGestureMouse
```

Create virtual environment:

```powershell
python -m venv .venv
```

Activate environment:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run application:

```powershell
python src/main.py
```

## Build Standalone EXE

The application can be packaged into a standalone Windows executable using PyInstaller.

Build command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1
```

Output:

```
dist/HandGestureMouse.exe
```

## Project Structure

```
HandGestureMouse/

├── src/
│   ├── main.py
│   ├── gui.py
│   ├── camera.py
│   ├── hand_tracker.py
│   ├── mouse_controller.py
│   ├── gesture_detector.py
│   ├── action_manager.py
│   ├── smoothing.py
│   ├── config.py
│   └── utils.py
│
├── config/
│   └── settings.json
│
├── scripts/
│   └── build_exe.ps1
│
├── requirements.txt
└── README.md
```

## Development Status

🚧 Active Development

Current focus:

- Standalone Windows executable packaging
- Performance improvement
- Gesture accuracy optimization

## Future Improvements

- Custom gesture training
- Better accuracy in different environments
- More gesture commands
- Cross-platform support
- AI-based personalized gesture recognition

## License

MIT License

## Author

**Mehedi Hasan**

GitHub:
https://github.com/Mehedi-Hasan30
