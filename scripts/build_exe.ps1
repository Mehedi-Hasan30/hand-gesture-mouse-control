# Build script for creating the standalone Windows executable.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "==> Hand Gesture Mouse - EXE Build" -ForegroundColor Cyan
Write-Host "Project root: $ProjectRoot"

if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "==> Activating virtual environment"
. .\.venv\Scripts\Activate.ps1

Write-Host "==> Installing dependencies"
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

Write-Host "==> Running tests"
python -m pytest tests/ -q
if ($LASTEXITCODE -ne 0) {
    throw "Tests failed. Aborting build."
}

Write-Host "==> Ensuring MediaPipe model is available for bundling"
$ModelDir = Join-Path $ProjectRoot "assets\models"
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
python -c "from pathlib import Path; from src.paths import get_runtime_root; from src.utils import ensure_hand_landmarker_model; ensure_hand_landmarker_model(get_runtime_root())"

Write-Host "==> Building executable with PyInstaller"
python -m PyInstaller HandGestureMouse.spec --noconfirm --clean
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller build failed."
}

$ExePath = Join-Path $ProjectRoot "dist\HandGestureMouse.exe"
if (-not (Test-Path $ExePath)) {
    throw "Expected executable not found at $ExePath"
}

Write-Host ""
Write-Host "Build complete!" -ForegroundColor Green
Write-Host "Executable: $ExePath"
Write-Host ""
Write-Host "Notes:"
Write-Host "  - First launch may take a few seconds while files extract."
Write-Host "  - Logs and saved settings are written next to the .exe."
Write-Host "  - Internet is only required if the model was not bundled."
