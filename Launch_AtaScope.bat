@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo The thesis Python environment is missing. See START_HERE.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -c "import PySide6.QtWidgets" >nul 2>nul
if errorlevel 1 (
    echo Installing the AtaScope native desktop interface dependency...
    ".venv\Scripts\python.exe" -m pip install -r requirements-desktop.txt
    if errorlevel 1 (
        pause
        exit /b 1
    )
)
start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0scripts\desktop_app.py"
