@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Create the thesis environment first. See START_HERE.md.
    pause
    exit /b 1
)
echo Installing AtaScope desktop dependencies into the existing thesis environment...
".venv\Scripts\python.exe" -m pip install -r requirements-desktop.txt
if errorlevel 1 (
    echo Desktop dependency installation failed. See the pip message above.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip check
if errorlevel 1 (
    echo The environment has dependency conflicts. See START_HERE.md.
    pause
    exit /b 1
)
echo Desktop dependencies verified. Open Launch_AtaScope.bat.
pause
