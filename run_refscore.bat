@echo off
setlocal EnableDelayedExpansion
title RefScore Academic Application

:: Get the directory of this script
cd /d "%~dp0"

echo ========================================================
echo               RefScore Academic Application
echo ========================================================
echo.

:: 1. Find Python Interpreter
set "PYTHON_CMD="

:: Check for standard virtual environment names
if exist "venv\Scripts\python.exe" (
    set "PYTHON_CMD=venv\Scripts\python.exe"
    echo [INFO] Using virtual environment: venv
) else if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
    echo [INFO] Using virtual environment: .venv
) else (
    :: Check if python is in PATH
    where python >nul 2>nul
    if !errorlevel! equ 0 (
        set "PYTHON_CMD=python"
        echo [INFO] Using system Python
    ) else (
        echo [ERROR] Python not found! Please install Python 3.8+ or create a virtual environment.
        pause
        exit /b 1
    )
)

:: 2. Check for RefScore package
if not exist "refscore\__init__.py" (
    echo [ERROR] RefScore package not found in current directory!
    echo         Expected to find: %~dp0refscore
    pause
    exit /b 1
)

:: 3. Run Application
echo [INFO] Launching RefScore...
echo.

:: Add current directory to PYTHONPATH to ensure imports work
set "PYTHONPATH=%~dp0;%PYTHONPATH%"

"%PYTHON_CMD%" -m refscore.main %*

if !errorlevel! neq 0 (
    echo.
    echo [ERROR] Application crashed or exited with error code !errorlevel!
    echo.
    echo Press any key to close...
    pause >nul
) else (
    echo.
    echo [INFO] Application closed successfully.
)

endlocal
