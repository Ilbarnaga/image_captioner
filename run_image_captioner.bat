@echo off
REM Run this file by double-clicking it in Windows or typing its name in Command Prompt

title "Image captioner"

REM Navigate to the directory where this script is located
cd /d "%~dp0"

REM Check if the virtual environment already exists
if not exist ".venv\" (
    echo First time setup: Creating virtual environment...
    python -m venv .venv
    
    REM Activate and install requirements
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
) else (
    REM It already exists, just activate it
    call .venv\Scripts\activate.bat
)

REM Run the code
python main.py

REM Keep the window open if the program crashes or finishes
pause