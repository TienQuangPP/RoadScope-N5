@echo off
setlocal
cd /d "%~dp0"
title RoadScope-N5 Static-Image Demo

echo ============================================================
echo RoadScope-N5 static-image inference
echo Source : %CD%\input
echo Results: %CD%\output
echo ============================================================

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python was not found on PATH.
    echo Install Python 3.10 or newer, then install requirements.txt.
    goto :finish
)

if not exist "roadscope_n5.onnx" (
    echo ERROR: roadscope_n5.onnx is missing from %CD%.
    goto :finish
)

if not exist "input" mkdir "input"
if not exist "output" mkdir "output"

python run.py --source input --output output

if errorlevel 1 (
    echo.
    echo Inference failed. Read the error above and check README.md.
) else (
    echo.
    echo Inference completed successfully.
    echo Open: %CD%\output
)

:finish
pause
