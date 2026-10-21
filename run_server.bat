@echo off
REM Terminal 1: segmentation server on http://127.0.0.1:8000
REM   Mazurowski nnU-Net (CT-Muscle-and-Fat-Segmentation) + TotalSegmentator T12/L3/L4
REM   set MODEL=demo   to test without GPU / model weights
cd /d "%~dp0inference"
if exist "..\..\venv\Scripts\activate.bat" (call "..\..\venv\Scripts\activate.bat") else if exist "..\..\.venv\Scripts\activate.bat" (call "..\..\.venv\Scripts\activate.bat") else (echo venv not found next to the project folder & pause & exit /b 1)
if "%SEG_CHECKPOINT%"=="" set SEG_CHECKPOINT=final
python server.py
pause
