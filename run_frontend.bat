@echo off
REM Terminal 2: Streamlit frontend -> http://localhost:8501   (settings: .env)
cd /d "%~dp0"
if exist "..\venv\Scripts\activate.bat" (call "..\venv\Scripts\activate.bat") else if exist "..\.venv\Scripts\activate.bat" (call "..\.venv\Scripts\activate.bat") else (echo venv not found next to this folder & pause & exit /b 1)
streamlit run app.py
pause
