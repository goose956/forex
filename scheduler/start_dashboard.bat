@echo off
echo ============================================
echo  Forex Signal Tracker -- Dashboard
echo ============================================
echo.

REM Change to project root (parent of scheduler folder)
cd /d "%~dp0.."

REM Activate virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment.
    echo Make sure you have run setup first.
    pause
    exit /b 1
)

echo Opening dashboard at http://localhost:8501
echo Press Ctrl+C in this window to stop the dashboard.
echo.

REM Open browser after a short delay
start "" /b timeout /t 2 /nobreak >nul && start http://localhost:8501

REM Start Streamlit
streamlit run dashboard\app.py --server.headless true

pause
