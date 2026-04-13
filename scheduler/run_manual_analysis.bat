@echo off
echo ============================================
echo  Forex Signal Tracker -- Manual Analysis
echo ============================================
echo.
echo This will run a full analysis for today.
echo Both Claude and GPT will be used.
echo Expected time: 5-15 minutes.
echo.

cd /d "%~dp0.."

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)

echo [1/2] Running daily signal analysis...
python scripts\run_daily.py
if errorlevel 1 (
    echo.
    echo ERROR: Analysis failed. Check tracker\logs\daily.log for details.
    pause
    exit /b 1
)

echo.
echo [2/2] Updating outcomes from 5 days ago...
python scripts\update_outcomes.py

echo.
echo ============================================
echo  Done! Open the dashboard to see results.
echo  Run: scheduler\start_dashboard.bat
echo ============================================
echo.
pause
