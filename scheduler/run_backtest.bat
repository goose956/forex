@echo off
echo ============================================
echo  Forex Signal Tracker -- Backtest Analysis
echo ============================================
echo.
echo This will run analysis for a specific past date.
echo.

set /p DATE="Enter date to analyse (YYYY-MM-DD): "

if "%DATE%"=="" (
    echo No date entered. Exiting.
    pause
    exit /b 1
)

echo.
echo Running analysis for %DATE%...
echo Expected time: 5-15 minutes.
echo.

cd /d "%~dp0.."

call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Could not activate virtual environment.
    pause
    exit /b 1
)

python scripts\run_daily.py --date %DATE%
if errorlevel 1 (
    echo.
    echo ERROR: Backtest failed. Check tracker\logs\daily.log for details.
    pause
    exit /b 1
)

echo.
echo Done. Report saved to tracker\reports\daily\%DATE%_GBPUSD.txt
echo Open the dashboard to see the backtest signal.
echo.
pause
