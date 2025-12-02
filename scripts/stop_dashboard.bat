@echo off
REM ============================================================================
REM AirSplatMap Dashboard Stop Script (Windows)
REM ============================================================================
REM Stops the dashboard that was started with start_dashboard_bg.bat

setlocal enabledelayedexpansion

REM Get script directory and project root
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%.."
set PROJECT_ROOT=%CD%
popd

REM PID file location
set PID_FILE=%PROJECT_ROOT%\output\.dashboard.pid

echo.
echo ========================================
echo   Stop AirSplatMap Dashboard
echo ========================================
echo.

REM Check if PID file exists
if not exist "%PID_FILE%" (
    echo No dashboard PID file found.
    echo Dashboard may not be running, or was started in foreground mode.
    echo.
    echo Attempting to find and stop any running dashboard processes...
    goto :kill_by_port
)

REM Read PID from file
set /p DASHBOARD_PID=<"%PID_FILE%"

echo Found PID file: %PID_FILE%
echo Dashboard PID: %DASHBOARD_PID%
echo.

REM Check if process is still running
tasklist /FI "PID eq %DASHBOARD_PID%" 2>nul | find "%DASHBOARD_PID%" >nul
if %ERRORLEVEL% neq 0 (
    echo Process %DASHBOARD_PID% is not running.
    del "%PID_FILE%" 2>nul
    echo Cleaned up stale PID file.
    goto :kill_by_port
)

REM Kill the process tree (dashboard and any child processes)
echo Stopping dashboard process %DASHBOARD_PID%...
taskkill /F /T /PID %DASHBOARD_PID% >nul 2>&1

if %ERRORLEVEL% equ 0 (
    echo Dashboard stopped successfully.
    del "%PID_FILE%" 2>nul
) else (
    echo Warning: Could not stop process %DASHBOARD_PID%
    goto :kill_by_port
)

goto :done

:kill_by_port
REM Fallback: Find and kill by port
echo.
echo Looking for processes using dashboard ports...

REM Load config for ports
if exist "%~dp0config.bat" call "%~dp0config.bat"
if not defined AIRSPLAT_HTTP_PORT set AIRSPLAT_HTTP_PORT=9002
if not defined AIRSPLAT_WS_PORT set AIRSPLAT_WS_PORT=9003

REM Find processes on HTTP port
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%AIRSPLAT_HTTP_PORT% " ^| findstr "LISTENING"') do (
    echo Found process %%a on HTTP port %AIRSPLAT_HTTP_PORT%
    taskkill /F /PID %%a >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo   Stopped process %%a
    )
)

REM Find processes on WebSocket port
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":%AIRSPLAT_WS_PORT% " ^| findstr "LISTENING"') do (
    echo Found process %%a on WebSocket port %AIRSPLAT_WS_PORT%
    taskkill /F /PID %%a >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo   Stopped process %%a
    )
)

REM Clean up PID file if exists
if exist "%PID_FILE%" del "%PID_FILE%" 2>nul

:done
echo.
echo Dashboard stopped.
echo.

endlocal
