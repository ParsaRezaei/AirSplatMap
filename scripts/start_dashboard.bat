@echo off
REM ============================================================================
REM AirSplatMap Dashboard Start Script (Windows)
REM ============================================================================
REM Starts the dashboard in the background.
REM Usage: start_dashboard.bat [--http-port PORT] [--ws-port PORT] [--foreground]

setlocal enabledelayedexpansion

REM Load config
if exist "%~dp0config.bat" call "%~dp0config.bat"

REM Default values
if not defined AIRSPLAT_HTTP_PORT set AIRSPLAT_HTTP_PORT=9002
if not defined AIRSPLAT_WS_PORT set AIRSPLAT_WS_PORT=9003
if not defined AIRSPLAT_CONDA_ENV set AIRSPLAT_CONDA_ENV=airsplatmap

set FOREGROUND=0

REM Parse command line arguments
:parse_args
if "%~1"=="" goto :done_args
if "%~1"=="--http-port" (
    set AIRSPLAT_HTTP_PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--ws-port" (
    set AIRSPLAT_WS_PORT=%~2
    shift
    shift
    goto :parse_args
)
if "%~1"=="--foreground" (
    set FOREGROUND=1
    shift
    goto :parse_args
)
if "%~1"=="-f" (
    set FOREGROUND=1
    shift
    goto :parse_args
)
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
shift
goto :parse_args

:show_help
echo Usage: %~nx0 [--http-port PORT] [--ws-port PORT] [--foreground]
echo.
echo   --http-port PORT   HTTP server port (default: %AIRSPLAT_HTTP_PORT%)
echo   --ws-port PORT     WebSocket server port (default: %AIRSPLAT_WS_PORT%)
echo   --foreground, -f   Run in foreground (default: background)
echo.
echo   Use stop_dashboard.bat to stop the background server.
exit /b 0

:done_args

REM Get script directory and project root
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%.."
set PROJECT_ROOT=%CD%
popd

REM PID and log file locations
set PID_FILE=%PROJECT_ROOT%\output\.dashboard.pid
set LOG_FILE=%PROJECT_ROOT%\output\dashboard.log

REM Create output directory if needed
if not exist "%PROJECT_ROOT%\output" mkdir "%PROJECT_ROOT%\output"

REM Check if already running
if exist "%PID_FILE%" (
    set /p OLD_PID=<"%PID_FILE%"
    tasklist /FI "PID eq !OLD_PID!" 2>nul | find "python" >nul
    if !ERRORLEVEL! equ 0 (
        echo Dashboard is already running with PID !OLD_PID!
        echo   Web UI: http://localhost:%AIRSPLAT_HTTP_PORT%
        echo   Use stop_dashboard.bat to stop it first.
        exit /b 1
    ) else (
        del "%PID_FILE%" 2>nul
    )
)

echo.
echo ========================================
echo   AirSplatMap Dashboard
echo ========================================
echo.
echo Project root: %PROJECT_ROOT%
echo HTTP Port: %AIRSPLAT_HTTP_PORT%
echo WebSocket Port: %AIRSPLAT_WS_PORT%

REM Change to project root
cd /d "%PROJECT_ROOT%"

if %FOREGROUND%==1 (
    echo.
    echo Starting dashboard in foreground...
    echo   Web UI: http://localhost:%AIRSPLAT_HTTP_PORT%
    echo   Press Ctrl+C to stop
    echo.
    call conda activate %AIRSPLAT_CONDA_ENV% 2>nul
    python scripts/web_dashboard.py --http-port %AIRSPLAT_HTTP_PORT% --ws-port %AIRSPLAT_WS_PORT%
    exit /b %ERRORLEVEL%
)

echo Log file: %LOG_FILE%
echo.
echo Starting dashboard in background...

REM Start the dashboard in background using PowerShell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& { $proc = Start-Process -FilePath 'cmd.exe' -ArgumentList '/c conda activate %AIRSPLAT_CONDA_ENV% && python scripts/web_dashboard.py --http-port %AIRSPLAT_HTTP_PORT% --ws-port %AIRSPLAT_WS_PORT% > \"%LOG_FILE%\" 2>&1' -WindowStyle Hidden -PassThru; $proc.Id | Out-File -FilePath '%PID_FILE%' -Encoding ascii -NoNewline; Write-Host ('Started with PID: ' + $proc.Id) }"

if %ERRORLEVEL% neq 0 (
    echo Failed to start dashboard!
    exit /b 1
)

REM Wait a moment for startup
timeout /t 2 /nobreak >nul

echo.
echo Dashboard is running!
echo   Web UI: http://localhost:%AIRSPLAT_HTTP_PORT%
echo   WebSocket: ws://localhost:%AIRSPLAT_WS_PORT%
echo   Log: %LOG_FILE%
echo.
echo To stop: scripts\stop_dashboard.bat
echo.

endlocal
