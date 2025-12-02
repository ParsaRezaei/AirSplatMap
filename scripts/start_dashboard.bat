@echo off
REM ============================================================================
REM AirSplatMap Dashboard Start Script (Windows)
REM ============================================================================
REM Usage: start_dashboard.bat [--http-port PORT] [--ws-port PORT]

setlocal enabledelayedexpansion

REM Load config
call "%~dp0config.bat"

REM Default values from config (if not already set)
if not defined AIRSPLAT_HTTP_PORT set AIRSPLAT_HTTP_PORT=9002
if not defined AIRSPLAT_WS_PORT set AIRSPLAT_WS_PORT=9003

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
if "%~1"=="-h" goto :show_help
if "%~1"=="--help" goto :show_help
shift
goto :parse_args

:show_help
echo Usage: %~nx0 [--http-port PORT] [--ws-port PORT]
echo   Default HTTP: %AIRSPLAT_HTTP_PORT%, WS: %AIRSPLAT_WS_PORT%
exit /b 0

:done_args

echo.
echo ========================================
echo   AirSplatMap Dashboard
echo ========================================
echo.

REM Get script directory and project root
set SCRIPT_DIR=%~dp0
pushd "%SCRIPT_DIR%.."
set PROJECT_ROOT=%CD%
popd

echo Project root: %PROJECT_ROOT%
echo HTTP Port: %AIRSPLAT_HTTP_PORT%
echo WebSocket Port: %AIRSPLAT_WS_PORT%
echo.

REM Check if conda is available
where conda >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Warning: conda not found in PATH
    echo Trying to activate conda environment anyway...
)

REM Activate conda environment
echo Activating conda environment: %AIRSPLAT_CONDA_ENV%
call conda activate %AIRSPLAT_CONDA_ENV% 2>nul
if %ERRORLEVEL% neq 0 (
    echo Warning: Could not activate conda environment '%AIRSPLAT_CONDA_ENV%'
    echo Continuing with current Python environment...
)

REM Change to project root
cd /d "%PROJECT_ROOT%"

echo.
echo Starting dashboard...
echo   Web UI: http://localhost:%AIRSPLAT_HTTP_PORT%
echo   Press Ctrl+C to stop
echo.

REM Start the dashboard
python scripts/web_dashboard.py --http-port %AIRSPLAT_HTTP_PORT% --ws-port %AIRSPLAT_WS_PORT%

if %ERRORLEVEL% neq 0 (
    echo.
    echo Dashboard exited with error code: %ERRORLEVEL%
    pause
)

endlocal
