@echo off
REM ============================================================================
REM Download TUM RGB-D Dataset Scenes (Windows)
REM https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
REM ============================================================================
REM Usage: download_tum.bat [output_directory]
REM   Default: .\datasets

setlocal enabledelayedexpansion

REM Set output directory (default to datasets folder in project)
if "%~1"=="" (
    set "TUM_DIR=%~dp0..\..\datasets"
) else (
    set "TUM_DIR=%~1"
)

REM Convert to absolute path
pushd "%TUM_DIR%" 2>nul || (mkdir "%TUM_DIR%" && pushd "%TUM_DIR%")
set "TUM_DIR=%CD%"
popd

echo.
echo ========================================
echo   TUM RGB-D Dataset Downloader
echo ========================================
echo.
echo Output directory: %TUM_DIR%
echo.

REM Create directory if it doesn't exist
if not exist "%TUM_DIR%" mkdir "%TUM_DIR%"
cd /d "%TUM_DIR%"

REM Define scenes to download
set SCENES[0]=https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz
set SCENES[1]=https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz
set SCENES[2]=https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz
set SCENES[3]=https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz
set SCENES[4]=https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
set SCENES[5]=https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz

set /a COUNT=6

echo Found %COUNT% scenes to download.
echo.

REM Check for curl (built into Windows 10+)
where curl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: curl not found. Please install curl or use Windows 10+
    exit /b 1
)

REM Check for tar (built into Windows 10+)
where tar >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: tar not found. Please install tar or use Windows 10+
    exit /b 1
)

REM Download each scene
for /L %%i in (0,1,5) do (
    set "URL=!SCENES[%%i]!"
    
    REM Extract filename from URL
    for %%F in (!URL!) do set "FILENAME=%%~nxF"
    set "DIRNAME=!FILENAME:.tgz=!"
    
    REM Check if already downloaded
    if exist "!DIRNAME!" (
        echo [SKIP] Already exists: !DIRNAME!
    ) else (
        echo.
        echo [%%i/5] Downloading: !FILENAME!
        curl -L -# "!URL!" -o "!FILENAME!"
        
        if !ERRORLEVEL! neq 0 (
            echo [ERROR] Failed to download !FILENAME!
        ) else (
            echo Extracting: !FILENAME!
            tar -xzf "!FILENAME!"
            
            if !ERRORLEVEL! neq 0 (
                echo [ERROR] Failed to extract !FILENAME!
            ) else (
                del "!FILENAME!"
                echo [OK] Done: !DIRNAME!
            )
        )
    )
)

echo.
echo ========================================
echo   Download Complete
echo ========================================
echo.
echo Available TUM scenes in %TUM_DIR%:
echo.
for /d %%D in (*) do echo   - %%D

echo.
echo To use with AirSplatMap dashboard, run:
echo   .\scripts\start_dashboard.bat
echo.

endlocal
