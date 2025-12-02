@echo off
REM ============================================================================
REM AirSplatMap Configuration (Windows)
REM ============================================================================

REM Conda environment name
if not defined AIRSPLAT_CONDA_ENV set AIRSPLAT_CONDA_ENV=airsplatmap

REM Default ports
if not defined AIRSPLAT_HTTP_PORT set AIRSPLAT_HTTP_PORT=9002
if not defined AIRSPLAT_WS_PORT set AIRSPLAT_WS_PORT=9003

REM Output directory
if not defined AIRSPLAT_OUTPUT_DIR set AIRSPLAT_OUTPUT_DIR=%~dp0..\output
