#!/bin/bash
# ============================================================================
# AirSplatMap Dashboard Configuration
# ============================================================================
# Source this file to get common settings, or it's auto-sourced by scripts

# Default ports
export AIRSPLAT_HTTP_PORT=${AIRSPLAT_HTTP_PORT:-9002}
export AIRSPLAT_WS_PORT=${AIRSPLAT_WS_PORT:-9003}

# File locations
export AIRSPLAT_LOG="/tmp/airsplatmap_dashboard.log"
export AIRSPLAT_PID="/tmp/airsplatmap_dashboard.pid"

# Conda environment
export AIRSPLAT_CONDA_ENV="airsplatmap"

# Script directory (auto-detected)
if [ -n "${BASH_SOURCE[0]}" ]; then
    export AIRSPLAT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export AIRSPLAT_ROOT="$(dirname "$AIRSPLAT_SCRIPT_DIR")"
fi
