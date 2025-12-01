#!/bin/bash
# ============================================================================
# AirSplatMap Dashboard Stop Script
# ============================================================================
# Usage: ./stop_dashboard.sh

# Load config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dashboard_config.sh"

echo "🛰️  Stopping AirSplatMap Dashboard"
echo "==================================="

STOPPED=false

# Function to kill process on port
kill_port() {
    local PORT=$1
    local PIDS=$(lsof -ti :$PORT 2>/dev/null || fuser $PORT/tcp 2>/dev/null | tr -d ' ')
    if [ -n "$PIDS" ]; then
        echo "🔄 Freeing port $PORT (PIDs: $PIDS)..."
        for PID in $PIDS; do
            kill $PID 2>/dev/null || true
        done
        sleep 1
        # Force kill
        PIDS=$(lsof -ti :$PORT 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            for PID in $PIDS; do
                kill -9 $PID 2>/dev/null || true
            done
        fi
        STOPPED=true
    fi
}

# Stop using PID file
if [ -f "$AIRSPLAT_PID" ]; then
    PID=$(cat "$AIRSPLAT_PID" 2>/dev/null)
    if [ -n "$PID" ] && ps -p "$PID" > /dev/null 2>&1; then
        echo "🔄 Stopping dashboard (PID: $PID)..."
        kill "$PID" 2>/dev/null || true
        sleep 2
        if ps -p "$PID" > /dev/null 2>&1; then
            kill -9 "$PID" 2>/dev/null || true
        fi
        STOPPED=true
    fi
    rm -f "$AIRSPLAT_PID"
fi

# Kill any web_dashboard processes
PIDS=$(pgrep -f "web_dashboard.py" 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    echo "🔄 Killing web_dashboard processes: $PIDS"
    pkill -f "web_dashboard.py" 2>/dev/null || true
    sleep 1
    pkill -9 -f "web_dashboard.py" 2>/dev/null || true
    STOPPED=true
fi

# Free ports
kill_port $AIRSPLAT_HTTP_PORT
kill_port $AIRSPLAT_WS_PORT

if [ "$STOPPED" = true ]; then
    echo "✅ Dashboard stopped"
else
    echo "ℹ️  No dashboard was running"
fi

# Verify ports are free
echo ""
for PORT in $AIRSPLAT_HTTP_PORT $AIRSPLAT_WS_PORT; do
    if lsof -ti :$PORT > /dev/null 2>&1; then
        echo "⚠️  Port $PORT still in use!"
    else
        echo "✓ Port $PORT is free"
    fi
done
