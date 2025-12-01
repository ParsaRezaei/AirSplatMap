#!/bin/bash
# ============================================================================
# AirSplatMap Dashboard Start Script
# ============================================================================
# Usage: ./start_dashboard.sh [--http-port PORT] [--ws-port PORT]

set -e

# Load config
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/dashboard_config.sh"

# Parse arguments (override config)
while [[ $# -gt 0 ]]; do
    case $1 in
        --http-port) AIRSPLAT_HTTP_PORT="$2"; shift 2 ;;
        --ws-port) AIRSPLAT_WS_PORT="$2"; shift 2 ;;
        -h|--help) 
            echo "Usage: $0 [--http-port PORT] [--ws-port PORT]"
            echo "  Default HTTP: $AIRSPLAT_HTTP_PORT, WS: $AIRSPLAT_WS_PORT"
            echo "  Config: $SCRIPT_DIR/dashboard_config.sh"
            exit 0 ;;
        *) shift ;;
    esac
done

echo "🛰️  AirSplatMap Dashboard"
echo "========================"

# Function to kill process on port
kill_port() {
    local PORT=$1
    local PIDS=$(lsof -ti :$PORT 2>/dev/null || fuser $PORT/tcp 2>/dev/null | tr -d ' ')
    if [ -n "$PIDS" ]; then
        echo "⚠️  Port $PORT in use, killing PIDs: $PIDS"
        for PID in $PIDS; do
            kill $PID 2>/dev/null || true
        done
        sleep 1
        # Force kill if still there
        PIDS=$(lsof -ti :$PORT 2>/dev/null || true)
        if [ -n "$PIDS" ]; then
            for PID in $PIDS; do
                kill -9 $PID 2>/dev/null || true
            done
            sleep 1
        fi
    fi
}

# Stop existing dashboard
if [ -f "$AIRSPLAT_PID" ]; then
    OLD_PID=$(cat "$AIRSPLAT_PID" 2>/dev/null)
    if [ -n "$OLD_PID" ] && ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Stopping existing dashboard (PID: $OLD_PID)..."
        kill "$OLD_PID" 2>/dev/null || true
        sleep 2
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            kill -9 "$OLD_PID" 2>/dev/null || true
            sleep 1
        fi
    fi
    rm -f "$AIRSPLAT_PID"
fi

# Kill any web_dashboard processes
pkill -f "web_dashboard.py" 2>/dev/null || true
sleep 1

# Free the ports
kill_port $AIRSPLAT_HTTP_PORT
kill_port $AIRSPLAT_WS_PORT

# Double-check ports are free
sleep 1
for PORT in $AIRSPLAT_HTTP_PORT $AIRSPLAT_WS_PORT; do
    if lsof -ti :$PORT > /dev/null 2>&1; then
        echo "❌ Port $PORT still in use after cleanup!"
        lsof -i :$PORT
        exit 1
    fi
done

# Activate conda environment
if [ -f ~/miniconda/etc/profile.d/conda.sh ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
conda activate $AIRSPLAT_CONDA_ENV 2>/dev/null || true

# Start the dashboard
cd "$AIRSPLAT_ROOT"
echo "📁 Working directory: $(pwd)"
echo "🚀 Starting on http://localhost:$AIRSPLAT_HTTP_PORT (WS: $AIRSPLAT_WS_PORT)"
echo "📝 Log: $AIRSPLAT_LOG"
echo ""

nohup python scripts/web_dashboard.py \
    --http-port $AIRSPLAT_HTTP_PORT \
    --ws-port $AIRSPLAT_WS_PORT \
    > "$AIRSPLAT_LOG" 2>&1 &

DASH_PID=$!
echo $DASH_PID > "$AIRSPLAT_PID"

# Wait and verify
sleep 3

if ps -p $DASH_PID > /dev/null 2>&1; then
    # Check if HTTP port is listening
    if lsof -ti :$AIRSPLAT_HTTP_PORT > /dev/null 2>&1 || \
       netstat -tln 2>/dev/null | grep -q ":$AIRSPLAT_HTTP_PORT "; then
        echo "✅ Dashboard started (PID: $DASH_PID)"
        echo ""
        echo "   🌐 Open: http://localhost:$AIRSPLAT_HTTP_PORT"
        echo ""
        echo "   📋 Logs: tail -f $AIRSPLAT_LOG"
        echo "   🛑 Stop: $SCRIPT_DIR/stop_dashboard.sh"
    else
        echo "⚠️  Process started but HTTP port not listening yet"
        echo "   Check logs: cat $AIRSPLAT_LOG"
        tail -20 "$AIRSPLAT_LOG"
    fi
else
    echo "❌ Failed to start dashboard"
    echo ""
    cat "$AIRSPLAT_LOG"
    rm -f "$AIRSPLAT_PID"
    exit 1
fi
