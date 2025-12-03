#!/usr/bin/env bash
# ============================================================================
# AirSplatMap Dashboard Start Script
# ============================================================================
# Usage: ./start_dashboard.sh [--http-port PORT] [--ws-port PORT] [--foreground]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default ports
HTTP_PORT=${AIRSPLAT_HTTP_PORT:-9002}
WS_PORT=${AIRSPLAT_WS_PORT:-9003}
FOREGROUND=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --http-port)
            HTTP_PORT="$2"
            shift 2
            ;;
        --ws-port)
            WS_PORT="$2"
            shift 2
            ;;
        -f|--foreground)
            FOREGROUND=1
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--http-port PORT] [--ws-port PORT] [--foreground]"
            echo ""
            echo "  --http-port PORT   HTTP server port (default: 9002)"
            echo "  --ws-port PORT     WebSocket server port (default: 9003)"
            echo "  --foreground, -f   Run in foreground"
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

echo ""
echo "========================================"
echo "  AirSplatMap Dashboard"
echo "========================================"
echo ""
echo "Project root: $PROJECT_ROOT"
echo "HTTP Port: $HTTP_PORT"
echo "WebSocket Port: $WS_PORT"

cd "$PROJECT_ROOT"

if [ "$FOREGROUND" -eq 1 ]; then
    echo ""
    echo "Starting dashboard in foreground..."
    echo "  Web UI: http://localhost:$HTTP_PORT"
    echo "  Press Ctrl+C to stop"
    echo ""
    python dashboard/web_dashboard.py --http-port "$HTTP_PORT" --ws-port "$WS_PORT"
else
    # Create output directory
    mkdir -p "$PROJECT_ROOT/output"
    
    PID_FILE="$PROJECT_ROOT/output/.dashboard.pid"
    LOG_FILE="$PROJECT_ROOT/output/dashboard.log"
    
    # Check if already running
    if [ -f "$PID_FILE" ]; then
        OLD_PID=$(cat "$PID_FILE")
        if ps -p "$OLD_PID" > /dev/null 2>&1; then
            echo "Dashboard is already running with PID $OLD_PID"
            echo "  Web UI: http://localhost:$HTTP_PORT"
            echo "  Use stop_dashboard.sh to stop it first."
            exit 1
        fi
        rm -f "$PID_FILE"
    fi
    
    echo "Log file: $LOG_FILE"
    echo ""
    echo "Starting dashboard in background..."
    
    nohup python dashboard/web_dashboard.py --http-port "$HTTP_PORT" --ws-port "$WS_PORT" > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    
    sleep 2
    
    echo ""
    echo "Dashboard is running!"
    echo "  Web UI: http://localhost:$HTTP_PORT"
    echo "  WebSocket: ws://localhost:$WS_PORT"
    echo "  Log: $LOG_FILE"
    echo ""
    echo "To stop: ./dashboard/stop_dashboard.sh"
    echo ""
fi
