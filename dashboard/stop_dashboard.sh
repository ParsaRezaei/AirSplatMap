#!/usr/bin/env bash
# ============================================================================
# AirSplatMap Dashboard Stop Script
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PID_FILE="$PROJECT_ROOT/output/.dashboard.pid"

echo ""
echo "========================================"
echo "  Stop AirSplatMap Dashboard"
echo "========================================"
echo ""

if [ ! -f "$PID_FILE" ]; then
    echo "No dashboard PID file found."
    echo "Dashboard may not be running."
    exit 0
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null 2>&1; then
    echo "Stopping dashboard (PID: $PID)..."
    kill "$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
    rm -f "$PID_FILE"
    echo "Dashboard stopped."
else
    echo "Process $PID is not running."
    rm -f "$PID_FILE"
    echo "Cleaned up stale PID file."
fi

echo ""
