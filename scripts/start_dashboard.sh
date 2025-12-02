#!/usr/bin/env bash
# ============================================================================
# AirSplatMap Dashboard Start Script (Cross-Platform)
# ============================================================================
# Automatically detects OS and runs the correct platform script
#
# Usage: ./start_dashboard.sh [--http-port PORT] [--ws-port PORT]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$(uname -s)" in
    Linux*)
        exec "$SCRIPT_DIR/platform/linux/start_dashboard.sh" "$@"
        ;;
    Darwin*)
        # macOS - use Linux script (compatible)
        exec "$SCRIPT_DIR/platform/linux/start_dashboard.sh" "$@"
        ;;
    CYGWIN*|MINGW*|MSYS*)
        # Windows Git Bash / MSYS - call the batch file
        cmd.exe /c "$SCRIPT_DIR\\platform\\windows\\start_dashboard.bat" "$@"
        ;;
    *)
        echo "Unknown OS: $(uname -s)"
        echo "Please run platform-specific script manually:"
        echo "  Linux: ./scripts/platform/linux/start_dashboard.sh"
        echo "  Windows: scripts\\platform\\windows\\start_dashboard.bat"
        exit 1
        ;;
esac
