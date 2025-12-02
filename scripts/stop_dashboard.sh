#!/usr/bin/env bash
# ============================================================================
# AirSplatMap Dashboard Stop Script (Cross-Platform)
# ============================================================================
# Automatically detects OS and runs the correct platform script
#
# Usage: ./stop_dashboard.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "$(uname -s)" in
    Linux*)
        exec "$SCRIPT_DIR/platform/linux/stop_dashboard.sh" "$@"
        ;;
    Darwin*)
        exec "$SCRIPT_DIR/platform/linux/stop_dashboard.sh" "$@"
        ;;
    CYGWIN*|MINGW*|MSYS*)
        cmd.exe /c "$SCRIPT_DIR\\platform\\windows\\stop_dashboard.bat" "$@"
        ;;
    *)
        echo "Unknown OS: $(uname -s)"
        exit 1
        ;;
esac
