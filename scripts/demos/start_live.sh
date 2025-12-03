#!/bin/bash
#
# Start Live TUM Stream
# =====================
# Adds a live TUM dataset stream to the dashboard.
# Starts dashboard/simulator only if not already running.
#
# Usage:
#   ./scripts/start_live.sh                    # Use default dataset
#   ./scripts/start_live.sh <dataset_name>     # Use specific dataset
#

# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_DATASET="rgbd_dataset_freiburg1_desk2"
TUM_DATASET_PATH="/home/past/parsa/datasets/tum"
SIMULATOR_FPS=15
SIMULATOR_PORT=8554
DASHBOARD_HTTP_PORT=9002
DASHBOARD_WS_PORT=9003
CONDA_ENV="airsplatmap"
LIVE_SOURCE_NAME="tum_live"

# =============================================================================
# SCRIPT
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

DATASET="${1:-$DEFAULT_DATASET}"
DATASET_FULL_PATH="$TUM_DATASET_PATH/$DATASET"

echo -e "${BLUE}=== AirSplatMap Live Stream ===${NC}"

# Check dataset exists
if [ ! -d "$DATASET_FULL_PATH" ]; then
    echo -e "${RED}Error: Dataset not found: $DATASET${NC}"
    echo "Available:"
    ls -1 "$TUM_DATASET_PATH" 2>/dev/null | grep rgbd_
    exit 1
fi

echo -e "${GREEN}Dataset:${NC} $DATASET"

# Activate conda
source ~/miniconda/etc/profile.d/conda.sh
conda activate $CONDA_ENV 2>/dev/null

# Check if dashboard is running
DASHBOARD_RUNNING=false
if curl -s --max-time 2 "http://localhost:$DASHBOARD_HTTP_PORT/" >/dev/null 2>&1; then
    DASHBOARD_RUNNING=true
    echo -e "${GREEN}✓ Dashboard already running${NC}"
else
    echo -e "${YELLOW}Starting dashboard...${NC}"
    cd "$PROJECT_DIR"
    nohup python scripts/web_dashboard.py > /tmp/dashboard.log 2>&1 &
    sleep 3
    if curl -s --max-time 2 "http://localhost:$DASHBOARD_HTTP_PORT/" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Dashboard started${NC}"
    else
        echo -e "${RED}Failed to start dashboard${NC}"
        exit 1
    fi
fi

# Check if simulator is running with correct dataset
RESTART_SIM=false
SIM_STATUS=$(curl -s --max-time 2 "http://localhost:$SIMULATOR_PORT/status" 2>/dev/null)
if [ -n "$SIM_STATUS" ]; then
    CUR_DS=$(echo "$SIM_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('dataset',''))" 2>/dev/null)
    if [ "$CUR_DS" = "$DATASET" ]; then
        echo -e "${GREEN}✓ Simulator already running with $DATASET${NC}"
    else
        echo -e "${YELLOW}Simulator has different dataset, restarting...${NC}"
        RESTART_SIM=true
    fi
else
    RESTART_SIM=true
fi

if [ "$RESTART_SIM" = true ]; then
    pkill -f "tum_rtsp_simulator" 2>/dev/null
    sleep 1
    echo -e "${YELLOW}Starting TUM simulator...${NC}"
    cd "$PROJECT_DIR"
    nohup python scripts/tools/tum_rtsp_simulator.py \
        --dataset "$DATASET_FULL_PATH" \
        --port $SIMULATOR_PORT \
        --fps $SIMULATOR_FPS \
        > /tmp/tum_simulator.log 2>&1 &
    sleep 3
    if curl -s --max-time 2 "http://localhost:$SIMULATOR_PORT/status" >/dev/null 2>&1; then
        echo -e "${GREEN}✓ Simulator started${NC}"
    else
        echo -e "${RED}Failed to start simulator${NC}"
        exit 1
    fi
fi

# Add live source via API
echo -e "${YELLOW}Configuring live source...${NC}"
STREAM_URL="http://localhost:$SIMULATOR_PORT/stream"

python3 << EOF
import websocket
import json
import time

def on_message(ws, msg):
    d = json.loads(msg)
    if d.get('type') == 'init':
        # Remove existing
        for ds in d.get('datasets', []):
            if ds.get('name') == '$LIVE_SOURCE_NAME':
                ws.send(json.dumps({'cmd': 'remove_live', 'name': '$LIVE_SOURCE_NAME'}))
                time.sleep(0.2)
        # Add new
        ws.send(json.dumps({
            'cmd': 'add_live',
            'name': '$LIVE_SOURCE_NAME',
            'source': '$STREAM_URL',
            'pose_method': 'ground_truth',
            'depth_method': 'ground_truth'
        }))
        time.sleep(0.3)
        print("✓ Live source configured")
        ws.close()

ws = websocket.WebSocketApp('ws://localhost:$DASHBOARD_WS_PORT', on_message=on_message)
import threading
threading.Timer(5, ws.close).start()
ws.run_forever()
EOF

echo ""
echo -e "${GREEN}Ready!${NC}"
echo -e "Dashboard: ${BLUE}http://localhost:$DASHBOARD_HTTP_PORT${NC}"
echo -e "Stream:    ${BLUE}http://localhost:$SIMULATOR_PORT/stream${NC}"
echo ""
echo "Select '$LIVE_SOURCE_NAME' in dashboard and click Start"
