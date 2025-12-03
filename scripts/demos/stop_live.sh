#!/bin/bash
#
# Stop AirSplatMap Live Services
#

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping AirSplatMap services...${NC}"

pkill -f "tum_rtsp_simulator" 2>/dev/null && echo -e "${GREEN}✓ Stopped TUM simulator${NC}"
pkill -f "web_dashboard.py" 2>/dev/null && echo -e "${GREEN}✓ Stopped dashboard${NC}"

fuser -k 8554/tcp 9002/tcp 9003/tcp 2>/dev/null

sleep 1
echo -e "${GREEN}Done.${NC}"
