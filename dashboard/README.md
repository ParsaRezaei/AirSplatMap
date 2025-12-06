# AirSplatMap Web Dashboard

Real-time 3D Gaussian Splatting visualization dashboard.

## Quick Start

```bash
# Navigate to dashboard directory first
cd dashboard

# Windows
start_dashboard.bat

# Linux/macOS
./start_dashboard.sh

# Then open browser
http://127.0.0.1:9002
```

## Features

- **Live 3DGS Visualization**: See Gaussian splatting in real-time
- **Multiple Sources**: TUM datasets, RTSP streams, webcams, video files
- **Pose Estimation**: ORB, SIFT, LoFTR, SuperPoint, and more
- **Depth Estimation**: MiDaS, Depth Anything, ZoeDepth
- **3D Point Cloud**: Interactive WebGL visualization
- **Session History**: Auto-save and restore sessions

## Files

- `web_dashboard.py` - Python backend (HTTP + WebSocket server)
- `web_dashboard.html` - Frontend UI
- `start_dashboard.bat/.sh` - Start scripts
- `stop_dashboard.bat/.sh` - Stop scripts

## Ports

- HTTP: 9002 (configurable with `--http-port`)
- WebSocket: 9003 (configurable with `--ws-port`)

## Adding Live Sources

### Via UI
1. Click "Add Live Source"
2. Enter source (RTSP URL, webcam index, video path)
3. Select pose/depth estimation methods
4. Click "Add"

### Via Script
```bash
python scripts/tools/add_realsense_to_dashboard.py
```

### Programmatically
```python
from dashboard.web_dashboard import Server

server = Server()
server.add_live_source(
    name="My Camera",
    source="rtsp://192.168.1.100:8554/camera",
    pose_method="orb",
    depth_method="midas"
)
```

## Configuration

Set environment variables or use command line:

```bash
# Environment
export AIRSPLAT_HTTP_PORT=9002
export AIRSPLAT_WS_PORT=9003

# Command line
python dashboard/web_dashboard.py --http-port 9002 --ws-port 9003
```
