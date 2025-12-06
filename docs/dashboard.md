# Web Dashboard

AirSplatMap includes a real-time web dashboard for visualizing 3D Gaussian Splatting.

## Quick Start

```bash
# Windows
cd dashboard
start_dashboard.bat

# Linux/macOS
cd dashboard
./start_dashboard.sh

# Open browser
http://127.0.0.1:9002
```

## Features

### Real-time Visualization

- **RGB Camera Feed** - Live input from camera or dataset
- **Depth View** - Depth map visualization
- **Rendered View** - Engine's rendered output
- **3D Point Cloud** - Interactive WebGL point cloud
- **3D Gaussians** - Colored Gaussian splats

### Metrics Charts

- FPS (frames per second)
- Loss curve
- PSNR (reconstruction quality)
- Gaussian count over time

### Dataset Management

- Auto-discover TUM RGB-D datasets
- Add custom live sources (RTSP, webcam, video)
- Run history with session replay

## Interface Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│  AirSplatMap Dashboard                               [Settings] [Help]  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │   Dataset    │  │    Engine    │  │   Controls   │                   │
│  │  [Dropdown]  │  │  [Dropdown]  │  │ [Start/Stop] │                   │
│  └──────────────┘  └──────────────┘  └──────────────┘                   │
│                                                                          │
│  ┌─────────────────────────┐  ┌─────────────────────────┐               │
│  │       RGB Feed          │  │      Rendered View       │               │
│  │                         │  │                          │               │
│  │                         │  │                          │               │
│  └─────────────────────────┘  └─────────────────────────┘               │
│                                                                          │
│  ┌─────────────────────────┐  ┌─────────────────────────┐               │
│  │    3D Point Cloud       │  │      Metrics Chart       │               │
│  │     (Interactive)       │  │                          │               │
│  │                         │  │                          │               │
│  └─────────────────────────┘  └─────────────────────────┘               │
│                                                                          │
│  Frame: 142/500  │  FPS: 17.2  │  Gaussians: 185,432  │  Loss: 0.0234   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Controls

### Dataset Selection

1. Choose dataset from dropdown
2. Datasets are auto-discovered from:
   - `datasets/` folder
   - `datasets/tum/` subfolder
   - Live sources you've added

### Engine Selection

| Engine | Best For |
|--------|----------|
| gsplat | Real-time, low memory |
| graphdeco | Best quality |
| monogs | SLAM with tracking |

### Start/Stop

- **Start**: Begin processing frames
- **Stop**: Pause processing
- **Reset**: Clear current scene

### Settings

| Setting | Description | Default |
|---------|-------------|---------|
| Target FPS | Frame processing rate | 30 |
| Max Points | 3D visualization limit | 20,000 |
| Splat Size | Gaussian render size | 2.0 |
| Splat Opacity | Gaussian transparency | 0.7 |
| Render Mode | Auto/Native/Software | Auto |

## Adding Live Sources

### Via Dashboard UI

1. Click "Add Live Source"
2. Fill in the form:
   - **Name**: Display name
   - **Source**: URL or camera index
   - **Pose Method**: Visual odometry method
   - **Depth Method**: Depth estimation method
3. Click "Add"

### Source Types

```
# Webcam
0                           # Camera index 0
1                           # Camera index 1

# RTSP stream
rtsp://192.168.1.100:8554/camera

# HTTP stream
http://localhost:8554/stream

# Video file
/path/to/video.mp4
C:\Videos\recording.avi
```

### Via Script

```bash
python scripts/tools/add_realsense_to_dashboard.py
```

### Programmatic

```python
import requests

# Add live source
response = requests.post("http://localhost:9002/api/add_source", json={
    "name": "my_camera",
    "source": "rtsp://192.168.1.100:8554/camera",
    "pose_method": "orb",
    "depth_method": "midas"
})
```

## Configuration

### Ports

Default ports:
- HTTP: 9002
- WebSocket: 9003

Custom ports:
```bash
# Via command line
python dashboard/web_dashboard.py --http-port 8080 --ws-port 8081

# Via script
dashboard\start_dashboard.bat --http-port 8080 --ws-port 8081
```

### Environment Variables

```bash
export AIRSPLAT_HTTP_PORT=9002
export AIRSPLAT_WS_PORT=9003
export AIRSPLAT_CONDA_ENV=airsplatmap
```

## WebSocket API

The dashboard uses WebSocket for real-time updates.

### Connect

```javascript
const ws = new WebSocket('ws://localhost:9003');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data.type, data);
};
```

### Message Types

**Server → Client:**

```javascript
// Initialization
{ "type": "init", "datasets": [...], "engines": [...] }

// Frame update
{ "type": "frame", "rgb": "base64...", "depth": "base64...", "frame_id": 42 }

// Metrics update
{ "type": "metrics", "fps": 17.2, "loss": 0.023, "gaussians": 185432, "psnr": 24.5 }

// Point cloud update
{ "type": "points", "positions": [...], "colors": [...] }

// Gaussians update
{ "type": "gaussians", "positions": [...], "colors": [...], "sizes": [...] }

// Status
{ "type": "status", "running": true, "frame": 142, "total": 500 }
```

**Client → Server:**

```javascript
// Start processing
{ "cmd": "start", "dataset": "fr1_desk", "engine": "gsplat" }

// Stop processing
{ "cmd": "stop" }

// Change settings
{ "cmd": "settings", "target_fps": 30, "max_points": 50000 }

// Add live source
{ "cmd": "add_live", "name": "camera", "source": "0", "pose_method": "orb" }
```

## Python API

### Server Class

```python
from dashboard.web_dashboard import Server

# Create server
server = Server(http_port=9002, ws_port=9003)

# Add live source programmatically
server.add_live_source(
    name="realsense",
    source="rtsp://localhost:8554/camera",
    pose_method="orb",
    depth_method="midas"
)

# Start server
server.start()  # Blocking
# or
server.start_background()  # Non-blocking
```

### Custom Callbacks

```python
# Custom frame callback
def on_frame(frame_data):
    print(f"Frame {frame_data['frame_id']}")

server.on_frame_callback = on_frame

# Custom metrics callback  
def on_metrics(metrics):
    print(f"FPS: {metrics['fps']}")

server.on_metrics_callback = on_metrics
```

## Run History

The dashboard saves run history automatically:

- **Location**: `output/.history/`
- **Contents**: Frame snapshots, metrics, Gaussian states
- **Replay**: Select from "History" dropdown

### View History

```python
from dashboard.web_dashboard import Server

server = Server()
history = server.get_history()

for run in history:
    print(f"{run['timestamp']}: {run['dataset']} - {run['frames']} frames")
```

## Troubleshooting

### Dashboard Won't Start

```bash
# Check if port is in use
netstat -an | findstr 9002

# Kill existing process
dashboard\stop_dashboard.bat
```

### WebSocket Connection Failed

```bash
# Verify WebSocket server
python -c "import websockets; print('websockets OK')"

# Check firewall settings
# Allow ports 9002 and 9003
```

### 3D View Not Loading

- Ensure WebGL is enabled in browser
- Try Chrome or Firefox (best WebGL support)
- Check browser console for errors

### Slow Performance

1. Reduce Max Points setting
2. Use gsplat engine
3. Lower Target FPS
4. Close other GPU applications

### No Datasets Found

```bash
# Check dataset location
ls datasets/
ls datasets/tum/

# Dataset should have:
# - rgb/ folder with images
# - rgb.txt file
```

## Browser Support

| Browser | Support | Notes |
|---------|---------|-------|
| Chrome | ✅ Best | Recommended |
| Firefox | ✅ Good | WebGL2 support |
| Edge | ✅ Good | Chromium-based |
| Safari | ⚠️ Limited | WebGL issues |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Start/Stop |
| R | Reset scene |
| F | Fullscreen |
| 1-4 | Switch view tabs |
| +/- | Zoom 3D view |
