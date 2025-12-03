# ArduPilot Integration for AirSplatMap

This guide explains how to use AirSplatMap with ArduPilot-based drones, rovers, boats, and other vehicles for real-time 3D Gaussian Splatting mapping.

## Overview

The ArduPilot integration allows you to:
- Stream camera video from a vehicle-mounted camera
- Get real-time pose (position + orientation) from the flight controller
- Build 3D Gaussian Splat maps in real-time or post-process recordings

## Requirements

### Python Dependencies
```bash
pip install pymavlink opencv-python numpy torch
```

### Hardware
- **Flight Controller**: Pixhawk, Cube, or any ArduPilot-compatible FC
- **Camera**: USB webcam, Raspberry Pi camera, IP camera, or RTSP stream
- **Connection**: Serial (USB), UDP, or TCP to the flight controller

## Connection Types

### Serial (USB)
```bash
# Linux
--mavlink /dev/ttyUSB0

# Windows
--mavlink COM3

# With baud rate
--mavlink /dev/ttyUSB0,115200
```

### UDP (Mission Planner / QGC)
```bash
# Receive from Mission Planner/QGC (they forward to this port)
--mavlink udpin:0.0.0.0:14550

# Send to a specific address
--mavlink udpout:192.168.1.100:14550
```

### TCP
```bash
# Connect to SITL
--mavlink tcp:127.0.0.1:5762

# Connect to MAVProxy
--mavlink tcp:localhost:5760
```

## Quick Start

### 1. Test Connection
First, verify you can connect to your vehicle:

```bash
python scripts/demos/live_ardupilot_demo.py --test --mavlink udpin:0.0.0.0:14550
```

Expected output:
```
Testing connection to: udpin:0.0.0.0:14550
Waiting for heartbeat (timeout 30s)...
✓ Connection successful!

Vehicle Status:
  Armed: False
  GPS Fix: 3 (12 satellites)
  Battery: 12.4V (87%)
  Attitude: Roll=0.5° Pitch=-2.1° Yaw=45.3°
  Position (NED): [0.12, -0.34, -0.05]
```

### 2. Pose Monitor Mode
View live pose data without mapping:

```bash
# With camera display
python scripts/demos/live_ardupilot_demo.py --pose-only --mavlink udpin:0.0.0.0:14550 --camera 0

# Without camera (console only)
python scripts/demos/live_ardupilot_demo.py --pose-only --mavlink udpin:0.0.0.0:14550
```

### 3. Live Mapping
Run full 3DGS mapping with ArduPilot poses:

```bash
python scripts/demos/live_ardupilot_demo.py \
    --mavlink udpin:0.0.0.0:14550 \
    --camera 0 \
    --output output/drone_mapping
```

## Camera Configuration

### USB Webcam
```bash
--camera 0              # First webcam
--camera 1              # Second webcam
--width 640 --height 480
```

### IP Camera / RTSP
```bash
--camera "rtsp://192.168.1.100:8554/main"
--camera "http://192.168.1.100:8080/video"
```

### GStreamer Pipeline
```bash
# Raspberry Pi Camera (libcamera)
--camera "libcamerasrc ! video/x-raw,width=640,height=480 ! videoconvert ! appsink"

# NVIDIA Jetson CSI camera
--camera "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=640,height=480 ! nvvidconv ! videoconvert ! appsink"
```

### Camera Mounting
Configure the camera mounting angle relative to the vehicle:

```bash
# Camera pointing straight down
--camera-pitch -90

# Camera at 45° angle (typical for mapping)
--camera-pitch -45

# Camera pointing forward
--camera-pitch 0
```

## Remote Streaming Setup

For drones with companion computers (Raspberry Pi, Jetson), run the server on the companion and connect from your workstation:

### On Companion Computer
```bash
python scripts/tools/ardupilot_server.py \
    --camera 0 \
    --mavlink /dev/ttyAMA0 \
    --port 8554
```

### On Workstation
```bash
# View stream in browser
http://companion-ip:8554/

# Run mapping
python scripts/demos/live_web_demo.py --url http://companion-ip:8554
```

### Server Endpoints
| Endpoint | Description |
|----------|-------------|
| `GET /` | MJPEG video stream |
| `GET /frame` | Single JPEG with pose in headers |
| `GET /bundle` | JSON with base64 image + pose |
| `GET /pose` | Current 4x4 pose matrix |
| `GET /status` | Vehicle and server status |
| `GET /intrinsics` | Camera intrinsics |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        Drone / Vehicle                        │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐      │
│  │   Camera    │    │  Pixhawk    │    │  Companion  │      │
│  │  (USB/CSI)  │───▶│ (ArduPilot) │───▶│   (RPi/    │      │
│  └─────────────┘    └─────────────┘    │   Jetson)  │      │
│                           │            └──────┬──────┘      │
│                           │ MAVLink           │ WiFi/LTE    │
│                           ▼                   ▼             │
└───────────────────────────┼───────────────────┼─────────────┘
                            │                   │
                    ┌───────┴───────┐   ┌───────┴───────┐
                    │ Ground Station│   │  HTTP Server  │
                    │(Mission Planner)  │  (port 8554)  │
                    └───────┬───────┘   └───────┬───────┘
                            │                   │
                            ▼                   ▼
                    ┌───────────────────────────────────┐
                    │         AirSplatMap               │
                    │  ┌─────────────────────────────┐  │
                    │  │   ArduPilotPoseProvider     │  │
                    │  │   (MAVLink pose data)       │  │
                    │  └─────────────────────────────┘  │
                    │  ┌─────────────────────────────┐  │
                    │  │      ArduPilotSource        │  │
                    │  │   (Camera + Pose fusion)    │  │
                    │  └─────────────────────────────┘  │
                    │  ┌─────────────────────────────┐  │
                    │  │      3DGS Pipeline          │  │
                    │  │   (Real-time mapping)       │  │
                    │  └─────────────────────────────┘  │
                    └───────────────────────────────────┘
```

## SITL Simulation

Test without hardware using ArduPilot SITL:

### Start SITL
```bash
# In ArduPilot directory
sim_vehicle.py -v ArduCopter --console --map
```

### Connect AirSplatMap
```bash
python scripts/demos/live_ardupilot_demo.py \
    --mavlink tcp:127.0.0.1:5762 \
    --camera 0 \
    --pose-only
```

## Coordinate Frames

### ArduPilot (NED)
- X = North
- Y = East  
- Z = Down

### AirSplatMap (Camera/OpenCV)
- X = Right
- Y = Down
- Z = Forward

The `ArduPilotPoseProvider` automatically converts between these frames.

## Troubleshooting

### No Heartbeat
```
Waiting for heartbeat...
✗ Connection failed!
```
- Check MAVLink connection string
- Verify ArduPilot is powered and running
- Check serial port permissions: `sudo usermod -a -G dialout $USER`
- Try different baud rates for serial

### No Position Data
- Ensure GPS has fix (3D fix recommended)
- Check if LOCAL_POSITION_NED messages are enabled
- Verify EKF is healthy in Mission Planner

### Camera Not Opening
- Check camera index/URL is correct
- Verify camera permissions
- Try different backends: `cv2.VideoCapture(0, cv2.CAP_V4L2)`

### High Latency
- Reduce resolution: `--width 320 --height 240`
- Lower target FPS: `--target-fps 10`
- Use wired connection instead of WiFi
- Enable hardware encoding on companion computer

## Python API Usage

```python
from src.pose.ardupilot_mavlink import ArduPilotPoseProvider
from src.pipeline.ardupilot_source import ArduPilotSource

# Just get poses from ArduPilot
provider = ArduPilotPoseProvider("udpin:0.0.0.0:14550")
provider.start()

pose = provider.get_pose()       # 4x4 camera-to-world matrix
state = provider.get_state()     # ArduPilotState with all telemetry
attitude = provider.get_attitude()  # (roll, pitch, yaw) in radians

provider.stop()

# Full frame source with camera + poses
source = ArduPilotSource(
    camera_source=0,
    mavlink_connection="udpin:0.0.0.0:14550",
    camera_pitch_deg=-45,
)

for frame in source:
    # frame.rgb - RGB image
    # frame.pose - 4x4 pose from ArduPilot
    # frame.depth - Estimated depth (if enabled)
    process(frame)

source.stop()
```

## Tips for Best Results

1. **GPS Quality**: Wait for good GPS fix (>10 satellites) before mapping
2. **Fly Slow**: Slower movement = better pose accuracy and less motion blur
3. **Overlap**: Ensure sufficient visual overlap between frames
4. **Lighting**: Consistent lighting improves tracking and mapping
5. **Camera Angle**: 45° down angle works well for terrain mapping
6. **Stabilization**: Use gimbal if available for smoother video

## Files

| File | Description |
|------|-------------|
| `src/pose/ardupilot_mavlink.py` | MAVLink pose provider |
| `src/pipeline/ardupilot_source.py` | Frame source combining camera + ArduPilot |
| `scripts/demos/live_ardupilot_demo.py` | Interactive demo script |
| `scripts/tools/ardupilot_server.py` | HTTP server for remote streaming |
