# AirSplatMap Documentation

Welcome to AirSplatMap - a real-time 3D Gaussian Splatting pipeline for drones, robots, and cameras.

## Quick Links

- [Getting Started](getting_started.md) - Installation and first run
- [Architecture](architecture.md) - System design overview
- [Engines](engines.md) - 3DGS engine comparison
- [Pose Estimation](pose_estimation.md) - Visual odometry methods
- [Depth Estimation](depth_estimation.md) - Monocular depth methods
- [Dashboard](dashboard.md) - Web dashboard usage
- [Benchmarks](benchmarks.md) - Running evaluations
- [API Reference](api_reference.md) - Python API docs
- [ArduPilot Integration](ardupilot_integration.md) - Drone/rover support

## What is AirSplatMap?

AirSplatMap is a modular framework for **real-time 3D reconstruction** using Gaussian Splatting. It's designed for:

- **Drones** - Aerial mapping with ArduPilot/MAVLink
- **Robots** - Mobile robot SLAM
- **Handheld cameras** - RealSense, webcams, video files
- **Research** - Benchmarking and algorithm development

## Key Features

| Feature | Description |
|---------|-------------|
| 🚀 **Multiple Engines** | GraphDeco, GSplat, MonoGS, SplaTAM, Photo-SLAM |
| 📍 **Pose Estimation** | ORB, SIFT, LoFTR, SuperPoint, RealSense VIO |
| 🎯 **Depth Estimation** | MiDaS, Depth Anything, ZoeDepth |
| 🌐 **Web Dashboard** | Real-time 3D visualization |
| 📊 **Benchmarks** | Automated evaluation with plots |
| 🤖 **ArduPilot** | MAVLink integration for drones |

## Project Structure

```
AirSplatMap/
├── src/                    # Core library
│   ├── engines/           # 3DGS backends
│   ├── pipeline/          # Frame sources & orchestration
│   ├── pose/              # Visual odometry
│   ├── depth/             # Depth estimation
│   └── viewer/            # Visualization
├── dashboard/             # Web dashboard
├── benchmarks/            # Evaluation suite
├── notebooks/             # Jupyter notebooks
├── scripts/               # Demos and tools
├── submodules/            # External dependencies
└── docs/                  # Documentation
```

## Supported Hardware

- **Intel RealSense** D435, D455, L515
- **Webcams** - Any USB camera
- **Drones** - ArduPilot-compatible (PX4, ArduCopter)
- **NVIDIA GPU** - Required for real-time performance

## License

MIT License - See [LICENSE](../LICENSE)
