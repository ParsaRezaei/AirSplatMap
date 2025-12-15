# AirSplatMap Documentation

Welcome to AirSplatMap - a real-time 3D Gaussian Splatting pipeline for drones, robots, and cameras.

---

## 🌐 Online Resources

| Resource | Link |
|----------|------|
| 📊 **Interactive Benchmark Viewer** | [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/) |
| 💻 **GitHub Repository** | [github.com/ParsaRezaei/AirSplatMap](https://github.com/ParsaRezaei/AirSplatMap) |
| 🔬 **Research Papers** | [papers/](../papers/README.md) |
| 📈 **Benchmark Results** | [benchmarks/results/](../benchmarks/results/index.html) |

---

## Quick Links

### Getting Started
- [Getting Started](getting_started.md) - Installation and first run
- [Architecture](architecture.md) - System design overview

### Core Components
- [Engines](engines.md) - 3DGS engine comparison (gsplat, graphdeco, monogs, etc.)
- [Pose Estimation](pose_estimation.md) - Visual odometry methods (ORB, SIFT, LoFTR, etc.)
- [Depth Estimation](depth_estimation.md) - Monocular depth methods (MiDaS, Depth Pro, etc.)

### Tools & Visualization
- [Dashboard](dashboard.md) - Web dashboard usage
- [Benchmarks](benchmarks.md) - Running evaluations
- [API Reference](api_reference.md) - Python API docs

### Integration
- [ArduPilot Integration](ardupilot_integration.md) - Drone/rover support

---

## What is AirSplatMap?

AirSplatMap is a modular framework for **real-time 3D reconstruction** using Gaussian Splatting. It's designed for:

- **Drones** - Aerial mapping with ArduPilot/MAVLink ([learn more](ardupilot_integration.md))
- **Robots** - Mobile robot SLAM
- **Handheld cameras** - RealSense, webcams, video files
- **Research** - Benchmarking and algorithm development ([view benchmarks](https://ParsaRezaei.github.io/AirSplatMap/))

## Key Features

| Feature | Description | Documentation |
|---------|-------------|---------------|
| 🚀 **Multiple Engines** | GraphDeco, GSplat, MonoGS, SplaTAM, Photo-SLAM, Gaussian-SLAM | [Engines Guide](engines.md) |
| 📍 **Pose Estimation** | ORB, SIFT, LoFTR, SuperPoint, LightGlue, RoMa, RAFT | [Pose Guide](pose_estimation.md) |
| 🎯 **Depth Estimation** | MiDaS, Depth Anything V2/V3, Apple Depth Pro | [Depth Guide](depth_estimation.md) |
| 🌐 **Web Dashboard** | Real-time 3D visualization | [Dashboard Guide](dashboard.md) |
| 📊 **Benchmarks** | Automated evaluation with interactive reports | [Benchmarks Guide](benchmarks.md) |
| 🤖 **ArduPilot** | MAVLink integration for drones | [ArduPilot Guide](ardupilot_integration.md) |

## Project Structure

```
AirSplatMap/
├── src/                    # Core library
│   ├── engines/           # 3DGS backends (gsplat, graphdeco, monogs...)
│   ├── pipeline/          # Frame sources & orchestration
│   ├── pose/              # Visual odometry
│   ├── depth/             # Depth estimation
│   └── viewer/            # Visualization
├── dashboard/             # Web dashboard
├── benchmarks/            # Evaluation suite
│   └── results/           # Interactive HTML reports
├── notebooks/             # Jupyter notebooks
├── scripts/               # Demos and tools
├── submodules/            # External dependencies
├── papers/                # Research papers (CVPR format)
└── docs/                  # Documentation (you are here)
```

## Supported Hardware

### Cameras
- **Intel RealSense** D435, D455, L515
- **Webcams** - Any USB camera
- **IP Cameras** - RTSP/HTTP streams

### Compute
- **Desktop** - NVIDIA RTX 20xx+ (CUDA 12.x)
- **Edge** - NVIDIA Jetson Orin (JetPack 6.x)
- **Experimental** - Apple Silicon (MPS)

### Vehicles
- **Drones** - ArduPilot-compatible (PX4, ArduCopter)
- **Rovers** - Ground vehicles with MAVLink

## Next Steps

1. 📥 **Install**: Follow the [Getting Started](getting_started.md) guide
2. 🎬 **Try the Dashboard**: Run the [Web Dashboard](dashboard.md)
3. 📊 **View Benchmarks**: Explore results at [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/)
4. 🔧 **Customize**: Read the [API Reference](api_reference.md)

## License

MIT License - See [LICENSE](../LICENSE)
