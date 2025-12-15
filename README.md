# AirSplatMap

Real-time 3D Gaussian Splatting for drones, robots, and cameras.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub%20Pages-blue)](https://ParsaRezaei.github.io/AirSplatMap/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![CUDA 12.x](https://img.shields.io/badge/CUDA-12.x-brightgreen.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 🌐 Live Demo & Resources

| Resource | Link |
|----------|------|
| 📊 **Interactive Benchmark Viewer** | [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/) |
| 📖 **Full Documentation** | [docs/](docs/index.md) |
| 🔬 **Research Papers** | [papers/](papers/README.md) |
| 📈 **Benchmark Results** | [benchmarks/results/](benchmarks/results/index.html) |
| 🎬 **Web Dashboard** | [dashboard/](dashboard/README.md) |
| 🐛 **Issue Tracker** | [GitHub Issues](https://github.com/ParsaRezaei/AirSplatMap/issues) |

---

## Overview

AirSplatMap is a modular framework for **real-time 3D reconstruction** using Gaussian Splatting. Stream RGB-D from cameras, drones, or datasets and watch 3D maps build in real-time.

```
Camera/Drone → Pose Estimation → 3D Gaussian Splatting → Web Dashboard
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🚀 **Multiple 3DGS Engines** | [GSplat, GraphDeco, MonoGS, SplaTAM, Photo-SLAM, Gaussian-SLAM](docs/engines.md) |
| 📍 **Visual Odometry** | [ORB, SIFT, LoFTR, SuperPoint, LightGlue, RoMa, RAFT](docs/pose_estimation.md) |
| 🎯 **Depth Estimation** | [MiDaS, Depth Anything V2/V3, Apple Depth Pro](docs/depth_estimation.md) |
| 🌐 **Web Dashboard** | [Real-time 3D visualization in browser](docs/dashboard.md) |
| 📊 **Benchmarks** | [Automated evaluation with interactive HTML reports](docs/benchmarks.md) |
| 🤖 **ArduPilot** | [MAVLink integration for drones/rovers](docs/ardupilot_integration.md) |
| 🖥️ **Edge Deployment** | Full Jetson Orin support (JetPack 6.x) |

## 🖥️ Supported Platforms

| Platform | GPU | Status |
|----------|-----|--------|
| **Ubuntu 20.04/22.04** | NVIDIA RTX 20xx+ | ✅ Fully supported |
| **Windows 10/11** | NVIDIA RTX 20xx+ | ✅ Fully supported |
| **NVIDIA Jetson** | Orin (JetPack 6.x) | ✅ Fully supported |
| macOS | Apple Silicon | ⚠️ Experimental (MPS) |

### Jetson Support

Full support for **NVIDIA Jetson Orin** devices (Nano, NX, AGX) running JetPack 6.x:
- Automatic detection of Jetson platform
- NVIDIA's Jetson PyTorch wheel with SM 8.7 support
- All 6 engines working: gsplat, graphdeco, monogs, splatam, gslam, da3gs

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Getting Started](docs/getting_started.md) | Installation and first run |
| [Architecture](docs/architecture.md) | System design overview |
| [Engines](docs/engines.md) | 3DGS engine comparison |
| [Pose Estimation](docs/pose_estimation.md) | Visual odometry methods |
| [Depth Estimation](docs/depth_estimation.md) | Monocular depth methods |
| [Dashboard](docs/dashboard.md) | Web dashboard usage |
| [Benchmarks](docs/benchmarks.md) | Running evaluations |
| [API Reference](docs/api_reference.md) | Python API docs |
| [ArduPilot](docs/ardupilot_integration.md) | Drone/rover integration |

> 📘 **New to AirSplatMap?** Start with [Getting Started](docs/getting_started.md) for installation and your first 3D reconstruction!

## 🚀 Quick Start

### Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/ParsaRezaei/AirSplatMap.git
cd AirSplatMap

# Run setup script (creates conda env + installs PyTorch with CUDA)
./setup_env.sh          # Linux/macOS/Jetson
# .\setup_env.ps1       # Windows (PowerShell)

# Activate environment
conda activate airsplatmap

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

> 📖 For detailed installation instructions, see [Getting Started](docs/getting_started.md).

### Run Web Dashboard

```bash
cd dashboard
./start_dashboard.sh      # Linux/macOS
# start_dashboard.bat     # Windows

# Open browser: http://127.0.0.1:9002
```

> 🎬 Learn more about the dashboard in the [Dashboard Guide](docs/dashboard.md).

### Run TUM Dataset Demo

```bash
python scripts/demos/live_tum_demo.py --sequence fr1_desk --engine gsplat
```

### Run with RealSense Camera

```bash
python scripts/demos/live_realsense_demo.py --engine gsplat
```

### Run Benchmarks

```bash
# Quick benchmark
python -m benchmarks.run --quick

# View interactive results
# Open benchmarks/results/<hostname>/benchmark_<timestamp>/report.html
# Or view online: https://ParsaRezaei.github.io/AirSplatMap/
```

> 📊 See [Benchmarks Guide](docs/benchmarks.md) for comprehensive evaluation options.

## 📖 Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   FrameSource   │────▶ OnlineGSPipeline  ────▶   BaseGSEngine   │
│  (TUM, COLMAP,  │     │                  │     │  (Graphdeco,    │
│   video, etc.)  │     │  - Frame loop    │     │   SplaTAM, etc) │
└─────────────────┘     │  - RS correction │     └─────────────────┘
                        │  - Metrics       │
                        └──────────────────┘
```

### Key Components

- **`src/engines/base.py`**: Abstract `BaseGSEngine` interface that all 3DGS backends must implement
- **`src/engines/graphdeco_engine.py`**: Implementation using the original Graphdeco 3DGS codebase
- **`src/pipeline/frames.py`**: `Frame` dataclass and `FrameSource` abstractions (includes `TumRGBDSource`)
- **`src/pipeline/online_gs.py`**: Main `OnlineGSPipeline` orchestrator
- **`src/pipeline/rs_corrector.py`**: Rolling shutter correction abstractions (placeholder for now)

## Installation

### Quick Start - Cross-Platform (Windows/Linux/Jetson)

```bash
# Clone with all submodules
git clone --recursive https://github.com/ParsaRezaei/AirSplatMap.git
cd AirSplatMap

# Run the setup script (creates conda env + installs PyTorch with CUDA)
./setup_env.sh          # Linux/macOS/Jetson
# .\setup_env.ps1       # Windows (PowerShell)

# Activate environment
conda activate airsplatmap

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from src.engines import list_engines; print([k for k,v in list_engines().items() if v['available']])"
```

The setup script handles:
- Creating the conda environment from `environment.yml`
- **Jetson**: Installs NVIDIA's Jetson PyTorch wheel with SM 8.7 (Orin) support
- **Desktop**: Installs PyTorch with CUDA 12.4 support
- Building gsplat from source (with correct CUDA architecture)
- Setting up libstdc++ compatibility on Jetson

### Linux-Only: Build CUDA Extensions (Optional)

Only needed if you want to use the `graphdeco` engine (original 3DGS):

```bash
cd submodules/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation -e .
cd ../simple-knn
pip install --no-build-isolation -e .
cd ../../../..
```

### Windows: Build CUDA Extensions (Optional)

To use the `graphdeco`, `monogs`, or `splatam` engines on Windows, you need to compile the CUDA extensions:

**Prerequisites:**
1. Install [Visual Studio Build Tools 2019](https://aka.ms/vs/16/release/vs_buildtools.exe)
   - During installation, select "C++ build tools" workload
   - Make sure "MSVC v142" and "Windows 10 SDK" are checked
2. CUDA Toolkit 12.1+ (should match your PyTorch CUDA version)

**Build the extensions:**
```cmd
cd submodules\gaussian-splatting\submodules\diff-gaussian-rasterization
pip install --no-build-isolation -e .
cd ..\simple-knn
pip install --no-build-isolation -e .
cd ..\..\..\..
```

**Verify installation:**
```python
from simple_knn._C import distCUDA2
print("simple_knn OK")
```

### RealSense Camera Setup

```bash
# Test camera detection
python scripts/demos/live_realsense_demo.py --list-devices

# Run live demo
python scripts/demos/live_realsense_demo.py
```

**Tips:**
- Use USB 3.0 ports (blue connector)
- On Windows, install [Intel RealSense Viewer](https://github.com/IntelRealSense/librealsense/releases) to verify camera works

### If Already Cloned Without Submodules

```bash
cd AirSplatMap
git submodule update --init --recursive
```

### Submodule-Only Mode (Recommended)

To ensure you're using the bundled submodules (with fixes applied):

```bash
export AIRSPLAT_USE_SUBMODULES=1
```

### Available Engines

| Engine | Speed | Real-time | Description | Requirements |
|--------|-------|-----------|-------------|--------------|
| `graphdeco` | ~2-5 FPS | ❌ | Original 3DGS from GRAPHDECO | Included as submodule |
| `gsplat` | ~17 FPS | ✅ | Nerfstudio's optimized 3DGS (4x less memory) | `pip install gsplat` (in environment.yml) |
| `splatam` | ~0.4 FPS | ❌ | RGB-D SLAM with Gaussian Splatting | Included as submodule |
| `monogs` | ~10 FPS | ✅ | Gaussian Splatting SLAM (CVPR'24 Highlight) | Included as submodule |
| `photoslam` | Real-time | ✅ | Photo-SLAM - Photorealistic SLAM (CVPR'24) | Included as submodule (requires C++ build) |
| `gslam` | ~5 FPS | ❌ | Gaussian-SLAM with submaps | Included as submodule |

```python
from src.engines import get_engine, list_engines

# List available engines with details
for name, info in list_engines().items():
    print(f"{name}: {'✅' if info['available'] else '❌'} {info['description']}")

# Get a specific engine
engine = get_engine("gsplat")  # or "graphdeco", "splatam", "monogs", "photoslam"
```

### Bundled Submodules

All external dependencies are included as git submodules with necessary fixes applied:

| Submodule | Upstream | Fixes Applied |
|-----------|----------|---------------|
| `submodules/gaussian-splatting` | graphdeco-inria/gaussian-splatting | Output path naming, grayscale SSIM |
| `submodules/MonoGS` | muskie82/MonoGS | CUDA multiprocessing, NumPy deprecation |
| `submodules/Gaussian-SLAM` | VladimirYugay/Gaussian-SLAM | NumPy deprecation |
| `submodules/Photo-SLAM` | HuajianUP/Photo-SLAM | PyTorch API fix |
| `submodules/SplaTAM` | spla-tam/SplaTAM | - |

**Photo-SLAM (C++ based, requires additional build):**
```bash
cd submodules/Photo-SLAM
./build.sh  # Requires LibTorch, OpenCV with CUDA
```

## Web Dashboard

AirSplatMap includes a real-time web dashboard for visualizing and controlling 3DGS mapping runs.

### Features

- **Real-time visualization**: 3D point cloud, Gaussian splats, camera feed, and rendered views
- **Multiple engines**: Switch between gsplat, graphdeco, splatam, monogs, and more
- **Live metrics**: FPS, loss, PSNR, Gaussian count with real-time charts
- **Run history**: Replay past runs frame-by-frame with full state reconstruction
- **Configurable settings**: Target FPS, point limits, splat size, render mode

### Quick Start

```bash
# Start the dashboard
cd dashboard
./start_dashboard.sh      # Linux/macOS
start_dashboard.bat       # Windows

# Open in browser
# http://localhost:9002

# Stop the dashboard
./stop_dashboard.sh       # Linux/macOS
stop_dashboard.bat        # Windows
```

### Dashboard Controls

| Panel | Description |
|-------|-------------|
| **Camera** | Live RGB feed from the dataset |
| **Render** | Engine's rendered view (native CUDA or software) |
| **3D Points** | Cumulative point cloud visualization |
| **3D Splats** | Colored Gaussian splats with depth |
| **Chart** | Real-time metrics (FPS, loss, PSNR, Gaussians) |

### Configuration

Dashboard ports can be customized via command line:

```bash
# Default configuration
HTTP_PORT=9002      # Web interface port
WS_PORT=9003        # WebSocket port for real-time updates
```

### Render Modes

- **Auto**: Use native engine renderer, fall back to software if unavailable
- **Native Only**: Force native CUDA renderer (blank if fails)
- **Software Only**: Point-cloud projection rendering (works with any engine)

### Custom Ports

```bash
cd dashboard
./start_dashboard.sh --http-port 8080 --ws-port 8081   # Linux/macOS
start_dashboard.bat --http-port 8080 --ws-port 8081    # Windows
```

## Usage

### Quick Start with TUM RGB-D Dataset

```bash
# Run from AirSplatMap directory
python scripts/demos/online_tum_graphdeco.py \
    --dataset-root ../datasets \
    --max-frames 200 \
    --steps-per-frame 5 \
    --render-every 50
```

### Python API

```python
from src.engines import GraphdecoEngine
from src.pipeline import OnlineGSPipeline
from src.pipeline.frames import TumRGBDSource

# Create components
engine = GraphdecoEngine()
source = TumRGBDSource(dataset_root="../datasets")

# Create pipeline
pipeline = OnlineGSPipeline(
    engine=engine,
    frame_source=source,
    steps_per_frame=5,
    render_every=50,
)

# Run
summary = pipeline.run(max_frames=200)

# Or step manually
for frame in source:
    metrics = pipeline.step(frame)
    if frame.idx % 50 == 0:
        preview = engine.render_view(frame.pose, frame.image_size)

# Save results
pipeline.save_final()
```

### Adding a New Engine

Implement the `BaseGSEngine` interface:

```python
from src.engines.base import BaseGSEngine

class MyEngine(BaseGSEngine):
    def initialize_scene(self, intrinsics, config):
        # Setup your 3DGS implementation
        pass
    
    def add_frame(self, frame_id, rgb, depth, pose_world_cam):
        # Add a new observation
        pass
    
    def optimize_step(self, n_steps=1):
        # Run optimization
        return {'loss': 0.0, 'num_gaussians': 0}
    
    def render_view(self, pose_world_cam, image_size):
        # Render from a pose
        return np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
    
    def save_state(self, path):
        # Save to disk
        pass
    
    def load_state(self, path):
        # Load from disk
        pass
```

### Adding a New Frame Source

Implement the `FrameSource` interface:

```python
from src.pipeline.frames import FrameSource, Frame

class MySource(FrameSource):
    def __iter__(self):
        for i, data in enumerate(self.data):
            yield Frame(
                idx=i,
                timestamp=data['ts'],
                rgb=data['rgb'],
                depth=data['depth'],
                pose=data['pose'],
                intrinsics=self.intrinsics,
            )
    
    def __len__(self):
        return len(self.data)
```

## 📦 Project Structure

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
├── notebooks/             # Jupyter notebooks
├── scripts/               # Demos and tools
├── submodules/            # External dependencies
└── docs/                  # Documentation
```

## 🎮 Available Engines

| Engine | Speed | Real-time | Best For | Docs |
|--------|-------|-----------|----------|------|
| `gsplat` | ⭐⭐⭐⭐⭐ | ✅ | Real-time, low memory | [Details](docs/engines.md#gsplat-recommended-for-real-time) |
| `graphdeco` | ⭐⭐⭐ | ❌ | Best quality | [Details](docs/engines.md#graphdeco-best-quality) |
| `monogs` | ⭐⭐⭐⭐ | ✅ | SLAM with tracking | [Details](docs/engines.md#monogs-real-time-slam) |
| `splatam` | ⭐⭐ | ❌ | RGB-D dense reconstruction | [Details](docs/engines.md#splatam-rgb-d-slam) |
| `da3gs` | ⭐⭐ | ❌ | End-to-end depth + Gaussians | [Details](docs/engines.md#da3gs) |
| `gslam` | ⭐⭐⭐ | ❌ | Large scenes with submaps | [Details](docs/engines.md#gaussian-slam-gslam) |

```python
from src.engines import get_engine, list_engines

# List available
print(list_engines())

# Use specific engine
engine = get_engine("gsplat")
```

> 📖 Full engine comparison and benchmarks in [Engines Guide](docs/engines.md).

## 📊 Benchmarks

Run comprehensive evaluations on pose estimation, depth estimation, and 3D Gaussian Splatting:

```bash
# Run full benchmark suite
python -m benchmarks.run --quick

# Run specific benchmarks
python -m benchmarks.run --pose     # Pose estimation only
python -m benchmarks.run --depth    # Depth estimation only
python -m benchmarks.run --gs       # Gaussian splatting only

# View interactive results
# Open benchmarks/results/<hostname>/benchmark_<timestamp>/report.html
```

### View Results Online

📊 **[Interactive Benchmark Viewer](https://ParsaRezaei.github.io/AirSplatMap/)** - Explore all benchmark results across different hardware configurations (Desktop, Jetson, etc.)

> 📖 See [Benchmarks Guide](docs/benchmarks.md) for detailed evaluation options and metrics.

## 🔧 Configuration

Configuration options and advanced usage are covered in the documentation:

- **Engine Configuration**: [docs/engines.md](docs/engines.md#custom-configuration)
- **Pipeline Configuration**: [docs/architecture.md](docs/architecture.md#configuration)
- **Dashboard Settings**: [docs/dashboard.md](docs/dashboard.md#configuration)
- **ArduPilot Setup**: [docs/ardupilot_integration.md](docs/ardupilot_integration.md)

## 📖 Research Papers

AirSplatMap includes two CVPR-format academic papers in the [papers/](papers/) directory:

1. **Deep Learning Paper**: Focus on 3DGS engines, optimization, and edge deployment
2. **Computer Vision Paper**: Comprehensive evaluation of depth and pose estimation methods

> 📄 See [papers/README.md](papers/README.md) for paper abstracts, key results, and building instructions.

## License

MIT License - See [LICENSE](LICENSE)

---

## 🛠️ Development

### Building CUDA Extensions (Optional)

For `graphdeco` and `monogs` engines:

```bash
cd submodules/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation -e .
```

### Running Tests

```bash
python -m pytest tests/
```

### Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

> 🐛 Found a bug? [Open an issue](https://github.com/ParsaRezaei/AirSplatMap/issues)

---

## 📚 Additional Resources

- **[TUM RGB-D Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset)** - Standard benchmark dataset
- **[3D Gaussian Splatting (Original Paper)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)** - SIGGRAPH 2023
- **[gsplat Library](https://docs.gsplat.studio/)** - Nerfstudio's optimized implementation
- **[ArduPilot Documentation](https://ardupilot.org/dev/)** - Drone/rover integration

---

<p align="center">
  Made with ❤️ for the 3D Gaussian Splatting community
</p>
<p align="center">
  <a href="https://ParsaRezaei.github.io/AirSplatMap/">View Benchmarks</a> •
  <a href="docs/getting_started.md">Get Started</a> •
  <a href="https://github.com/ParsaRezaei/AirSplatMap/issues">Report Bug</a>
</p>
