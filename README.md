# AirSplatMap

Real-time 3D Gaussian Splatting for drones, robots, and cameras.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

AirSplatMap is a modular framework for **real-time 3D reconstruction** using Gaussian Splatting. Stream RGB-D from cameras, drones, or datasets and watch 3D maps build in real-time.

```
Camera/Drone → Pose Estimation → 3D Gaussian Splatting → Web Dashboard
```

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🚀 **Multiple 3DGS Engines** | GSplat, GraphDeco, MonoGS, SplaTAM, Photo-SLAM |
| 📍 **Visual Odometry** | ORB, SIFT, LoFTR, SuperPoint, RealSense VIO |
| 🎯 **Depth Estimation** | MiDaS, Depth Anything, ZoeDepth |
| 🌐 **Web Dashboard** | Real-time 3D visualization in browser |
| 📊 **Benchmarks** | Automated evaluation with plots |
| 🤖 **ArduPilot** | MAVLink integration for drones/rovers |

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

## 🚀 Quick Start

### Installation

```bash
# Clone with submodules
git clone --recursive https://github.com/ParsaRezaei/AirSplatMap.git
cd AirSplatMap

# Create conda environment
conda env create -f environment_crossplatform.yml
conda activate airsplatmap

# Verify
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Run Web Dashboard

```bash
# Start dashboard
dashboard\start_dashboard.bat  # Windows
./dashboard/start_dashboard.sh  # Linux

# Open browser: http://127.0.0.1:9002
```

### Run TUM Dataset Demo

```bash
python scripts/demos/live_tum_demo.py --sequence fr1_desk --engine gsplat
```

### Run with RealSense Camera

```bash
python scripts/demos/live_realsense_demo.py --engine gsplat
```

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

### Quick Start - Cross-Platform (Windows/Linux)

```bash
# Clone with all submodules
git clone --recursive https://github.com/ParsaRezaei/AirSplatMap.git
cd AirSplatMap

# Create conda environment (includes PyTorch + CUDA + all dependencies)
conda env create -f environment_crossplatform.yml
conda activate airsplatmap

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from src.pipeline import RealSenseSource; print('RealSenseSource OK')"
```

That's it! The `environment_crossplatform.yml` includes everything:
- PyTorch with CUDA 12.1
- OpenCV, NumPy, SciPy, etc.
- pyrealsense2 for RealSense cameras
- gsplat for fast 3DGS

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
./scripts/start_dashboard.sh

# Open in browser
# http://localhost:9002

# Stop the dashboard
./scripts/stop_dashboard.sh
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

Edit `scripts/dashboard_config.sh` to change default ports:

```bash
# Dashboard configuration
HTTP_PORT=9002      # Web interface port
WS_PORT=9003        # WebSocket port for real-time updates
LOG_FILE=/tmp/airsplatmap_dashboard.log
PID_FILE=/tmp/airsplatmap_dashboard.pid
CONDA_ENV=airsplatmap
```

### Render Modes

- **Auto**: Use native engine renderer, fall back to software if unavailable
- **Native Only**: Force native CUDA renderer (blank if fails)
- **Software Only**: Point-cloud projection rendering (works with any engine)

### Custom Ports

```bash
./scripts/start_dashboard.sh --http-port 8080 --ws-port 8081
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

| Engine | Speed | Real-time | Best For |
|--------|-------|-----------|----------|
| `gsplat` | ⭐⭐⭐⭐⭐ | ✅ | Real-time, low memory |
| `graphdeco` | ⭐⭐⭐ | ❌ | Best quality |
| `monogs` | ⭐⭐⭐⭐ | ✅ | SLAM with tracking |
| `splatam` | ⭐⭐ | ❌ | RGB-D dense reconstruction |
| `photoslam` | ⭐⭐⭐⭐ | ✅ | Photorealistic |

```python
from src.engines import get_engine, list_engines

# List available
print(list_engines())

# Use specific engine
engine = get_engine("gsplat")
```

## 📊 Benchmarks

```bash
# Run full benchmark suite
python benchmarks/run_all.py --quick

# View results
# Open benchmarks/results/report.html
```

## 🔧 Configuration

See full details in [docs/](docs/).

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

---

<p align="center">
  Made with ❤️ for the 3D Gaussian Splatting community
</p>
