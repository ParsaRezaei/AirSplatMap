# AirSplatMap

An extensible, online 3D Gaussian Splatting mapping pipeline.

## Overview

AirSplatMap provides a modular framework for incremental 3D Gaussian Splatting (3DGS) from live or replayed streams of RGB/RGB-D frames with poses. The main contribution is the pipeline/orchestration layer; the actual 3DGS implementation sits behind a clean interface allowing different backends to be swapped.

## Architecture

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

1. **Clone and setup the workspace** (if not already done):
   ```bash
   cd /path/to/workspace
   ```

2. **Install dependencies**:
   ```bash
   cd AirSplatMap
   pip install -r requirements.txt
   ```

3. **Ensure Graphdeco 3DGS is available**:
   The `GraphdecoEngine` expects the Graphdeco gaussian-splatting repository to be available. It searches in:
   - `GRAPHDECO_PATH` environment variable
   - `../gaussian-splatting` (relative to AirSplatMap)
   
   Make sure the CUDA extensions are compiled:
   ```bash
   cd gaussian-splatting/submodules/diff-gaussian-rasterization
   pip install --no-build-isolation -e .
   cd ../simple-knn
   pip install --no-build-isolation -e .
   ```

4. **For all engines (recommended)**: Use the `airsplatmap` conda environment which has:
   - Python 3.10 + PyTorch 2.5.1 + CUDA 12.1
   - All three engines: graphdeco, gsplat, splatam
   
   ```bash
   conda activate airsplatmap
   ```

### Available Engines

| Engine | Speed | Real-time | Description | Requirements |
|--------|-------|-----------|-------------|--------------|
| `graphdeco` | ~2-5 FPS | ❌ | Original 3DGS from GRAPHDECO | gaussian-splatting repo with CUDA extensions |
| `gsplat` | ~17 FPS | ✅ | Nerfstudio's optimized 3DGS (4x less memory) | `pip install gsplat` |
| `splatam` | ~0.4 FPS | ❌ | RGB-D SLAM with Gaussian Splatting | SplaTAM repo at ~/SplaTAM |
| `monogs` | ~10 FPS | ✅ | Gaussian Splatting SLAM (CVPR'24 Highlight) | [MonoGS](https://github.com/muskie82/MonoGS) |
| `photoslam` | Real-time | ✅ | Photo-SLAM - Photorealistic SLAM (CVPR'24) | [Photo-SLAM](https://github.com/HuajianUP/Photo-SLAM) |

```python
from src.engines import get_engine, list_engines

# List available engines with details
for name, info in list_engines().items():
    print(f"{name}: {'✅' if info['available'] else '❌'} {info['description']}")

# Get a specific engine
engine = get_engine("gsplat")  # or "graphdeco", "splatam", "monogs", "photoslam"
```

### Engine Installation

**GSplat (recommended for real-time):**
```bash
pip install gsplat
```

**MonoGS (real-time mono/stereo/RGB-D SLAM):**
```bash
git clone https://github.com/muskie82/MonoGS.git --recursive
cd MonoGS
pip install --no-build-isolation -e submodules/simple-knn
pip install --no-build-isolation -e submodules/diff-gaussian-rasterization
pip install munch open3d PyOpenGL glfw PyGLM rich
```

**Photo-SLAM (C++ based, requires build):**
```bash
git clone https://github.com/HuajianUP/Photo-SLAM.git --recursive
cd Photo-SLAM
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

## Project Structure

```
AirSplatMap/
├── src/
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── base.py                  # BaseGSEngine interface
│   │   ├── graphdeco_engine.py      # Original 3DGS (GRAPHDECO)
│   │   ├── gsplat_engine.py         # Nerfstudio gsplat
│   │   ├── splatam_engine.py        # SplaTAM RGB-D SLAM
│   │   ├── monogs_engine.py         # MonoGS SLAM
│   │   └── gslam_engine.py          # Gaussian-SLAM
│   ├── pipeline/
│   │   ├── frames.py                # Frame, FrameSource, TumRGBDSource
│   │   ├── online_gs.py             # OnlineGSPipeline
│   │   └── rs_corrector.py          # Rolling shutter correction
│   └── data/                        # Dataset loaders
├── scripts/
│   ├── start_dashboard.sh           # Start web dashboard
│   ├── stop_dashboard.sh            # Stop web dashboard
│   ├── dashboard_config.sh          # Dashboard configuration
│   ├── web_dashboard.py             # Dashboard backend
│   ├── web_dashboard.html           # Dashboard frontend
│   ├── benchmarks/                  # Benchmarking scripts
│   │   ├── batch_gsplat_tum.py
│   │   ├── run_monogs_tum.py
│   │   └── run_splatam_tum.py
│   ├── demos/                       # Demo/example scripts
│   │   ├── live_demo.py
│   │   ├── live_tum_demo.py
│   │   └── online_tum_graphdeco.py
│   └── tools/                       # Utility scripts
│       ├── download_tum.sh
│       ├── generate_meshes.py
│       └── generate_voxel_grid.py
├── output/                          # Results and benchmarks
├── requirements.txt
└── README.md
```

## Configuration

### Engine Config (passed to `initialize_scene`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sh_degree` | 3 | Spherical harmonics degree |
| `white_background` | False | Use white background |
| `position_lr_init` | 0.00016 | Initial position learning rate |
| `densify_grad_threshold` | 0.0002 | Gradient threshold for densification |
| `densify_until_iter` | 15000 | Stop densification after N iterations |
| `lambda_dssim` | 0.2 | Weight for SSIM loss |
| `recency_weight` | 0.7 | Weight for recent frames in sampling |

### Pipeline Config

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps_per_frame` | 5 | Optimization steps per new frame |
| `warmup_frames` | 1 | Frames before starting optimization |
| `render_every` | 0 | Render preview every N frames |
| `save_every` | 0 | Save checkpoint every N frames |

## Supported Datasets

- **TUM RGB-D**: Use `TumRGBDSource` - expects TUM benchmark format with `rgb/`, `depth/`, `rgb.txt`, `depth.txt`, `groundtruth.txt`
- **COLMAP**: (Planned) `ColmapSource` for COLMAP sparse reconstructions
- **Video**: (Planned) `VideoSource` for video files with external pose tracking

## License

See LICENSE file for details.
