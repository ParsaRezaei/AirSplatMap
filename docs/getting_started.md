# Getting Started

This guide will help you install AirSplatMap and run your first 3D Gaussian Splatting reconstruction.

## Prerequisites

- **OS**: Windows 10/11, Ubuntu 20.04/22.04, or NVIDIA Jetson (JetPack 6.x)
- **GPU**: NVIDIA GPU with CUDA support
  - Desktop: RTX 20xx or newer recommended
  - Jetson: Orin Nano, Orin NX, AGX Orin (JetPack 6.0+)
- **RAM**: 16GB+ recommended (8GB minimum on Jetson)
- **Storage**: 10GB+ for datasets and outputs

## Installation

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/ParsaRezaei/AirSplatMap.git
cd AirSplatMap
```

### 2. Run Setup Script

The setup script creates the conda environment and installs PyTorch with CUDA support.

**Linux / macOS / Jetson:**
```bash
./setup_env.sh
conda activate airsplatmap
```

**Windows (PowerShell):**
```powershell
# If you get an execution policy error, run this first:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_env.ps1
conda activate airsplatmap
```

### 3. Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check CUDA architecture support (Jetson Orin should show sm_87)
python -c "import torch; print('CUDA archs:', torch.cuda.get_arch_list())"

# Check AirSplatMap engines
python -c "from src.engines import list_engines; print('Engines:', list(list_engines().keys()))"
```

### Jetson-Specific Notes

On NVIDIA Jetson (JetPack 6.x), the setup script automatically:
- Installs NVIDIA's Jetson PyTorch wheel with SM 8.7 support
- Builds gsplat from source with Jetson CUDA kernels
- Sets up `LD_PRELOAD` for libstdc++ compatibility

Verify Jetson Orin support:
```bash
python -c "import torch; print('SM 8.7 supported:', any('87' in a for a in torch.cuda.get_arch_list()))"
```

## Quick Start

### Option 1: Web Dashboard (Recommended)

```bash
# Start the dashboard
cd dashboard
start_dashboard.bat       # Windows
./start_dashboard.sh      # Linux/macOS

# Open browser
# http://127.0.0.1:9002
```

### Option 2: TUM Dataset Demo

```bash
# Download TUM dataset manually from:
# https://vision.in.tum.de/data/datasets/rgbd-dataset/download
# Extract to datasets/tum/

# Run demo
python scripts/demos/live_tum_demo.py --sequence fr1_desk --engine gsplat
```

### Option 3: RealSense Camera

```bash
# Test camera
python scripts/demos/live_realsense_demo.py --list-devices

# Run live 3DGS
python scripts/demos/live_realsense_demo.py --engine gsplat
```

### Option 4: Python API

```python
from src.engines import get_engine
from src.pipeline import OnlineGSPipeline
from src.pipeline.frames import TumRGBDSource

# Setup
engine = get_engine("gsplat")
source = TumRGBDSource("path/to/tum/sequence")

# Create pipeline
pipeline = OnlineGSPipeline(
    engine=engine,
    frame_source=source,
    steps_per_frame=5
)

# Run
summary = pipeline.run(max_frames=200)
print(f"Processed {summary['frames']} frames, {summary['gaussians']} Gaussians")
```

## Download Datasets

### TUM RGB-D

Download manually from https://vision.in.tum.de/data/datasets/rgbd-dataset/download

```bash
# Extract to datasets/tum/ directory
mkdir -p datasets/tum
cd datasets/tum
# Download and extract sequences like rgbd_dataset_freiburg1_desk.tgz
```

Available sequences:
- `fr1_desk`, `fr1_desk2`, `fr1_room`
- `fr2_desk`, `fr2_xyz`
- `fr3_office`, `fr3_structure`

### Custom Data

Place your data in the `datasets/` folder:

```
datasets/
└── my_sequence/
    ├── rgb/
    │   ├── 0001.png
    │   ├── 0002.png
    │   └── ...
    ├── depth/
    │   ├── 0001.png
    │   └── ...
    ├── rgb.txt
    ├── depth.txt
    └── groundtruth.txt  # Optional
```

## Next Steps

- [Architecture](architecture.md) - Understand how it works
- [Engines](engines.md) - Choose the right 3DGS engine
- [Dashboard](dashboard.md) - Use the web interface
- [Benchmarks](benchmarks.md) - Evaluate performance

## Troubleshooting

### CUDA Out of Memory

```python
# Use gsplat engine (4x less memory)
engine = get_engine("gsplat")

# Or reduce resolution
engine = get_engine("graphdeco", image_scale=0.5)
```

### Import Errors

```bash
# Rebuild CUDA extensions
cd submodules/gaussian-splatting/submodules/diff-gaussian-rasterization
pip install --no-build-isolation -e .
```

### RealSense Not Detected

1. Use USB 3.0 port (blue connector)
2. Install Intel RealSense Viewer to verify camera
3. Check `pyrealsense2` installation: `python -c "import pyrealsense2"`
