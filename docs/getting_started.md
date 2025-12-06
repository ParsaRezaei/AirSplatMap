# Getting Started

This guide will help you install AirSplatMap and run your first 3D Gaussian Splatting reconstruction.

## Prerequisites

- **OS**: Windows 10/11 or Ubuntu 20.04/22.04
- **GPU**: NVIDIA GPU with CUDA support (RTX 20xx or newer recommended)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for datasets and outputs

## Installation

### 1. Clone the Repository

```bash
git clone --recursive https://github.com/ParsaRezaei/AirSplatMap.git
cd AirSplatMap
```

### 2. Create Conda Environment

```bash
# Create environment with all dependencies
conda env create -f environment.yml
conda activate airsplatmap
```

### 3. Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check AirSplatMap
python -c "from src.engines import list_engines; print('Engines:', list(list_engines().keys()))"
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
# Download TUM dataset
cd scripts/tools
./download_tum.sh           # Linux/macOS
download_tum.bat            # Windows
cd ../..

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

```bash
# Download all default sequences
cd scripts/tools
./download_tum.sh           # Linux/macOS
download_tum.bat            # Windows
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
