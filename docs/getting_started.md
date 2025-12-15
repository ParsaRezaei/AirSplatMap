# Getting Started

This guide will help you install AirSplatMap and run your first 3D Gaussian Splatting reconstruction.

> 📊 **See it in action**: View benchmark results at [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/)

---

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

---

## Quick Start

### Option 1: Web Dashboard (Recommended)

The web dashboard provides the easiest way to visualize 3D Gaussian Splatting in real-time.

```bash
# Start the dashboard
cd dashboard
./start_dashboard.sh      # Linux/macOS
# start_dashboard.bat     # Windows

# Open browser
# http://127.0.0.1:9002
```

> 📖 Learn more in the [Dashboard Guide](dashboard.md)

### Option 2: TUM Dataset Demo

```bash
# Download TUM dataset first (see datasets section below)
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

> 📖 Full API documentation in [API Reference](api_reference.md)

---

## Download Datasets

### TUM RGB-D (Recommended for Testing)

The TUM RGB-D dataset is the standard benchmark for RGB-D SLAM. Download using the provided script:

```bash
# Download minimal set for testing (~1.5GB)
./scripts/tools/datasets/download_datasets.sh ./datasets tum minimal

# Download standard benchmark set (~6GB)
./scripts/tools/datasets/download_datasets.sh ./datasets tum

# Download all TUM scenes (~15GB)
./scripts/tools/datasets/download_datasets.sh ./datasets tum all
```

Or download manually from [TUM RGB-D Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/download).

> 📖 Full dataset documentation in [datasets/README.md](../datasets/README.md)

### Other Datasets

```bash
# ICL-NUIM (synthetic)
./scripts/tools/datasets/download_datasets.sh ./datasets icl

# Replica
./scripts/tools/datasets/download_datasets.sh ./datasets replica

# 7-Scenes
./scripts/tools/datasets/download_datasets.sh ./datasets 7scenes
```

---

## Run Benchmarks

Evaluate pose estimation, depth estimation, and 3DGS engines:

```bash
# Quick benchmark
python -m benchmarks.run --quick

# Comprehensive benchmark
python -m benchmarks.run --pipeline --multi-dataset

# View results
# Open benchmarks/results/<hostname>/benchmark_<timestamp>/report.html
```

📊 **View benchmark results online**: [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/)

> 📖 Full benchmarking documentation in [Benchmarks Guide](benchmarks.md)

---

## Next Steps

| Goal | Guide |
|------|-------|
| Understand the system | [Architecture](architecture.md) |
| Choose the right engine | [Engines Guide](engines.md) |
| Configure pose estimation | [Pose Estimation](pose_estimation.md) |
| Configure depth estimation | [Depth Estimation](depth_estimation.md) |
| Use the web interface | [Dashboard Guide](dashboard.md) |
| Integrate with drones | [ArduPilot Integration](ardupilot_integration.md) |
| Run evaluations | [Benchmarks Guide](benchmarks.md) |
| Use the Python API | [API Reference](api_reference.md) |

---

## Troubleshooting

### CUDA Out of Memory

```python
# Use gsplat engine (4x less memory than graphdeco)
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

### Need Help?

- 🐛 [Report a bug](https://github.com/ParsaRezaei/AirSplatMap/issues)
- 📖 Check other [documentation pages](index.md)
