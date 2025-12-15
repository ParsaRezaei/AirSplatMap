# AirSplatMap Benchmark Suite

Comprehensive benchmarking tools for evaluating pose estimation, depth estimation, and 3D Gaussian Splatting methods against RGB-D ground truth datasets.

> 📊 **View Results Online**: [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/) - Interactive benchmark viewer

---

## Quick Start

```bash
# Run all benchmarks on default dataset (quick mode)
python -m benchmarks.run --quick

# Run on ALL available TUM datasets
python -m benchmarks.run --multi-dataset --quick

# Run on specific datasets
python -m benchmarks.run --datasets fr1_desk fr2_desk fr3_office

# Run comprehensive benchmark (all methods, all datasets)
python -m benchmarks.run --comprehensive --multi-dataset
```

---

## Documentation

| Resource | Link |
|----------|------|
| 📖 **Full Benchmarks Guide** | [docs/benchmarks.md](../docs/benchmarks.md) |
| 📊 **Interactive Results Viewer** | [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/) |
| 📁 **Datasets Guide** | [datasets/README.md](../datasets/README.md) |
| 📍 **Pose Estimation Docs** | [docs/pose_estimation.md](../docs/pose_estimation.md) |
| 🎯 **Depth Estimation Docs** | [docs/depth_estimation.md](../docs/depth_estimation.md) |
| 🚀 **Engines Guide** | [docs/engines.md](../docs/engines.md) |

---

## Supported Datasets

### TUM RGB-D Dataset
Real-world indoor scenes with motion capture ground truth.

```bash
# Download all TUM datasets (~15GB)
./scripts/tools/download_datasets.sh ./datasets tum all

# Download specific subset
./scripts/tools/download_datasets.sh ./datasets tum fr1_xyz,fr1_desk,fr2_desk
```

| Sequence | Description | Size |
|----------|-------------|------|
| fr1_xyz | Simple translations | 0.5GB |
| fr1_desk, fr1_desk2 | Office desk scenes | 0.4GB |
| fr1_room | Full room with loop closure | 0.8GB |
| fr2_desk | High-quality office scene | 2.0GB |
| fr3_long_office | Large office trajectory | 1.6GB |

### ICL-NUIM Dataset
Synthetic scenes with perfect ground truth.

```bash
./scripts/tools/download_datasets.sh ./datasets icl
```

### Replica Dataset
High-quality synthetic reconstructions.

```bash
./scripts/tools/download_datasets.sh ./datasets replica
```

## Structure

```
benchmarks/
├── run.py              # Main benchmark runner
├── html_report.py      # HTML report generation
├── pose/               # Pose estimation benchmark
├── depth/              # Depth estimation benchmark
├── gaussian_splatting/ # GS benchmark
├── visualization/      # Plotting utilities
├── batch/              # Batch processing scripts
└── results/            # Output directory
    ├── benchmark_YYYYMMDD_HHMMSS/
    │   ├── results.json
    │   ├── report.html
    │   ├── benchmark.log
    │   └── plots/
    └── latest -> benchmark_...
```

## Usage

### Command Line Options

```bash
python -m benchmarks.run [OPTIONS]

# What to run
--pose              Run pose estimation only
--depth             Run depth estimation only
--gs                Run Gaussian splatting only
--pipeline          Run combined pipeline tests
--all               Run all benchmarks (default)

# Dataset selection
--dataset NAME      Run on specific dataset
--datasets A B C    Run on multiple specific datasets
--multi-dataset     Run on ALL available datasets

# Configuration
--quick             Quick mode (fewer frames, faster)
--max-frames N      Maximum frames per sequence
--comprehensive     Test ALL available methods

# Method selection
--pose-methods A B  Pose methods (default: orb sift robust_flow)
--depth-methods A B Depth methods (default: midas depth_anything_v2)
--gs-engines A B    GS engines (default: graphdeco gsplat)
```

### Examples

```bash
# Quick single-dataset benchmark
python -m benchmarks.run --quick

# Full multi-dataset benchmark
python -m benchmarks.run --multi-dataset

# Pose-only benchmark on all datasets
python -m benchmarks.run --pose --multi-dataset --pose-methods orb sift keyframe

# Comprehensive benchmark (takes hours!)
python -m benchmarks.run --comprehensive --multi-dataset --max-frames 200
```

### Python API

```python
from benchmarks.pose.benchmark_pose import run_benchmark as pose_benchmark
from benchmarks.depth.benchmark_depth import run_depth_benchmark
from benchmarks.gaussian_splatting.benchmark_gs import run_gs_benchmark

# Run pose benchmark
result = pose_benchmark(
    method='orb',
    dataset_path='datasets/tum/rgbd_dataset_freiburg1_desk',
    max_frames=100
)
print(f"ATE: {result.ate_rmse:.4f}m, FPS: {result.fps:.1f}")

# Run depth benchmark
result = run_depth_benchmark(
    method='midas',
    dataset_path='datasets/tum/rgbd_dataset_freiburg1_desk',
    max_frames=50
)
print(f"AbsRel: {result.abs_rel:.4f}, δ1: {result.delta1:.2%}")

# Run GS benchmark
result = run_gs_benchmark(
    engine_name='gsplat',
    dataset_path='datasets/tum/rgbd_dataset_freiburg1_desk',
    max_frames=50
)
print(f"PSNR: {result.psnr:.2f}dB, SSIM: {result.ssim:.4f}")
```

## Available Methods

### Pose Estimation

| Method | Description | Speed |
|--------|-------------|-------|
| `orb` | ORB features + PnP | Fast |
| `sift` | SIFT features + PnP | Moderate |
| `robust_flow` | Optical flow with outlier rejection | Fast |
| `keyframe` | Keyframe-based ORB | Moderate |

### Depth Estimation

| Method | Description | Metric Depth |
|--------|-------------|--------------|
| `midas` | MiDaS DPT-Large | No |
| `midas_small` | MiDaS Small (fast) | No |
| `depth_anything` | Depth Anything V2 | No |
| `zoedepth` | ZoeDepth | Yes |

### Gaussian Splatting Engines

| Engine | Description | Real-time |
|--------|-------------|-----------|
| `gsplat` | Nerfstudio's optimized 3DGS | Yes |
| `graphdeco` | Original 3DGS from INRIA | No |
| `monogs` | MonoGS SLAM | Yes |
| `splatam` | SplaTAM RGB-D SLAM | No |

## Metrics

### Pose Estimation
- **ATE RMSE**: Absolute Trajectory Error (root mean square) in meters
- **RPE Trans**: Relative Pose Error for translation
- **RPE Rot**: Relative Pose Error for rotation (degrees)
- **FPS**: Processing speed
- **Latency**: Per-frame latency (avg, p95, p99)

### Depth Estimation
- **AbsRel**: Absolute relative error |d - d*| / d*
- **RMSE**: Root mean square error in meters
- **delta < 1.25**: Threshold accuracy (% within 1.25x of ground truth)

### Gaussian Splatting
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **Render FPS**: Novel view synthesis speed

## Dataset Setup

Benchmarks use TUM RGB-D datasets by default. Download them:

```bash
cd scripts/tools
./download_tum.sh           # Linux/macOS
download_tum.bat            # Windows
```

Datasets will be downloaded to `datasets/tum/`.

## Results

Results are saved to `benchmarks/results/`:

- `pose_benchmark.json` - Pose estimation results
- `depth_benchmark.json` - Depth estimation results  
- `gs_benchmark.json` - Gaussian splatting results
- `report.html` - Interactive HTML report
- `plots/` - Generated visualization plots

## Example Output

```
================================================================================
POSE ESTIMATION BENCHMARK RESULTS
================================================================================

Dataset: rgbd_dataset_freiburg1_desk
--------------------------------------------------------------------------------
Method          ATE_RMSE   ATE_Mean   RPE_Trans  RPE_Rot    FPS    Inliers
--------------------------------------------------------------------------------
orb             0.0892     0.0743     0.0234     0.892      45.2   342
sift            0.0721     0.0612     0.0198     0.754      12.1   489
robust_flow     0.0654     0.0534     0.0176     0.623      28.4   512
================================================================================
```

## Batch Processing

The `batch/` directory contains scripts for processing multiple datasets:

```bash
# Process all TUM datasets with GSplat
python benchmarks/batch/batch_gsplat_tum.py

# Run MonoGS on TUM
python benchmarks/batch/run_monogs_tum.py
```

## See Also

- [Detailed Benchmarks Documentation](../docs/benchmarks.md)
- [Pose Estimation Guide](../docs/pose_estimation.md)
- [Depth Estimation Guide](../docs/depth_estimation.md)
- [Engines Comparison](../docs/engines.md)
- [Interactive Benchmark Viewer](https://ParsaRezaei.github.io/AirSplatMap/)
