# AirSplatMap Benchmarks

Comprehensive benchmarking suite for evaluating pose estimation, depth estimation, and Gaussian splatting performance.

## Directory Structure

```
benchmarks/
├── README.md                 # This file
├── run_all.py               # Run all benchmarks
├── pose/                    # Pose estimation benchmarks
│   ├── benchmark_pose.py    # Pose benchmark runner
│   └── plots/               # Generated pose plots
├── depth/                   # Depth estimation benchmarks
│   ├── benchmark_depth.py   # Depth benchmark runner
│   └── plots/               # Generated depth plots
├── gaussian_splatting/      # 3DGS benchmarks
│   ├── benchmark_gs.py      # GS benchmark runner
│   └── plots/               # Generated GS plots
├── visualization/           # Common visualization utilities
│   └── plot_utils.py        # Plotting helpers
├── batch/                   # Batch processing scripts
│   ├── batch_gsplat_tum.py  # Batch GSplat on TUM
│   ├── batch_process_scenes.py
│   ├── run_monogs_tum.py    # MonoGS on TUM
│   └── run_splatam_tum.py   # SplaTAM on TUM
└── results/                 # Combined benchmark results
    ├── report.html          # Interactive HTML report
    └── *.json               # Raw results
```

## Quick Start

```bash
# Run all benchmarks
python benchmarks/run_all.py

# Run specific benchmarks
python benchmarks/pose/benchmark_pose.py --methods orb sift loftr
python benchmarks/depth/benchmark_depth.py --methods depth_anything midas
python benchmarks/gaussian_splatting/benchmark_gs.py --engines graphdeco gsplat

# Generate plots only (from existing results)
python benchmarks/run_all.py --plots-only
```

## Metrics

### Pose Estimation
- **ATE (Absolute Trajectory Error)**: RMSE, mean, median (meters)
- **RPE (Relative Pose Error)**: Translation (m/frame), Rotation (deg/frame)
- **Speed**: FPS, total time
- **Robustness**: Lost frames, average inliers

### Depth Estimation
- **Accuracy**: AbsRel, SqRel, RMSE, RMSElog
- **Completeness**: δ < 1.25, δ < 1.25², δ < 1.25³
- **Speed**: FPS, memory usage

### Gaussian Splatting
- **Quality**: PSNR, SSIM, LPIPS
- **Efficiency**: Training time, #Gaussians, memory
- **Real-time**: Render FPS, convergence time

## Visualization

All benchmarks generate:
1. **Bar charts** comparing methods
2. **Trajectory plots** (pose)
3. **Error heatmaps** (depth)
4. **Training curves** (GS)
5. **Interactive HTML report**

## Requirements

```bash
pip install matplotlib seaborn pandas plotly
```
