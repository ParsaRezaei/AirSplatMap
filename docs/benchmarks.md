# Benchmarks

AirSplatMap includes a comprehensive benchmarking suite for evaluating pose estimation, depth estimation, and 3D Gaussian Splatting performance.

## Quick Start

```bash
# Run pipeline benchmark (most common)
python -m benchmarks.run --pipeline --quick

# Run specific benchmarks
python -m benchmarks.run --pose                 # Pose only
python -m benchmarks.run --depth                # Depth only  
python -m benchmarks.run --gs                   # Gaussian splatting only
python -m benchmarks.run --pipeline             # Combined pipeline tests

# Specify methods and datasets
python -m benchmarks.run --pipeline --datasets freiburg1_desk --depth-methods midas depth_pro --pose-methods orb --gs-engines gsplat graphdeco

# View interactive HTML report
# Open benchmarks/results/<hostname>/benchmark_<timestamp>/report.html
```

## Benchmark Suite Structure

```
benchmarks/
├── __init__.py           # Package init with paths
├── run.py                # Main CLI entry point
├── hardware_monitor.py   # GPU/CPU monitoring
├── pose/
│   └── benchmark_pose.py   # Pose estimation evaluation
├── depth/
│   └── benchmark_depth.py  # Depth estimation evaluation
├── gaussian_splatting/
│   └── benchmark_gs.py     # 3DGS engine evaluation
├── visualization/
│   ├── html_report.py      # HTML report generation
│   └── plot_utils.py       # Plotting utilities
└── results/
    └── <hostname>/         # Results per machine
        └── benchmark_<timestamp>/
            ├── report.html    # Interactive HTML report
            ├── results.json   # Raw results
            └── plots/         # Generated visualizations
```

## Pose Estimation Benchmark

Evaluates visual odometry methods against ground truth trajectories.

### Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| ATE RMSE | Absolute Trajectory Error (root mean square) | meters |
| ATE Mean | Average absolute error | meters |
| RPE Trans | Relative Pose Error (translation) | m/frame |
| RPE Rot | Relative Pose Error (rotation) | deg/frame |
| FPS | Processing speed | frames/sec |
| Inliers | Average matched features | count |

### Run

```bash
# All methods
python benchmarks/pose/benchmark_pose.py

# Specific methods
python benchmarks/pose/benchmark_pose.py --methods orb sift loftr

# Custom dataset
python benchmarks/pose/benchmark_pose.py --dataset-root /path/to/datasets

# Specific sequence
python benchmarks/pose/benchmark_pose.py --dataset fr1_desk
```

### Output

```
================================================================================
POSE ESTIMATION BENCHMARK RESULTS
================================================================================

📁 Dataset: rgbd_dataset_freiburg1_desk
--------------------------------------------------------------------------------
Method          ATE_RMSE   ATE_Mean   RPE_Trans  RPE_Rot    FPS    Inliers
--------------------------------------------------------------------------------
orb             0.0892     0.0743     0.0234     0.892      45.2   342
sift            0.0721     0.0612     0.0198     0.754      12.1   489
robust_flow     0.0654     0.0534     0.0176     0.623      28.4   512
loftr           0.0483     0.0398     0.0123     0.445       3.2   892
================================================================================
```

## Depth Estimation Benchmark

Evaluates monocular depth methods against sensor depth.

### Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| AbsRel | Absolute relative error | Lower |
| SqRel | Squared relative error | Lower |
| RMSE | Root mean squared error (m) | Lower |
| RMSElog | RMSE in log space | Lower |
| δ < 1.25 | % within 25% of ground truth | Higher |
| δ < 1.25² | % within 56% of ground truth | Higher |
| δ < 1.25³ | % within 95% of ground truth | Higher |

### Run

```bash
# All methods
python benchmarks/depth/benchmark_depth.py

# Specific methods
python benchmarks/depth/benchmark_depth.py --methods midas zoedepth depth_anything

# Fewer frames (faster)
python benchmarks/depth/benchmark_depth.py --max-frames 50 --skip-frames 10
```

### Output

```
================================================================================
DEPTH ESTIMATION BENCHMARK RESULTS
================================================================================

📁 Dataset: rgbd_dataset_freiburg1_desk
--------------------------------------------------------------------------------
Method              AbsRel   SqRel    RMSE     RMSElog  δ<1.25   δ<1.25²  δ<1.25³   FPS
--------------------------------------------------------------------------------
midas               0.1234   0.0892   0.3821   0.1892   0.8542   0.9234   0.9678   15.2
midas_small         0.1823   0.1234   0.5234   0.2341   0.7823   0.8912   0.9412   35.1
depth_anything      0.0923   0.0623   0.3124   0.1523   0.9123   0.9623   0.9823   12.4
zoedepth            0.1123   0.0789   0.3523   0.1723   0.8823   0.9523   0.9723    8.1
================================================================================
```

## Gaussian Splatting Benchmark

Evaluates 3DGS engines on reconstruction quality and efficiency.

### Metrics

| Metric | Description | Unit |
|--------|-------------|------|
| PSNR | Peak Signal-to-Noise Ratio | dB |
| SSIM | Structural Similarity Index | 0-1 |
| LPIPS | Learned Perceptual Similarity | 0-1 (lower better) |
| Gaussians | Final Gaussian count | count |
| Train Time | Total optimization time | seconds |
| Memory | Peak GPU memory usage | MB |
| FPS | Processing speed | frames/sec |

### Run

```bash
# All engines
python benchmarks/gaussian_splatting/benchmark_gs.py

# Specific engines
python benchmarks/gaussian_splatting/benchmark_gs.py --engines gsplat graphdeco

# Custom settings
python benchmarks/gaussian_splatting/benchmark_gs.py --max-frames 100 --iterations 100
```

### Output

```
================================================================================
GAUSSIAN SPLATTING BENCHMARK RESULTS
================================================================================

📁 Dataset: rgbd_dataset_freiburg1_desk
--------------------------------------------------------------------------------
Engine       Frames    PSNR     SSIM     LPIPS    Gaussians      Time     FPS  Memory
--------------------------------------------------------------------------------
gsplat          50   24.82   0.8523   0.1234      185,432     12.3s   17.21   2134MB
graphdeco       50   25.34   0.8712   0.1023      210,234     58.2s    3.52   8423MB
monogs          50   24.52   0.8423   0.1323      175,234     24.5s   10.12   4234MB
================================================================================
```

## Running Full Benchmark Suite

```bash
# Run everything
python -m benchmarks all

# Quick mode (fewer frames)
python -m benchmarks all --quick

# Specific benchmarks
python -m benchmarks pose --methods orb sift
python -m benchmarks depth --methods midas
python -m benchmarks gs --engines gsplat

# Skip specific benchmarks
python -m benchmarks all --skip-pose
python -m benchmarks all --skip-depth
python -m benchmarks all --skip-gs
```

## HTML Report

Generate an interactive HTML report:

```bash
python -m benchmarks report --plots
```

Opens `benchmarks/results/report.html` with:
- Interactive charts (Chart.js)
- Method comparison tables
- Best performer highlights
- Export options

## Visualization

### Generate Plots Only

```bash
python -m benchmarks report --plots
```

### Available Plots

| Plot | Location |
|------|----------|
| Pose comparison | `benchmarks/results/plots/pose_comparison.png` |
| Depth comparison | `benchmarks/results/plots/depth_comparison.png` |
| GS comparison | `benchmarks/results/plots/gs_comparison.png` |
| Overall summary | `benchmarks/results/plots/overall_summary.png` |
| Trajectory plots | `benchmarks/pose/plots/` |
| Training curves | `benchmarks/gaussian_splatting/plots/` |

### Custom Visualization

```python
from benchmarks.visualization.plot_utils import (
    plot_pose_metrics_bar,
    plot_depth_metrics_bar,
    plot_gs_metrics_bar,
    generate_html_report
)

# Load results
import json
with open('benchmarks/results/pose_benchmark.json') as f:
    pose_results = json.load(f)

# Generate custom plot
plot_pose_metrics_bar(
    pose_results,
    metrics=['ate_rmse', 'fps'],
    title='My Custom Comparison',
    output_path='my_plot.png'
)
```

## Datasets

### TUM RGB-D

Standard benchmark sequences:

| Sequence | Frames | Description |
|----------|--------|-------------|
| fr1_desk | 573 | Desktop scene |
| fr1_desk2 | 620 | Desktop scene 2 |
| fr1_room | 1352 | Room exploration |
| fr2_desk | 2870 | Longer desktop |
| fr2_xyz | 3669 | XYZ motion |
| fr3_office | 2488 | Office scene |

Download:
```bash
cd scripts/tools
./download_tum.sh           # Linux/macOS
download_tum.bat            # Windows
```

### Custom Datasets

Place in `datasets/` with TUM format:
```
datasets/my_sequence/
├── rgb/
├── depth/
├── rgb.txt
├── depth.txt
└── groundtruth.txt
```

## API Reference

### BenchmarkResult (Pose)

```python
@dataclass
class BenchmarkResult:
    method: str
    dataset: str
    num_frames: int
    total_time: float
    fps: float
    ate_rmse: float
    ate_mean: float
    ate_median: float
    ate_std: float
    ate_max: float
    rpe_trans_rmse: float
    rpe_trans_mean: float
    rpe_rot_rmse: float
    rpe_rot_mean: float
    avg_inliers: float
```

### DepthBenchmarkResult

```python
@dataclass
class DepthBenchmarkResult:
    method: str
    dataset: str
    num_frames: int
    total_time: float
    fps: float
    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    delta1: float
    delta2: float
    delta3: float
```

### GSBenchmarkResult

```python
@dataclass
class GSBenchmarkResult:
    engine: str
    dataset: str
    num_frames: int
    total_time: float
    train_time: float
    avg_frame_time: float
    fps: float
    psnr: float
    ssim: float
    lpips: float
    final_gaussians: int
    peak_memory_mb: float
```

## Configuration

### Environment Variables

```bash
# Dataset location
export AIRSPLAT_DATASET_ROOT=/path/to/datasets

# Output directory
export AIRSPLAT_OUTPUT_ROOT=/path/to/output
```

### Benchmark Config

```python
# benchmarks/config.py
POSE_METHODS = ['orb', 'sift', 'robust_flow', 'loftr']
DEPTH_METHODS = ['midas', 'midas_small', 'depth_anything', 'zoedepth']
GS_ENGINES = ['gsplat', 'graphdeco', 'monogs']

MAX_FRAMES_QUICK = 50
MAX_FRAMES_FULL = 500
```

## Batch Processing

For running on multiple sequences:

```bash
# Process all TUM sequences with GSplat
python benchmarks/batch/batch_gsplat_tum.py \
    --dataset-root /path/to/tum \
    --output-root ./output/gsplat

# Generate report from batch results
python benchmarks/batch/generate_benchmark_report.py \
    --output-root ./output
```

## Tips

### Faster Benchmarks

```bash
# Use quick mode
python -m benchmarks all --quick

# Run only pose with specific methods
python -m benchmarks pose --methods orb sift

# Reduce frames
python -m benchmarks all --max-frames 50
```

### Reproducibility

```bash
# Set random seed
export PYTHONHASHSEED=42

# Seeds can be configured in benchmark code
```

### GPU Memory

```bash
# Monitor GPU during benchmark
watch -n 1 nvidia-smi

# Use only low-memory engines
python -m benchmarks gs --engines gsplat  # Skip memory-heavy engines
```
