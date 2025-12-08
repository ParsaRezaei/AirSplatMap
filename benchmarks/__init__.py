"""
AirSplatMap Benchmark Suite
===========================

Comprehensive benchmarking for pose estimation, depth estimation, and 
Gaussian splatting. Core metrics and evaluation logic lives in `src/evaluation/`.

Structure:
    benchmarks/
        __init__.py              - This file
        run.py                   - Main benchmark runner (CLI entry point)
        pose/
            benchmark_pose.py    - Pose estimation benchmark logic
        depth/
            benchmark_depth.py   - Depth estimation benchmark logic
        gaussian_splatting/
            benchmark_gs.py      - Gaussian splatting benchmark logic
        visualization/           - Plotting utilities
        results/<hostname>/      - Output directory (per-machine)

Usage:
    # Run all benchmarks on all datasets
    python -m benchmarks.run --comprehensive --multi-dataset
    
    # Quick benchmark (fewer frames)
    python -m benchmarks.run --comprehensive --multi-dataset --quick
    
    # Specific benchmark types only
    python -m benchmarks.run --pose --multi-dataset
    python -m benchmarks.run --depth --multi-dataset
    python -m benchmarks.run --gs --multi-dataset
    
    # Specific datasets
    python -m benchmarks.run --comprehensive --datasets fr1_desk room0

Results are saved to:
    benchmarks/results/<hostname>/benchmark_<timestamp>/
        ├── results.json    - Raw benchmark data
        ├── report.html     - Interactive HTML report
        ├── benchmark.log   - Run log
        └── plots/          - Visualization charts

Core evaluation functions are in src/evaluation/:
    from src.evaluation import compute_image_metrics, compute_trajectory_metrics
"""

__version__ = "1.0.0"

import socket
from pathlib import Path

# Paths
BENCHMARKS_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
RESULTS_DIR = BENCHMARKS_DIR / "results"
DATASETS_DIR = PROJECT_ROOT / "datasets"


def get_hostname() -> str:
    """Get the current hostname."""
    return socket.gethostname()


def get_host_results_dir(hostname: str = None) -> Path:
    """
    Get the results directory for a specific hostname.
    
    Args:
        hostname: The hostname to get results for. Defaults to current host.
    
    Returns:
        Path to the hostname-specific results directory.
    """
    if hostname is None:
        hostname = get_hostname()
    return RESULTS_DIR / hostname


def get_host_latest_dir(hostname: str = None) -> Path:
    """
    Get the latest results directory for a specific hostname.
    
    Args:
        hostname: The hostname to get results for. Defaults to current host.
    
    Returns:
        Path to the latest symlink for the hostname.
    """
    return get_host_results_dir(hostname) / "latest"
