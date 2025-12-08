"""
AirSplatMap Benchmark Suite
===========================

Lightweight CLI runners for benchmarking. Core metrics and evaluation
logic lives in `src/evaluation/`.

Structure:
    benchmarks/
        __init__.py       - This file
        __main__.py       - CLI entry point
        runners/          - Modular benchmark runners
            __init__.py
            pose.py       - Pose estimation benchmarks
            depth.py      - Depth estimation benchmarks  
            gs.py         - Gaussian splatting benchmarks
        visualization/    - Plotting utilities
        results/          - Output directory

Usage:
    # Run all benchmarks
    python -m benchmarks --all
    
    # Run specific benchmark
    python -m benchmarks pose --methods orb sift
    python -m benchmarks depth --methods midas
    python -m benchmarks gs --engines graphdeco
    
    # Generate report from existing results
    python -m benchmarks report --input results/

Core evaluation functions are in src/evaluation/:
    from src.evaluation import compute_image_metrics, compute_trajectory_metrics
    from src.evaluation import BenchmarkRunner, BenchmarkResult
"""

__version__ = "1.0.0"

import socket
from pathlib import Path

# Paths
BENCHMARKS_DIR = Path(__file__).parent
PROJECT_ROOT = BENCHMARKS_DIR.parent
RESULTS_DIR = BENCHMARKS_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
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
