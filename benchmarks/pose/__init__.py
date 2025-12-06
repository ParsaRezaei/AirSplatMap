"""
Pose Estimation Benchmarks
==========================

Legacy benchmark module. Use `benchmarks.runners.pose` for the new modular API.

For CLI usage:
    python -m benchmarks pose --methods orb sift
"""

# Re-export from runners for backward compatibility
from benchmarks.runners.pose import PoseBenchmark, PoseResult

# Legacy exports from old benchmark_pose.py
from .benchmark_pose import (
    run_benchmark,
    find_tum_datasets,
    print_results_table,
    BenchmarkResult,
)

__all__ = [
    'PoseBenchmark', 'PoseResult',
    'run_benchmark', 'find_tum_datasets', 'print_results_table', 'BenchmarkResult',
]
