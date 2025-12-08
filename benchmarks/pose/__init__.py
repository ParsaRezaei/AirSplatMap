"""
Pose Estimation Benchmarks
==========================

Benchmark pose estimation methods against ground truth trajectories.

For CLI usage:
    python -m benchmarks.run --pose --multi-dataset
"""

from .benchmark_pose import (
    run_benchmark,
    find_tum_datasets,
    print_results_table,
    BenchmarkResult,
)

__all__ = [
    'run_benchmark', 'find_tum_datasets', 'print_results_table', 'BenchmarkResult',
]
