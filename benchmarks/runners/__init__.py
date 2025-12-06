"""
Modular Benchmark Runners
=========================

Each runner handles one type of benchmark:
- PoseBenchmark: Visual odometry / SLAM pose estimation
- DepthBenchmark: Monocular depth estimation
- GSBenchmark: 3D Gaussian Splatting quality

All runners use metrics from src/evaluation/ for consistency.
"""

from .pose import PoseBenchmark, PoseResult
from .depth import DepthBenchmark, DepthResult
from .gs import GSBenchmark, GSResult

__all__ = [
    'PoseBenchmark', 'PoseResult',
    'DepthBenchmark', 'DepthResult', 
    'GSBenchmark', 'GSResult',
]
