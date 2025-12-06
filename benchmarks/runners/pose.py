"""
Pose Estimation Benchmark Runner
================================

Evaluates visual odometry / pose estimation methods against TUM RGB-D ground truth.
Uses ATE and RPE metrics from src/evaluation/metrics.py for consistency.
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import compute_ate, compute_rpe

logger = logging.getLogger(__name__)


@dataclass
class PoseResult:
    """Result from pose estimation benchmark."""
    method: str
    dataset: str
    num_frames: int
    
    # Timing
    total_time: float
    fps: float
    
    # ATE metrics
    ate_rmse: float
    ate_mean: float
    ate_median: float
    ate_std: float
    ate_max: float
    
    # RPE metrics
    rpe_trans_rmse: float
    rpe_trans_mean: float
    rpe_rot_rmse: float
    rpe_rot_mean: float
    
    # Tracking stats
    avg_inliers: float
    avg_confidence: float
    lost_frames: int
    
    # Latency (ms)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PoseBenchmark:
    """
    Benchmark runner for pose estimation methods.
    
    Example:
        benchmark = PoseBenchmark(
            methods=['orb', 'sift', 'robust_flow'],
            dataset_root='datasets/tum'
        )
        results = benchmark.run(max_frames=200)
        benchmark.save_results('results/pose_benchmark.json')
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        dataset_root: Path = None,
        datasets: List[str] = None,
    ):
        # Default methods include both traditional and deep learning approaches
        self.methods = methods or [
            'orb', 'sift', 'robust_flow', 'keyframe',  # Traditional
            'loftr', 'lightglue', 'raft',  # Deep learning
        ]
        self.dataset_root = Path(dataset_root) if dataset_root else PROJECT_ROOT / 'datasets'
        self.datasets = datasets  # If None, will auto-discover
        self.results: List[PoseResult] = []
    
    def run(
        self,
        max_frames: int = 200,
        skip_frames: int = 1,
    ) -> List[PoseResult]:
        """Run benchmark on all methods and datasets.
        
        Optimized to load each dataset only once, then run all methods on it.
        """
        from src.pipeline.frames import TumRGBDSource
        from src.pose import get_pose_estimator
        
        # Find datasets
        datasets = self._find_datasets()
        if not datasets:
            logger.error(f"No TUM datasets found in {self.dataset_root}")
            return []
        
        logger.info(f"Running pose benchmark: {len(self.methods)} methods × {len(datasets)} datasets")
        
        self.results = []
        
        for dataset_path in datasets:
            # Load dataset ONCE per dataset
            logger.info(f"Loading dataset: {dataset_path.name}")
            source = TumRGBDSource(str(dataset_path))
            frames = list(source)
            
            if max_frames:
                frames = frames[:max_frames]
            if skip_frames > 1:
                frames = frames[::skip_frames]
            
            logger.info(f"  Loaded {len(frames)} frames (max={max_frames}, skip={skip_frames})")
            
            # Run all methods on this dataset
            for method in self.methods:
                try:
                    result = self._run_single_with_frames(
                        method, dataset_path.name, frames
                    )
                    self.results.append(result)
                    logger.info(f"  {method}: ATE={result.ate_rmse:.4f}m, FPS={result.fps:.1f}")
                except Exception as e:
                    logger.error(f"  {method}: FAILED - {e}")
        
        return self.results
    
    def _run_single_with_frames(
        self,
        method: str,
        dataset_name: str,
        frames: List,
    ) -> PoseResult:
        """Run benchmark for single method on pre-loaded frames."""
        from src.pose import get_pose_estimator
        
        # Initialize estimator
        estimator = get_pose_estimator(method)
        estimator.set_intrinsics_from_dict(frames[0].intrinsics)
        
        # Run estimation with timing
        estimated_poses = []
        gt_poses = []
        inliers_list = []
        confidence_list = []
        latencies_ms = []
        lost_frames = 0
        
        t0 = time.time()
        
        for frame in frames:
            frame_start = time.time()
            result = estimator.estimate(frame.rgb)
            latencies_ms.append((time.time() - frame_start) * 1000)
            
            estimated_poses.append(result.pose.copy())
            gt_poses.append(frame.pose.copy())
            inliers_list.append(result.num_inliers)
            confidence_list.append(result.confidence)
            
            if result.num_inliers < 8:
                lost_frames += 1
        
        total_time = time.time() - t0
        
        # Convert to numpy
        estimated_poses = np.array(estimated_poses)
        gt_poses = np.array(gt_poses)
        latencies_ms = np.array(latencies_ms)
        
        # Compute metrics using src/evaluation
        ate = compute_ate(estimated_poses, gt_poses, align=True)
        rpe = compute_rpe(estimated_poses, gt_poses)
        
        return PoseResult(
            method=method,
            dataset=dataset_name,
            num_frames=len(frames),
            total_time=round(total_time, 2),
            fps=round(len(frames) / total_time, 2),
            ate_rmse=round(ate['ate_rmse'], 4),
            ate_mean=round(ate['ate_mean'], 4),
            ate_median=round(ate['ate_median'], 4),
            ate_std=round(ate['ate_std'], 4),
            ate_max=round(ate['ate_max'], 4),
            rpe_trans_rmse=round(rpe['rpe_trans_rmse'], 4),
            rpe_trans_mean=round(rpe['rpe_trans_mean'], 4),
            rpe_rot_rmse=round(rpe['rpe_rot_rmse'], 4),
            rpe_rot_mean=round(rpe['rpe_rot_mean'], 4),
            avg_inliers=round(np.mean(inliers_list), 1),
            avg_confidence=round(np.mean(confidence_list), 3),
            lost_frames=lost_frames,
            avg_latency_ms=round(np.mean(latencies_ms), 2),
            min_latency_ms=round(np.min(latencies_ms), 2),
            max_latency_ms=round(np.max(latencies_ms), 2),
            p95_latency_ms=round(np.percentile(latencies_ms, 95), 2),
            p99_latency_ms=round(np.percentile(latencies_ms, 99), 2),
        )
    
    def _find_datasets(self) -> List[Path]:
        """Find TUM datasets in dataset_root."""
        if self.datasets:
            return [self.dataset_root / d for d in self.datasets if (self.dataset_root / d).exists()]
        
        datasets = []
        search_paths = [self.dataset_root, self.dataset_root / 'tum']
        
        for search in search_paths:
            if not search.exists():
                continue
            for item in sorted(search.iterdir()):
                if item.is_dir() and (item / 'rgb.txt').exists() and (item / 'groundtruth.txt').exists():
                    datasets.append(item)
        
        return datasets
    
    def save_results(self, filepath: Path) -> None:
        """Save results to JSON."""
        import json
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        logger.info(f"Saved {len(self.results)} results to {filepath}")
    
    def print_table(self) -> None:
        """Print results as formatted table."""
        if not self.results:
            print("No results to display")
            return
        
        print("\n" + "=" * 100)
        print("POSE ESTIMATION BENCHMARK RESULTS")
        print("=" * 100)
        
        # Group by dataset
        by_dataset = {}
        for r in self.results:
            by_dataset.setdefault(r.dataset, []).append(r)
        
        for dataset, results in by_dataset.items():
            print(f"\n📁 {dataset}")
            print("-" * 100)
            print(f"{'Method':<15} {'ATE RMSE':>10} {'RPE Trans':>10} {'FPS':>8} {'Latency':>10} {'Lost':>6}")
            print("-" * 100)
            
            for r in sorted(results, key=lambda x: x.ate_rmse):
                print(f"{r.method:<15} {r.ate_rmse:>10.4f}m {r.rpe_trans_rmse:>10.4f}m "
                      f"{r.fps:>8.1f} {r.avg_latency_ms:>9.1f}ms {r.lost_frames:>6}")
        
        print("=" * 100)
