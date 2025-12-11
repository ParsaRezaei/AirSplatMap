#!/usr/bin/env python3
"""
Pose Estimation Benchmark
=========================

Benchmark pose estimators against TUM RGB-D ground truth trajectories.
Computes ATE (Absolute Trajectory Error) and RPE (Relative Pose Error).

Usage:
    python benchmarks/pose/benchmark_pose.py                    # Run all estimators
    python benchmarks/pose/benchmark_pose.py --methods orb sift # Specific methods
    python benchmarks/pose/benchmark_pose.py --dataset fr1_desk # Specific dataset
    python benchmarks/pose/benchmark_pose.py --output results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose import get_pose_estimator, list_pose_estimators
from src.pipeline.frames import TumRGBDSource, SevenScenesSource, ReplicaSource

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _detect_dataset_type(dataset_path: str) -> str:
    """Auto-detect dataset type based on directory structure."""
    path = Path(dataset_path)
    
    # Check for TUM format (rgb.txt file)
    if (path / "rgb.txt").exists():
        return 'tum'
    
    # Check for 7-Scenes format (seq-XX directories with frame-*.color.png)
    seq_dirs = list(path.glob("seq-*"))
    if seq_dirs and any(list(s.glob("frame-*.color.png")) for s in seq_dirs):
        return '7scenes'
    
    # Check for Replica format (traj.txt file)
    if (path / "traj.txt").exists():
        return 'replica'
    
    # Default to TUM
    logger.warning(f"Could not detect dataset type for {path}, assuming TUM format")
    return 'tum'


def _get_dataset_source(dataset_path: str):
    """Get appropriate FrameSource for a dataset."""
    dataset_type = _detect_dataset_type(dataset_path)
    
    if dataset_type == 'tum':
        return TumRGBDSource(dataset_path)
    elif dataset_type == '7scenes':
        return SevenScenesSource(dataset_path)
    elif dataset_type == 'replica':
        return ReplicaSource(dataset_path)
    else:
        return TumRGBDSource(dataset_path)


@dataclass
class BenchmarkResult:
    """Result from benchmarking a pose estimator."""
    method: str
    dataset: str
    num_frames: int
    total_time: float
    fps: float
    
    # Absolute Trajectory Error (ATE)
    ate_rmse: float  # Root mean square error in meters
    ate_mean: float
    ate_median: float
    ate_std: float
    ate_max: float
    
    # Relative Pose Error (RPE) - translation
    rpe_trans_rmse: float
    rpe_trans_mean: float
    
    # Relative Pose Error (RPE) - rotation (degrees)
    rpe_rot_rmse: float
    rpe_rot_mean: float
    
    # Tracking stats
    avg_inliers: float
    avg_confidence: float
    lost_frames: int  # Frames with <8 inliers
    
    # Fields with defaults must come last
    # Method category
    is_monocular: bool = True  # True for monocular-only, False for RGB-D capable
    uses_scale_alignment: bool = True  # Whether scale alignment was applied
    scale_factor: float = 1.0  # Scale factor from Umeyama alignment
    
    # Latency metrics (ms per frame)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0  # 95th percentile latency
    p99_latency_ms: float = 0.0  # 99th percentile latency


def align_trajectories(estimated: np.ndarray, ground_truth: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Align estimated trajectory to ground truth using Umeyama alignment.
    
    Args:
        estimated: Nx3 estimated positions
        ground_truth: Nx3 ground truth positions
        
    Returns:
        (aligned_estimated, scale, rotation, translation)
    """
    # Center both trajectories
    est_mean = estimated.mean(axis=0)
    gt_mean = ground_truth.mean(axis=0)
    
    est_centered = estimated - est_mean
    gt_centered = ground_truth - gt_mean
    
    # Compute scale
    est_var = np.sum(est_centered ** 2)
    gt_var = np.sum(gt_centered ** 2)
    
    # Cross-covariance matrix
    H = est_centered.T @ gt_centered
    
    # SVD
    U, S, Vt = np.linalg.svd(H)
    
    # Rotation
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Scale
    scale = np.sum(S) / est_var if est_var > 0 else 1.0
    
    # Translation
    t = gt_mean - scale * R @ est_mean
    
    # Apply transformation
    aligned = scale * (estimated @ R.T) + t
    
    return aligned, scale, R, t


def compute_ate(estimated: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Compute Absolute Trajectory Error (ATE).
    
    Args:
        estimated: Nx3 estimated positions (already aligned)
        ground_truth: Nx3 ground truth positions
        
    Returns:
        Dict with RMSE, mean, median, std, max
    """
    errors = np.linalg.norm(estimated - ground_truth, axis=1)
    
    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'max': float(np.max(errors)),
    }


def compute_rpe(estimated_poses: List[np.ndarray], gt_poses: List[np.ndarray], 
                delta: int = 1) -> Dict[str, float]:
    """
    Compute Relative Pose Error (RPE).
    
    Args:
        estimated_poses: List of 4x4 pose matrices
        gt_poses: List of 4x4 ground truth poses
        delta: Frame delta for relative pose computation
        
    Returns:
        Dict with translation and rotation errors
    """
    trans_errors = []
    rot_errors = []
    
    for i in range(len(estimated_poses) - delta):
        # Estimated relative pose
        est_rel = np.linalg.inv(estimated_poses[i]) @ estimated_poses[i + delta]
        
        # Ground truth relative pose
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + delta]
        
        # Error in relative pose
        error = np.linalg.inv(gt_rel) @ est_rel
        
        # Translation error
        trans_error = np.linalg.norm(error[:3, 3])
        trans_errors.append(trans_error)
        
        # Rotation error (convert to angle)
        R_error = error[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
        rot_errors.append(np.degrees(angle))
    
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    
    return {
        'trans_rmse': float(np.sqrt(np.mean(trans_errors ** 2))),
        'trans_mean': float(np.mean(trans_errors)),
        'rot_rmse': float(np.sqrt(np.mean(rot_errors ** 2))),
        'rot_mean': float(np.mean(rot_errors)),
    }


def run_benchmark(method: str, dataset_path: str, max_frames: Optional[int] = None,
                  skip_frames: int = 1) -> BenchmarkResult:
    """
    Run benchmark for a single pose estimator on a dataset.
    
    Args:
        method: Pose estimator name ('orb', 'sift', 'loftr', etc.)
        dataset_path: Path to dataset (TUM, 7-Scenes, Replica, or ICL-NUIM)
        max_frames: Maximum frames to process (None = all)
        skip_frames: Process every Nth frame
        
    Returns:
        BenchmarkResult
    """
    dataset_name = Path(dataset_path).name
    logger.info(f"Benchmarking {method} on {dataset_name}...")
    
    # Auto-detect dataset type and load appropriate source
    source = _get_dataset_source(dataset_path)
    
    # Get total frame count and intrinsics without loading all frames
    total_frames = len(source)
    intrinsics = source.get_intrinsics()
    
    # Calculate which frame indices to process
    if max_frames and max_frames < total_frames:
        frame_indices = set(range(0, max_frames, skip_frames))
    else:
        frame_indices = set(range(0, total_frames, skip_frames))
    
    if len(frame_indices) == 0:
        raise ValueError(f"No frames found in {dataset_path}")
    
    # Initialize estimator
    estimator = get_pose_estimator(method)
    
    # Check if estimator is available (some require specific packages)
    if hasattr(estimator, '_available') and not estimator._available:
        raise RuntimeError(f"Pose estimator '{method}' failed to initialize (missing dependencies)")
    
    estimator.set_intrinsics_from_dict(intrinsics)
    
    # Run estimation - stream frames to avoid loading all into memory
    estimated_poses = []
    gt_poses = []
    inliers_list = []
    confidence_list = []
    latencies_ms = []
    lost_frames = 0
    frames_processed = 0
    
    t0 = time.time()
    
    for frame in source:
        # Skip frames not in our indices
        if frame.idx not in frame_indices:
            continue
            
        frame_start = time.time()
        result = estimator.estimate(frame.rgb)
        frame_end = time.time()
        
        latencies_ms.append((frame_end - frame_start) * 1000)
        
        estimated_poses.append(result.pose.copy())
        gt_poses.append(frame.pose.copy())
        inliers_list.append(result.num_inliers)
        confidence_list.append(result.confidence)
        
        if result.num_inliers < 8:
            lost_frames += 1
        
        frames_processed += 1
    
    total_time = time.time() - t0
    fps = frames_processed / total_time
    
    # Compute latency statistics
    latencies_ms = np.array(latencies_ms)
    avg_latency_ms = float(np.mean(latencies_ms))
    min_latency_ms = float(np.min(latencies_ms))
    max_latency_ms = float(np.max(latencies_ms))
    p95_latency_ms = float(np.percentile(latencies_ms, 95))
    p99_latency_ms = float(np.percentile(latencies_ms, 99))
    
    # Extract positions
    est_positions = np.array([p[:3, 3] for p in estimated_poses])
    gt_positions = np.array([p[:3, 3] for p in gt_poses])
    
    # Align trajectories
    aligned_positions, scale, R, t = align_trajectories(est_positions, gt_positions)
    
    # Also align the full poses for RPE
    aligned_poses = []
    for pose in estimated_poses:
        aligned = np.eye(4)
        aligned[:3, :3] = R @ pose[:3, :3]
        aligned[:3, 3] = scale * (R @ pose[:3, 3]) + t
        aligned_poses.append(aligned)
    
    # Compute errors
    ate = compute_ate(aligned_positions, gt_positions)
    rpe = compute_rpe(aligned_poses, gt_poses)
    
    # Determine if method is monocular-only
    # All our visual odometry methods are monocular (don't use depth)
    # RGB-D SLAM methods like ORB-SLAM3 would be is_monocular=False
    is_monocular = True  # All current methods are monocular
    
    # Cleanup estimator to free GPU memory
    if hasattr(estimator, 'cleanup'):
        estimator.cleanup()
    del estimator
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    except ImportError:
        pass
    
    return BenchmarkResult(
        method=method,
        dataset=dataset_name,
        num_frames=frames_processed,
        total_time=round(total_time, 2),
        fps=round(fps, 2),
        is_monocular=is_monocular,
        uses_scale_alignment=True,  # Always using scale alignment for monocular
        ate_rmse=round(ate['rmse'], 4),
        ate_mean=round(ate['mean'], 4),
        ate_median=round(ate['median'], 4),
        ate_std=round(ate['std'], 4),
        ate_max=round(ate['max'], 4),
        scale_factor=round(scale, 4),
        rpe_trans_rmse=round(rpe['trans_rmse'], 4),
        rpe_trans_mean=round(rpe['trans_mean'], 4),
        rpe_rot_rmse=round(rpe['rot_rmse'], 4),
        rpe_rot_mean=round(rpe['rot_mean'], 4),
        avg_inliers=round(np.mean(inliers_list), 1),
        avg_confidence=round(np.mean(confidence_list), 3),
        lost_frames=lost_frames,
        avg_latency_ms=round(avg_latency_ms, 2),
        min_latency_ms=round(min_latency_ms, 2),
        max_latency_ms=round(max_latency_ms, 2),
        p95_latency_ms=round(p95_latency_ms, 2),
        p99_latency_ms=round(p99_latency_ms, 2),
    )


def find_tum_datasets(base_path: Path) -> List[Path]:
    """Find all TUM datasets in a directory."""
    datasets = []
    
    search_paths = [base_path, base_path / "tum"]
    
    for search in search_paths:
        if not search.exists():
            continue
        for item in sorted(search.iterdir()):
            if item.is_dir() and (item / "rgb.txt").exists() and (item / "groundtruth.txt").exists():
                datasets.append(item)
    
    return datasets


def print_results_table(results: List[BenchmarkResult]):
    """Print results in a nice table format."""
    if not results:
        print("No results to display")
        return
    
    # Group by dataset
    by_dataset = {}
    for r in results:
        if r.dataset not in by_dataset:
            by_dataset[r.dataset] = []
        by_dataset[r.dataset].append(r)
    
    print("\n" + "=" * 100)
    print("POSE ESTIMATION BENCHMARK RESULTS")
    print("=" * 100)
    
    for dataset, dataset_results in by_dataset.items():
        print(f"\n📁 Dataset: {dataset}")
        print("-" * 100)
        print(f"{'Method':<10} {'FPS':>8} {'ATE RMSE':>10} {'ATE Mean':>10} {'RPE Trans':>10} {'RPE Rot':>10} {'Inliers':>8} {'Lost':>6}")
        print("-" * 100)
        
        # Sort by ATE RMSE
        dataset_results.sort(key=lambda x: x.ate_rmse)
        
        for r in dataset_results:
            print(f"{r.method:<10} {r.fps:>8.1f} {r.ate_rmse:>10.4f} {r.ate_mean:>10.4f} "
                  f"{r.rpe_trans_rmse:>10.4f} {r.rpe_rot_rmse:>10.2f}° {r.avg_inliers:>8.1f} {r.lost_frames:>6}")
    
    # Overall summary
    print("\n" + "=" * 100)
    print("SUMMARY (averaged across all datasets)")
    print("-" * 100)
    
    by_method = {}
    for r in results:
        if r.method not in by_method:
            by_method[r.method] = []
        by_method[r.method].append(r)
    
    print(f"{'Method':<10} {'Avg FPS':>10} {'Avg ATE':>12} {'Avg RPE Trans':>14} {'Avg RPE Rot':>12}")
    print("-" * 100)
    
    method_avgs = []
    for method, method_results in by_method.items():
        avg_fps = np.mean([r.fps for r in method_results])
        avg_ate = np.mean([r.ate_rmse for r in method_results])
        avg_rpe_t = np.mean([r.rpe_trans_rmse for r in method_results])
        avg_rpe_r = np.mean([r.rpe_rot_rmse for r in method_results])
        method_avgs.append((method, avg_fps, avg_ate, avg_rpe_t, avg_rpe_r))
    
    # Sort by ATE
    method_avgs.sort(key=lambda x: x[2])
    
    for method, avg_fps, avg_ate, avg_rpe_t, avg_rpe_r in method_avgs:
        print(f"{method:<10} {avg_fps:>10.1f} {avg_ate:>12.4f}m {avg_rpe_t:>14.4f}m {avg_rpe_r:>12.2f}°")
    
    print("=" * 100)
