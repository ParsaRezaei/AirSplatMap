#!/usr/bin/env python3
"""
Pose Estimation Benchmark
=========================

Benchmark pose estimators against TUM RGB-D ground truth trajectories.
Computes ATE (Absolute Trajectory Error) and RPE (Relative Pose Error).

Usage:
    python scripts/benchmark_pose.py                    # Run all estimators
    python scripts/benchmark_pose.py --methods orb sift # Specific methods
    python scripts/benchmark_pose.py --dataset fr1_desk # Specific dataset
    python scripts/benchmark_pose.py --output results.json
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
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pose import get_pose_estimator, list_pose_estimators
from src.pipeline.frames import TumRGBDSource

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
        dataset_path: Path to TUM dataset
        max_frames: Maximum frames to process (None = all)
        skip_frames: Process every Nth frame
        
    Returns:
        BenchmarkResult
    """
    dataset_name = Path(dataset_path).name
    logger.info(f"Benchmarking {method} on {dataset_name}...")
    
    # Load dataset
    source = TumRGBDSource(dataset_path)
    frames = list(source)
    
    if max_frames:
        frames = frames[:max_frames]
    
    if skip_frames > 1:
        frames = frames[::skip_frames]
    
    if len(frames) == 0:
        raise ValueError(f"No frames found in {dataset_path}")
    
    # Initialize estimator
    estimator = get_pose_estimator(method)
    intrinsics = frames[0].intrinsics
    estimator.set_intrinsics(
        intrinsics['fx'], intrinsics['fy'],
        intrinsics['cx'], intrinsics['cy']
    )
    
    # Run estimation
    estimated_poses = []
    gt_poses = []
    inliers_list = []
    confidence_list = []
    lost_frames = 0
    
    t0 = time.time()
    
    for frame in frames:
        result = estimator.estimate(frame.rgb)
        
        estimated_poses.append(result.pose.copy())
        gt_poses.append(frame.pose.copy())
        inliers_list.append(result.num_inliers)
        confidence_list.append(result.confidence)
        
        if result.num_inliers < 8:
            lost_frames += 1
    
    total_time = time.time() - t0
    fps = len(frames) / total_time
    
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
    
    return BenchmarkResult(
        method=method,
        dataset=dataset_name,
        num_frames=len(frames),
        total_time=round(total_time, 2),
        fps=round(fps, 2),
        ate_rmse=round(ate['rmse'], 4),
        ate_mean=round(ate['mean'], 4),
        ate_median=round(ate['median'], 4),
        ate_std=round(ate['std'], 4),
        ate_max=round(ate['max'], 4),
        rpe_trans_rmse=round(rpe['trans_rmse'], 4),
        rpe_trans_mean=round(rpe['trans_mean'], 4),
        rpe_rot_rmse=round(rpe['rot_rmse'], 4),
        rpe_rot_mean=round(rpe['rot_mean'], 4),
        avg_inliers=round(np.mean(inliers_list), 1),
        avg_confidence=round(np.mean(confidence_list), 3),
        lost_frames=lost_frames,
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


def main():
    parser = argparse.ArgumentParser(description='Benchmark pose estimators against TUM ground truth')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Pose methods to benchmark (default: all CPU methods)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset name to use')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Root directory containing TUM datasets')
    parser.add_argument('--max-frames', type=int, default=None,
                        help='Maximum frames to process per dataset')
    parser.add_argument('--skip-frames', type=int, default=1,
                        help='Process every Nth frame')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    parser.add_argument('--include-gpu', action='store_true',
                        help='Include GPU methods (loftr)')
    
    args = parser.parse_args()
    
    # Find dataset root
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        # Default locations
        for candidate in [
            Path(__file__).parent.parent.parent / "datasets",
            Path.home() / "datasets",
        ]:
            if candidate.exists():
                dataset_root = candidate
                break
        else:
            logger.error("Could not find datasets directory. Use --dataset-root")
            sys.exit(1)
    
    # Find datasets
    datasets = find_tum_datasets(dataset_root)
    
    if args.dataset:
        datasets = [d for d in datasets if args.dataset in d.name]
    
    if not datasets:
        logger.error(f"No TUM datasets found in {dataset_root}")
        sys.exit(1)
    
    logger.info(f"Found {len(datasets)} dataset(s)")
    
    # Get methods to benchmark
    available = list_pose_estimators()
    
    if args.methods:
        methods = args.methods
    else:
        # Default: CPU methods only
        methods = [name for name, info in available.items() 
                   if not info['requires_gpu'] or args.include_gpu]
    
    logger.info(f"Benchmarking methods: {methods}")
    
    # Run benchmarks
    results = []
    
    for dataset in datasets:
        for method in methods:
            try:
                result = run_benchmark(
                    method=method,
                    dataset_path=str(dataset),
                    max_frames=args.max_frames,
                    skip_frames=args.skip_frames,
                )
                results.append(result)
                logger.info(f"  {method}: ATE={result.ate_rmse:.4f}m, FPS={result.fps:.1f}")
            except Exception as e:
                logger.error(f"  {method}: FAILED - {e}")
    
    # Print results
    print_results_table(results)
    
    # Save to JSON if requested
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        logger.info(f"Results saved to {output_path}")
    
    # Also save to default location
    default_output = Path(__file__).parent.parent / "output" / "pose_benchmark.json"
    default_output.parent.mkdir(parents=True, exist_ok=True)
    with open(default_output, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"Results also saved to {default_output}")


if __name__ == '__main__':
    main()
