#!/usr/bin/env python3
"""
AirSplatMap Comprehensive Benchmark Suite
==========================================

Benchmarks pose estimation, depth estimation, and Gaussian splatting,
including combined pipeline tests with different GT/estimated combinations.

Usage:
    python -m benchmarks.run                        # Run all benchmarks
    python -m benchmarks.run --pose                 # Pose only
    python -m benchmarks.run --depth                # Depth only  
    python -m benchmarks.run --gs                   # Gaussian splatting only
    python -m benchmarks.run --pipeline             # Combined pipeline tests
    python -m benchmarks.run --quick                # Quick mode (fewer frames)
"""

import argparse
import json
import logging
import sys
import time
import os
import socket
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

# Curated list of representative datasets (avoids redundant similar scenes)
# This reduces benchmark time from ~85 hours to ~27 hours while maintaining coverage
CURATED_DATASETS = {
    # TUM RGB-D - core benchmarks
    'rgbd_dataset_freiburg1_desk',      # Standard benchmark scene
    'rgbd_dataset_freiburg1_room',      # Larger motion, loop closure
    'rgbd_dataset_freiburg1_xyz',       # Axis-aligned baseline
    'rgbd_dataset_freiburg3_long_office_household',  # Long sequence, drift test
    # 7-Scenes - indoor variety
    'chess',                            # Good texture, geometric
    'office',                           # Standard indoor
    # Replica - clean synthetic GT
    'office0',                          # Synthetic office
    'room0',                            # Synthetic room
}
sys.path.insert(0, str(PROJECT_ROOT))

# Basic console logging - file handler added later when output_dir is known
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def log_gpu_memory(prefix: str = ""):
    """Log current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB peak")
    except Exception as e:
        logger.debug(f"Could not get GPU memory: {e}")


def check_cuda_health() -> bool:
    """
    Check if CUDA is in a healthy state.
    Returns True if CUDA is working, False if corrupted.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return True  # No CUDA, but not an error
        
        # Try basic CUDA operations
        torch.cuda.synchronize()
        test_tensor = torch.zeros(1, device='cuda')
        del test_tensor
        torch.cuda.empty_cache()
        return True
    except RuntimeError as e:
        if "CUDA error" in str(e) or "illegal memory" in str(e).lower():
            logger.error(f"CUDA health check failed: {e}")
            return False
        raise
    except Exception:
        return True  # Non-CUDA errors are fine


def safe_cuda_cleanup():
    """Safely attempt CUDA cleanup, ignoring errors if context is corrupted."""
    try:
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass  # CUDA may already be corrupted
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass  # CUDA may already be corrupted
        gc.collect()
    except Exception:
        pass


def setup_file_logging(output_dir: Path):
    """Add file handler to log to output directory."""
    log_file = output_dir / "benchmark.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Add to root logger so all modules log to file
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging to: {log_file}")


# =============================================================================
# Utility Functions
# =============================================================================

def find_tum_datasets(dataset_root: Path) -> List[Path]:
    """Find all TUM RGB-D datasets."""
    datasets = []
    for search_dir in [dataset_root, dataset_root / "tum"]:
        if search_dir.exists():
            for item in sorted(search_dir.iterdir()):
                if item.is_dir() and (item / "rgb.txt").exists():
                    datasets.append(item)
    return datasets


def find_7scenes_datasets(dataset_root: Path) -> List[Path]:
    """Find all 7-Scenes datasets."""
    datasets = []
    search_dir = dataset_root / "7scenes"
    if search_dir.exists():
        for scene_dir in sorted(search_dir.iterdir()):
            if scene_dir.is_dir():
                # Check for seq-XX subdirectories with frame files
                seq_dirs = list(scene_dir.glob("seq-*"))
                if seq_dirs:
                    # Check if any seq dir has frame files
                    for seq in seq_dirs:
                        if list(seq.glob("frame-*.color.png")):
                            datasets.append(scene_dir)
                            break
    return datasets


def find_replica_datasets(dataset_root: Path) -> List[Path]:
    """Find all Replica datasets."""
    datasets = []
    search_dir = dataset_root / "replica"
    if search_dir.exists():
        for scene_dir in sorted(search_dir.iterdir()):
            if scene_dir.is_dir():
                # Check for traj.txt and results/ or frames
                if (scene_dir / "traj.txt").exists():
                    datasets.append(scene_dir)
    return datasets


def find_icl_nuim_datasets(dataset_root: Path) -> List[Path]:
    """Find all ICL-NUIM datasets (TUM-compatible format)."""
    datasets = []
    search_dir = dataset_root / "icl_nuim"
    if search_dir.exists():
        for item in sorted(search_dir.iterdir()):
            if item.is_dir() and (item / "rgb.txt").exists():
                datasets.append(item)
    return datasets


def find_all_datasets(dataset_root: Path) -> List[Tuple[Path, str]]:
    """
    Find all supported datasets and return with their type.
    
    Returns:
        List of (path, dataset_type) tuples where dataset_type is one of:
        'tum', '7scenes', 'replica', 'icl_nuim'
    """
    all_datasets = []
    
    # TUM RGB-D
    for path in find_tum_datasets(dataset_root):
        all_datasets.append((path, 'tum'))
    
    # 7-Scenes
    for path in find_7scenes_datasets(dataset_root):
        all_datasets.append((path, '7scenes'))
    
    # Replica
    for path in find_replica_datasets(dataset_root):
        all_datasets.append((path, 'replica'))
    
    # ICL-NUIM (uses TUM format)
    for path in find_icl_nuim_datasets(dataset_root):
        all_datasets.append((path, 'icl_nuim'))
    
    return all_datasets


def get_dataset_source(dataset_path: Path, dataset_type: str):
    """
    Get the appropriate FrameSource for a dataset.
    
    Args:
        dataset_path: Path to the dataset
        dataset_type: One of 'tum', '7scenes', 'replica', 'icl_nuim'
    
    Returns:
        FrameSource instance
    """
    from src.pipeline.frames import TumRGBDSource, SevenScenesSource, ReplicaSource
    
    if dataset_type == 'tum' or dataset_type == 'icl_nuim':
        return TumRGBDSource(str(dataset_path))
    elif dataset_type == '7scenes':
        return SevenScenesSource(str(dataset_path))
    elif dataset_type == 'replica':
        return ReplicaSource(str(dataset_path))
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def save_results(results: Any, path: Path):
    """Save results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_results(path: Path) -> Any:
    """Load results from JSON."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# =============================================================================
# Pose/Depth Caching for Pipeline Reuse
# =============================================================================

def save_pose_cache(
    estimated_poses: Dict[str, List[np.ndarray]], 
    gt_poses: List[np.ndarray],
    pose_errors: Dict[str, float],
    output_dir: Path, 
    dataset_name: str
):
    """Save estimated poses and ground truth for pipeline reuse and trajectory visualization."""
    cache_dir = output_dir / "cache" / dataset_name / "pose"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ground truth poses (N, 4, 4)
    gt_array = np.array(gt_poses)
    np.save(cache_dir / "gt_poses.npy", gt_array)
    
    # Save each method's estimated poses
    for method, poses in estimated_poses.items():
        pose_array = np.array(poses)  # (N, 4, 4)
        np.save(cache_dir / f"{method}_poses.npy", pose_array)
    
    # Save pose errors for reference (convert numpy types to Python types)
    pose_errors_serializable = {k: float(v) for k, v in pose_errors.items()}
    with open(cache_dir / "pose_errors.json", 'w') as f:
        json.dump(pose_errors_serializable, f)
    
    logger.info(f"Saved pose cache to {cache_dir} ({len(estimated_poses)} methods)")


def save_depth_cache(
    estimated_depths: Dict[str, List[np.ndarray]], 
    gt_depths: List[np.ndarray],
    depth_errors: Dict[str, float],
    output_dir: Path, 
    dataset_name: str
):
    """Save estimated depths and ground truth for pipeline reuse."""
    cache_dir = output_dir / "cache" / dataset_name / "depth"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Save ground truth depths (compressed, can be large)
    np.savez_compressed(cache_dir / "gt_depths.npz", 
                        **{f"frame_{i}": d for i, d in enumerate(gt_depths) if d is not None})
    
    # Save each method's estimated depths
    for method, depths in estimated_depths.items():
        np.savez_compressed(cache_dir / f"{method}_depths.npz",
                           **{f"frame_{i}": d for i, d in enumerate(depths)})
    
    # Save depth errors for reference (convert numpy types to Python types)
    depth_errors_serializable = {k: float(v) for k, v in depth_errors.items()}
    with open(cache_dir / "depth_errors.json", 'w') as f:
        json.dump(depth_errors_serializable, f)
    
    logger.info(f"Saved depth cache to {cache_dir} ({len(estimated_depths)} methods)")


def load_pose_cache(output_dir: Path, dataset_name: str) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray], Dict[str, float], Optional[np.ndarray]]:
    """Load cached pose estimates if available.
    
    Returns:
        (gt_poses, estimated_poses, pose_errors, frame_indices)
    """
    cache_dir = output_dir / "cache" / dataset_name / "pose"
    
    if not cache_dir.exists():
        return None, {}, {}, None
    
    # Load ground truth
    gt_poses = None
    gt_file = cache_dir / "gt_poses.npy"
    if gt_file.exists():
        gt_poses = np.load(gt_file)
    
    # Load frame indices (which frames from the dataset were processed)
    frame_indices = None
    indices_file = cache_dir / "frame_indices.npy"
    if indices_file.exists():
        frame_indices = np.load(indices_file)
    
    # Load estimated poses
    estimated_poses = {}
    for pose_file in cache_dir.glob("*_poses.npy"):
        if pose_file.name == "gt_poses.npy":
            continue
        method = pose_file.stem.replace("_poses", "")
        estimated_poses[method] = np.load(pose_file)
    
    # Load pose errors
    pose_errors = {}
    errors_file = cache_dir / "pose_errors.json"
    if errors_file.exists():
        with open(errors_file) as f:
            pose_errors = json.load(f)
    
    if estimated_poses:
        n_frames = len(frame_indices) if frame_indices is not None else "?"
        logger.info(f"Loaded cached poses for {dataset_name}: {list(estimated_poses.keys())} ({n_frames} frames)")
    
    return gt_poses, estimated_poses, pose_errors, frame_indices


def load_depth_cache(output_dir: Path, dataset_name: str) -> Tuple[Optional[List[np.ndarray]], Dict[str, List[np.ndarray]], Dict[str, float]]:
    """Load cached depth estimates if available."""
    cache_dir = output_dir / "cache" / dataset_name / "depth"
    
    if not cache_dir.exists():
        return None, {}, {}
    
    # Load ground truth depths
    gt_depths = None
    gt_file = cache_dir / "gt_depths.npz"
    if gt_file.exists():
        data = np.load(gt_file)
        gt_depths = [data[k] for k in sorted(data.files, key=lambda x: int(x.split('_')[1]))]
    
    # Load estimated depths
    estimated_depths = {}
    for depth_file in cache_dir.glob("*_depths.npz"):
        if depth_file.name == "gt_depths.npz":
            continue
        method = depth_file.stem.replace("_depths", "")
        data = np.load(depth_file)
        estimated_depths[method] = [data[k] for k in sorted(data.files, key=lambda x: int(x.split('_')[1]))]
    
    # Load depth errors
    depth_errors = {}
    errors_file = cache_dir / "depth_errors.json"
    if errors_file.exists():
        with open(errors_file) as f:
            depth_errors = json.load(f)
    
    if estimated_depths:
        logger.info(f"Loaded cached depths for {dataset_name}: {list(estimated_depths.keys())}")
    
    return gt_depths, estimated_depths, depth_errors


# =============================================================================
# Pose Estimation Benchmark
# =============================================================================

def run_pose_benchmark(
    dataset_path: Path,
    methods: List[str] = None,
    max_frames: int = None,
    skip_frames: int = 1,
    cache_dir: Path = None,
) -> List[Dict]:
    """Run pose estimation benchmark on a single dataset.
    
    Args:
        dataset_path: Path to dataset
        methods: List of pose methods to benchmark
        max_frames: Maximum frames to process (None = ALL frames for best accuracy)
        skip_frames: Process every Nth frame
        cache_dir: If provided, save poses to this directory for pipeline reuse
        
    Returns:
        List of result dicts
    """
    from benchmarks.pose.benchmark_pose import run_benchmark
    from dataclasses import asdict
    
    if methods is None:
        methods = ['orb', 'sift', 'robust_flow']
    
    results = []
    cached_poses = {}
    
    for method in methods:
        try:
            # Clear GPU memory before each method (safely)
            safe_cuda_cleanup()
            
            logger.info(f"  Testing {method}...")
            
            # Request poses if we need to cache
            if cache_dir is not None:
                result, poses = run_benchmark(
                    method=method,
                    dataset_path=str(dataset_path),
                    max_frames=max_frames,
                    skip_frames=skip_frames,
                    return_poses=True,
                )
                cached_poses[method] = poses
            else:
                result = run_benchmark(
                    method=method,
                    dataset_path=str(dataset_path),
                    max_frames=max_frames,
                    skip_frames=skip_frames,
                )
            
            result_dict = asdict(result)
            results.append(result_dict)
            logger.info(f"    ATE={result.ate_rmse:.4f}m, FPS={result.fps:.1f}")
            
            # Cache immediately after each method completes (incremental caching)
            if cache_dir is not None and method in cached_poses:
                dataset_name = dataset_path.name
                dataset_cache = cache_dir / dataset_name / "pose"
                dataset_cache.mkdir(parents=True, exist_ok=True)
                
                # Save GT poses and frame indices (only once, from first method)
                gt_file = dataset_cache / 'gt_poses.npy'
                if not gt_file.exists():
                    gt_poses = cached_poses[method]['gt']
                    np.save(gt_file, gt_poses)
                    frame_indices = cached_poses[method]['frame_indices']
                    np.save(dataset_cache / 'frame_indices.npy', np.array(frame_indices))
                
                # Save this method's aligned poses
                np.save(dataset_cache / f'{method}_poses.npy', cached_poses[method]['aligned'])
                
                # Update pose errors JSON with all completed methods
                pose_errors = {r['method']: r['ate_rmse'] for r in results}
                with open(dataset_cache / 'pose_errors.json', 'w') as f:
                    json.dump(pose_errors, f)
                
                logger.info(f"    Cached {method} poses to {dataset_cache}")
                
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Force cleanup after each method (safely)
            safe_cuda_cleanup()
    
    # Log final cache status
    if cache_dir is not None and cached_poses:
        dataset_name = dataset_path.name
        dataset_cache = cache_dir / dataset_name / "pose"
        logger.info(f"  Cached poses to {dataset_cache} ({len(cached_poses)} methods)")
    
    return results


# =============================================================================
# Depth Estimation Benchmark
# =============================================================================

def run_depth_benchmark(
    dataset_path: Path,
    methods: List[str] = None,
    cache_dir: Path = None,
) -> List[Dict]:
    """Run depth estimation benchmark on a single dataset.
    
    Processes ALL frames and saves to cache for pipeline reuse.
    Benchmark metrics are computed on a subset for speed.
    """
    from benchmarks.depth.benchmark_depth import run_depth_benchmark as _run_depth
    from src.depth import get_depth_estimator
    from src.pipeline.frames import TumRGBDSource, SevenScenesSource, ReplicaSource
    from dataclasses import asdict
    
    if methods is None:
        methods = ['midas', 'midas_small']
    
    results = []
    estimated_depths = {}
    gt_depths = []
    depth_errors = {}
    
    # Load ALL frames for caching
    dataset_type = 'tum'
    if (dataset_path / "traj.txt").exists():
        dataset_type = 'replica'
    elif list(dataset_path.glob("seq-*")):
        dataset_type = '7scenes'
    
    if dataset_type == 'tum':
        source = TumRGBDSource(str(dataset_path))
    elif dataset_type == '7scenes':
        source = SevenScenesSource(str(dataset_path))
    else:
        source = ReplicaSource(str(dataset_path))
    
    all_frames = list(source)
    total_frames = len(all_frames)
    logger.info(f"  Loaded {total_frames} frames for depth estimation")
    
    # For benchmark metrics, use a subset (every Nth frame, max 100)
    metric_skip = max(1, total_frames // 100)
    
    for method in methods:
        try:
            safe_cuda_cleanup()
            logger.info(f"  Testing {method}...")
            
            # Run benchmark on subset for metrics
            result = _run_depth(
                method=method,
                dataset_path=str(dataset_path),
                max_frames=100,
                skip_frames=metric_skip,
            )
            result_dict = asdict(result)
            results.append(result_dict)
            logger.info(f"    AbsRel={result.abs_rel:.4f}, d1={result.delta1:.2%}, FPS={result.fps:.1f}")
            
            # Now estimate ALL frames for caching
            if cache_dir is not None:
                logger.info(f"    Caching all {total_frames} depth maps...")
                estimator = get_depth_estimator(method)
                
                # Check if this is a metric depth estimator (no scaling needed)
                is_metric = estimator.is_metric() if callable(getattr(estimator, 'is_metric', None)) else False
                if is_metric:
                    logger.info(f"      {method} outputs metric depth - no scaling needed")
                
                # First pass: estimate all depths and compute global scale from frames with GT
                raw_depths = []
                scales = []
                
                for i, frame in enumerate(all_frames):
                    if i % 100 == 0 and i > 0:
                        logger.info(f"      Estimating frame {i}/{total_frames}")
                    
                    pred = estimator.estimate(frame.rgb)
                    pred_depth = pred.depth.copy() if hasattr(pred, 'depth') else pred.copy()
                    raw_depths.append(pred_depth)
                    
                    # Compute scale from frames that have GT depth (only for non-metric methods)
                    if not is_metric:
                        gt_depth = frame.depth
                        if gt_depth is not None:
                            valid_gt = (gt_depth > 0.1) & (gt_depth < 10)
                            valid_pred = pred_depth > 0.01
                            valid = valid_gt & valid_pred
                            
                            if valid.sum() > 100:
                                gt_median = np.median(gt_depth[valid])
                                pred_median = np.median(pred_depth[valid])
                                if pred_median > 0.01:
                                    scale = gt_median / pred_median
                                    if 0.1 < scale < 100:  # Reasonable scale range
                                        scales.append(scale)
                
                # Compute global scale (median of per-frame scales) - only for non-metric
                if is_metric:
                    global_scale = 1.0  # No scaling for metric depth
                elif scales:
                    global_scale = np.median(scales)
                    logger.info(f"      Global depth scale: {global_scale:.4f} (from {len(scales)} frames with GT)")
                else:
                    global_scale = 1.0
                    logger.warning(f"      No valid GT frames for scaling, using scale=1.0")
                
                # Second pass: apply global scale to all frames
                depths = []
                abs_rels = []
                
                for i, (frame, raw_depth) in enumerate(zip(all_frames, raw_depths)):
                    # Check for degenerate MiDaS output (negative values or near-zero)
                    # MiDaS can fail on some frames, producing garbage output
                    if raw_depth.min() < -0.1 or raw_depth.mean() < 0.1:
                        # Skip degenerate frames - pass None to let GS handle it
                        pred_depth = None
                    else:
                        pred_depth = raw_depth * global_scale
                        pred_depth = np.clip(pred_depth, 0.0, 15.0)
                    
                    # Compute AbsRel only for valid frames with GT
                    gt_depth = frame.depth
                    if gt_depth is not None and pred_depth is not None:
                        valid_gt = (gt_depth > 0.1) & (gt_depth < 10)
                        valid_pred = pred_depth > 0.05
                        valid = valid_gt & valid_pred
                        if valid.sum() > 100:
                            abs_rel = np.mean(np.abs(pred_depth[valid] - gt_depth[valid]) / gt_depth[valid])
                            abs_rels.append(abs_rel)
                    
                    depths.append(pred_depth)
                
                estimated_depths[method] = depths
                depth_errors[method] = np.mean(abs_rels) if abs_rels else float('inf')
                logger.info(f"      Avg AbsRel: {depth_errors[method]:.4f}")
                
                # Store GT depths once
                if not gt_depths:
                    gt_depths = [f.depth for f in all_frames]
                
                if hasattr(estimator, 'cleanup'):
                    estimator.cleanup()
                del estimator
                safe_cuda_cleanup()
                logger.info(f"    Cached {len(depths)} depth maps")
                
        except Exception as e:
            logger.error(f"    {method} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save cached depths
    if cache_dir is not None and estimated_depths:
        dataset_name = dataset_path.name
        save_depth_cache(estimated_depths, gt_depths, depth_errors, cache_dir, dataset_name)
    
    return results

# =============================================================================
# Gaussian Splatting Benchmark  
# =============================================================================

def run_gs_benchmark(
    dataset_path: Path,
    engines: List[str] = None,
    max_frames: int = 50,
    iterations: int = 5,
    output_dir: Path = None,
) -> List[Dict]:
    """Run Gaussian splatting benchmark on a single dataset."""
    from benchmarks.gaussian_splatting.benchmark_gs import run_gs_benchmark as _run_gs
    from dataclasses import asdict
    
    if engines is None:
        engines = ['graphdeco', 'gsplat']
    
    results = []
    for engine in engines:
        try:
            logger.info(f"  Testing {engine}...")
            log_gpu_memory(f"    [Before {engine}] ")
            result = _run_gs(
                engine_name=engine,
                dataset_path=str(dataset_path),
                max_frames=max_frames,
                n_iterations=iterations,
                output_dir=output_dir,
            )
            result_dict = asdict(result)
            results.append(result_dict)
            logger.info(f"    PSNR={result.psnr:.2f}dB, SSIM={result.ssim:.4f}, Gaussians={result.final_gaussians:,}")
            log_gpu_memory(f"    [After {engine}] ")
        except Exception as e:
            logger.error(f"    {engine} failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Critical: Free GPU memory after each engine test
            import torch
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    return results


# =============================================================================
# Combined Pipeline Benchmark (Pose + Depth + GS)
# =============================================================================

@dataclass
class PipelineResult:
    """Results from combined pipeline benchmark."""
    name: str
    pose_source: str  # 'gt' or estimator name
    depth_source: str  # 'gt' or estimator name
    gs_engine: str
    dataset: str
    
    # Quality metrics
    psnr: float
    ssim: float
    
    # Efficiency
    total_time: float
    fps: float
    num_gaussians: int
    
    # Optional metrics (must have defaults)
    lpips: float = 0.0  # Added for paper
    pose_ate: float = 0.0
    depth_abs_rel: float = 0.0


def run_pipeline_benchmark(
    dataset_path: Path,
    dataset_type: str = 'tum',
    gs_engine: str = 'gsplat',
    pose_methods: List[str] = None,
    depth_methods: List[str] = None,
    max_frames: int = 100,
    iterations: int = 5,
    # Cached data from pose/depth benchmarks (ALL frames)
    cached_poses: Dict[str, np.ndarray] = None,
    cached_depths: Dict[str, List[np.ndarray]] = None,
    cached_pose_errors: Dict[str, float] = None,
    cached_depth_errors: Dict[str, float] = None,
) -> Tuple[List[Dict], bool]:
    """
    Run combined pipeline benchmark with different pose/depth sources.
    
    Tests combinations:
    - GT pose + GT depth (upper bound)
    - GT pose + estimated depth
    - Estimated pose + GT depth  
    - Estimated pose + estimated depth (real-world scenario)
    
    Returns:
        Tuple of (results list, cuda_healthy boolean)
    
    Uses cached poses/depths from pose and depth benchmarks.
    Uniformly samples frames for GS testing.
    """
    from src.engines import get_engine
    from src.pose import get_pose_estimator
    from src.depth import get_depth_estimator
    from src.evaluation.metrics import compute_image_metrics
    import cv2
    import torch
    
    if pose_methods is None:
        pose_methods = ['orb']
    if depth_methods is None:
        depth_methods = ['midas_small']
    
    dataset_name = dataset_path.name
    logger.info(f"Running pipeline benchmark on {dataset_name}")
    
    # Load dataset using appropriate source for dataset type
    source = get_dataset_source(dataset_path, dataset_type)
    all_frames = list(source)
    total_frames = len(all_frames)
    
    # Uniform sampling: pick frames evenly distributed across sequence
    if total_frames <= max_frames:
        indices = np.arange(total_frames)
        frames = all_frames
        logger.info(f"  Using all {total_frames} frames")
    else:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
        frames = [all_frames[i] for i in indices]
        logger.info(f"  Uniformly sampled {max_frames} frames from {total_frames} total")
    
    if not frames:
        logger.error(f"No frames found in {dataset_path}")
        return [], True
    
    intrinsics = frames[0].intrinsics
    image_size = (frames[0].rgb.shape[1], frames[0].rgb.shape[0])
    
    results = []
    
    # Define test configurations
    configs = [
        ('GT_pose_GT_depth', 'gt', 'gt'),
    ]
    
    # Add estimated depth with GT pose
    for depth_method in depth_methods:
        configs.append((f'GT_pose_{depth_method}_depth', 'gt', depth_method))
    
    # Add estimated pose with GT depth
    for pose_method in pose_methods:
        configs.append((f'{pose_method}_pose_GT_depth', pose_method, 'gt'))
    
    # Add fully estimated combinations
    for pose_method in pose_methods:
        for depth_method in depth_methods:
            configs.append((f'{pose_method}_pose_{depth_method}_depth', pose_method, depth_method))
    
    # Pre-compute estimated poses and depths (or use cached)
    estimated_poses = {}
    estimated_depths = {}
    pose_errors = cached_pose_errors if cached_pose_errors else {}
    depth_errors = cached_depth_errors if cached_depth_errors else {}
    
    # Use cached poses if available
    # The cached poses are indexed the same as our frames (we use cached_frame_indices)
    if cached_poses:
        for method in pose_methods:
            if method in cached_poses:
                cached = cached_poses[method]
                # Cache has ALL frames, extract the ones we need by index
                if len(cached) >= total_frames:
                    estimated_poses[method] = [cached[i] for i in indices]
                    logger.info(f"  Using cached poses for {method} ({len(estimated_poses[method])} poses from {len(cached)} cached)")
                elif len(cached) >= len(frames):
                    estimated_poses[method] = cached[:len(frames)]
                    logger.info(f"  Using cached poses for {method} ({len(estimated_poses[method])} poses)")
                else:
                    logger.warning(f"  Cached poses for {method} have {len(cached)} frames, need {len(frames)}")
            else:
                logger.warning(f"  No cached poses for {method}, will compute")
    
    # Use cached depths if available (cache has ALL frames)
    if cached_depths:
        for method in depth_methods:
            if method in cached_depths:
                cached = cached_depths[method]
                # Cache has ALL frames, extract the ones we need by index
                if len(cached) >= total_frames:
                    estimated_depths[method] = [cached[i] for i in indices]
                    logger.info(f"  Using cached depths for {method} ({len(estimated_depths[method])} depths from {len(cached)} cached)")
                elif len(cached) >= len(frames):
                    estimated_depths[method] = cached[:len(frames)]
                    logger.info(f"  Using cached depths for {method}")
                else:
                    logger.warning(f"  Cached depths for {method} have {len(cached)} frames, need {len(frames)}")
            else:
                logger.warning(f"  No cached depths for {method}, will compute")
    
    # Estimate poses (only for methods not in cache)
    for pose_method in pose_methods:
        if pose_method in estimated_poses:
            continue  # Already loaded from cache
        logger.info(f"  Estimating poses with {pose_method}...")
        estimator = None
        try:
            # Clear GPU memory before loading each estimator
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc
                gc.collect()
            except ImportError:
                pass
            
            estimator = get_pose_estimator(pose_method)
            estimator.set_intrinsics_from_dict(intrinsics)
            poses = []
            
            for frame in frames:
                result = estimator.estimate(frame.rgb)
                poses.append(result.pose.copy())
            
            estimated_poses[pose_method] = poses
            
            # Compute ATE with Umeyama alignment (same as pose benchmark)
            gt_positions = np.array([f.pose[:3, 3] for f in frames])
            est_positions = np.array([p[:3, 3] for p in poses])
            
            # Umeyama alignment for monocular VO (handles unknown scale)
            from benchmarks.pose.benchmark_pose import align_trajectories
            aligned_est, scale, rotation, translation = align_trajectories(est_positions, gt_positions)
            ate = np.sqrt(np.mean(np.sum((aligned_est - gt_positions) ** 2, axis=1)))
            pose_errors[pose_method] = ate
            logger.info(f"    ATE: {ate:.4f}m (scale={scale:.4f})")
        except Exception as e:
            logger.error(f"    Failed: {e}")
            import traceback
            traceback.print_exc()
            estimated_poses[pose_method] = [f.pose for f in frames]  # Fallback to GT
            pose_errors[pose_method] = float('inf')
        finally:
            # Cleanup estimator to free GPU memory
            if estimator is not None:
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
    
    # Estimate depths (only for methods not in cache)
    for depth_method in depth_methods:
        if depth_method in estimated_depths:
            continue  # Already loaded from cache
        logger.info(f"  Estimating depths with {depth_method}...")
        estimator = None
        try:
            # Clear GPU memory before loading each estimator
            safe_cuda_cleanup()
            
            estimator = get_depth_estimator(depth_method)
            
            # Check if this is a metric depth estimator (no scaling needed)
            is_metric = estimator.is_metric() if callable(getattr(estimator, 'is_metric', None)) else False
            if is_metric:
                logger.info(f"    {depth_method} outputs metric depth - no scaling needed")
            
            # First pass: estimate all depths and compute global scale
            raw_depths = []
            scales = []
            
            for frame in frames:
                result = estimator.estimate(frame.rgb)
                pred_depth = result.depth.copy()
                raw_depths.append(pred_depth)
                
                # Compute scale from frames with GT depth (only for non-metric methods)
                if not is_metric:
                    gt_depth = frame.depth
                    if gt_depth is not None:
                        valid_gt = (gt_depth > 0.1) & (gt_depth < 10)
                        valid_pred = pred_depth > 0.01
                        valid = valid_gt & valid_pred
                        
                        if valid.sum() > 100:
                            gt_median = np.median(gt_depth[valid])
                            pred_median = np.median(pred_depth[valid])
                            if pred_median > 0.01:
                                scale = gt_median / pred_median
                                if 0.1 < scale < 100:
                                    scales.append(scale)
            
            # Compute global scale - only for non-metric methods
            if is_metric:
                global_scale = 1.0  # No scaling for metric depth
            elif scales:
                global_scale = np.median(scales)
                logger.info(f"    Global depth scale: {global_scale:.4f}")
            else:
                global_scale = 1.0
                logger.warning(f"    No valid GT frames for scaling, using scale=1.0")
            
            # Second pass: apply global scale to all frames
            depths = []
            abs_rels = []
            
            for frame, raw_depth in zip(frames, raw_depths):
                # Check for degenerate MiDaS output (negative values or near-zero)
                if raw_depth.min() < -0.1 or raw_depth.mean() < 0.1:
                    # Skip degenerate frames - pass None to let GS handle it
                    pred_depth = None
                else:
                    pred_depth = raw_depth * global_scale
                    pred_depth = np.clip(pred_depth, 0.0, 15.0)
                
                # Compute AbsRel only for valid frames
                gt_depth = frame.depth
                if gt_depth is not None and pred_depth is not None:
                    valid_gt = (gt_depth > 0.1) & (gt_depth < 10)
                    valid_pred = pred_depth > 0.05
                    valid = valid_gt & valid_pred
                    if valid.sum() > 100:
                        abs_rel = np.mean(np.abs(pred_depth[valid] - gt_depth[valid]) / gt_depth[valid])
                        abs_rels.append(abs_rel)
                
                depths.append(pred_depth)
            
            estimated_depths[depth_method] = depths
            depth_errors[depth_method] = np.mean(abs_rels) if abs_rels else float('inf')
            logger.info(f"    AbsRel: {depth_errors[depth_method]:.4f}")
        except Exception as e:
            logger.error(f"    Failed: {e}")
            import traceback
            traceback.print_exc()
            estimated_depths[depth_method] = [frame.depth for frame in frames]  # Fallback to GT
            depth_errors[depth_method] = float('inf')
        finally:
            # Cleanup estimator to free GPU memory
            if estimator is not None:
                if hasattr(estimator, 'cleanup'):
                    estimator.cleanup()
                del estimator
            safe_cuda_cleanup()
    
    # Track CUDA health across tests - check initial state
    cuda_healthy = check_cuda_health()
    if not cuda_healthy:
        logger.error("CUDA already unhealthy before GS tests - skipping all configurations")
        for config_name, pose_source, depth_source in configs:
            results.append({
                'name': config_name,
                'pose_source': pose_source,
                'depth_source': depth_source,
                'gs_engine': gs_engine,
                'dataset': dataset_name,
                'psnr': 0, 'ssim': 0, 'lpips': 0,
                'total_time': 0, 'fps': 0, 'num_gaussians': 0,
                'pose_ate': 0, 'depth_abs_rel': 0,
                'error': 'CUDA context corrupted before test'
            })
        return results, False
    
    # Run GS with each configuration
    for config_name, pose_source, depth_source in configs:
        logger.info(f"  Testing {config_name} with {gs_engine}...")
        
        # Skip if CUDA is in a bad state
        if not cuda_healthy:
            logger.warning(f"  Skipping {config_name} due to previous CUDA error")
            results.append({
                'name': config_name,
                'pose_source': pose_source,
                'depth_source': depth_source,
                'gs_engine': gs_engine,
                'dataset': dataset_name,
                'psnr': 0, 'ssim': 0, 'lpips': 0,
                'total_time': 0, 'fps': 0, 'num_gaussians': 0,
                'pose_ate': 0, 'depth_abs_rel': 0,
                'error': 'Skipped due to CUDA error'
            })
            continue
        
        # Check GPU health before each test
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                # Quick test to verify CUDA is working
                test_tensor = torch.zeros(1, device='cuda')
                del test_tensor
                # Quick memory check
                free_mem = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
                if free_mem < 1e9:  # Less than 1GB free
                    logger.warning(f"Low GPU memory ({free_mem/1e9:.1f}GB free), forcing cleanup...")
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                logger.error(f"CUDA error before test: {e}")
                cuda_healthy = False
                results.append({
                    'name': config_name,
                    'pose_source': pose_source,
                    'depth_source': depth_source,
                    'gs_engine': gs_engine,
                    'dataset': dataset_name,
                    'psnr': 0, 'ssim': 0, 'lpips': 0,
                    'total_time': 0, 'fps': 0, 'num_gaussians': 0,
                    'pose_ate': 0, 'depth_abs_rel': 0,
                    'error': f'CUDA error: {str(e)}'
                })
                continue
        
        try:
            torch.cuda.empty_cache()
            
            engine = get_engine(gs_engine)
            config = {'num_frames': len(frames), 'max_gaussians': 200000}
            engine.initialize_scene(intrinsics, config)
            
            t_start = time.time()
            
            # Check if this is a batch-based engine (like DA3GS)
            is_batch_engine = getattr(engine, 'is_batch_engine', False)
            
            for i, frame in enumerate(frames):
                # Get pose
                if pose_source == 'gt':
                    pose = frame.pose
                else:
                    pose = estimated_poses[pose_source][i]
                
                # Get depth
                if depth_source == 'gt':
                    depth = frame.depth
                else:
                    depth = estimated_depths[depth_source][i]
                
                engine.add_frame(
                    frame_id=i,
                    rgb=frame.rgb,
                    depth=depth,
                    pose_world_cam=pose
                )
                
                # For incremental engines, optimize after each frame
                # For batch engines (DA3GS), defer optimization until all frames added
                if not is_batch_engine:
                    for _ in range(iterations):
                        try:
                            engine.optimize_step(n_steps=1)
                        except:
                            pass
            
            # For batch engines, run optimization once after all frames
            if is_batch_engine:
                try:
                    engine.optimize_step(n_steps=1)
                except Exception as e:
                    logger.warning(f"Batch optimization failed: {e}")
            
            total_time = time.time() - t_start
            
            # Evaluate quality
            psnr_scores, ssim_scores, lpips_scores = [], [], []
            eval_indices = np.linspace(0, len(frames)-1, min(5, len(frames)), dtype=int)
            
            for idx in eval_indices:
                frame = frames[idx]
                pose = frame.pose if pose_source == 'gt' else estimated_poses[pose_source][idx]
                
                try:
                    rendered = engine.render_view(pose, image_size)
                    if rendered is not None and rendered.shape[0] > 0:
                        gt = frame.rgb
                        if rendered.shape != gt.shape:
                            rendered = cv2.resize(rendered, (gt.shape[1], gt.shape[0]))
                        metrics = compute_image_metrics(rendered, gt, compute_lpips_metric=True)
                        psnr_scores.append(metrics['psnr'])
                        ssim_scores.append(metrics['ssim'])
                        if 'lpips' in metrics and metrics['lpips'] is not None:
                            lpips_scores.append(metrics['lpips'])
                except:
                    pass
            
            result = PipelineResult(
                name=config_name,
                pose_source=pose_source,
                depth_source=depth_source,
                gs_engine=gs_engine,
                dataset=dataset_name,
                psnr=np.mean(psnr_scores) if psnr_scores else 0,
                ssim=np.mean(ssim_scores) if ssim_scores else 0,
                lpips=np.mean(lpips_scores) if lpips_scores else 0,
                total_time=total_time,
                fps=len(frames) / total_time,
                num_gaussians=engine.get_num_gaussians(),
                pose_ate=float(pose_errors.get(pose_source, 0)) if pose_source != 'gt' else 0.0,
                depth_abs_rel=float(depth_errors.get(depth_source, 0)) if depth_source != 'gt' else 0.0,
            )
            results.append(asdict(result))
            
            logger.info(f"    PSNR={result.psnr:.2f}dB, SSIM={result.ssim:.4f}, LPIPS={result.lpips:.4f}, {result.num_gaussians:,} Gaussians")
            log_gpu_memory(f"    ")
            
        except Exception as e:
            logger.error(f"    Failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Add failed result
            results.append({
                'name': config_name,
                'pose_source': pose_source,
                'depth_source': depth_source,
                'gs_engine': gs_engine,
                'dataset': dataset_name,
                'psnr': 0, 'ssim': 0, 'lpips': 0,
                'total_time': 0, 'fps': 0, 'num_gaussians': 0,
                'pose_ate': float(pose_errors.get(pose_source, 0)) if pose_source != 'gt' else 0.0,
                'depth_abs_rel': float(depth_errors.get(depth_source, 0)) if depth_source != 'gt' else 0.0,
                'error': str(e)
            })
            
            # Check if it's a CUDA error - if so, we may need to skip remaining tests
            if "CUDA error" in str(e) or "illegal memory access" in str(e).lower():
                logger.warning("CUDA error detected - attempting aggressive cleanup...")
                cuda_healthy = False
        finally:
            # Critical: Delete engine and free GPU memory aggressively
            engine_instance = locals().get('engine', None)
            try:
                if engine_instance is not None:
                    # Call cleanup method if available
                    if hasattr(engine_instance, 'cleanup'):
                        try:
                            engine_instance.cleanup()
                        except Exception as cleanup_err:
                            logger.warning(f"Engine cleanup error: {cleanup_err}")
                    elif hasattr(engine_instance, 'reset'):
                        try:
                            engine_instance.reset()
                        except Exception as reset_err:
                            logger.warning(f"Engine reset error: {reset_err}")
                    del engine_instance
            except Exception as del_err:
                logger.warning(f"Error deleting engine: {del_err}")
            
            # Ensure engine variable is cleared
            engine = None
            
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                # Try to synchronize and clear CUDA state
                try:
                    torch.cuda.synchronize()
                except RuntimeError as sync_err:
                    logger.warning(f"CUDA sync error during cleanup: {sync_err}")
                    cuda_healthy = False
                    try:
                        torch.cuda.reset_peak_memory_stats()
                    except Exception:
                        pass
                
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                
                # Quick test to see if CUDA is still working
                try:
                    test_tensor = torch.zeros(1, device='cuda')
                    del test_tensor
                    torch.cuda.empty_cache()
                except RuntimeError as recovery_err:
                    logger.error(f"CUDA not recovered: {recovery_err}")
                    cuda_healthy = False
    
    return results, cuda_healthy


# =============================================================================
# Report Generation
# =============================================================================

def print_pose_results(results: List[Dict]):
    """Print pose benchmark results."""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("POSE ESTIMATION RESULTS")
    print("=" * 80)
    print(f"{'Method':<15} {'ATE (m)':<12} {'RPE Trans':<12} {'RPE Rot':<10} {'FPS':<8}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x['ate_rmse']):
        print(f"{r['method']:<15} {r['ate_rmse']:<12.4f} {r['rpe_trans_rmse']:<12.4f} "
              f"{r['rpe_rot_rmse']:<10.2f}° {r['fps']:<8.1f}")


def print_depth_results(results: List[Dict]):
    """Print depth benchmark results."""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("DEPTH ESTIMATION RESULTS")
    print("=" * 80)
    print(f"{'Method':<15} {'AbsRel':<10} {'RMSE':<10} {'δ<1.25':<10} {'FPS':<8}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x['abs_rel']):
        print(f"{r['method']:<15} {r['abs_rel']:<10.4f} {r['rmse']:<10.4f} "
              f"{r['delta1']:<10.2%} {r['fps']:<8.1f}")


def print_gs_results(results: List[Dict]):
    """Print Gaussian splatting results."""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("GAUSSIAN SPLATTING RESULTS")
    print("=" * 80)
    print(f"{'Engine':<12} {'PSNR (dB)':<12} {'SSIM':<10} {'Gaussians':<12} {'FPS':<8}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: -x['psnr']):
        print(f"{r['engine']:<12} {r['psnr']:<12.2f} {r['ssim']:<10.4f} "
              f"{r['final_gaussians']:<12,} {r['fps']:<8.1f}")


def print_pipeline_results(results: List[Dict]):
    """Print combined pipeline results."""
    if not results:
        return
    
    print("\n" + "=" * 80)
    print("COMBINED PIPELINE RESULTS")
    print("=" * 80)
    print(f"{'Configuration':<35} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10} {'Pose ATE':<12} {'Depth Err':<10}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: -x['psnr']):
        pose_ate = f"{r['pose_ate']:.4f}m" if r['pose_ate'] > 0 else "GT"
        depth_err = f"{r['depth_abs_rel']:.4f}" if r['depth_abs_rel'] > 0 else "GT"
        lpips_val = f"{r.get('lpips', 0):.4f}" if r.get('lpips', 0) > 0 else "-"
        print(f"{r['name']:<35} {r['psnr']:<10.2f} {r['ssim']:<10.4f} {lpips_val:<10} "
              f"{pose_ate:<12} {depth_err:<10}")


# =============================================================================
# Render Comparison Grid
# =============================================================================

def create_render_comparison_grid(renders_dir: Path, output_path: Path, gs_results: List[Dict], dataset_name: str = None):
    """
    Create side-by-side comparison of GT vs rendered images for each engine.
    
    Args:
        renders_dir: Base renders directory
        output_path: Where to save the comparison grid
        gs_results: GS benchmark results
        dataset_name: If provided, look in renders_dir/dataset_name/engine
                     Otherwise look in renders_dir/engine (legacy)
    """
    import matplotlib.pyplot as plt
    import cv2
    
    engines = [r['engine'] for r in gs_results]
    
    # Determine render paths - check for dataset subdirectory structure
    def get_engine_dir(engine):
        if dataset_name:
            # New structure: renders/dataset/engine
            path = renders_dir / dataset_name / engine
            if path.exists():
                return path
        # Legacy structure: renders/engine
        return renders_dir / engine
    
    # Find how many sample images we have
    sample_count = 0
    for engine in engines:
        engine_dir = get_engine_dir(engine)
        if engine_dir.exists():
            gt_files = sorted(engine_dir.glob("gt_*.png"))
            sample_count = max(sample_count, len(gt_files))
    
    if sample_count == 0:
        return
    
    # Create grid: rows = samples, cols = GT + engines + error maps
    n_rows = min(sample_count, 5)
    n_cols = 1 + len(engines) * 2  # GT + (rendered + error) per engine
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for row in range(n_rows):
        col = 0
        
        # GT image (from first engine that has it)
        gt_img = None
        for engine in engines:
            engine_dir = get_engine_dir(engine)
            gt_path = engine_dir / f"gt_{row}.png"
            if gt_path.exists():
                gt_img = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB)
                break
        
        if gt_img is not None:
            axes[row, col].imshow(gt_img)
            if row == 0:
                axes[row, col].set_title("Ground Truth", fontweight='bold')
        axes[row, col].axis('off')
        col += 1
        
        # Each engine's rendered + error
        for engine in engines:
            engine_dir = get_engine_dir(engine)
            render_path = engine_dir / f"rendered_{row}.png"
            error_path = engine_dir / f"error_{row}.png"
            
            # Rendered
            if render_path.exists():
                rendered = cv2.cvtColor(cv2.imread(str(render_path)), cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(rendered)
                # Get metrics for this engine
                eng_result = next((r for r in gs_results if r['engine'] == engine), {})
                if row == 0:
                    axes[row, col].set_title(f"{engine}\nPSNR: {eng_result.get('psnr', 0):.1f}dB", fontweight='bold')
            axes[row, col].axis('off')
            col += 1
            
            # Error map
            if error_path.exists():
                error_img = cv2.cvtColor(cv2.imread(str(error_path)), cv2.COLOR_BGR2RGB)
                axes[row, col].imshow(error_img)
                if row == 0:
                    axes[row, col].set_title("Error", fontsize=10)
            axes[row, col].axis('off')
            col += 1
    
    plt.suptitle("Gaussian Splatting: Ground Truth vs Rendered Comparison", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Comprehensive Visualization Generation
# =============================================================================

def generate_all_visualizations(
    all_results: Dict,
    output_dir: Path,
    dataset_name: str,
):
    """
    Generate comprehensive visualizations organized by category.
    
    Output structure:
        output_dir/
        ├── plots/
        │   ├── pose/
        │   │   ├── metrics_bar.png
        │   │   ├── latency.png
        │   │   ├── trajectory_2d.png
        │   │   ├── trajectory_3d.png
        │   │   ├── ate_over_time.png
        │   │   ├── rpe_analysis.png
        │   │   ├── drift_vs_distance.png
        │   │   ├── orientation.png
        │   │   └── radar.png
        │   ├── depth/
        │   │   ├── metrics_bar.png
        │   │   ├── latency.png
        │   │   ├── accuracy_thresholds.png
        │   │   ├── error_heatmaps/
        │   │   └── histograms.png
        │   ├── gs/
        │   │   ├── quality_metrics.png
        │   │   ├── gaussian_statistics.png
        │   │   ├── training_curves.png
        │   │   ├── comprehensive_summary.png
        │   │   └── pipeline_comparison.png
        │   └── summary/
        │       ├── combined_accuracy_panel.png
        │       ├── all_metrics_overview.png
        │       └── radar_charts/
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        logger.warning("Matplotlib not available, skipping visualizations")
        return
    
    # Import visualization modules
    try:
        from benchmarks.visualization.plot_utils import (
            setup_plot_style, plot_pose_metrics_bar, plot_depth_metrics_bar,
            plot_gs_metrics_bar, plot_latency_comparison, plot_overall_summary,
        )
        from benchmarks.visualization.pose_visualizations import (
            plot_trajectory_2d, plot_trajectory_3d, plot_ate_over_time,
            plot_rpe_analysis, plot_drift_vs_distance, plot_orientation_over_time,
            plot_pose_error_summary,
        )
        from benchmarks.visualization.depth_visualizations import (
            plot_depth_histograms, plot_depth_comparison_grid,
        )
        from benchmarks.visualization.gs_visualizations import (
            plot_render_quality_metrics, plot_gaussian_statistics,
            plot_training_curves, plot_gs_comprehensive_summary, plot_pipeline_comparison,
        )
        from benchmarks.visualization.cross_metric_plots import (
            plot_combined_accuracy_panel, plot_all_metrics_violin, plot_method_radar_chart,
        )
        setup_plot_style()
    except ImportError as e:
        logger.warning(f"Could not import visualization modules: {e}")
        return
    
    pose_results = all_results.get('pose', [])
    depth_results = all_results.get('depth', [])
    gs_results = all_results.get('gs', [])
    pipeline_results = all_results.get('pipeline', [])
    
    # Create organized directory structure
    plots_dir = output_dir / "plots"
    pose_dir = plots_dir / "pose"
    depth_dir = plots_dir / "depth"
    gs_dir = plots_dir / "gs"
    summary_dir = plots_dir / "summary"
    
    for d in [plots_dir, pose_dir, depth_dir, gs_dir, summary_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    total_plots = 0
    
    # =========================================================================
    # POSE ESTIMATION PLOTS
    # =========================================================================
    if pose_results:
        logger.info("📍 Generating pose estimation plots...")
        
        # Metrics bar chart
        try:
            plot_pose_metrics_bar(
                pose_results,
                metrics=['ate_rmse', 'rpe_trans_rmse', 'fps'],
                title=f"Pose Estimation - {dataset_name}",
                output_path=pose_dir / "metrics_bar",
            )
            total_plots += 1
            logger.info("    ✓ pose/metrics_bar.png")
        except Exception as e:
            logger.warning(f"    ✗ metrics_bar failed: {e}")
        
        # Latency comparison
        try:
            plot_latency_comparison(
                pose_results,
                benchmark_type='pose',
                title="Pose Estimation Latency",
                output_path=pose_dir / "latency",
            )
            total_plots += 1
            logger.info("    ✓ pose/latency.png")
        except Exception as e:
            logger.warning(f"    ✗ latency failed: {e}")
        
        # Error summary
        try:
            plot_pose_error_summary(
                pose_results,
                output_path=pose_dir / "error_summary.png",
                title=f"Pose Error Summary - {dataset_name}",
            )
            total_plots += 1
            logger.info("    ✓ pose/error_summary.png")
        except Exception as e:
            logger.warning(f"    ✗ error_summary failed: {e}")
        
        # Radar chart
        try:
            plot_method_radar_chart(pose_results, 'pose', pose_dir / "radar.png")
            total_plots += 1
            logger.info("    ✓ pose/radar.png")
        except Exception as e:
            logger.warning(f"    ✗ radar failed: {e}")
    
    # =========================================================================
    # DEPTH ESTIMATION PLOTS
    # =========================================================================
    if depth_results:
        logger.info("📏 Generating depth estimation plots...")
        
        # Metrics bar chart
        try:
            plot_depth_metrics_bar(
                depth_results,
                metrics=['abs_rel', 'rmse', 'delta1'],
                title=f"Depth Estimation - {dataset_name}",
                output_path=depth_dir / "metrics_bar",
            )
            total_plots += 1
            logger.info("    ✓ depth/metrics_bar.png")
        except Exception as e:
            logger.warning(f"    ✗ metrics_bar failed: {e}")
        
        # Latency comparison
        try:
            plot_latency_comparison(
                depth_results,
                benchmark_type='depth',
                title="Depth Estimation Latency",
                output_path=depth_dir / "latency",
            )
            total_plots += 1
            logger.info("    ✓ depth/latency.png")
        except Exception as e:
            logger.warning(f"    ✗ latency failed: {e}")
        
        # Radar chart
        try:
            plot_method_radar_chart(depth_results, 'depth', depth_dir / "radar.png")
            total_plots += 1
            logger.info("    ✓ depth/radar.png")
        except Exception as e:
            logger.warning(f"    ✗ radar failed: {e}")
    
    # =========================================================================
    # GAUSSIAN SPLATTING PLOTS
    # =========================================================================
    if gs_results:
        logger.info("🎨 Generating Gaussian splatting plots...")
        
        # Quality metrics
        try:
            plot_render_quality_metrics(gs_results, gs_dir / "quality_metrics.png")
            total_plots += 1
            logger.info("    ✓ gs/quality_metrics.png")
        except Exception as e:
            logger.warning(f"    ✗ quality_metrics failed: {e}")
        
        # Gaussian statistics
        try:
            plot_gaussian_statistics(gs_results, gs_dir / "gaussian_statistics.png")
            total_plots += 1
            logger.info("    ✓ gs/gaussian_statistics.png")
        except Exception as e:
            logger.warning(f"    ✗ gaussian_statistics failed: {e}")
        
        # Training curves
        try:
            plot_training_curves(gs_results, gs_dir / "training_curves.png")
            total_plots += 1
            logger.info("    ✓ gs/training_curves.png")
        except Exception as e:
            logger.warning(f"    ✗ training_curves failed: {e}")
        
        # Comprehensive summary
        try:
            plot_gs_comprehensive_summary(gs_results, gs_dir / "comprehensive_summary.png")
            total_plots += 1
            logger.info("    ✓ gs/comprehensive_summary.png")
        except Exception as e:
            logger.warning(f"    ✗ comprehensive_summary failed: {e}")
        
        # Radar chart
        try:
            plot_method_radar_chart(gs_results, 'gs', gs_dir / "radar.png")
            total_plots += 1
            logger.info("    ✓ gs/radar.png")
        except Exception as e:
            logger.warning(f"    ✗ radar failed: {e}")
        
        # Render comparison grid (GT vs rendered for each engine)
        try:
            renders_dir = output_dir / "renders"
            if renders_dir.exists():
                # Create comparison per dataset
                create_render_comparison_grid(
                    renders_dir, 
                    gs_dir / "render_comparison.png", 
                    gs_results,
                    dataset_name=dataset_name
                )
                total_plots += 1
                logger.info("    ✓ gs/render_comparison.png")
        except Exception as e:
            logger.warning(f"    ✗ render_comparison failed: {e}")
    
    # =========================================================================
    # PIPELINE COMPARISON PLOTS
    # =========================================================================
    if pipeline_results:
        logger.info("🔄 Generating pipeline comparison plots...")
        
        try:
            plot_pipeline_comparison(pipeline_results, gs_dir / "pipeline_comparison.png")
            total_plots += 1
            logger.info("    ✓ gs/pipeline_comparison.png")
        except Exception as e:
            logger.warning(f"    ✗ pipeline_comparison failed: {e}")
    
    # =========================================================================
    # SUMMARY / CROSS-METRIC PLOTS
    # =========================================================================
    logger.info("📊 Generating summary plots...")
    
    # Combined accuracy panel (main figure for papers)
    try:
        plot_combined_accuracy_panel(
            pose_results, depth_results, gs_results, pipeline_results,
            summary_dir / "combined_accuracy_panel.png",
            dataset_name=dataset_name,
        )
        total_plots += 1
        logger.info("    ✓ summary/combined_accuracy_panel.png")
    except Exception as e:
        logger.warning(f"    ✗ combined_accuracy_panel failed: {e}")
    
    # All metrics overview
    try:
        plot_all_metrics_violin(
            pose_results, depth_results, gs_results,
            summary_dir / "all_metrics_overview.png",
        )
        total_plots += 1
        logger.info("    ✓ summary/all_metrics_overview.png")
    except Exception as e:
        logger.warning(f"    ✗ all_metrics_overview failed: {e}")
    
    # Overall summary (original)
    try:
        plot_overall_summary(
            pose_results=pose_results,
            depth_results=depth_results,
            gs_results=gs_results,
            title=f"AirSplatMap Benchmark Summary - {dataset_name}",
            output_path=summary_dir / "overall_summary",
        )
        total_plots += 1
        logger.info("    ✓ summary/overall_summary.png")
    except Exception as e:
        logger.warning(f"    ✗ overall_summary failed: {e}")
    
    logger.info(f"✅ Generated {total_plots} visualization plots")
    
    # Print directory structure
    print(f"\n📁 Plots directory structure:")
    print(f"   {plots_dir}/")
    for subdir in ['pose', 'depth', 'gs', 'summary']:
        subpath = plots_dir / subdir
        if subpath.exists():
            files = list(subpath.glob('*.png'))
            print(f"   ├── {subdir}/ ({len(files)} plots)")
            for f in sorted(files)[:5]:
                print(f"   │   ├── {f.name}")
            if len(files) > 5:
                print(f"   │   └── ... and {len(files) - 5} more")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='AirSplatMap Benchmark Suite')
    
    # What to run
    parser.add_argument('--pose', action='store_true', help='Run pose benchmark')
    parser.add_argument('--depth', action='store_true', help='Run depth benchmark')
    parser.add_argument('--gs', action='store_true', help='Run Gaussian splatting benchmark')
    parser.add_argument('--pipeline', action='store_true', help='Run combined pipeline benchmark')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    
    # Configuration
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer frames)')
    parser.add_argument('--dataset', type=str, help='Specific dataset name')
    parser.add_argument('--datasets', nargs='+', help='Multiple dataset names to benchmark')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames')
    parser.add_argument('--multi-dataset', action='store_true', 
                        help='Run benchmark across ALL available datasets')
    parser.add_argument('--comprehensive', action='store_true', help='Test ALL available methods')
    parser.add_argument('--jetson', action='store_true', 
                        help='Jetson mode: all TUM freiburg datasets with memory optimizations')
    
    # Methods - defaults are fast/reliable options
    parser.add_argument('--pose-methods', nargs='+', 
                        default=['orb', 'sift', 'robust_flow'],
                        help='Pose methods to test')
    parser.add_argument('--depth-methods', nargs='+', 
                        default=['midas', 'depth_anything_v3', 'depth_pro'],
                        help='Depth methods to test')
    parser.add_argument('--gs-engines', nargs='+', 
                        default=['graphdeco'],
                        help='GS engines to test (graphdeco, gsplat, splatam, gslam, monogs, da3gs)')
    
    args = parser.parse_args()
    
    # Check GPU availability upfront
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"✓ GPU available: {gpu_name} ({gpu_mem:.1f} GB)")
            print(f"\n🎮 GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        else:
            logger.warning("⚠️ No GPU available - benchmarks will be slow!")
            print("\n⚠️  WARNING: No GPU detected - benchmarks will run on CPU (very slow)")
    except ImportError:
        logger.warning("PyTorch not available - cannot check GPU")
    
    # Comprehensive mode - test ALL available methods
    if args.comprehensive:
        from src.pose import list_pose_estimators
        from src.depth import list_depth_estimators
        from src.engines import list_engines
        
        args.pose_methods = list(list_pose_estimators().keys())
        # Exclude stereo methods (require stereo camera pairs, not available in TUM)
        # Exclude depth_pro and depth_pro_lite - both OOM on Jetson 8GB (1.9GB model)
        # Exclude depth_anything_v2 since v3 is available
        excluded_depth = ('stereo', 'stereo_fast', 'stereo_sgbm', 'stereo_bm', 'none',
                         'depth_pro', 'depth_pro_lite',  # OOM on Jetson
                         'depth_anything_v2', 'dav2', 'depth_anything')
        args.depth_methods = [k for k, v in list_depth_estimators().items() 
                             if k not in excluded_depth
                             and v.get('available', True)]
        # Exclude da3gs - experimental, GS rendering not fully working yet
        excluded_engines = ('da3gs',)
        args.gs_engines = [k for k, v in list_engines().items() 
                          if k not in excluded_engines
                          and v.get('available', True)]
        
        logger.info(f"Comprehensive mode: {len(args.pose_methods)} pose, {len(args.depth_methods)} depth, {len(args.gs_engines)} GS")
    
    # Default to all if nothing specified
    run_all = args.all or not (args.pose or args.depth or args.gs or args.pipeline)
    
    # Quick mode settings
    # Pose and Depth benchmarks run on ALL frames to build complete cache
    # Pipeline uniformly samples from cache for GS testing
    if args.quick:
        pose_frames = None  # None = ALL frames
        depth_frames = None  # None = ALL frames
        pipeline_frames = 100  # Uniform sample for GS
        gs_iterations = 3
    else:
        pose_frames = None  # None = ALL frames for accurate ATE
        depth_frames = None  # None = ALL frames for complete cache
        pipeline_frames = min(args.max_frames, 200)  # Uniform sample for GS
        gs_iterations = 5
    
    # Find datasets
    dataset_root = PROJECT_ROOT / "datasets"
    
    # Find all supported datasets with their types
    all_datasets_with_types = find_all_datasets(dataset_root)
    
    # Filter datasets based on arguments
    if args.datasets:
        # Multiple specific datasets by name
        all_datasets_with_types = [
            (d, t) for d, t in all_datasets_with_types 
            if any(name in d.name for name in args.datasets)
        ]
    elif args.dataset:
        # Single specific dataset
        all_datasets_with_types = [
            (d, t) for d, t in all_datasets_with_types 
            if args.dataset in d.name
        ]
    elif not args.multi_dataset:
        # Default: use first dataset only
        all_datasets_with_types = all_datasets_with_types[:1]
    else:
        # Multi-dataset mode: use curated list to avoid redundant scenes
        # This reduces ~19 datasets to ~8 representative ones
        all_datasets_with_types = [
            (d, t) for d, t in all_datasets_with_types 
            if d.name in CURATED_DATASETS
        ]
        if not all_datasets_with_types:
            logger.warning("No curated datasets found, using all available")
            all_datasets_with_types = find_all_datasets(dataset_root)
    
    # Jetson mode: include all TUM datasets (optimizations allow this now)
    # GPU peak was only 0.75GB with freiburg1, plenty of headroom for larger datasets
    if args.jetson:
        # Include freiburg1, freiburg2, and freiburg3 datasets
        all_datasets_with_types = [
            (d, t) for d, t in all_datasets_with_types
            if 'freiburg' in d.name  # All TUM RGB-D datasets
        ]
        logger.info(f"Jetson mode: {len(all_datasets_with_types)} TUM datasets")
    
    # Extract just the paths for backward compatibility
    datasets = [d for d, t in all_datasets_with_types]
    dataset_types = {str(d): t for d, t in all_datasets_with_types}
    
    if not datasets:
        logger.error(f"No datasets found in {dataset_root}")
        return
    
    logger.info(f"Found {len(datasets)} datasets to benchmark")
    for d in datasets:
        logger.info(f"  - {d.name}")
    
    # Create timestamped output directory under hostname folder
    hostname = socket.gethostname()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    host_results_dir = PROJECT_ROOT / "benchmarks" / "results" / hostname
    output_dir = host_results_dir / f"benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Setup file logging to output directory (logs to both console and file)
    setup_file_logging(output_dir)
    
    # Initialize hardware monitoring
    try:
        from benchmarks.hardware_monitor import HardwareMonitor, get_system_info, print_hardware_stats
        hardware_monitor = HardwareMonitor(sample_interval=0.5)
        hardware_monitor.start()
        hardware_available = True
        logger.info("Hardware monitoring enabled")
    except Exception as e:
        logger.warning(f"Hardware monitoring not available: {e}")
        hardware_monitor = None
        hardware_available = False
    
    print("\n" + "=" * 80)
    print("  AIRSPLATMAP BENCHMARK SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Datasets: {len(datasets)} ({', '.join(d.name[:20] for d in datasets[:3])}{'...' if len(datasets) > 3 else ''})")
    print(f"  Output:  {output_dir}")
    print("=" * 80)
    
    # Print system info
    if hardware_available:
        try:
            sys_info = get_system_info()
            print(f"\n  System: {sys_info.get('cpu', {}).get('cores_logical', '?')} CPU cores, "
                  f"{sys_info.get('memory', {}).get('total_gb', '?')} GB RAM")
            if 'gpu' in sys_info and sys_info['gpu']:
                print(f"  GPU: {sys_info['gpu'].get('name', 'Unknown')} "
                      f"({sys_info['gpu'].get('memory_total_gb', '?')} GB)")
        except:
            sys_info = {}
    else:
        sys_info = {}
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'datasets': [d.name for d in datasets],
        'system_info': sys_info,
        'pose': [],
        'depth': [],
        'gs': [],
        'pipeline': [],
        'hardware': {},
        'per_dataset': {},  # Results grouped by dataset
    }
    
    # Track global CUDA health across datasets
    global_cuda_healthy = True
    
    # Run benchmarks on each dataset
    for dataset_idx, dataset in enumerate(datasets):
        dataset_name = dataset.name
        print(f"\n{'='*80}")
        print(f"  DATASET {dataset_idx + 1}/{len(datasets)}: {dataset_name}")
        print(f"{'='*80}")
        
        # Check CUDA health before starting a new dataset
        if not global_cuda_healthy:
            logger.error(f"Skipping {dataset_name} - CUDA context corrupted from previous dataset")
            print(f"  SKIPPED - CUDA context corrupted")
            continue
        
        if not check_cuda_health():
            logger.error(f"CUDA unhealthy before {dataset_name} - skipping remaining datasets")
            global_cuda_healthy = False
            print(f"  SKIPPED - CUDA context corrupted")
            continue
        
        dataset_results = {
            'pose': [],
            'depth': [],
            'gs': [],
            'pipeline': [],
        }
        
        if run_all or args.pose:
            print("\n" + "-" * 40)
            print(f"POSE ESTIMATION - {dataset_name}")
            print("-" * 40)
            if hardware_monitor:
                hardware_monitor.mark(f"pose_{dataset_name}")
            
            # Cache poses for later use by pipeline
            pose_cache_dir = output_dir / 'cache' if output_dir else None
            pose_results = run_pose_benchmark(
                dataset, args.pose_methods, pose_frames, skip_frames=1,
                cache_dir=pose_cache_dir
            )
            # Add dataset info to each result
            for r in pose_results:
                r['dataset'] = dataset_name
            dataset_results['pose'] = pose_results
            all_results['pose'].extend(pose_results)
            print_pose_results(pose_results)
        
        if run_all or args.depth:
            print("\n" + "-" * 40)
            print(f"DEPTH ESTIMATION - {dataset_name}")
            print("-" * 40)
            if hardware_monitor:
                hardware_monitor.mark(f"depth_{dataset_name}")
            
            # Cache depths for later use by pipeline (processes ALL frames)
            depth_cache_dir = output_dir if output_dir else None
            depth_results = run_depth_benchmark(
                dataset, args.depth_methods,
                cache_dir=depth_cache_dir
            )
            for r in depth_results:
                r['dataset'] = dataset_name
            dataset_results['depth'] = depth_results
            all_results['depth'].extend(depth_results)
            print_depth_results(depth_results)
        
        if run_all or args.gs:
            print("\n" + "-" * 40)
            print(f"GAUSSIAN SPLATTING - {dataset_name}")
            print("-" * 40)
            if hardware_monitor:
                hardware_monitor.mark(f"gs_{dataset_name}")
            gs_results = run_gs_benchmark(
                dataset, args.gs_engines, pipeline_frames, gs_iterations, output_dir
            )
            for r in gs_results:
                r['dataset'] = dataset_name
            dataset_results['gs'] = gs_results
            all_results['gs'].extend(gs_results)
            print_gs_results(gs_results)
        
        if run_all or args.pipeline:
            print("\n" + "-" * 40)
            print(f"COMBINED PIPELINE - {dataset_name}")
            print("-" * 40)
            if hardware_monitor:
                hardware_monitor.mark(f"pipeline_{dataset_name}")
            
            # Limit pipeline to fast methods to avoid OOM on Jetson
            # Heavy GPU methods (loftr, superpoint, lightglue, raft, r2d2, roma) are tested in pose benchmark
            fast_pose_methods = ['orb', 'sift', 'robust_flow', 'keyframe']
            pipeline_pose_methods = [m for m in fast_pose_methods if m in args.pose_methods] or ['orb']
            pipeline_depth_methods = args.depth_methods  # Always use user-specified depth methods
            
            # Load cached poses/depths from pose and depth benchmarks (ALL frames)
            _, cached_poses, cached_pose_errors, _ = load_pose_cache(output_dir, dataset_name)
            _, cached_depths, cached_depth_errors = load_depth_cache(output_dir, dataset_name)
            
            if cached_poses:
                logger.info(f"  Loaded cached poses: {list(cached_poses.keys())}")
            if cached_depths:
                logger.info(f"  Loaded cached depths: {list(cached_depths.keys())}")
            
            # Run pipeline benchmark for EACH GS engine
            pipeline_engines = args.gs_engines if args.gs_engines else ['graphdeco']
            pipeline_results = []
            for i, engine in enumerate(pipeline_engines):
                logger.info(f"  Running pipeline with {engine}...")
                
                engine_results, cuda_ok = run_pipeline_benchmark(
                    dataset,
                    dataset_type=dataset_types.get(str(dataset), 'tum'),
                    gs_engine=engine,
                    pose_methods=pipeline_pose_methods,
                    depth_methods=pipeline_depth_methods,
                    max_frames=pipeline_frames,
                    iterations=gs_iterations,
                    cached_poses=cached_poses,
                    cached_depths=cached_depths,
                    cached_pose_errors=cached_pose_errors,
                    cached_depth_errors=cached_depth_errors,
                )
                pipeline_results.extend(engine_results)
                
                # If CUDA became unhealthy, skip remaining engines AND mark globally unhealthy
                if not cuda_ok:
                    logger.warning(f"CUDA unhealthy after {engine}, skipping remaining pipeline engines")
                    global_cuda_healthy = False
                    break
            
            dataset_results['pipeline'] = pipeline_results
            all_results['pipeline'].extend(pipeline_results)
            print_pipeline_results(pipeline_results)
        
        # Check CUDA health after each dataset
        if not check_cuda_health():
            logger.warning(f"CUDA became unhealthy after {dataset_name}")
            global_cuda_healthy = False
        
        # Store per-dataset results
        all_results['per_dataset'][dataset_name] = dataset_results
    
    # Stop hardware monitoring and collect results
    if hardware_monitor:
        hardware_monitor.stop()
        all_results['hardware'] = hardware_monitor.get_summary()
        
        # Print hardware summary
        print("\n" + "-" * 40)
        print("HARDWARE USAGE SUMMARY")
        print("-" * 40)
        overall = hardware_monitor.get_stats()
        print_hardware_stats(overall, "Overall Benchmark")
        
        # Per-phase stats
        for phase in ['pose', 'depth', 'gs', 'pipeline']:
            phase_stats = hardware_monitor.get_stats(phase)
            if phase_stats.num_samples > 0:
                print(f"\n  {phase.upper():10} - GPU: {phase_stats.gpu_utilization_mean:.0f}% util, "
                      f"{phase_stats.gpu_memory_used_gb_max:.1f}GB max | "
                      f"CPU: {phase_stats.cpu_percent_mean:.0f}% | "
                      f"RAM: {phase_stats.ram_used_gb_max:.1f}GB")
    
    # Generate plots with organized folder structure
    print("\n" + "-" * 40)
    print("GENERATING VISUALIZATIONS")
    print("-" * 40)
    
    # Use the new comprehensive visualization generator
    try:
        from benchmarks.visualization.generate_visualizations import generate_comprehensive_visualizations
        generate_comprehensive_visualizations(all_results, output_dir)
    except ImportError as e:
        logger.warning(f"Could not import new visualization module: {e}")
        # Fall back to old visualization system
        dataset_label = f"{len(datasets)} datasets" if len(datasets) > 1 else datasets[0].name
        generate_all_visualizations(all_results, output_dir, dataset_label)
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save results JSON
    save_results(all_results, output_dir / "results.json")
    
    # Generate HTML report
    try:
        from benchmarks.visualization.html_report import (
            generate_html_report, generate_simple_html_report
        )
        generate_html_report(
            all_results['pose'],
            all_results['depth'],
            all_results['gs'],
            all_results['pipeline'],
            all_results.get('hardware', {}),
            output_dir / "report.html",
        )
        logger.info("Generated interactive HTML report")
    except Exception as e:
        logger.warning(f"HTML report failed: {e}, trying simple fallback")
        import traceback
        traceback.print_exc()
        try:
            from benchmarks.visualization.html_report import generate_simple_html_report
            generate_simple_html_report(
                all_results['pose'],
                all_results['depth'],
                all_results['gs'],
                all_results['pipeline'],
                output_dir / "report.html",
            )
            logger.info("Generated simple HTML report (fallback)")
        except Exception as e2:
            logger.error(f"HTML report generation failed: {e2}")
    
    # Create a latest symlink within the hostname folder
    latest_link = host_results_dir / "latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(output_dir.name)
        logger.info(f"Updated latest symlink: {latest_link} -> {output_dir.name}")
    except Exception as e:
        logger.warning(f"Could not create latest symlink: {e}")
    
    # List plots
    plots_dir = output_dir / "plots"
    total_plots = sum(1 for _ in plots_dir.rglob("*.png")) if plots_dir.exists() else 0
    
    # Count per-dataset vs general plots
    general_plots = sum(1 for _ in (plots_dir / "general").rglob("*.png")) if (plots_dir / "general").exists() else 0
    dataset_dirs = [d for d in plots_dir.iterdir() if d.is_dir() and d.name != "general"] if plots_dir.exists() else []
    
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\n📁 Output Directory: {output_dir}")
    print(f"   ├── results.json      - Raw benchmark data")
    print(f"   ├── report.html       - Interactive HTML report with:")
    print(f"   │                        • Dataset dropdowns for filtering")
    print(f"   │                        • Method/engine checkboxes")
    print(f"   │                        • Interactive 3D trajectory viewer")
    print(f"   │                        • Cross-dataset aggregated summaries")
    print(f"   ├── benchmark.log     - Full log output")
    print(f"   ├── renders/          - GS render comparisons (per-dataset)")
    print(f"   └── plots/            - {total_plots} visualization charts")
    print(f"       ├── general/      - {general_plots} aggregated cross-dataset plots")
    for ddir in sorted(dataset_dirs)[:3]:
        dcount = sum(1 for _ in ddir.rglob("*.png"))
        print(f"       ├── {ddir.name[:20]+'...' if len(ddir.name) > 20 else ddir.name}/  - {dcount} plots")
    if len(dataset_dirs) > 3:
        print(f"       └── ... and {len(dataset_dirs) - 3} more dataset folders")
    print(f"\n🔗 Latest results: benchmarks/results/{hostname}/latest/")


if __name__ == '__main__':
    main()
