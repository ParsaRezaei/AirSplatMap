#!/usr/bin/env python3
"""
Depth Estimation Benchmark
==========================

Benchmark depth estimators against TUM RGB-D, 7-Scenes, Replica, or other datasets 
with ground truth depth.
Computes standard depth metrics: AbsRel, SqRel, RMSE, RMSElog, δ < 1.25^n

Usage:
    python benchmarks/depth/benchmark_depth.py                           # Run all methods
    python benchmarks/depth/benchmark_depth.py --methods midas zoedepth  # Specific methods
    python benchmarks/depth/benchmark_depth.py --dataset fr1_desk        # Specific dataset
    python benchmarks/depth/benchmark_depth.py --output results.json
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
    from src.pipeline.frames import TumRGBDSource, SevenScenesSource, ReplicaSource
    
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
class DepthBenchmarkResult:
    """Result from benchmarking a depth estimator."""
    method: str
    dataset: str
    num_frames: int
    total_time: float
    fps: float
    
    # Standard depth metrics - ALIGNED (lower is better except δ)
    abs_rel: float  # |d - d*| / d*
    sq_rel: float   # |d - d*|² / d*
    rmse: float     # √(mean((d - d*)²))
    rmse_log: float # √(mean((log(d) - log(d*))²))
    
    # Threshold accuracy - ALIGNED (higher is better)
    delta1: float  # % with max(d/d*, d*/d) < 1.25
    delta2: float  # % with max(d/d*, d*/d) < 1.25²
    delta3: float  # % with max(d/d*, d*/d) < 1.25³
    
    # Fields with defaults must come last
    is_metric: bool = False  # Whether this method outputs metric depth (vs relative)
    silog: float = 0.0  # Scale-invariant log error (Eigen et al.)
    edge_f1: float = 0.0  # F1 score for depth discontinuities
    
    # RAW metrics (without alignment) - only meaningful for metric depth
    raw_abs_rel: float = 0.0
    raw_rmse: float = 0.0
    raw_delta1: float = 0.0
    
    # Additional stats
    scale: float = 1.0  # Scale factor used for alignment
    shift: float = 0.0  # Shift used for alignment
    valid_pixels: int = 0
    
    # Latency metrics (ms per frame)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


def compute_depth_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    align: bool = True,
) -> Dict[str, float]:
    """
    Compute standard depth estimation metrics.
    
    Args:
        pred: HxW predicted depth
        gt: HxW ground truth depth
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        align: Whether to scale-shift align prediction to GT
        
    Returns:
        Dict with all metrics
    """
    # Create valid mask
    valid = (gt > min_depth) & (gt < max_depth) & np.isfinite(gt)
    valid &= (pred > 0) & np.isfinite(pred)
    
    if np.sum(valid) < 100:
        return {
            'abs_rel': float('nan'),
            'sq_rel': float('nan'),
            'rmse': float('nan'),
            'rmse_log': float('nan'),
            'delta1': 0.0,
            'delta2': 0.0,
            'delta3': 0.0,
            'scale': 1.0,
            'shift': 0.0,
            'valid_pixels': int(np.sum(valid)),
        }
    
    pred_valid = pred[valid]
    gt_valid = gt[valid]
    
    scale, shift = 1.0, 0.0
    
    if align:
        # Least squares alignment: pred_aligned = scale * pred + shift
        # Minimize ||scale * pred + shift - gt||²
        A = np.vstack([pred_valid, np.ones_like(pred_valid)]).T
        result = np.linalg.lstsq(A, gt_valid, rcond=None)
        scale, shift = result[0]
        
        # Ensure positive scale
        if scale < 0:
            scale = np.median(gt_valid) / np.median(pred_valid)
            shift = 0
        
        pred_aligned = scale * pred_valid + shift
    else:
        pred_aligned = pred_valid
    
    # Clamp aligned predictions
    pred_aligned = np.clip(pred_aligned, min_depth, max_depth)
    
    # Compute metrics
    thresh = np.maximum(pred_aligned / gt_valid, gt_valid / pred_aligned)
    
    abs_rel = np.mean(np.abs(pred_aligned - gt_valid) / gt_valid)
    sq_rel = np.mean(((pred_aligned - gt_valid) ** 2) / gt_valid)
    rmse = np.sqrt(np.mean((pred_aligned - gt_valid) ** 2))
    
    # Log metrics (avoid log of zero)
    pred_log = np.log(np.maximum(pred_aligned, 1e-6))
    gt_log = np.log(np.maximum(gt_valid, 1e-6))
    rmse_log = np.sqrt(np.mean((pred_log - gt_log) ** 2))
    
    # Scale-invariant log error (SILog) - Eigen et al.
    # silog = sqrt(mean(d²) - λ*mean(d)²) where d = log(pred) - log(gt)
    log_diff = pred_log - gt_log
    silog = np.sqrt(np.mean(log_diff ** 2) - 0.5 * (np.mean(log_diff) ** 2))
    
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25 ** 2)
    delta3 = np.mean(thresh < 1.25 ** 3)
    
    return {
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'silog': float(silog),
        'delta1': float(delta1),
        'delta2': float(delta2),
        'delta3': float(delta3),
        'scale': float(scale),
        'shift': float(shift),
        'valid_pixels': int(np.sum(valid)),
    }


def compute_edge_accuracy(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.25) -> Dict[str, float]:
    """
    Compute edge accuracy metrics - how well depth discontinuities are preserved.
    
    Args:
        pred: HxW predicted depth
        gt: HxW ground truth depth  
        threshold: Edge detection threshold (relative depth change)
        
    Returns:
        Dict with precision, recall, F1 for edges
    """
    try:
        import cv2
    except ImportError:
        return {'edge_precision': 0.0, 'edge_recall': 0.0, 'edge_f1': 0.0}
    
    # Compute depth gradients
    def get_edges(depth, threshold):
        # Sobel gradients
        gx = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        
        # Normalize by local depth to get relative gradient
        depth_safe = np.maximum(depth, 0.1)
        rel_grad = grad_mag / depth_safe
        
        return rel_grad > threshold
    
    # Valid mask
    valid = (gt > 0.1) & (gt < 10) & np.isfinite(gt) & (pred > 0) & np.isfinite(pred)
    
    gt_edges = get_edges(gt, threshold)
    pred_edges = get_edges(pred, threshold)
    
    # Apply valid mask
    gt_edges = gt_edges & valid
    pred_edges = pred_edges & valid
    
    # Compute precision, recall, F1
    tp = np.sum(gt_edges & pred_edges)
    fp = np.sum(pred_edges & ~gt_edges)
    fn = np.sum(gt_edges & ~pred_edges)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    return {
        'edge_precision': float(precision),
        'edge_recall': float(recall),
        'edge_f1': float(f1),
    }


class BaseDepthEstimator:
    """Base class for depth estimators."""
    
    def __init__(self):
        self._device = 'cuda'
    
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """Estimate depth from RGB image. Returns HxW depth map."""
        raise NotImplementedError
    
    def get_name(self) -> str:
        raise NotImplementedError


class MiDaSDepthEstimator(BaseDepthEstimator):
    """MiDaS depth estimation (DPT-Large or smaller models)."""
    
    def __init__(self, model_type: str = 'DPT_Large'):
        super().__init__()
        self._model_type = model_type
        self._model = None
        self._transform = None
        
        try:
            import torch
            self._torch = torch
            
            # Load MiDaS
            self._model = torch.hub.load('intel-isl/MiDaS', model_type)
            self._model.to(self._device).eval()
            
            # Get transforms
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms')
            if model_type in ['DPT_Large', 'DPT_Hybrid']:
                self._transform = midas_transforms.dpt_transform
            else:
                self._transform = midas_transforms.small_transform
            
            logger.info(f"MiDaS {model_type} initialized on {self._device}")
            self._available = True
        except Exception as e:
            logger.warning(f"MiDaS not available: {e}")
            self._available = False
    
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        if not self._available:
            return np.zeros(rgb.shape[:2], dtype=np.float32)
        
        import torch
        
        # Transform
        input_batch = self._transform(rgb).to(self._device)
        
        # Inference
        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode='bicubic',
                align_corners=False,
            ).squeeze()
        
        depth = prediction.cpu().numpy()
        
        # MiDaS outputs inverse depth (disparity-like)
        # Convert to depth: higher value = closer
        depth = depth.max() - depth + 1e-6
        
        return depth.astype(np.float32)
    
    def get_name(self) -> str:
        return f"midas_{self._model_type.lower()}"


class DepthAnythingEstimator(BaseDepthEstimator):
    """Depth Anything V2 estimator."""
    
    def __init__(self, model_size: str = 'vitl'):
        super().__init__()
        self._model_size = model_size
        self._model = None
        self._available = False
        
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
            
            self._torch = torch
            
            model_name = f"depth-anything/Depth-Anything-V2-{model_size.capitalize()}-hf"
            
            self._processor = AutoImageProcessor.from_pretrained(model_name)
            self._model = AutoModelForDepthEstimation.from_pretrained(model_name)
            self._model.to(self._device).eval()
            
            logger.info(f"Depth Anything V2 ({model_size}) initialized on {self._device}")
            self._available = True
        except Exception as e:
            logger.warning(f"Depth Anything not available: {e}")
    
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        if not self._available:
            return np.zeros(rgb.shape[:2], dtype=np.float32)
        
        import torch
        from PIL import Image
        
        # Convert to PIL
        image = Image.fromarray(rgb)
        
        # Process
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model(**inputs)
            predicted_depth = outputs.predicted_depth
        
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=rgb.shape[:2],
            mode='bicubic',
            align_corners=False,
        ).squeeze()
        
        depth = prediction.cpu().numpy()
        
        return depth.astype(np.float32)
    
    def get_name(self) -> str:
        return f"depth_anything_{self._model_size}"


class ZoeDepthEstimator(BaseDepthEstimator):
    """ZoeDepth metric depth estimator."""
    
    def __init__(self, model_type: str = 'ZoeD_NK'):
        super().__init__()
        self._model_type = model_type
        self._model = None
        self._available = False
        
        try:
            import torch
            self._torch = torch
            
            # Load ZoeDepth
            self._model = torch.hub.load('isl-org/ZoeDepth', model_type, pretrained=True)
            self._model.to(self._device).eval()
            
            logger.info(f"ZoeDepth ({model_type}) initialized on {self._device}")
            self._available = True
        except Exception as e:
            logger.warning(f"ZoeDepth not available: {e}")
    
    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        if not self._available:
            return np.zeros(rgb.shape[:2], dtype=np.float32)
        
        import torch
        from PIL import Image
        
        image = Image.fromarray(rgb)
        
        with torch.no_grad():
            depth = self._model.infer_pil(image)
        
        # Resize to original
        if depth.shape != rgb.shape[:2]:
            import cv2
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        return depth.astype(np.float32)
    
    def get_name(self) -> str:
        return f"zoedepth_{self._model_type.lower()}"


# Use depth estimators from src.depth module
def get_depth_estimator(name: str):
    """Get depth estimator by name from src.depth module."""
    from src.depth import get_depth_estimator as _get_depth_estimator
    return _get_depth_estimator(name)


def list_depth_estimators() -> Dict[str, Dict]:
    """List available depth estimators from src.depth module."""
    from src.depth import list_depth_estimators as _list_depth_estimators
    return _list_depth_estimators()


def run_depth_benchmark(
    method: str,
    dataset_path: str,
    max_frames: Optional[int] = None,
    skip_frames: int = 1,
) -> DepthBenchmarkResult:
    """
    Run benchmark for a single depth estimator on a dataset.
    """
    from src.pipeline.frames import TumRGBDSource, SevenScenesSource, ReplicaSource
    from src.depth import list_depth_estimators
    
    dataset_name = Path(dataset_path).name
    logger.info(f"Benchmarking {method} on {dataset_name}...")
    
    # Check if this is a metric depth estimator
    estimator_info = list_depth_estimators().get(method, {})
    is_metric = estimator_info.get('metric', False)
    
    # Auto-detect dataset type and load appropriate source
    source = _get_dataset_source(dataset_path)
    frames = list(source)
    
    if max_frames:
        frames = frames[:max_frames]
    
    if skip_frames > 1:
        frames = frames[::skip_frames]
    
    if len(frames) == 0:
        raise ValueError(f"No frames found in {dataset_path}")
    
    # Clear GPU memory before initializing new estimator
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    except ImportError:
        pass
    
    # Initialize estimator
    estimator = get_depth_estimator(method)
    
    # Get the actual model name (handles fallbacks like V3->V2)
    actual_method = estimator.get_name() if hasattr(estimator, 'get_name') else method
    logger.info(f"  Depth estimator '{actual_method}' using device: {getattr(estimator, 'device', 'unknown')}")
    
    # Run estimation
    all_metrics = []
    raw_metrics = []  # For metric depth estimators
    latencies_ms = []
    
    t0 = time.time()
    
    for frame in frames:
        if frame.depth is None:
            continue
        
        # Estimate depth with timing
        frame_start = time.time()
        result = estimator.estimate(frame.rgb)
        frame_end = time.time()
        
        # Extract depth from result (may be DepthResult or ndarray)
        if hasattr(result, 'depth'):
            pred_depth = result.depth
        else:
            pred_depth = result
        
        latencies_ms.append((frame_end - frame_start) * 1000)
        
        # Compute aligned metrics (standard)
        metrics = compute_depth_metrics(pred_depth, frame.depth, align=True)
        all_metrics.append(metrics)
        
        # Compute edge accuracy
        edge_metrics = compute_edge_accuracy(pred_depth, frame.depth)
        metrics.update(edge_metrics)
        
        # Compute raw metrics (for metric depth estimators)
        if is_metric:
            raw = compute_depth_metrics(pred_depth, frame.depth, align=False)
            raw_metrics.append(raw)
    
    total_time = time.time() - t0
    fps = len(frames) / total_time if total_time > 0 else 0
    
    if not all_metrics:
        raise ValueError("No valid frames with depth found")
    
    # Average metrics
    avg_metrics = {
        k: np.nanmean([m[k] for m in all_metrics])
        for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'silog', 'delta1', 'delta2', 'delta3', 'scale', 'shift', 'edge_f1']
    }
    
    # Average raw metrics (for metric depth)
    raw_abs_rel, raw_rmse, raw_delta1 = 0.0, 0.0, 0.0
    if raw_metrics:
        raw_abs_rel = np.nanmean([m['abs_rel'] for m in raw_metrics])
        raw_rmse = np.nanmean([m['rmse'] for m in raw_metrics])
        raw_delta1 = np.nanmean([m['delta1'] for m in raw_metrics])
    
    # Compute latency statistics
    latencies_ms = np.array(latencies_ms)
    avg_latency_ms = float(np.mean(latencies_ms)) if len(latencies_ms) > 0 else 0.0
    min_latency_ms = float(np.min(latencies_ms)) if len(latencies_ms) > 0 else 0.0
    max_latency_ms = float(np.max(latencies_ms)) if len(latencies_ms) > 0 else 0.0
    p95_latency_ms = float(np.percentile(latencies_ms, 95)) if len(latencies_ms) > 0 else 0.0
    p99_latency_ms = float(np.percentile(latencies_ms, 99)) if len(latencies_ms) > 0 else 0.0
    
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
    
    return DepthBenchmarkResult(
        method=actual_method,  # Use actual model name (handles fallbacks)
        dataset=dataset_name,
        num_frames=len(frames),
        total_time=round(total_time, 2),
        fps=round(fps, 2),
        is_metric=is_metric,
        abs_rel=round(avg_metrics['abs_rel'], 4),
        sq_rel=round(avg_metrics['sq_rel'], 4),
        rmse=round(avg_metrics['rmse'], 4),
        rmse_log=round(avg_metrics['rmse_log'], 4),
        silog=round(avg_metrics['silog'], 4),
        delta1=round(avg_metrics['delta1'], 4),
        delta2=round(avg_metrics['delta2'], 4),
        delta3=round(avg_metrics['delta3'], 4),
        edge_f1=round(avg_metrics.get('edge_f1', 0.0), 4),
        raw_abs_rel=round(raw_abs_rel, 4),
        raw_rmse=round(raw_rmse, 4),
        raw_delta1=round(raw_delta1, 4),
        scale=round(avg_metrics['scale'], 4),
        shift=round(avg_metrics['shift'], 4),
        valid_pixels=int(np.mean([m['valid_pixels'] for m in all_metrics])),
        avg_latency_ms=round(avg_latency_ms, 2),
        min_latency_ms=round(min_latency_ms, 2),
        max_latency_ms=round(max_latency_ms, 2),
        p95_latency_ms=round(p95_latency_ms, 2),
        p99_latency_ms=round(p99_latency_ms, 2),
    )


def print_results_table(results: List[DepthBenchmarkResult]):
    """Print results in a nice table format."""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "=" * 130)
    print("DEPTH ESTIMATION BENCHMARK RESULTS")
    print("=" * 130)
    
    # Separate metric vs relative depth estimators
    metric_results = [r for r in results if r.is_metric]
    relative_results = [r for r in results if not r.is_metric]
    
    # Print relative depth results (aligned metrics only)
    if relative_results:
        print("\n📊 RELATIVE DEPTH ESTIMATORS (aligned to GT)")
        print("-" * 130)
        print(f"{'Method':<20} {'AbsRel':>8} {'SqRel':>8} {'RMSE':>8} {'SILog':>8} {'δ<1.25':>8} {'δ<1.25²':>8} {'Scale':>8} {'FPS':>6}")
        print("-" * 130)
        
        for r in sorted(relative_results, key=lambda x: x.abs_rel):
            print(f"{r.method:<20} {r.abs_rel:>8.4f} {r.sq_rel:>8.4f} {r.rmse:>8.4f} {r.silog:>8.4f} "
                  f"{r.delta1:>8.3f} {r.delta2:>8.3f} {r.scale:>8.2f} {r.fps:>6.1f}")
    
    # Print metric depth results (both raw and aligned)
    if metric_results:
        print("\n📏 METRIC DEPTH ESTIMATORS")
        print("-" * 130)
        print(f"{'Method':<20} {'AbsRel':>8} {'RMSE':>8} {'δ<1.25':>8} │ {'Raw AbsRel':>10} {'Raw RMSE':>10} {'Raw δ':>8} {'FPS':>6}")
        print(f"{'':20} {'(aligned)':>8} {'(aligned)':>8} {'(aligned)':>8} │ {'(no align)':>10} {'(no align)':>10} {'':>8} {'':>6}")
        print("-" * 130)
        
        for r in sorted(metric_results, key=lambda x: x.raw_abs_rel if x.raw_abs_rel > 0 else x.abs_rel):
            print(f"{r.method:<20} {r.abs_rel:>8.4f} {r.rmse:>8.4f} {r.delta1:>8.3f} │ "
                  f"{r.raw_abs_rel:>10.4f} {r.raw_rmse:>10.4f} {r.raw_delta1:>8.3f} {r.fps:>6.1f}")
    
    print("=" * 130)
