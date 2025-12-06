"""
Depth Estimation Benchmark Runner
=================================

Evaluates monocular depth estimation methods against TUM RGB-D ground truth.
Uses standard depth metrics: AbsRel, SqRel, RMSE, RMSElog, δ < 1.25^n
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Any

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


@dataclass
class DepthResult:
    """Result from depth estimation benchmark."""
    method: str
    dataset: str
    num_frames: int
    
    # Depth metrics (lower is better) - required fields first
    abs_rel: float
    sq_rel: float
    rmse: float
    rmse_log: float
    
    # Threshold accuracy (higher is better)
    delta1: float  # δ < 1.25
    delta2: float  # δ < 1.25²
    delta3: float  # δ < 1.25³
    
    # Model info - optional with defaults
    model_name: str = ""
    model_size: str = ""
    is_metric: bool = False
    
    # Timing
    total_time: float = 0.0
    fps: float = 0.0
    
    # Scale alignment
    scale: float = 1.0
    shift: float = 0.0
    valid_pixels: int = 0
    
    # Latency (ms)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
        pred: Predicted depth map (H, W)
        gt: Ground truth depth map (H, W)
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        align: Whether to scale-align predictions
        
    Returns:
        Dict with depth metrics
    """
    # Create valid mask
    valid = (gt > min_depth) & (gt < max_depth) & (pred > 0)
    
    if valid.sum() < 100:
        return {
            'abs_rel': np.nan, 'sq_rel': np.nan, 'rmse': np.nan, 'rmse_log': np.nan,
            'delta1': 0, 'delta2': 0, 'delta3': 0, 'scale': 1.0, 'shift': 0.0,
            'valid_pixels': int(valid.sum())
        }
    
    pred_valid = pred[valid]
    gt_valid = gt[valid]
    
    # Scale alignment (median scaling)
    scale = 1.0
    shift = 0.0
    if align:
        scale = np.median(gt_valid) / np.median(pred_valid)
        pred_valid = pred_valid * scale
    
    # Compute metrics
    thresh = np.maximum(gt_valid / pred_valid, pred_valid / gt_valid)
    
    abs_rel = np.mean(np.abs(gt_valid - pred_valid) / gt_valid)
    sq_rel = np.mean(((gt_valid - pred_valid) ** 2) / gt_valid)
    rmse = np.sqrt(np.mean((gt_valid - pred_valid) ** 2))
    
    # RMSE log (avoid log of 0)
    pred_log = np.log(np.clip(pred_valid, 1e-6, None))
    gt_log = np.log(np.clip(gt_valid, 1e-6, None))
    rmse_log = np.sqrt(np.mean((pred_log - gt_log) ** 2))
    
    delta1 = (thresh < 1.25).mean()
    delta2 = (thresh < 1.25 ** 2).mean()
    delta3 = (thresh < 1.25 ** 3).mean()
    
    return {
        'abs_rel': float(abs_rel),
        'sq_rel': float(sq_rel),
        'rmse': float(rmse),
        'rmse_log': float(rmse_log),
        'delta1': float(delta1),
        'delta2': float(delta2),
        'delta3': float(delta3),
        'scale': float(scale),
        'shift': float(shift),
        'valid_pixels': int(valid.sum()),
    }


class DepthBenchmark:
    """
    Benchmark runner for depth estimation methods.
    
    Example:
        benchmark = DepthBenchmark(
            methods=['midas', 'depth_anything'],
            dataset_root='datasets/tum'
        )
        results = benchmark.run(max_frames=50)
        benchmark.save_results('results/depth_benchmark.json')
    """
    
    def __init__(
        self,
        methods: List[str] = None,
        dataset_root: Path = None,
        datasets: List[str] = None,
    ):
        self.methods = methods or ['midas_small']
        self.dataset_root = Path(dataset_root) if dataset_root else PROJECT_ROOT / 'datasets'
        self.datasets = datasets
        self.results: List[DepthResult] = []
    
    def run(
        self,
        max_frames: int = 50,
        skip_frames: int = 5,
    ) -> List[DepthResult]:
        """Run benchmark on all methods and datasets.
        
        Optimized to load each dataset only once, then run all methods on it.
        """
        from src.pipeline.frames import TumRGBDSource
        from src.depth import get_depth_estimator
        
        datasets = self._find_datasets()
        if not datasets:
            logger.error(f"No TUM datasets found in {self.dataset_root}")
            return []
        
        logger.info(f"Running depth benchmark: {len(self.methods)} methods × {len(datasets)} datasets")
        
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
            
            # Filter frames with depth ground truth
            frames = [f for f in frames if f.depth is not None]
            
            if not frames:
                logger.warning(f"  No frames with depth ground truth, skipping")
                continue
            
            logger.info(f"  Loaded {len(frames)} frames with depth GT")
            
            # Run all methods on this dataset
            for method in self.methods:
                try:
                    result = self._run_single_with_frames(method, dataset_path.name, frames)
                    self.results.append(result)
                    logger.info(f"  {method}: AbsRel={result.abs_rel:.4f}, δ1={result.delta1:.3f}")
                except Exception as e:
                    logger.error(f"  {method}: FAILED - {e}")
                finally:
                    # Clear GPU memory between methods to avoid OOM
                    self._clear_gpu_memory()
        
        return self.results
    
    def _clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM errors."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
        
        import gc
        gc.collect()
    
    def _run_single_with_frames(
        self,
        method: str,
        dataset_name: str,
        frames: List,
    ) -> DepthResult:
        """Run benchmark for single method on pre-loaded frames."""
        from src.depth import get_depth_estimator
        
        estimator = get_depth_estimator(method)
        
        # Get model info
        model_name = estimator.get_name() if hasattr(estimator, 'get_name') else method
        model_size = ""
        if hasattr(estimator, 'model_size'):
            model_size = estimator.model_size
        elif hasattr(estimator, 'model_type'):
            model_size = estimator.model_type
        is_metric = estimator.is_metric() if hasattr(estimator, 'is_metric') else False
        
        all_metrics = []
        latencies_ms = []
        failed_frames = 0
        
        t0 = time.time()
        
        for frame in frames:
            frame_start = time.time()
            try:
                result = estimator.estimate(frame.rgb)
                latencies_ms.append((time.time() - frame_start) * 1000)
                
                # Extract depth array from DepthResult
                pred_depth = result.depth if hasattr(result, 'depth') else result
                
                # Validate prediction
                if pred_depth is None or not np.any(np.isfinite(pred_depth)):
                    failed_frames += 1
                    continue
                
                metrics = compute_depth_metrics(pred_depth, frame.depth)
                
                # Skip obviously bad results (scale issues)
                if metrics['abs_rel'] > 100:
                    logger.warning(f"    Skipping frame with extreme AbsRel: {metrics['abs_rel']:.2f}")
                    failed_frames += 1
                    continue
                    
                all_metrics.append(metrics)
                
            except Exception as e:
                logger.error(f"  {method} estimation failed: {e}")
                failed_frames += 1
                continue
        
        total_time = time.time() - t0
        
        # Handle case where all frames failed
        if not all_metrics:
            logger.warning(f"  All frames failed for {method}")
            return DepthResult(
                method=method,
                dataset=dataset_name,
                num_frames=len(frames),
                abs_rel=float('inf'),
                sq_rel=float('inf'),
                rmse=float('inf'),
                rmse_log=float('inf'),
                delta1=0.0,
                delta2=0.0,
                delta3=0.0,
                model_name=model_name,
                model_size=model_size,
                is_metric=is_metric,
                total_time=round(total_time, 2),
                fps=0.0,
            )
        
        latencies_ms = np.array(latencies_ms) if latencies_ms else np.array([0.0])
        
        # Average metrics
        avg = {k: np.nanmean([m[k] for m in all_metrics]) 
               for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'delta1', 'delta2', 'delta3', 'scale', 'shift']}
        
        return DepthResult(
            method=method,
            dataset=dataset_name,
            num_frames=len(frames),
            # Required metrics first
            abs_rel=round(avg['abs_rel'], 4),
            sq_rel=round(avg['sq_rel'], 4),
            rmse=round(avg['rmse'], 4),
            rmse_log=round(avg['rmse_log'], 4),
            delta1=round(avg['delta1'], 4),
            delta2=round(avg['delta2'], 4),
            delta3=round(avg['delta3'], 4),
            # Optional with defaults
            model_name=model_name,
            model_size=model_size,
            is_metric=is_metric,
            total_time=round(total_time, 2),
            fps=round(len(frames) / total_time, 2),
            scale=round(avg['scale'], 4),
            shift=round(avg['shift'], 4),
            valid_pixels=int(np.mean([m['valid_pixels'] for m in all_metrics])),
            avg_latency_ms=round(np.mean(latencies_ms), 2),
            min_latency_ms=round(np.min(latencies_ms), 2),
            max_latency_ms=round(np.max(latencies_ms), 2),
            p95_latency_ms=round(np.percentile(latencies_ms, 95), 2),
            p99_latency_ms=round(np.percentile(latencies_ms, 99), 2),
        )
    
    def _find_datasets(self) -> List[Path]:
        """Find TUM datasets with depth ground truth."""
        if self.datasets:
            return [self.dataset_root / d for d in self.datasets if (self.dataset_root / d).exists()]
        
        datasets = []
        search_paths = [self.dataset_root, self.dataset_root / 'tum']
        
        for search in search_paths:
            if not search.exists():
                continue
            for item in sorted(search.iterdir()):
                if item.is_dir() and (item / 'depth.txt').exists():
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
        print("DEPTH ESTIMATION BENCHMARK RESULTS")
        print("=" * 100)
        
        by_dataset = {}
        for r in self.results:
            by_dataset.setdefault(r.dataset, []).append(r)
        
        for dataset, results in by_dataset.items():
            print(f"\n📁 {dataset}")
            print("-" * 100)
            print(f"{'Method':<20} {'AbsRel':>8} {'RMSE':>8} {'δ<1.25':>8} {'FPS':>8} {'Latency':>10}")
            print("-" * 100)
            
            for r in sorted(results, key=lambda x: x.abs_rel):
                print(f"{r.method:<20} {r.abs_rel:>8.4f} {r.rmse:>8.4f} {r.delta1:>8.3f} "
                      f"{r.fps:>8.1f} {r.avg_latency_ms:>9.1f}ms")
        
        print("=" * 100)
