#!/usr/bin/env python3
"""
Depth Estimation Benchmark
==========================

Benchmark depth estimators against TUM RGB-D or other datasets with ground truth depth.
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


@dataclass
class DepthBenchmarkResult:
    """Result from benchmarking a depth estimator."""
    method: str
    dataset: str
    num_frames: int
    total_time: float
    fps: float
    
    # Standard depth metrics (lower is better except δ)
    abs_rel: float  # |d - d*| / d*
    sq_rel: float   # |d - d*|² / d*
    rmse: float     # √(mean((d - d*)²))
    rmse_log: float # √(mean((log(d) - log(d*))²))
    
    # Threshold accuracy (higher is better)
    delta1: float  # % with max(d/d*, d*/d) < 1.25
    delta2: float  # % with max(d/d*, d*/d) < 1.25²
    delta3: float  # % with max(d/d*, d*/d) < 1.25³
    
    # Additional stats
    scale: float = 1.0  # Scale factor used for alignment
    shift: float = 0.0  # Shift used for alignment
    valid_pixels: int = 0


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
    
    delta1 = np.mean(thresh < 1.25)
    delta2 = np.mean(thresh < 1.25 ** 2)
    delta3 = np.mean(thresh < 1.25 ** 3)
    
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
        'valid_pixels': int(np.sum(valid)),
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


# Registry of depth estimators
def get_depth_estimator(name: str) -> BaseDepthEstimator:
    """Get depth estimator by name."""
    name = name.lower()
    
    if name == 'midas' or name == 'midas_dpt_large':
        return MiDaSDepthEstimator('DPT_Large')
    elif name == 'midas_small':
        return MiDaSDepthEstimator('MiDaS_small')
    elif name == 'midas_hybrid':
        return MiDaSDepthEstimator('DPT_Hybrid')
    elif name in ['depth_anything', 'depth_anything_v2', 'depth_anything_vitl']:
        return DepthAnythingEstimator('vitl')
    elif name == 'depth_anything_vits':
        return DepthAnythingEstimator('vits')
    elif name == 'depth_anything_vitb':
        return DepthAnythingEstimator('vitb')
    elif name in ['zoedepth', 'zoedepth_nk']:
        return ZoeDepthEstimator('ZoeD_NK')
    elif name == 'zoedepth_k':
        return ZoeDepthEstimator('ZoeD_K')
    else:
        raise ValueError(f"Unknown depth estimator: {name}")


def list_depth_estimators() -> Dict[str, Dict]:
    """List available depth estimators."""
    return {
        'midas': {'description': 'MiDaS DPT-Large (relative depth)', 'metric': False},
        'midas_small': {'description': 'MiDaS Small (fast, relative)', 'metric': False},
        'depth_anything': {'description': 'Depth Anything V2 Large', 'metric': False},
        'depth_anything_vits': {'description': 'Depth Anything V2 Small (fast)', 'metric': False},
        'zoedepth': {'description': 'ZoeDepth NK (metric depth)', 'metric': True},
    }


def run_depth_benchmark(
    method: str,
    dataset_path: str,
    max_frames: Optional[int] = None,
    skip_frames: int = 1,
) -> DepthBenchmarkResult:
    """
    Run benchmark for a single depth estimator on a dataset.
    """
    from src.pipeline.frames import TumRGBDSource
    
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
    estimator = get_depth_estimator(method)
    
    # Run estimation
    all_metrics = []
    
    t0 = time.time()
    
    for frame in frames:
        if frame.depth is None:
            continue
        
        # Estimate depth
        pred_depth = estimator.estimate(frame.rgb)
        
        # Compute metrics
        metrics = compute_depth_metrics(pred_depth, frame.depth)
        all_metrics.append(metrics)
    
    total_time = time.time() - t0
    fps = len(frames) / total_time if total_time > 0 else 0
    
    if not all_metrics:
        raise ValueError("No valid frames with depth found")
    
    # Average metrics
    avg_metrics = {
        k: np.nanmean([m[k] for m in all_metrics])
        for k in ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'delta1', 'delta2', 'delta3', 'scale', 'shift']
    }
    
    return DepthBenchmarkResult(
        method=method,
        dataset=dataset_name,
        num_frames=len(frames),
        total_time=round(total_time, 2),
        fps=round(fps, 2),
        abs_rel=round(avg_metrics['abs_rel'], 4),
        sq_rel=round(avg_metrics['sq_rel'], 4),
        rmse=round(avg_metrics['rmse'], 4),
        rmse_log=round(avg_metrics['rmse_log'], 4),
        delta1=round(avg_metrics['delta1'], 4),
        delta2=round(avg_metrics['delta2'], 4),
        delta3=round(avg_metrics['delta3'], 4),
        scale=round(avg_metrics['scale'], 4),
        shift=round(avg_metrics['shift'], 4),
        valid_pixels=int(np.mean([m['valid_pixels'] for m in all_metrics])),
    )


def print_results_table(results: List[DepthBenchmarkResult]):
    """Print results in a nice table format."""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "=" * 110)
    print("DEPTH ESTIMATION BENCHMARK RESULTS")
    print("=" * 110)
    
    # Group by dataset
    by_dataset = {}
    for r in results:
        if r.dataset not in by_dataset:
            by_dataset[r.dataset] = []
        by_dataset[r.dataset].append(r)
    
    for dataset, dataset_results in by_dataset.items():
        print(f"\n📁 Dataset: {dataset}")
        print("-" * 110)
        print(f"{'Method':<20} {'AbsRel':>8} {'SqRel':>8} {'RMSE':>8} {'RMSElog':>8} {'δ<1.25':>8} {'δ<1.25²':>8} {'δ<1.25³':>8} {'FPS':>6}")
        print("-" * 110)
        
        # Sort by AbsRel
        dataset_results.sort(key=lambda x: x.abs_rel)
        
        for r in dataset_results:
            print(f"{r.method:<20} {r.abs_rel:>8.4f} {r.sq_rel:>8.4f} {r.rmse:>8.4f} {r.rmse_log:>8.4f} "
                  f"{r.delta1:>8.3f} {r.delta2:>8.3f} {r.delta3:>8.3f} {r.fps:>6.1f}")
    
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description='Benchmark depth estimators')
    parser.add_argument('--methods', nargs='+', default=None,
                        help='Depth methods to benchmark')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset name to use')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Root directory containing datasets')
    parser.add_argument('--max-frames', type=int, default=100,
                        help='Maximum frames to process per dataset')
    parser.add_argument('--skip-frames', type=int, default=5,
                        help='Process every Nth frame')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Find dataset root
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
    else:
        for candidate in [
            PROJECT_ROOT / "datasets",
            Path.home() / "datasets",
        ]:
            if candidate.exists():
                dataset_root = candidate
                break
        else:
            logger.error("Could not find datasets directory. Use --dataset-root")
            sys.exit(1)
    
    # Find TUM datasets
    from src.pipeline.frames import TumRGBDSource
    
    datasets = []
    for item in sorted(dataset_root.iterdir()):
        if item.is_dir() and (item / "rgb.txt").exists() and (item / "depth.txt").exists():
            datasets.append(item)
    
    # Also check tum subdirectory
    tum_dir = dataset_root / "tum"
    if tum_dir.exists():
        for item in sorted(tum_dir.iterdir()):
            if item.is_dir() and (item / "rgb.txt").exists() and (item / "depth.txt").exists():
                datasets.append(item)
    
    if args.dataset:
        datasets = [d for d in datasets if args.dataset in d.name]
    
    if not datasets:
        logger.error(f"No datasets with depth found in {dataset_root}")
        sys.exit(1)
    
    logger.info(f"Found {len(datasets)} dataset(s)")
    
    # Get methods to benchmark
    if args.methods:
        methods = args.methods
    else:
        methods = ['midas', 'midas_small']
    
    logger.info(f"Benchmarking methods: {methods}")
    
    # Run benchmarks
    results = []
    
    for dataset in datasets[:2]:  # Limit datasets for speed
        for method in methods:
            try:
                result = run_depth_benchmark(
                    method=method,
                    dataset_path=str(dataset),
                    max_frames=args.max_frames,
                    skip_frames=args.skip_frames,
                )
                results.append(result)
                logger.info(f"  {method}: AbsRel={result.abs_rel:.4f}, FPS={result.fps:.1f}")
            except Exception as e:
                logger.error(f"  {method}: FAILED - {e}")
    
    # Print results
    print_results_table(results)
    
    # Save to JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "benchmarks" / "results" / "depth_benchmark.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"Results saved to {output_path}")
    
    # Generate plots
    try:
        from benchmarks.visualization.plot_utils import plot_depth_metrics_bar
        plot_path = PROJECT_ROOT / "benchmarks" / "depth" / "plots" / "depth_comparison.png"
        plot_depth_metrics_bar([asdict(r) for r in results], output_path=plot_path)
        logger.info(f"Plot saved to {plot_path}")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")


if __name__ == '__main__':
    main()
