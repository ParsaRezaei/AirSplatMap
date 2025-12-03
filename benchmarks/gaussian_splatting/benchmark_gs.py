#!/usr/bin/env python3
"""
Gaussian Splatting Benchmark
============================

Benchmark 3DGS engines on RGB-D datasets.
Computes quality metrics (PSNR, SSIM, LPIPS) and efficiency metrics.

Usage:
    python benchmarks/gaussian_splatting/benchmark_gs.py                    # Run all engines
    python benchmarks/gaussian_splatting/benchmark_gs.py --engines graphdeco gsplat
    python benchmarks/gaussian_splatting/benchmark_gs.py --dataset fr1_desk
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class GSBenchmarkResult:
    """Result from benchmarking a Gaussian splatting engine."""
    engine: str
    dataset: str
    num_frames: int
    
    # Timing
    total_time: float  # Total processing time
    train_time: float  # Time spent in optimization
    avg_frame_time: float  # Average time per frame
    fps: float  # Processing FPS
    
    # Quality metrics
    psnr: float  # Peak Signal-to-Noise Ratio (dB)
    ssim: float  # Structural Similarity Index
    lpips: float  # Learned Perceptual Image Patch Similarity (lower is better)
    
    # Efficiency metrics
    final_gaussians: int  # Number of Gaussians at end
    peak_memory_mb: float  # Peak GPU memory usage
    
    # Training history (for plotting)
    loss_history: List[float] = field(default_factory=list)
    psnr_history: List[float] = field(default_factory=list)
    gaussian_history: List[int] = field(default_factory=list)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images."""
    try:
        from skimage.metrics import structural_similarity as ssim
        # Convert to grayscale if needed
        if img1.ndim == 3:
            import cv2
            img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        else:
            img1_gray = img1
            img2_gray = img2
        return ssim(img1_gray, img2_gray)
    except ImportError:
        logger.warning("scikit-image not available for SSIM, using simplified version")
        # Simplified SSIM approximation
        mu1 = np.mean(img1)
        mu2 = np.mean(img2)
        sigma1 = np.std(img1)
        sigma2 = np.std(img2)
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
        
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                   ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
        return float(ssim_val)


def compute_lpips(img1: np.ndarray, img2: np.ndarray, lpips_model=None) -> float:
    """Compute LPIPS between two images."""
    try:
        import torch
        import lpips
        
        if lpips_model is None:
            lpips_model = lpips.LPIPS(net='alex').cuda()
        
        # Convert to tensor
        def to_tensor(img):
            t = torch.from_numpy(img).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            t = t * 2 - 1  # [0, 1] -> [-1, 1]
            return t.cuda()
        
        t1 = to_tensor(img1)
        t2 = to_tensor(img2)
        
        with torch.no_grad():
            score = lpips_model(t1, t2)
        
        return float(score.item())
    except ImportError:
        return 0.0  # LPIPS not available


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except:
        pass
    return 0.0


def run_gs_benchmark(
    engine_name: str,
    dataset_path: str,
    max_frames: Optional[int] = None,
    n_iterations: int = 100,
    eval_interval: int = 20,
) -> GSBenchmarkResult:
    """
    Run benchmark for a single 3DGS engine on a dataset.
    
    Args:
        engine_name: Engine name ('graphdeco', 'gsplat', 'monogs', 'splatam')
        dataset_path: Path to TUM dataset
        max_frames: Maximum frames to process
        n_iterations: Number of optimization iterations per frame
        eval_interval: Evaluate quality every N iterations
        
    Returns:
        GSBenchmarkResult
    """
    from src.pipeline.frames import TumRGBDSource
    from src.engines import get_engine
    
    dataset_name = Path(dataset_path).name
    logger.info(f"Benchmarking {engine_name} on {dataset_name}...")
    
    # Load dataset
    source = TumRGBDSource(dataset_path)
    frames = list(source)
    
    if max_frames:
        frames = frames[:max_frames]
    
    if len(frames) == 0:
        raise ValueError(f"No frames found in {dataset_path}")
    
    # Get intrinsics from first frame
    intrinsics = frames[0].intrinsics
    
    # Initialize engine
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
    except:
        pass
    
    engine = get_engine(engine_name)
    engine.initialize_scene(intrinsics)
    
    # Training history
    loss_history = []
    psnr_history = []
    gaussian_history = []
    
    # Render frames for quality evaluation
    eval_frames = frames[::max(1, len(frames) // 5)]  # Sample 5 frames for eval
    
    t0 = time.time()
    train_time = 0
    
    # Process frames
    for i, frame in enumerate(frames):
        # Add frame
        engine.add_frame(
            frame_id=frame.idx,
            rgb=frame.rgb,
            depth=frame.depth,
            pose_world_cam=frame.pose
        )
        
        # Optimize
        t_opt_start = time.time()
        for _ in range(n_iterations):
            try:
                metrics = engine.optimize_step(n_steps=1)
                if metrics and 'loss' in metrics:
                    loss_history.append(metrics['loss'])
            except:
                try:
                    metrics = engine.optimize_step()
                except:
                    pass
        train_time += time.time() - t_opt_start
        
        # Record Gaussian count
        gaussian_history.append(engine.get_num_gaussians())
        
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(f"  Frame {i+1}/{len(frames)}, Gaussians: {engine.get_num_gaussians():,}")
    
    total_time = time.time() - t0
    
    # Evaluate quality on held-out frames
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    lpips_model = None
    try:
        import lpips
        lpips_model = lpips.LPIPS(net='alex').cuda()
    except:
        pass
    
    for frame in eval_frames:
        try:
            # Render from this viewpoint
            rendered = engine.render(frame.pose)
            
            if rendered is not None:
                # Ensure same size
                gt = frame.rgb
                if rendered.shape != gt.shape:
                    import cv2
                    rendered = cv2.resize(rendered, (gt.shape[1], gt.shape[0]))
                
                psnr_scores.append(compute_psnr(gt, rendered))
                ssim_scores.append(compute_ssim(gt, rendered))
                
                if lpips_model is not None:
                    lpips_scores.append(compute_lpips(gt, rendered, lpips_model))
        except Exception as e:
            logger.warning(f"Render failed: {e}")
    
    # Compute averages
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0.0
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_lpips = np.mean(lpips_scores) if lpips_scores else 0.0
    
    # Get memory
    peak_memory = get_gpu_memory_mb()
    
    return GSBenchmarkResult(
        engine=engine_name,
        dataset=dataset_name,
        num_frames=len(frames),
        total_time=round(total_time, 2),
        train_time=round(train_time, 2),
        avg_frame_time=round(total_time / len(frames), 3),
        fps=round(len(frames) / total_time, 2),
        psnr=round(avg_psnr, 2),
        ssim=round(avg_ssim, 4),
        lpips=round(avg_lpips, 4),
        final_gaussians=engine.get_num_gaussians(),
        peak_memory_mb=round(peak_memory, 1),
        loss_history=loss_history[-100:] if loss_history else [],  # Keep last 100
        psnr_history=psnr_history,
        gaussian_history=gaussian_history,
    )


def print_results_table(results: List[GSBenchmarkResult]):
    """Print results in a nice table format."""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "=" * 100)
    print("GAUSSIAN SPLATTING BENCHMARK RESULTS")
    print("=" * 100)
    
    # Group by dataset
    by_dataset = {}
    for r in results:
        if r.dataset not in by_dataset:
            by_dataset[r.dataset] = []
        by_dataset[r.dataset].append(r)
    
    for dataset, dataset_results in by_dataset.items():
        print(f"\n📁 Dataset: {dataset}")
        print("-" * 100)
        print(f"{'Engine':<12} {'Frames':>6} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'Gaussians':>12} {'Time':>8} {'FPS':>6} {'Memory':>8}")
        print("-" * 100)
        
        # Sort by PSNR
        dataset_results.sort(key=lambda x: -x.psnr)
        
        for r in dataset_results:
            print(f"{r.engine:<12} {r.num_frames:>6} {r.psnr:>8.2f} {r.ssim:>8.4f} {r.lpips:>8.4f} "
                  f"{r.final_gaussians:>12,} {r.total_time:>7.1f}s {r.fps:>6.2f} {r.peak_memory_mb:>7.0f}MB")
    
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description='Benchmark Gaussian splatting engines')
    parser.add_argument('--engines', nargs='+', default=['graphdeco'],
                        help='Engines to benchmark (graphdeco, gsplat, monogs, splatam)')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Specific dataset name to use')
    parser.add_argument('--dataset-root', type=str, default=None,
                        help='Root directory containing datasets')
    parser.add_argument('--max-frames', type=int, default=50,
                        help='Maximum frames to process per dataset')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Optimization iterations per frame')
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
    datasets = []
    for item in sorted(dataset_root.iterdir()):
        if item.is_dir() and (item / "rgb.txt").exists():
            datasets.append(item)
    
    # Also check tum subdirectory
    tum_dir = dataset_root / "tum"
    if tum_dir.exists():
        for item in sorted(tum_dir.iterdir()):
            if item.is_dir() and (item / "rgb.txt").exists():
                datasets.append(item)
    
    if args.dataset:
        datasets = [d for d in datasets if args.dataset in d.name]
    
    if not datasets:
        logger.error(f"No datasets found in {dataset_root}")
        sys.exit(1)
    
    logger.info(f"Found {len(datasets)} dataset(s)")
    logger.info(f"Benchmarking engines: {args.engines}")
    
    # Run benchmarks
    results = []
    
    for dataset in datasets[:1]:  # Limit to 1 dataset for speed
        for engine in args.engines:
            try:
                result = run_gs_benchmark(
                    engine_name=engine,
                    dataset_path=str(dataset),
                    max_frames=args.max_frames,
                    n_iterations=args.iterations,
                )
                results.append(result)
                logger.info(f"  {engine}: PSNR={result.psnr:.2f}dB, Gaussians={result.final_gaussians:,}")
            except Exception as e:
                logger.error(f"  {engine}: FAILED - {e}")
                import traceback
                traceback.print_exc()
    
    # Print results
    print_results_table(results)
    
    # Save to JSON
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = PROJECT_ROOT / "benchmarks" / "results" / "gs_benchmark.json"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert dataclass to dict, handling lists
    results_dict = []
    for r in results:
        d = asdict(r)
        results_dict.append(d)
    
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    logger.info(f"Results saved to {output_path}")
    
    # Generate plots
    try:
        from benchmarks.visualization.plot_utils import plot_gs_metrics_bar, plot_gs_training_curves
        
        # Metrics bar chart
        plot_path = PROJECT_ROOT / "benchmarks" / "gaussian_splatting" / "plots" / "gs_comparison.png"
        plot_gs_metrics_bar(results_dict, output_path=plot_path)
        logger.info(f"Metrics plot saved to {plot_path}")
        
        # Training curves
        if any(r.loss_history for r in results):
            training_data = {
                r.engine: {
                    'loss': r.loss_history,
                    'gaussians': r.gaussian_history,
                }
                for r in results if r.loss_history
            }
            curves_path = PROJECT_ROOT / "benchmarks" / "gaussian_splatting" / "plots" / "training_curves.png"
            plot_gs_training_curves(training_data, output_path=curves_path)
            logger.info(f"Training curves saved to {curves_path}")
    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")


if __name__ == '__main__':
    main()
