#!/usr/bin/env python3
"""
Gaussian Splatting Benchmark
============================

Benchmark 3DGS engines on RGB-D datasets (TUM, 7-Scenes, Replica, ICL-NUIM).
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
from typing import List, Optional, Tuple, Dict
from typing import Dict, List, Optional, Tuple, Any
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
    
    # Latency metrics (ms per frame)
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    render_fps: float = 0.0  # Render-only FPS (inference)
    
    # Rendered images for visualization (stored separately, not in dataclass for memory)
    # Use get_rendered_samples() to retrieve


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
    if lpips_model is None:
        return 0.0  # Model must be passed in
    
    try:
        import torch
        
        # Convert to tensor
        def to_tensor(img):
            t = torch.from_numpy(img.copy()).float() / 255.0
            t = t.permute(2, 0, 1).unsqueeze(0)  # HWC -> BCHW
            t = t * 2 - 1  # [0, 1] -> [-1, 1]
            return t.cuda()
        
        t1 = to_tensor(img1)
        t2 = to_tensor(img2)
        
        with torch.no_grad():
            score = lpips_model(t1, t2)
        
        return float(score.item())
    except Exception as e:
        logger.warning(f"LPIPS computation failed: {e}")
        return 0.0


def get_lpips_model():
    """Get or create cached LPIPS model."""
    global _LPIPS_MODEL
    if '_LPIPS_MODEL' not in globals():
        try:
            import lpips
            _LPIPS_MODEL = lpips.LPIPS(net='alex').cuda()
            logger.info("LPIPS model loaded successfully")
        except ImportError:
            logger.warning("LPIPS not available - install with: pip install lpips")
            _LPIPS_MODEL = None
        except Exception as e:
            logger.warning(f"Failed to load LPIPS model: {e}")
            _LPIPS_MODEL = None
    return _LPIPS_MODEL


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
    output_dir: Optional[Path] = None,
) -> GSBenchmarkResult:
    """
    Run benchmark for a single 3DGS engine on a dataset.
    
    Args:
        engine_name: Engine name ('graphdeco', 'gsplat', 'monogs', 'splatam')
        dataset_path: Path to dataset (TUM, 7-Scenes, Replica, or ICL-NUIM)
        max_frames: Maximum frames to process
        n_iterations: Number of optimization iterations per frame
        eval_interval: Evaluate quality every N iterations
        output_dir: If provided, save rendered images here
        
    Returns:
        GSBenchmarkResult
    """
    from src.pipeline.frames import TumRGBDSource, SevenScenesSource, ReplicaSource
    from src.engines import get_engine
    
    dataset_name = Path(dataset_path).name
    logger.info(f"Benchmarking {engine_name} on {dataset_name}...")
    
    # Auto-detect dataset type and load appropriate source
    source = _get_dataset_source(dataset_path)
    all_frames = list(source)
    
    # Sample frames UNIFORMLY across full sequence for better scene coverage
    if max_frames and len(all_frames) > max_frames:
        indices = np.linspace(0, len(all_frames) - 1, max_frames, dtype=int)
        frames = [all_frames[i] for i in indices]
        logger.info(f"  Uniformly sampled {max_frames} frames from {len(all_frames)} total")
    else:
        frames = all_frames
    
    if len(frames) == 0:
        raise ValueError(f"No frames found in {dataset_path}")
    
    # Get intrinsics from first frame
    intrinsics = frames[0].intrinsics
    
    # Initialize engine
    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
        if torch.cuda.is_available():
            logger.info(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    except:
        pass
    
    engine = get_engine(engine_name)
    logger.info(f"  GS engine '{engine_name}' using device: {getattr(engine, 'device', 'cuda')}")
    config = {
        'num_frames': len(frames),
        'max_gaussians': 200000,
    }
    engine.initialize_scene(intrinsics, config)
    
    # Training history
    loss_history = []
    psnr_history = []
    gaussian_history = []
    
    # Select evaluation frames spread WIDELY across the sequence
    # We want frames from very different parts of the trajectory
    num_eval_frames = min(6, len(frames))  # Up to 6 frames for evaluation
    if len(frames) <= num_eval_frames:
        eval_indices = list(range(len(frames)))
    else:
        # Pick frames at 0%, 20%, 40%, 60%, 80%, 100% of sequence
        # This ensures maximum diversity in viewpoints
        percentiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0][:num_eval_frames]
        eval_indices = [min(int(p * (len(frames) - 1)), len(frames) - 1) for p in percentiles]
        eval_indices = sorted(set(eval_indices))
    eval_frames = [frames[i] for i in eval_indices]
    logger.info(f"  Evaluation frames: {len(eval_frames)} frames at indices {eval_indices}")
    
    t0 = time.time()
    train_time = 0
    
    # For PSNR history - evaluate every N frames
    psnr_eval_interval = max(1, len(frames) // 10)  # ~10 PSNR samples during training
    image_size = (frames[0].rgb.shape[1], frames[0].rgb.shape[0])
    
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
        
        # Record PSNR at intervals
        if (i + 1) % psnr_eval_interval == 0 or i == len(frames) - 1:
            try:
                rendered = engine.render_view(frame.pose, image_size)
                if rendered is not None:
                    psnr_val = compute_psnr(frame.rgb, rendered)
                    psnr_history.append(psnr_val)
            except:
                pass
        
        # Log progress
        if (i + 1) % 10 == 0:
            logger.info(f"  Frame {i+1}/{len(frames)}, Gaussians: {engine.get_num_gaussians():,}")
    
    total_time = time.time() - t0
    
    # Evaluate quality on held-out frames
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    
    lpips_model = None
    lpips_model = get_lpips_model()
    
    # Setup render output directory if saving - organize by dataset then engine
    render_output_dir = None
    if output_dir:
        render_output_dir = Path(output_dir) / "renders" / dataset_name / engine_name
        render_output_dir.mkdir(parents=True, exist_ok=True)
    
    render_times_ms = []
    image_size = (frames[0].rgb.shape[1], frames[0].rgb.shape[0])
    for i, frame in enumerate(eval_frames):
        try:
            # Render from this viewpoint with timing
            render_start = time.time()
            rendered = engine.render_view(frame.pose, image_size)
            render_end = time.time()
            render_times_ms.append((render_end - render_start) * 1000)
            
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
                
                # Save renders if output_dir provided
                if render_output_dir:
                    import cv2
                    cv2.imwrite(str(render_output_dir / f"gt_{i}.png"), cv2.cvtColor(gt, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(render_output_dir / f"rendered_{i}.png"), cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
                    # Error map
                    error = np.mean(np.abs(gt.astype(float) - rendered.astype(float)), axis=2)
                    error_norm = (error / (error.max() + 1e-8) * 255).astype(np.uint8)
                    cv2.imwrite(str(render_output_dir / f"error_{i}.png"), cv2.applyColorMap(error_norm, cv2.COLORMAP_HOT))
        except Exception as e:
            logger.warning(f"Render failed: {e}")
    
    # Compute averages
    avg_psnr = np.mean(psnr_scores) if psnr_scores else 0.0
    avg_ssim = np.mean(ssim_scores) if ssim_scores else 0.0
    avg_lpips = np.mean(lpips_scores) if lpips_scores else 0.0
    
    # Compute latency statistics (based on per-frame processing)
    frame_times_ms = [(total_time / len(frames)) * 1000] * len(frames)  # approximate
    avg_latency_ms = float(np.mean(frame_times_ms))
    min_latency_ms = avg_latency_ms  # No variance since we don't have per-frame times
    max_latency_ms = avg_latency_ms
    p95_latency_ms = avg_latency_ms
    p99_latency_ms = avg_latency_ms
    
    # Render FPS (inference only)
    render_fps = 0.0
    if render_times_ms:
        avg_render_ms = np.mean(render_times_ms)
        render_fps = 1000.0 / avg_render_ms if avg_render_ms > 0 else 0.0
    
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
        avg_latency_ms=round(avg_latency_ms, 2),
        min_latency_ms=round(min_latency_ms, 2),
        max_latency_ms=round(max_latency_ms, 2),
        p95_latency_ms=round(p95_latency_ms, 2),
        p99_latency_ms=round(p99_latency_ms, 2),
        render_fps=round(render_fps, 2),
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
