"""
Gaussian Splatting Benchmark Runner
===================================

Evaluates 3D Gaussian Splatting engines for rendering quality and performance.
Uses PSNR, SSIM, LPIPS metrics from src/evaluation/metrics.py
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

from src.evaluation.metrics import compute_image_metrics

logger = logging.getLogger(__name__)


@dataclass
class GSResult:
    """Result from Gaussian splatting benchmark."""
    engine: str
    dataset: str
    num_frames: int
    
    # Engine info
    engine_type: str = ""  # "optimization", "feed-forward", "slam"
    model_name: str = ""   # e.g., "DA3-LARGE", "graphdeco"
    
    # Timing
    total_time: float = 0.0
    train_time: float = 0.0
    avg_frame_time: float = 0.0
    fps: float = 0.0
    
    # Quality metrics
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0
    
    # Efficiency
    final_gaussians: int = 0
    peak_memory_mb: float = 0.0
    
    # Latency
    avg_latency_ms: float = 0.0
    render_fps: float = 0.0
    
    # Training history (for plots)
    loss_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GSBenchmark:
    """
    Benchmark runner for Gaussian splatting engines.
    
    Example:
        benchmark = GSBenchmark(
            engines=['graphdeco', 'gsplat'],
            dataset_root='datasets/tum'
        )
        results = benchmark.run(max_frames=50, iterations=30)
        benchmark.save_results('results/gs_benchmark.json')
    """
    
    def __init__(
        self,
        engines: List[str] = None,
        dataset_root: Path = None,
        datasets: List[str] = None,
    ):
        self.engines = engines or ['graphdeco']
        self.dataset_root = Path(dataset_root) if dataset_root else PROJECT_ROOT / 'datasets'
        self.datasets = datasets
        self.results: List[GSResult] = []
    
    def run(
        self,
        max_frames: int = 50,
        iterations: int = 30,
        eval_frames: int = 5,
    ) -> List[GSResult]:
        """Run benchmark on all engines and datasets.
        
        Optimized to load each dataset only once, then run all engines on it.
        """
        from src.engines import get_engine
        from src.pipeline.frames import TumRGBDSource
        
        datasets = self._find_datasets()
        if not datasets:
            logger.error(f"No datasets found in {self.dataset_root}")
            return []
        
        logger.info(f"Running GS benchmark: {len(self.engines)} engines × {len(datasets)} datasets")
        
        self.results = []
        
        for dataset_path in datasets:
            # Load dataset ONCE per dataset
            logger.info(f"Loading dataset: {dataset_path.name}")
            source = TumRGBDSource(str(dataset_path))
            frames = list(source)
            
            if max_frames:
                frames = frames[:max_frames]
            
            logger.info(f"  Loaded {len(frames)} frames")
            
            # Run all engines on this dataset
            for engine_name in self.engines:
                try:
                    result = self._run_single_with_frames(engine_name, dataset_path.name, frames, iterations, eval_frames)
                    self.results.append(result)
                    logger.info(f"  {engine_name}: PSNR={result.psnr:.2f}dB, Gaussians={result.final_gaussians:,}")
                except Exception as e:
                    logger.error(f"  {engine_name}: FAILED - {e}")
                    import traceback
                    traceback.print_exc()
        
        return self.results
    
    def _run_single_with_frames(
        self,
        engine_name: str,
        dataset_name: str,
        frames: List,
        iterations: int,
        eval_frames: int,
    ) -> GSResult:
        """Run benchmark for single engine on pre-loaded frames."""
        from src.engines import get_engine
        
        # Track GPU memory
        try:
            import torch
            torch.cuda.reset_peak_memory_stats()
        except:
            pass
        
        # Initialize engine
        engine = get_engine(engine_name)
        engine.initialize_scene(frames[0].intrinsics, {'num_frames': len(frames)})
        
        # Determine engine type and model name
        engine_type = "optimization"  # default
        model_name = engine_name
        if engine_name == 'da3gs':
            engine_type = "feed-forward"
            if hasattr(engine, 'model_name'):
                model_name = engine.model_name
        elif engine_name in ('monogs', 'splatam', 'gslam', 'photoslam'):
            engine_type = "slam"
        if hasattr(engine, 'config') and hasattr(engine.config, 'model_name'):
            model_name = engine.config.model_name
        
        loss_history = []
        t0 = time.time()
        train_time = 0
        
        # Process frames
        for i, frame in enumerate(frames):
            engine.add_frame(
                frame_id=frame.idx,
                rgb=frame.rgb,
                depth=frame.depth,
                pose_world_cam=frame.pose
            )
            
            # Optimize
            t_opt = time.time()
            for _ in range(iterations):
                try:
                    metrics = engine.optimize_step(n_steps=1)
                    if metrics and 'loss' in metrics:
                        loss_history.append(metrics['loss'])
                except:
                    pass
            train_time += time.time() - t_opt
            
            if (i + 1) % 10 == 0:
                logger.info(f"    Frame {i+1}/{len(frames)}, Gaussians: {engine.get_num_gaussians():,}")
        
        total_time = time.time() - t0
        
        # Evaluate rendering quality
        test_frames = frames[::max(1, len(frames) // eval_frames)][:eval_frames]
        psnr_scores, ssim_scores = [], []
        render_times = []
        
        # Get image size from first frame
        image_size = (frames[0].rgb.shape[1], frames[0].rgb.shape[0])  # (width, height)
        
        for frame in test_frames:
            try:
                t_render = time.time()
                rendered = engine.render_view(frame.pose, image_size)
                render_times.append(time.time() - t_render)
                
                if rendered is not None:
                    gt = frame.rgb
                    if rendered.shape != gt.shape:
                        import cv2
                        rendered = cv2.resize(rendered, (gt.shape[1], gt.shape[0]))
                    
                    metrics = compute_image_metrics(rendered, gt, compute_lpips_metric=False)
                    psnr_scores.append(metrics['psnr'])
                    ssim_scores.append(metrics['ssim'])
            except Exception as e:
                logger.warning(f"Render failed: {e}")
        
        # Get memory
        peak_memory = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        except:
            pass
        
        render_fps = 0.0
        if render_times:
            render_fps = 1.0 / np.mean(render_times)
        
        return GSResult(
            engine=engine_name,
            dataset=dataset_name,
            num_frames=len(frames),
            engine_type=engine_type,
            model_name=model_name,
            total_time=round(total_time, 2),
            train_time=round(train_time, 2),
            avg_frame_time=round(total_time / len(frames), 3),
            fps=round(len(frames) / total_time, 2),
            psnr=round(np.mean(psnr_scores) if psnr_scores else 0, 2),
            ssim=round(np.mean(ssim_scores) if ssim_scores else 0, 4),
            lpips=0.0,  # Expensive to compute
            final_gaussians=engine.get_num_gaussians(),
            peak_memory_mb=round(peak_memory, 1),
            avg_latency_ms=round(total_time / len(frames) * 1000, 2),
            render_fps=round(render_fps, 2),
            loss_history=loss_history[-100:] if loss_history else [],
        )
    
    def _find_datasets(self) -> List[Path]:
        """Find datasets for GS benchmark."""
        if self.datasets:
            return [self.dataset_root / d for d in self.datasets if (self.dataset_root / d).exists()]
        
        datasets = []
        search_paths = [self.dataset_root, self.dataset_root / 'tum']
        
        for search in search_paths:
            if not search.exists():
                continue
            for item in sorted(search.iterdir()):
                if item.is_dir() and (item / 'rgb.txt').exists():
                    datasets.append(item)
        
        return datasets[:1]  # Just first dataset by default
    
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
        print("GAUSSIAN SPLATTING BENCHMARK RESULTS")
        print("=" * 100)
        
        print(f"\n{'Engine':<12} {'Dataset':<35} {'PSNR':>8} {'SSIM':>8} {'Gaussians':>12} {'FPS':>8} {'Memory':>10}")
        print("-" * 100)
        
        for r in sorted(self.results, key=lambda x: -x.psnr):
            print(f"{r.engine:<12} {r.dataset:<35} {r.psnr:>8.2f} {r.ssim:>8.4f} "
                  f"{r.final_gaussians:>12,} {r.fps:>8.2f} {r.peak_memory_mb:>9.0f}MB")
        
        print("=" * 100)
