"""
Unified benchmarking system for comparing 3DGS SLAM engines.

Provides:
- BenchmarkRunner: Run multiple engines on multiple datasets
- BenchmarkResult: Store and compare results
- Report generation: Markdown tables, plots, comparisons
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    """Results from a single engine run on a single dataset."""
    engine_name: str
    dataset_name: str
    
    # Timing
    total_time_sec: float = 0.0
    fps: float = 0.0
    num_frames: int = 0
    
    # Model stats
    num_gaussians: int = 0
    model_size_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Rendering metrics
    psnr: float = 0.0
    ssim: float = 0.0
    lpips: float = 0.0
    
    # Trajectory metrics
    ate_rmse: float = 0.0
    ate_mean: float = 0.0
    rpe_trans_rmse: float = 0.0
    rpe_rot_rmse: float = 0.0
    
    # Additional info
    config: Dict[str, Any] = field(default_factory=dict)
    output_path: str = ""
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EngineResult':
        return cls(**data)


@dataclass
class BenchmarkResult:
    """Collection of results from benchmark runs."""
    name: str
    description: str = ""
    timestamp: str = ""
    results: List[EngineResult] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def add_result(self, result: EngineResult) -> None:
        self.results.append(result)
    
    def get_by_engine(self, engine_name: str) -> List[EngineResult]:
        return [r for r in self.results if r.engine_name == engine_name]
    
    def get_by_dataset(self, dataset_name: str) -> List[EngineResult]:
        return [r for r in self.results if r.dataset_name == dataset_name]
    
    def get_engines(self) -> List[str]:
        return list(set(r.engine_name for r in self.results))
    
    def get_datasets(self) -> List[str]:
        return list(set(r.dataset_name for r in self.results))
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save benchmark results to JSON."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'name': self.name,
            'description': self.description,
            'timestamp': self.timestamp,
            'results': [r.to_dict() for r in self.results]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved benchmark results to {filepath}")
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'BenchmarkResult':
        """Load benchmark results from JSON."""
        with open(filepath) as f:
            data = json.load(f)
        
        return cls(
            name=data['name'],
            description=data.get('description', ''),
            timestamp=data.get('timestamp', ''),
            results=[EngineResult.from_dict(r) for r in data['results']]
        )


class BenchmarkRunner:
    """
    Run benchmarks across multiple engines and datasets.
    
    Example usage:
        runner = BenchmarkRunner(
            engines=['gsplat', 'splatam', 'gslam'],
            datasets=['fr1_desk', 'fr1_room'],
            dataset_root='/path/to/tum'
        )
        results = runner.run()
        results.save('benchmark_results.json')
    """
    
    def __init__(
        self,
        engines: List[str],
        datasets: List[str],
        dataset_root: Union[str, Path],
        output_root: Optional[Union[str, Path]] = None,
        max_frames: int = 0,  # 0 = all frames
        engine_configs: Optional[Dict[str, Dict]] = None
    ):
        """
        Initialize benchmark runner.
        
        Args:
            engines: List of engine names to benchmark
            datasets: List of dataset names (TUM scene names)
            dataset_root: Root directory containing datasets
            output_root: Directory to save outputs (default: dataset_root/benchmark_output)
            max_frames: Maximum frames to process (0 = all)
            engine_configs: Optional per-engine configurations
        """
        self.engines = engines
        self.datasets = datasets
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root) if output_root else self.dataset_root / "benchmark_output"
        self.max_frames = max_frames
        self.engine_configs = engine_configs or {}
        
        self.output_root.mkdir(parents=True, exist_ok=True)
    
    def run(self, benchmark_name: str = "benchmark") -> BenchmarkResult:
        """
        Run all benchmarks.
        
        Returns:
            BenchmarkResult containing all results
        """
        results = BenchmarkResult(
            name=benchmark_name,
            description=f"Benchmark of {', '.join(self.engines)} on {', '.join(self.datasets)}"
        )
        
        for dataset in self.datasets:
            for engine in self.engines:
                try:
                    logger.info(f"Running {engine} on {dataset}...")
                    result = self._run_single(engine, dataset)
                    results.add_result(result)
                except Exception as e:
                    logger.error(f"Failed {engine} on {dataset}: {e}")
                    # Add empty result to mark failure
                    results.add_result(EngineResult(
                        engine_name=engine,
                        dataset_name=dataset,
                        config={'error': str(e)}
                    ))
        
        return results
    
    def _run_single(self, engine_name: str, dataset_name: str) -> EngineResult:
        """Run a single engine on a single dataset."""
        from src.engines import get_engine
        from src.pipeline.frames import TumRGBDSource
        
        # Create engine
        config = self.engine_configs.get(engine_name, {})
        engine = get_engine(engine_name, **config)
        
        # Create data source
        dataset_path = self.dataset_root / dataset_name
        source = TumRGBDSource(
            dataset_path=str(dataset_path),
            max_frames=self.max_frames if self.max_frames > 0 else None
        )
        
        # Initialize
        intrinsics = {
            'fx': source.fx, 'fy': source.fy,
            'cx': source.cx, 'cy': source.cy,
            'width': source.width, 'height': source.height
        }
        engine.initialize_scene(intrinsics, {'num_frames': len(source)})
        
        # Run
        start_time = time.time()
        
        for frame in source:
            engine.add_frame(
                frame_id=frame.idx,
                rgb=frame.rgb,
                depth=frame.depth,
                pose_world_cam=frame.pose
            )
            engine.optimize_step(n_steps=5)
        
        # Final optimization
        for _ in range(100):
            engine.optimize_step(n_steps=1)
        
        total_time = time.time() - start_time
        
        # Collect results
        result = EngineResult(
            engine_name=engine_name,
            dataset_name=dataset_name,
            total_time_sec=total_time,
            fps=len(source) / total_time if total_time > 0 else 0,
            num_frames=len(source),
            num_gaussians=engine.get_num_gaussians(),
            config=config,
            timestamp=datetime.now().isoformat()
        )
        
        # Save output
        output_path = self.output_root / engine_name / dataset_name
        output_path.mkdir(parents=True, exist_ok=True)
        engine.save_state(str(output_path))
        result.output_path = str(output_path)
        
        # Compute metrics if possible
        result = self._compute_metrics(result, engine, source)
        
        return result
    
    def _compute_metrics(
        self,
        result: EngineResult,
        engine,
        source
    ) -> EngineResult:
        """Compute rendering and trajectory metrics."""
        from .metrics import compute_image_metrics, compute_trajectory_metrics
        
        # Compute rendering metrics on a few test frames
        psnrs, ssims, lpipss = [], [], []
        test_frames = list(source)[::10][:10]  # Sample every 10th frame, max 10
        
        for frame in test_frames:
            rendered = engine.render_view(frame.pose, (source.width, source.height))
            if rendered is not None and rendered.max() > 0:
                metrics = compute_image_metrics(rendered, frame.rgb, compute_lpips_metric=False)
                psnrs.append(metrics['psnr'])
                ssims.append(metrics['ssim'])
        
        if psnrs:
            result.psnr = float(np.mean(psnrs))
            result.ssim = float(np.mean(ssims))
        
        return result


def generate_comparison_report(
    benchmark_result: BenchmarkResult,
    output_path: Optional[Union[str, Path]] = None
) -> str:
    """
    Generate a Markdown comparison report from benchmark results.
    
    Args:
        benchmark_result: BenchmarkResult to report on
        output_path: Optional path to save report
        
    Returns:
        Markdown string
    """
    lines = []
    
    # Header
    lines.append(f"# {benchmark_result.name}")
    lines.append(f"\nGenerated: {benchmark_result.timestamp}")
    lines.append(f"\n{benchmark_result.description}\n")
    
    engines = benchmark_result.get_engines()
    datasets = benchmark_result.get_datasets()
    
    # Summary table
    lines.append("## Performance Summary\n")
    lines.append("| Engine | Avg FPS | Avg PSNR | Avg SSIM | Avg ATE (m) | Avg Gaussians |")
    lines.append("|--------|---------|----------|----------|-------------|---------------|")
    
    for engine in sorted(engines):
        results = benchmark_result.get_by_engine(engine)
        if not results:
            continue
        
        fps_vals = [r.fps for r in results if r.fps > 0]
        psnr_vals = [r.psnr for r in results if r.psnr > 0]
        ssim_vals = [r.ssim for r in results if r.ssim > 0]
        ate_vals = [r.ate_rmse for r in results if r.ate_rmse > 0]
        gauss_vals = [r.num_gaussians for r in results if r.num_gaussians > 0]
        
        avg_fps = np.mean(fps_vals) if fps_vals else 0
        avg_psnr = np.mean(psnr_vals) if psnr_vals else 0
        avg_ssim = np.mean(ssim_vals) if ssim_vals else 0
        avg_ate = np.mean(ate_vals) if ate_vals else 0
        avg_gauss = np.mean(gauss_vals) if gauss_vals else 0
        
        lines.append(
            f"| {engine} | {avg_fps:.1f} | {avg_psnr:.2f} | {avg_ssim:.3f} | "
            f"{avg_ate:.4f} | {int(avg_gauss):,} |"
        )
    
    # Per-dataset breakdown
    lines.append("\n## Per-Dataset Results\n")
    
    for dataset in sorted(datasets):
        lines.append(f"### {dataset}\n")
        lines.append("| Engine | FPS | PSNR | SSIM | ATE RMSE | Gaussians | Time (s) |")
        lines.append("|--------|-----|------|------|----------|-----------|----------|")
        
        results = benchmark_result.get_by_dataset(dataset)
        for result in sorted(results, key=lambda r: r.engine_name):
            lines.append(
                f"| {result.engine_name} | {result.fps:.1f} | {result.psnr:.2f} | "
                f"{result.ssim:.3f} | {result.ate_rmse:.4f} | {result.num_gaussians:,} | "
                f"{result.total_time_sec:.1f} |"
            )
        lines.append("")
    
    # Best results highlight
    lines.append("## Best Results\n")
    
    if benchmark_result.results:
        best_fps = max(benchmark_result.results, key=lambda r: r.fps)
        best_psnr = max(benchmark_result.results, key=lambda r: r.psnr)
        best_ssim = max(benchmark_result.results, key=lambda r: r.ssim)
        best_ate = min([r for r in benchmark_result.results if r.ate_rmse > 0], 
                       key=lambda r: r.ate_rmse, default=None)
        
        lines.append(f"- **Fastest**: {best_fps.engine_name} ({best_fps.fps:.1f} FPS on {best_fps.dataset_name})")
        lines.append(f"- **Best PSNR**: {best_psnr.engine_name} ({best_psnr.psnr:.2f} dB on {best_psnr.dataset_name})")
        lines.append(f"- **Best SSIM**: {best_ssim.engine_name} ({best_ssim.ssim:.3f} on {best_ssim.dataset_name})")
        if best_ate:
            lines.append(f"- **Best Tracking**: {best_ate.engine_name} ({best_ate.ate_rmse:.4f}m ATE on {best_ate.dataset_name})")
    
    report = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved report to {output_path}")
    
    return report


def load_existing_results(output_root: Union[str, Path]) -> BenchmarkResult:
    """
    Load existing benchmark results from output directories.
    
    Scans output directories for metrics files and constructs a BenchmarkResult.
    
    Args:
        output_root: Root directory containing engine outputs
        
    Returns:
        BenchmarkResult populated from existing files
    """
    output_root = Path(output_root)
    results = BenchmarkResult(name="Loaded Results")
    
    # Scan for engine directories
    for engine_dir in output_root.iterdir():
        if not engine_dir.is_dir():
            continue
        
        engine_name = engine_dir.name
        
        # Scan for dataset directories
        for dataset_dir in engine_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            
            dataset_name = dataset_dir.name
            result = EngineResult(
                engine_name=engine_name,
                dataset_name=dataset_name,
                output_path=str(dataset_dir)
            )
            
            # Load metrics from various sources
            
            # ATE from Gaussian-SLAM format
            ate_file = dataset_dir / "ate_aligned.json"
            if ate_file.exists():
                with open(ate_file) as f:
                    ate_data = json.load(f)
                result.ate_rmse = ate_data.get('rmse', 0.0)
                result.ate_mean = ate_data.get('mean', 0.0)
            
            # Rendering metrics
            render_file = dataset_dir / "rendering_metrics.json"
            if render_file.exists():
                with open(render_file) as f:
                    render_data = json.load(f)
                result.psnr = render_data.get('psnr', 0.0)
                result.ssim = render_data.get('ssim', 0.0)
                result.lpips = render_data.get('lpips', 0.0)
            
            # Metrics summary (GSplat format)
            summary_file = dataset_dir / "metrics_summary.txt"
            if summary_file.exists():
                with open(summary_file) as f:
                    content = f.read()
                
                # Parse key metrics
                import re
                fps_match = re.search(r'Average FPS:\s*([\d.]+)', content)
                if fps_match:
                    result.fps = float(fps_match.group(1))
                
                gauss_match = re.search(r'Final Gaussians:\s*([\d,]+)', content)
                if gauss_match:
                    result.num_gaussians = int(gauss_match.group(1).replace(',', ''))
                
                frames_match = re.search(r'Frames processed:\s*(\d+)', content)
                if frames_match:
                    result.num_frames = int(frames_match.group(1))
                
                time_match = re.search(r'Total time:\s*([\d.]+)s', content)
                if time_match:
                    result.total_time_sec = float(time_match.group(1))
            
            # Config
            config_file = dataset_dir / "config.yaml"
            if config_file.exists():
                try:
                    import yaml
                    with open(config_file) as f:
                        result.config = yaml.safe_load(f)
                except Exception:
                    pass
            
            results.add_result(result)
    
    return results
