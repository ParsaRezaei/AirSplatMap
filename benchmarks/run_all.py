#!/usr/bin/env python3
"""
AirSplatMap Benchmark Suite
===========================

Run all benchmarks and generate comprehensive reports.

Usage:
    python benchmarks/run_all.py                    # Run everything
    python benchmarks/run_all.py --pose             # Pose only
    python benchmarks/run_all.py --depth            # Depth only  
    python benchmarks/run_all.py --gs               # Gaussian splatting only
    python benchmarks/run_all.py --plots-only       # Generate plots from existing results
    python benchmarks/run_all.py --report           # Generate HTML report only
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_results(results_path: Path) -> List[Dict]:
    """Load results from JSON file."""
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return []


def save_results(results: List[Dict], results_path: Path):
    """Save results to JSON file."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)


def run_pose_benchmark(
    methods: Optional[List[str]] = None,
    dataset_root: Optional[Path] = None,
    max_frames: int = 200,
    skip_frames: int = 2,
) -> List[Dict]:
    """Run pose estimation benchmark."""
    logger.info("=" * 60)
    logger.info("RUNNING POSE ESTIMATION BENCHMARK")
    logger.info("=" * 60)
    
    from benchmarks.pose.benchmark_pose import run_benchmark, find_tum_datasets, print_results_table
    from dataclasses import asdict
    
    if dataset_root is None:
        dataset_root = PROJECT_ROOT / "datasets"
    
    datasets = find_tum_datasets(dataset_root)
    if not datasets:
        logger.warning(f"No TUM datasets found in {dataset_root}")
        return []
    
    # Use first dataset for quick benchmark
    datasets = datasets[:1]
    
    if methods is None:
        methods = ['orb', 'sift', 'robust_flow', 'keyframe']
    
    results = []
    for dataset in datasets:
        for method in methods:
            try:
                result = run_benchmark(
                    method=method,
                    dataset_path=str(dataset),
                    max_frames=max_frames,
                    skip_frames=skip_frames,
                )
                results.append(asdict(result))
                logger.info(f"  {method}: ATE={result.ate_rmse:.4f}m, FPS={result.fps:.1f}")
            except Exception as e:
                logger.error(f"  {method}: FAILED - {e}")
    
    # Print and save
    if results:
        # Convert back for print function
        from benchmarks.pose.benchmark_pose import BenchmarkResult
        result_objs = [BenchmarkResult(**r) for r in results]
        print_results_table(result_objs)
    
    return results


def run_depth_benchmark(
    methods: Optional[List[str]] = None,
    dataset_root: Optional[Path] = None,
    max_frames: int = 50,
    skip_frames: int = 10,
) -> List[Dict]:
    """Run depth estimation benchmark."""
    logger.info("=" * 60)
    logger.info("RUNNING DEPTH ESTIMATION BENCHMARK")
    logger.info("=" * 60)
    
    from benchmarks.depth.benchmark_depth import run_depth_benchmark as run_bench, print_results_table
    from dataclasses import asdict
    
    if dataset_root is None:
        dataset_root = PROJECT_ROOT / "datasets"
    
    # Find datasets with depth
    datasets = []
    for item in sorted(dataset_root.iterdir()):
        if item.is_dir() and (item / "depth.txt").exists():
            datasets.append(item)
    
    tum_dir = dataset_root / "tum"
    if tum_dir.exists():
        for item in sorted(tum_dir.iterdir()):
            if item.is_dir() and (item / "depth.txt").exists():
                datasets.append(item)
    
    if not datasets:
        logger.warning(f"No datasets with depth found in {dataset_root}")
        return []
    
    datasets = datasets[:1]  # Use first dataset
    
    if methods is None:
        methods = ['midas', 'midas_small']
    
    results = []
    for dataset in datasets:
        for method in methods:
            try:
                result = run_bench(
                    method=method,
                    dataset_path=str(dataset),
                    max_frames=max_frames,
                    skip_frames=skip_frames,
                )
                results.append(asdict(result))
                logger.info(f"  {method}: AbsRel={result.abs_rel:.4f}, FPS={result.fps:.1f}")
            except Exception as e:
                logger.error(f"  {method}: FAILED - {e}")
    
    return results


def run_gs_benchmark(
    engines: Optional[List[str]] = None,
    dataset_root: Optional[Path] = None,
    max_frames: int = 30,
    iterations: int = 30,
) -> List[Dict]:
    """Run Gaussian splatting benchmark."""
    logger.info("=" * 60)
    logger.info("RUNNING GAUSSIAN SPLATTING BENCHMARK")
    logger.info("=" * 60)
    
    from benchmarks.gaussian_splatting.benchmark_gs import run_gs_benchmark as run_bench, print_results_table
    from dataclasses import asdict
    
    if dataset_root is None:
        dataset_root = PROJECT_ROOT / "datasets"
    
    # Find datasets
    datasets = []
    for item in sorted(dataset_root.iterdir()):
        if item.is_dir() and (item / "rgb.txt").exists():
            datasets.append(item)
    
    tum_dir = dataset_root / "tum"
    if tum_dir.exists():
        for item in sorted(tum_dir.iterdir()):
            if item.is_dir() and (item / "rgb.txt").exists():
                datasets.append(item)
    
    if not datasets:
        logger.warning(f"No datasets found in {dataset_root}")
        return []
    
    datasets = datasets[:1]  # Use first dataset
    
    if engines is None:
        engines = ['graphdeco']
    
    results = []
    for dataset in datasets:
        for engine in engines:
            try:
                result = run_bench(
                    engine_name=engine,
                    dataset_path=str(dataset),
                    max_frames=max_frames,
                    n_iterations=iterations,
                )
                results.append(asdict(result))
                logger.info(f"  {engine}: PSNR={result.psnr:.2f}dB, Gaussians={result.final_gaussians:,}")
            except Exception as e:
                logger.error(f"  {engine}: FAILED - {e}")
                import traceback
                traceback.print_exc()
    
    return results


def generate_plots(
    pose_results: List[Dict],
    depth_results: List[Dict],
    gs_results: List[Dict],
):
    """Generate all benchmark plots."""
    logger.info("=" * 60)
    logger.info("GENERATING PLOTS")
    logger.info("=" * 60)
    
    from benchmarks.visualization.plot_utils import (
        plot_pose_metrics_bar,
        plot_depth_metrics_bar,
        plot_gs_metrics_bar,
        plot_overall_summary,
    )
    
    plots_dir = PROJECT_ROOT / "benchmarks" / "results" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Pose plots
    if pose_results:
        try:
            plot_pose_metrics_bar(
                pose_results,
                metrics=['ate_rmse', 'rpe_trans_rmse', 'fps'],
                title="Pose Estimation Comparison",
                output_path=plots_dir / "pose_comparison",
            )
            logger.info(f"  Saved pose_comparison.png")
        except Exception as e:
            logger.warning(f"  Pose plot failed: {e}")
    
    # Depth plots
    if depth_results:
        try:
            plot_depth_metrics_bar(
                depth_results,
                metrics=['abs_rel', 'rmse', 'delta1'],
                title="Depth Estimation Comparison",
                output_path=plots_dir / "depth_comparison",
            )
            logger.info(f"  Saved depth_comparison.png")
        except Exception as e:
            logger.warning(f"  Depth plot failed: {e}")
    
    # GS plots
    if gs_results:
        try:
            plot_gs_metrics_bar(
                gs_results,
                metrics=['psnr', 'ssim', 'fps'],
                title="Gaussian Splatting Comparison",
                output_path=plots_dir / "gs_comparison",
            )
            logger.info(f"  Saved gs_comparison.png")
        except Exception as e:
            logger.warning(f"  GS plot failed: {e}")
    
    # Overall summary
    try:
        plot_overall_summary(
            pose_results=pose_results,
            depth_results=depth_results,
            gs_results=gs_results,
            title="AirSplatMap Benchmark Summary",
            output_path=plots_dir / "overall_summary",
        )
        logger.info(f"  Saved overall_summary.png")
    except Exception as e:
        logger.warning(f"  Summary plot failed: {e}")


def generate_report(
    pose_results: List[Dict],
    depth_results: List[Dict],
    gs_results: List[Dict],
):
    """Generate HTML report."""
    logger.info("=" * 60)
    logger.info("GENERATING HTML REPORT")
    logger.info("=" * 60)
    
    from benchmarks.visualization.plot_utils import generate_html_report
    
    report_path = PROJECT_ROOT / "benchmarks" / "results" / "report.html"
    
    generate_html_report(
        pose_results=pose_results,
        depth_results=depth_results,
        gs_results=gs_results,
        output_path=report_path,
        title="AirSplatMap Benchmark Report",
    )
    
    logger.info(f"  Report saved to: {report_path}")
    
    # Try to open in browser
    try:
        import webbrowser
        webbrowser.open(f"file://{report_path.absolute()}")
    except:
        pass


def main():
    parser = argparse.ArgumentParser(description='Run AirSplatMap benchmarks')
    
    # Benchmark selection
    parser.add_argument('--pose', action='store_true', help='Run pose benchmark')
    parser.add_argument('--depth', action='store_true', help='Run depth benchmark')
    parser.add_argument('--gs', action='store_true', help='Run Gaussian splatting benchmark')
    parser.add_argument('--all', action='store_true', help='Run all benchmarks')
    
    # Output options
    parser.add_argument('--plots-only', action='store_true', help='Only generate plots from existing results')
    parser.add_argument('--report', action='store_true', help='Generate HTML report')
    parser.add_argument('--no-plots', action='store_true', help='Skip plot generation')
    
    # Configuration
    parser.add_argument('--dataset-root', type=str, default=None, help='Dataset root directory')
    parser.add_argument('--pose-methods', nargs='+', default=None, help='Pose methods to benchmark')
    parser.add_argument('--depth-methods', nargs='+', default=None, help='Depth methods to benchmark')
    parser.add_argument('--gs-engines', nargs='+', default=None, help='GS engines to benchmark')
    parser.add_argument('--max-frames', type=int, default=100, help='Max frames per dataset')
    parser.add_argument('--quick', action='store_true', help='Quick mode (fewer frames/iterations)')
    
    args = parser.parse_args()
    
    # Determine what to run
    run_all = args.all or not (args.pose or args.depth or args.gs or args.plots_only or args.report)
    
    # Quick mode settings
    if args.quick:
        pose_max_frames = 50
        pose_skip = 5
        depth_max_frames = 20
        depth_skip = 10
        gs_max_frames = 20
        gs_iterations = 20
    else:
        pose_max_frames = args.max_frames
        pose_skip = 2
        depth_max_frames = min(args.max_frames, 100)
        depth_skip = 5
        gs_max_frames = min(args.max_frames, 50)
        gs_iterations = 50
    
    dataset_root = Path(args.dataset_root) if args.dataset_root else None
    
    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load existing results
    pose_results = load_results(results_dir / "pose_benchmark.json")
    depth_results = load_results(results_dir / "depth_benchmark.json")
    gs_results = load_results(results_dir / "gs_benchmark.json")
    
    # Run benchmarks
    if not args.plots_only and not args.report:
        print("\n" + "=" * 70)
        print("  AIRSPLATMAP BENCHMARK SUITE")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 70 + "\n")
        
        if run_all or args.pose:
            new_results = run_pose_benchmark(
                methods=args.pose_methods,
                dataset_root=dataset_root,
                max_frames=pose_max_frames,
                skip_frames=pose_skip,
            )
            if new_results:
                pose_results = new_results
                save_results(pose_results, results_dir / "pose_benchmark.json")
        
        if run_all or args.depth:
            new_results = run_depth_benchmark(
                methods=args.depth_methods,
                dataset_root=dataset_root,
                max_frames=depth_max_frames,
                skip_frames=depth_skip,
            )
            if new_results:
                depth_results = new_results
                save_results(depth_results, results_dir / "depth_benchmark.json")
        
        if run_all or args.gs:
            new_results = run_gs_benchmark(
                engines=args.gs_engines,
                dataset_root=dataset_root,
                max_frames=gs_max_frames,
                iterations=gs_iterations,
            )
            if new_results:
                gs_results = new_results
                save_results(gs_results, results_dir / "gs_benchmark.json")
    
    # Generate plots
    if not args.no_plots:
        generate_plots(pose_results, depth_results, gs_results)
    
    # Generate report
    if args.report or args.plots_only or run_all:
        generate_report(pose_results, depth_results, gs_results)
    
    print("\n" + "=" * 70)
    print("  BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {results_dir}")
    print(f"HTML report: {results_dir / 'report.html'}")
    print(f"Plots: {results_dir / 'plots'}")


if __name__ == '__main__':
    main()
