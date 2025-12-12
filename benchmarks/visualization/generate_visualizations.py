"""
Comprehensive Visualization Generator
=====================================

Generates organized visualizations with:
- Per-dataset folders with individual results
- Aggregated "general" folder with cross-dataset summaries
- Clean, publication-quality plots
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

# Check for matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("Matplotlib not available")


def setup_style():
    """Setup clean plot style."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#1e293b',
        'axes.facecolor': '#1e293b',
        'axes.edgecolor': '#475569',
        'axes.labelcolor': '#e2e8f0',
        'text.color': '#e2e8f0',
        'xtick.color': '#94a3b8',
        'ytick.color': '#94a3b8',
        'grid.color': '#334155',
        'grid.alpha': 0.5,
        'legend.facecolor': '#1e293b',
        'legend.edgecolor': '#475569',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.2,
    })


# Color palette
COLORS = {
    'primary': '#6366f1',
    'secondary': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'purple': '#8b5cf6',
    'pink': '#ec4899',
    'cyan': '#06b6d4',
    'slate': '#64748b',
}
COLOR_LIST = list(COLORS.values())


def aggregate_by_method(results: List[Dict], method_field: str = 'method') -> Dict[str, Dict]:
    """Aggregate results by method across datasets."""
    by_method = defaultdict(list)
    for r in results:
        method = r.get(method_field, 'unknown')
        by_method[method].append(r)
    
    aggregated = {}
    if not results:
        return aggregated
    
    # Get numeric fields
    numeric_fields = [k for k, v in results[0].items() 
                     if isinstance(v, (int, float)) and k not in [method_field, 'dataset', 'num_frames']]
    
    for method, method_results in by_method.items():
        stats = {
            'count': len(method_results),
            'datasets': list(set(r.get('dataset', 'unknown') for r in method_results))
        }
        for field in numeric_fields:
            values = [r.get(field) for r in method_results if r.get(field) is not None]
            if values:
                stats[f'{field}_mean'] = float(np.mean(values))
                stats[f'{field}_std'] = float(np.std(values))
                stats[f'{field}_min'] = float(np.min(values))
                stats[f'{field}_max'] = float(np.max(values))
        aggregated[method] = stats
    
    return aggregated


def group_by_dataset(results: List[Dict]) -> Dict[str, List[Dict]]:
    """Group results by dataset."""
    by_dataset = defaultdict(list)
    for r in results:
        dataset = r.get('dataset', 'unknown')
        by_dataset[dataset].append(r)
    return dict(by_dataset)


def generate_comprehensive_visualizations(
    all_results: Dict[str, List[Dict]],
    output_dir: Path,
    trajectory_data: Optional[Dict] = None,
):
    """
    Generate comprehensive visualizations organized by dataset.
    
    Output structure:
        output_dir/
        ├── plots/
        │   ├── general/           # Aggregated cross-dataset results
        │   │   ├── pose_summary.png
        │   │   ├── depth_summary.png
        │   │   ├── gs_summary.png
        │   │   ├── combined_overview.png
        │   │   └── method_comparison.png
        │   ├── {dataset_name}/    # Per-dataset results
        │   │   ├── pose/
        │   │   ├── depth/
        │   │   └── gs/
        │   └── ...
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available, skipping visualizations")
        return
    
    setup_style()
    
    # Handle None/missing results gracefully
    pose_results = all_results.get('pose') or []
    depth_results = all_results.get('depth') or []
    gs_results = all_results.get('gs') or []
    pipeline_results = all_results.get('pipeline') or []
    
    # Early exit if no results at all
    if not any([pose_results, depth_results, gs_results]):
        logger.warning("No results to visualize")
        return
    
    # Create directory structure
    plots_dir = output_dir / "plots"
    general_dir = plots_dir / "general"
    general_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all datasets
    all_datasets = set()
    for r in pose_results + depth_results + gs_results:
        dataset = r.get('dataset')
        if dataset:
            all_datasets.add(dataset)
    
    if not all_datasets:
        logger.warning("No datasets found in results")
        return
    
    logger.info(f"📊 Generating visualizations for {len(all_datasets)} datasets...")
    
    # =========================================================================
    # GENERAL (AGGREGATED) VISUALIZATIONS
    # =========================================================================
    logger.info("📈 Generating aggregated cross-dataset plots...")
    
    try:
        _generate_aggregated_pose_summary(pose_results, general_dir / "pose_summary.png")
        logger.info("    ✓ general/pose_summary.png")
    except Exception as e:
        logger.warning(f"    ✗ pose_summary failed: {e}")
    
    try:
        _generate_aggregated_depth_summary(depth_results, general_dir / "depth_summary.png")
        logger.info("    ✓ general/depth_summary.png")
    except Exception as e:
        logger.warning(f"    ✗ depth_summary failed: {e}")
    
    try:
        _generate_aggregated_gs_summary(gs_results, general_dir / "gs_summary.png")
        logger.info("    ✓ general/gs_summary.png")
    except Exception as e:
        logger.warning(f"    ✗ gs_summary failed: {e}")
    
    try:
        _generate_combined_overview(pose_results, depth_results, gs_results, 
                                    general_dir / "combined_overview.png")
        logger.info("    ✓ general/combined_overview.png")
    except Exception as e:
        logger.warning(f"    ✗ combined_overview failed: {e}")
    
    try:
        _generate_method_comparison_radar(pose_results, depth_results, gs_results,
                                          general_dir / "method_comparison_radar.png")
        logger.info("    ✓ general/method_comparison_radar.png")
    except Exception as e:
        logger.warning(f"    ✗ method_comparison_radar failed: {e}")
    
    # =========================================================================
    # PER-DATASET VISUALIZATIONS
    # =========================================================================
    pose_by_dataset = group_by_dataset(pose_results)
    depth_by_dataset = group_by_dataset(depth_results)
    gs_by_dataset = group_by_dataset(gs_results)
    
    for dataset in sorted(all_datasets):
        logger.info(f"[+] Generating plots for {dataset}...")
        dataset_dir = plots_dir / dataset
        
        # Pose plots
        dataset_pose = pose_by_dataset.get(dataset, [])
        if dataset_pose:
            pose_dir = dataset_dir / "pose"
            pose_dir.mkdir(parents=True, exist_ok=True)
            try:
                _generate_pose_metrics_bar(dataset_pose, pose_dir / "metrics_bar.png", dataset)
                _generate_pose_accuracy_plot(dataset_pose, pose_dir / "accuracy.png", dataset)
                logger.info(f"    [OK] {dataset}/pose/")
            except Exception as e:
                logger.warning(f"    [FAIL] {dataset}/pose failed: {e}")
        
        # Depth plots
        dataset_depth = depth_by_dataset.get(dataset, [])
        if dataset_depth:
            depth_dir = dataset_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            try:
                _generate_depth_metrics_bar(dataset_depth, depth_dir / "metrics_bar.png", dataset)
                _generate_depth_threshold_plot(dataset_depth, depth_dir / "thresholds.png", dataset)
                logger.info(f"    [OK] {dataset}/depth/")
            except Exception as e:
                logger.warning(f"    [FAIL] {dataset}/depth failed: {e}")
        
        # GS plots
        dataset_gs = gs_by_dataset.get(dataset, [])
        if dataset_gs:
            gs_dir = dataset_dir / "gs"
            gs_dir.mkdir(parents=True, exist_ok=True)
            try:
                _generate_gs_quality_bar(dataset_gs, gs_dir / "quality.png", dataset)
                _generate_gs_efficiency_plot(dataset_gs, gs_dir / "efficiency.png", dataset)
                logger.info(f"    [OK] {dataset}/gs/")
            except Exception as e:
                logger.warning(f"    [FAIL] {dataset}/gs failed: {e}")
    
    # 3D trajectory plots if data available
    if trajectory_data:
        try:
            _generate_trajectory_plots(trajectory_data, general_dir)
            logger.info("    [OK] general/trajectories/")
        except Exception as e:
            logger.warning(f"    [FAIL] trajectories failed: {e}")
    
    logger.info("Visualization generation complete!")


# =============================================================================
# AGGREGATED PLOTS
# =============================================================================

def _generate_aggregated_pose_summary(results: List[Dict], output_path: Path):
    """Generate aggregated pose summary with error bars."""
    if not results:
        return
    
    aggregated = aggregate_by_method(results, 'method')
    methods = sorted(aggregated.keys())
    
    if not methods:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    # ATE
    ate_means = [aggregated[m].get('ate_rmse_mean', 0) for m in methods]
    ate_stds = [aggregated[m].get('ate_rmse_std', 0) for m in methods]
    colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(methods))]
    
    bars = axes[0].bar(methods, ate_means, yerr=ate_stds, capsize=5, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_ylabel('ATE RMSE (m)')
    axes[0].set_title('Pose Accuracy (lower is better)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Highlight best
    best_idx = np.argmin(ate_means)
    bars[best_idx].set_edgecolor(COLORS['secondary'])
    bars[best_idx].set_linewidth(3)
    
    # RPE
    rpe_means = [aggregated[m].get('rpe_trans_rmse_mean', 0) for m in methods]
    rpe_stds = [aggregated[m].get('rpe_trans_rmse_std', 0) for m in methods]
    
    bars = axes[1].bar(methods, rpe_means, yerr=rpe_stds, capsize=5, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_ylabel('RPE Trans RMSE (m)')
    axes[1].set_title('Relative Pose Error')
    axes[1].tick_params(axis='x', rotation=45)
    
    # FPS
    fps_means = [aggregated[m].get('fps_mean', 0) for m in methods]
    fps_stds = [aggregated[m].get('fps_std', 0) for m in methods]
    
    bars = axes[2].bar(methods, fps_means, yerr=fps_stds, capsize=5, color=colors, edgecolor='white', linewidth=0.5)
    axes[2].set_ylabel('FPS')
    axes[2].set_title('Processing Speed')
    axes[2].tick_params(axis='x', rotation=45)
    
    # Highlight best
    best_idx = np.argmax(fps_means)
    bars[best_idx].set_edgecolor(COLORS['secondary'])
    bars[best_idx].set_linewidth(3)
    
    plt.suptitle('Pose Estimation Summary (Averaged Across Datasets)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _generate_aggregated_depth_summary(results: List[Dict], output_path: Path):
    """Generate aggregated depth summary."""
    if not results:
        return
    
    aggregated = aggregate_by_method(results, 'method')
    methods = sorted(aggregated.keys())
    
    if not methods:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(methods))]
    
    # AbsRel
    absrel_means = [aggregated[m].get('abs_rel_mean', 0) for m in methods]
    absrel_stds = [aggregated[m].get('abs_rel_std', 0) for m in methods]
    
    bars = axes[0].bar(methods, absrel_means, yerr=absrel_stds, capsize=5, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_ylabel('AbsRel')
    axes[0].set_title('Absolute Relative Error (lower is better)')
    axes[0].tick_params(axis='x', rotation=45)
    
    best_idx = np.argmin(absrel_means)
    bars[best_idx].set_edgecolor(COLORS['secondary'])
    bars[best_idx].set_linewidth(3)
    
    # Delta1
    delta_means = [aggregated[m].get('delta1_mean', 0) * 100 for m in methods]
    delta_stds = [aggregated[m].get('delta1_std', 0) * 100 for m in methods]
    
    bars = axes[1].bar(methods, delta_means, yerr=delta_stds, capsize=5, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_ylabel('δ < 1.25 (%)')
    axes[1].set_title('Threshold Accuracy (higher is better)')
    axes[1].tick_params(axis='x', rotation=45)
    
    best_idx = np.argmax(delta_means)
    bars[best_idx].set_edgecolor(COLORS['secondary'])
    bars[best_idx].set_linewidth(3)
    
    # FPS
    fps_means = [aggregated[m].get('fps_mean', 0) for m in methods]
    fps_stds = [aggregated[m].get('fps_std', 0) for m in methods]
    
    bars = axes[2].bar(methods, fps_means, yerr=fps_stds, capsize=5, color=colors, edgecolor='white', linewidth=0.5)
    axes[2].set_ylabel('FPS')
    axes[2].set_title('Processing Speed')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Depth Estimation Summary (Averaged Across Datasets)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _generate_aggregated_gs_summary(results: List[Dict], output_path: Path):
    """Generate aggregated GS summary."""
    if not results:
        return
    
    aggregated = aggregate_by_method(results, 'engine')
    engines = sorted(aggregated.keys())
    
    if not engines:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(engines))]
    
    # PSNR
    psnr_means = [aggregated[e].get('psnr_mean', 0) for e in engines]
    psnr_stds = [aggregated[e].get('psnr_std', 0) for e in engines]
    
    bars = axes[0].bar(engines, psnr_means, yerr=psnr_stds, capsize=5, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].set_title('Render Quality (higher is better)')
    axes[0].tick_params(axis='x', rotation=45)
    
    best_idx = np.argmax(psnr_means)
    bars[best_idx].set_edgecolor(COLORS['secondary'])
    bars[best_idx].set_linewidth(3)
    
    # SSIM
    ssim_means = [aggregated[e].get('ssim_mean', 0) for e in engines]
    ssim_stds = [aggregated[e].get('ssim_std', 0) for e in engines]
    
    bars = axes[1].bar(engines, ssim_means, yerr=ssim_stds, capsize=5, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_ylabel('SSIM')
    axes[1].set_title('Structural Similarity')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Gaussians
    gauss_means = [aggregated[e].get('final_gaussians_mean', 0) / 1000 for e in engines]
    gauss_stds = [aggregated[e].get('final_gaussians_std', 0) / 1000 for e in engines]
    
    bars = axes[2].bar(engines, gauss_means, yerr=gauss_stds, capsize=5, color=colors, edgecolor='white', linewidth=0.5)
    axes[2].set_ylabel('Gaussians (K)')
    axes[2].set_title('Model Complexity')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.suptitle('Gaussian Splatting Summary (Averaged Across Datasets)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _generate_combined_overview(pose_results: List[Dict], depth_results: List[Dict], 
                                gs_results: List[Dict], output_path: Path):
    """Generate combined overview panel."""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Pose - ATE by dataset and method
    ax1 = fig.add_subplot(gs[0, 0])
    if pose_results:
        by_dataset = group_by_dataset(pose_results)
        datasets = sorted(by_dataset.keys())
        methods = sorted(set(r['method'] for r in pose_results))
        
        x = np.arange(len(datasets))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            values = []
            for d in datasets:
                method_results = [r for r in by_dataset[d] if r['method'] == method]
                values.append(method_results[0]['ate_rmse'] if method_results else 0)
            ax1.bar(x + i * width, values, width, label=method, color=COLOR_LIST[i % len(COLOR_LIST)])
        
        ax1.set_xticks(x + width * len(methods) / 2)
        ax1.set_xticklabels([d[:15] + '...' if len(d) > 15 else d for d in datasets], rotation=45, ha='right')
        ax1.set_ylabel('ATE RMSE (m)')
        ax1.set_title('Pose: ATE by Dataset')
        ax1.legend(fontsize=8, loc='upper right')
    
    # Depth - AbsRel by dataset and method
    ax2 = fig.add_subplot(gs[0, 1])
    if depth_results:
        by_dataset = group_by_dataset(depth_results)
        datasets = sorted(by_dataset.keys())
        methods = sorted(set(r['method'] for r in depth_results))
        
        x = np.arange(len(datasets))
        width = 0.8 / len(methods)
        
        for i, method in enumerate(methods):
            values = []
            for d in datasets:
                method_results = [r for r in by_dataset[d] if r['method'] == method]
                values.append(method_results[0]['abs_rel'] if method_results else 0)
            ax2.bar(x + i * width, values, width, label=method, color=COLOR_LIST[i % len(COLOR_LIST)])
        
        ax2.set_xticks(x + width * len(methods) / 2)
        ax2.set_xticklabels([d[:15] + '...' if len(d) > 15 else d for d in datasets], rotation=45, ha='right')
        ax2.set_ylabel('AbsRel')
        ax2.set_title('Depth: AbsRel by Dataset')
        ax2.legend(fontsize=8, loc='upper right')
    
    # GS - PSNR by dataset and engine
    ax3 = fig.add_subplot(gs[0, 2])
    if gs_results:
        by_dataset = group_by_dataset(gs_results)
        datasets = sorted(by_dataset.keys())
        engines = sorted(set(r['engine'] for r in gs_results))
        
        x = np.arange(len(datasets))
        width = 0.8 / len(engines)
        
        for i, engine in enumerate(engines):
            values = []
            for d in datasets:
                engine_results = [r for r in by_dataset[d] if r['engine'] == engine]
                values.append(engine_results[0]['psnr'] if engine_results else 0)
            ax3.bar(x + i * width, values, width, label=engine, color=COLOR_LIST[i % len(COLOR_LIST)])
        
        ax3.set_xticks(x + width * len(engines) / 2)
        ax3.set_xticklabels([d[:15] + '...' if len(d) > 15 else d for d in datasets], rotation=45, ha='right')
        ax3.set_ylabel('PSNR (dB)')
        ax3.set_title('GS: PSNR by Dataset')
        ax3.legend(fontsize=8, loc='upper right')
    
    # FPS comparison across all methods
    ax4 = fig.add_subplot(gs[1, :])
    all_methods = []
    all_fps = []
    all_colors = []
    all_categories = []
    
    if pose_results:
        pose_agg = aggregate_by_method(pose_results, 'method')
        for m, stats in pose_agg.items():
            all_methods.append(f"Pose: {m}")
            all_fps.append(stats.get('fps_mean', 0))
            all_colors.append(COLORS['primary'])
            all_categories.append('Pose')
    
    if depth_results:
        depth_agg = aggregate_by_method(depth_results, 'method')
        for m, stats in depth_agg.items():
            all_methods.append(f"Depth: {m}")
            all_fps.append(stats.get('fps_mean', 0))
            all_colors.append(COLORS['secondary'])
            all_categories.append('Depth')
    
    if gs_results:
        gs_agg = aggregate_by_method(gs_results, 'engine')
        for e, stats in gs_agg.items():
            all_methods.append(f"GS: {e}")
            all_fps.append(stats.get('fps_mean', 0))
            all_colors.append(COLORS['warning'])
            all_categories.append('GS')
    
    if all_methods:
        bars = ax4.barh(all_methods, all_fps, color=all_colors)
        ax4.set_xlabel('FPS (higher is better)')
        ax4.set_title('Processing Speed Comparison')
        ax4.axvline(x=30, color=COLORS['secondary'], linestyle='--', alpha=0.5, label='Real-time (30 FPS)')
        ax4.legend(loc='lower right')
    
    plt.suptitle('Benchmark Overview', fontsize=18, fontweight='bold', y=1.02)
    plt.savefig(output_path)
    plt.close()


def _generate_method_comparison_radar(pose_results: List[Dict], depth_results: List[Dict],
                                      gs_results: List[Dict], output_path: Path):
    """Generate radar chart comparing best methods."""
    if not (pose_results and depth_results and gs_results):
        return
    
    # Get best method from each category
    pose_agg = aggregate_by_method(pose_results, 'method')
    depth_agg = aggregate_by_method(depth_results, 'method')
    gs_agg = aggregate_by_method(gs_results, 'engine')
    
    best_pose = min(pose_agg.items(), key=lambda x: x[1].get('ate_rmse_mean', float('inf')))
    best_depth = min(depth_agg.items(), key=lambda x: x[1].get('abs_rel_mean', float('inf')))
    best_gs = max(gs_agg.items(), key=lambda x: x[1].get('psnr_mean', 0))
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    categories = ['Pose\nAccuracy', 'Pose\nSpeed', 'Depth\nAccuracy', 'Depth\nSpeed', 'GS\nQuality', 'GS\nSpeed']
    N = len(categories)
    
    # Normalize values to 0-1 (higher is better)
    values = [
        1 - min(best_pose[1].get('ate_rmse_mean', 1) / 1.0, 1),  # Pose accuracy (invert)
        min(best_pose[1].get('fps_mean', 0) / 100, 1),  # Pose speed
        1 - min(best_depth[1].get('abs_rel_mean', 1) / 1.0, 1),  # Depth accuracy (invert)
        min(best_depth[1].get('fps_mean', 0) / 50, 1),  # Depth speed
        min(best_gs[1].get('psnr_mean', 0) / 40, 1),  # GS quality
        min(best_gs[1].get('fps_mean', 0) / 100, 1),  # GS speed
    ]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, color=COLORS['primary'])
    ax.fill(angles, values, alpha=0.25, color=COLORS['primary'])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    
    plt.title(f'Best Methods Performance\nPose: {best_pose[0]}, Depth: {best_depth[0]}, GS: {best_gs[0]}',
              fontsize=12, fontweight='bold', pad=20)
    plt.savefig(output_path)
    plt.close()


# =============================================================================
# PER-DATASET PLOTS
# =============================================================================

def _generate_pose_metrics_bar(results: List[Dict], output_path: Path, dataset: str):
    """Generate pose metrics bar chart for a single dataset."""
    methods = [r['method'] for r in results]
    ate = [r.get('ate_rmse', 0) for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(methods))]
    
    bars = ax.bar(methods, ate, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('ATE RMSE (m)')
    ax.set_title(f'Pose Estimation Accuracy - {dataset}')
    ax.tick_params(axis='x', rotation=45)
    
    # Highlight best
    if ate:
        best_idx = np.argmin(ate)
        bars[best_idx].set_edgecolor(COLORS['secondary'])
        bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _generate_pose_accuracy_plot(results: List[Dict], output_path: Path, dataset: str):
    """Generate pose accuracy scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, r in enumerate(results):
        ax.scatter(r.get('fps', 0), r.get('ate_rmse', 0),
                  s=100, color=COLOR_LIST[i % len(COLOR_LIST)], 
                  label=r['method'], edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('FPS (higher is better)')
    ax.set_ylabel('ATE RMSE (m) (lower is better)')
    ax.set_title(f'Accuracy vs Speed Trade-off - {dataset}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _generate_depth_metrics_bar(results: List[Dict], output_path: Path, dataset: str):
    """Generate depth metrics bar chart for a single dataset."""
    methods = [r['method'] for r in results]
    absrel = [r.get('abs_rel', 0) for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(methods))]
    
    bars = ax.bar(methods, absrel, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('AbsRel')
    ax.set_title(f'Depth Estimation Accuracy - {dataset}')
    ax.tick_params(axis='x', rotation=45)
    
    # Highlight best
    if absrel:
        best_idx = np.argmin(absrel)
        bars[best_idx].set_edgecolor(COLORS['secondary'])
        bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _generate_depth_threshold_plot(results: List[Dict], output_path: Path, dataset: str):
    """Generate depth threshold accuracy plot."""
    methods = [r['method'] for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    thresholds = ['delta1', 'delta2', 'delta3']
    x = np.arange(len(methods))
    width = 0.25
    
    for i, thresh in enumerate(thresholds):
        values = [r.get(thresh, 0) * 100 for r in results]
        ax.bar(x + i * width, values, width, label=f'δ < 1.25^{i+1}', 
               color=COLOR_LIST[i % len(COLOR_LIST)])
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'Depth Threshold Accuracy - {dataset}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _generate_gs_quality_bar(results: List[Dict], output_path: Path, dataset: str):
    """Generate GS quality bar chart for a single dataset."""
    engines = [r['engine'] for r in results]
    psnr = [r.get('psnr', 0) for r in results]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [COLOR_LIST[i % len(COLOR_LIST)] for i in range(len(engines))]
    
    bars = ax.bar(engines, psnr, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'Gaussian Splatting Quality - {dataset}')
    ax.tick_params(axis='x', rotation=45)
    
    # Highlight best
    if psnr:
        best_idx = np.argmax(psnr)
        bars[best_idx].set_edgecolor(COLORS['secondary'])
        bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _generate_gs_efficiency_plot(results: List[Dict], output_path: Path, dataset: str):
    """Generate GS efficiency scatter plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, r in enumerate(results):
        gaussians_k = r.get('final_gaussians', 0) / 1000
        ax.scatter(gaussians_k, r.get('psnr', 0),
                  s=150, color=COLOR_LIST[i % len(COLOR_LIST)],
                  label=r['engine'], edgecolor='white', linewidth=1)
    
    ax.set_xlabel('Gaussians (K)')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title(f'Quality vs Complexity - {dataset}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _generate_trajectory_plots(trajectory_data: Dict, output_dir: Path):
    """Generate 3D trajectory plots."""
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        logger.warning("3D plotting not available")
        return
    
    for method, data in trajectory_data.items():
        poses = data.get('poses', [])
        gt_poses = data.get('gt_poses', [])
        
        if not poses:
            continue
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Estimated trajectory
        poses_arr = np.array(poses)
        ax.plot(poses_arr[:, 0], poses_arr[:, 1], poses_arr[:, 2],
               'b-', linewidth=2, label=f'{method} (Estimated)')
        
        # Ground truth
        if gt_poses:
            gt_arr = np.array(gt_poses)
            ax.plot(gt_arr[:, 0], gt_arr[:, 1], gt_arr[:, 2],
                   'g-', linewidth=2, alpha=0.7, label='Ground Truth')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Trajectory - {method}')
        ax.legend()
        
        plt.savefig(traj_dir / f"{method}_trajectory.png")
        plt.close()
