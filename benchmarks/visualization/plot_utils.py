"""
Visualization Utilities for Benchmarks
=======================================

Common plotting functions for benchmark visualization.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Install with: pip install matplotlib")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# Color scheme for consistency
COLORS = {
    'orb': '#1f77b4',
    'sift': '#ff7f0e',
    'robust_flow': '#2ca02c',
    'flow': '#d62728',
    'keyframe': '#9467bd',
    'loftr': '#8c564b',
    'superpoint': '#e377c2',
    'graphdeco': '#7f7f7f',
    'gsplat': '#bcbd22',
    'splatam': '#17becf',
    'monogs': '#ff9896',
    'depth_anything': '#1f77b4',
    'midas': '#ff7f0e',
    'zoedepth': '#2ca02c',
}

def get_color(name: str) -> str:
    """Get consistent color for a method/engine."""
    return COLORS.get(name.lower(), '#333333')


def setup_plot_style():
    """Set up consistent plot styling."""
    if not HAS_MATPLOTLIB:
        return
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'seaborn-whitegrid')
    
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'figure.dpi': 100,
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
    })


def save_figure(fig, path: Path, formats: List[str] = ['png', 'pdf']):
    """Save figure in multiple formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        fig.savefig(path.with_suffix(f'.{fmt}'), bbox_inches='tight', dpi=150)
    
    plt.close(fig)


# =============================================================================
# POSE ESTIMATION PLOTS
# =============================================================================

def plot_trajectory_comparison(
    trajectories: Dict[str, np.ndarray],
    ground_truth: np.ndarray,
    title: str = "Trajectory Comparison",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Plot 2D trajectory comparison (top-down view).
    
    Args:
        trajectories: Dict mapping method name to Nx3 positions
        ground_truth: Nx3 ground truth positions
        title: Plot title
        output_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top-down view (XZ plane)
    ax1 = axes[0]
    ax1.plot(ground_truth[:, 0], ground_truth[:, 2], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    
    for name, traj in trajectories.items():
        color = get_color(name)
        ax1.plot(traj[:, 0], traj[:, 2], '-', color=color, linewidth=1.5, label=name, alpha=0.7)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Z (m)')
    ax1.set_title('Top-Down View (XZ)')
    ax1.legend(loc='best')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Side view (XY plane)
    ax2 = axes[1]
    ax2.plot(ground_truth[:, 0], ground_truth[:, 1], 'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    
    for name, traj in trajectories.items():
        color = get_color(name)
        ax2.plot(traj[:, 0], traj[:, 1], '-', color=color, linewidth=1.5, label=name, alpha=0.7)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Side View (XY)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_pose_error_over_time(
    errors: Dict[str, np.ndarray],
    title: str = "Trajectory Error Over Time",
    ylabel: str = "Error (m)",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Plot pose error evolution over time.
    
    Args:
        errors: Dict mapping method name to error array
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, err in errors.items():
        color = get_color(name)
        frames = np.arange(len(err))
        ax.plot(frames, err, '-', color=color, linewidth=1.5, label=name, alpha=0.8)
    
    ax.set_xlabel('Frame')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_pose_metrics_bar(
    results: List[Dict[str, Any]],
    metrics: List[str] = ['ate_rmse', 'rpe_trans_rmse', 'fps'],
    title: str = "Pose Estimation Comparison",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Plot bar chart comparing pose estimation metrics.
    
    Args:
        results: List of benchmark result dicts
        metrics: Metrics to plot
        title: Plot title
        output_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    methods = sorted(set(r['method'] for r in results))
    
    metric_labels = {
        'ate_rmse': 'ATE RMSE (m)',
        'ate_mean': 'ATE Mean (m)',
        'rpe_trans_rmse': 'RPE Trans (m/frame)',
        'rpe_rot_rmse': 'RPE Rot (°/frame)',
        'fps': 'Speed (FPS)',
        'avg_inliers': 'Avg Inliers',
    }
    
    for ax, metric in zip(axes, metrics):
        values = []
        colors = []
        
        for method in methods:
            method_results = [r for r in results if r['method'] == method]
            if method_results:
                avg = np.mean([r[metric] for r in method_results])
                values.append(avg)
                colors.append(get_color(method))
            else:
                values.append(0)
                colors.append('#cccccc')
        
        bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric))
        
        # Rotate x labels
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}' if val < 10 else f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# =============================================================================
# DEPTH ESTIMATION PLOTS
# =============================================================================

def plot_depth_error_heatmap(
    error_map: np.ndarray,
    title: str = "Depth Error Heatmap",
    output_path: Optional[Path] = None,
    vmax: float = None,
) -> Optional[plt.Figure]:
    """
    Plot depth error heatmap.
    
    Args:
        error_map: HxW error values
        title: Plot title
        output_path: Path to save figure
        vmax: Maximum value for colormap
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if vmax is None:
        vmax = np.percentile(error_map[np.isfinite(error_map)], 95)
    
    im = ax.imshow(error_map, cmap='hot', vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Error (m)')
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_depth_metrics_bar(
    results: List[Dict[str, Any]],
    metrics: List[str] = ['abs_rel', 'rmse', 'delta1'],
    title: str = "Depth Estimation Comparison",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Plot bar chart comparing depth estimation metrics.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    methods = sorted(set(r['method'] for r in results))
    
    metric_labels = {
        'abs_rel': 'Abs Rel ↓',
        'sq_rel': 'Sq Rel ↓',
        'rmse': 'RMSE (m) ↓',
        'rmse_log': 'RMSE log ↓',
        'delta1': 'δ < 1.25 ↑',
        'delta2': 'δ < 1.25² ↑',
        'delta3': 'δ < 1.25³ ↑',
        'fps': 'Speed (FPS) ↑',
    }
    
    for ax, metric in zip(axes, metrics):
        values = []
        colors = []
        
        for method in methods:
            method_results = [r for r in results if r['method'] == method]
            if method_results:
                avg = np.mean([r.get(metric, 0) for r in method_results])
                values.append(avg)
                colors.append(get_color(method))
            else:
                values.append(0)
                colors.append('#cccccc')
        
        bars = ax.bar(methods, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_depth_comparison(
    rgb: np.ndarray,
    gt_depth: np.ndarray,
    predicted_depths: Dict[str, np.ndarray],
    title: str = "Depth Comparison",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Plot side-by-side depth comparison.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    n_methods = len(predicted_depths)
    fig, axes = plt.subplots(2, n_methods + 2, figsize=(4 * (n_methods + 2), 8))
    
    # RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Input')
    axes[0, 0].axis('off')
    
    # Ground truth
    vmax = np.percentile(gt_depth[gt_depth > 0], 95)
    axes[0, 1].imshow(gt_depth, cmap='plasma', vmin=0, vmax=vmax)
    axes[0, 1].set_title('Ground Truth')
    axes[0, 1].axis('off')
    
    # Predictions
    for i, (name, pred) in enumerate(predicted_depths.items()):
        axes[0, i + 2].imshow(pred, cmap='plasma', vmin=0, vmax=vmax)
        axes[0, i + 2].set_title(name)
        axes[0, i + 2].axis('off')
        
        # Error map
        error = np.abs(pred - gt_depth)
        error[gt_depth == 0] = 0
        axes[1, i + 2].imshow(error, cmap='hot', vmin=0, vmax=vmax * 0.3)
        axes[1, i + 2].set_title(f'{name} Error')
        axes[1, i + 2].axis('off')
    
    # Hide unused subplots
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# =============================================================================
# GAUSSIAN SPLATTING PLOTS
# =============================================================================

def plot_gs_training_curves(
    training_data: Dict[str, Dict[str, List[float]]],
    title: str = "3DGS Training Curves",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Plot Gaussian splatting training curves.
    
    Args:
        training_data: Dict mapping engine name to dict with 'loss', 'psnr', 'gaussians' lists
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Loss curve
    ax1 = axes[0]
    for name, data in training_data.items():
        if 'loss' in data:
            color = get_color(name)
            ax1.plot(data['loss'], '-', color=color, linewidth=1.5, label=name, alpha=0.8)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend(loc='best')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)
    
    # PSNR curve
    ax2 = axes[1]
    for name, data in training_data.items():
        if 'psnr' in data:
            color = get_color(name)
            ax2.plot(data['psnr'], '-', color=color, linewidth=1.5, label=name, alpha=0.8)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('Reconstruction Quality (PSNR)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Gaussians count
    ax3 = axes[2]
    for name, data in training_data.items():
        if 'gaussians' in data:
            color = get_color(name)
            ax3.plot(data['gaussians'], '-', color=color, linewidth=1.5, label=name, alpha=0.8)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('# Gaussians')
    ax3.set_title('Gaussian Count')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_gs_metrics_bar(
    results: List[Dict[str, Any]],
    metrics: List[str] = ['psnr', 'ssim', 'fps'],
    title: str = "3DGS Engine Comparison",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Plot bar chart comparing 3DGS engine metrics.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    
    if n_metrics == 1:
        axes = [axes]
    
    engines = sorted(set(r['engine'] for r in results))
    
    metric_labels = {
        'psnr': 'PSNR (dB) ↑',
        'ssim': 'SSIM ↑',
        'lpips': 'LPIPS ↓',
        'fps': 'Render FPS ↑',
        'train_time': 'Train Time (s) ↓',
        'gaussians': '# Gaussians',
        'memory_mb': 'Memory (MB) ↓',
    }
    
    for ax, metric in zip(axes, metrics):
        values = []
        colors = []
        
        for engine in engines:
            engine_results = [r for r in results if r['engine'] == engine]
            if engine_results:
                avg = np.mean([r.get(metric, 0) for r in engine_results])
                values.append(avg)
                colors.append(get_color(engine))
            else:
                values.append(0)
                colors.append('#cccccc')
        
        bars = ax.bar(engines, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel(metric_labels.get(metric, metric))
        ax.set_title(metric_labels.get(metric, metric))
        ax.set_xticklabels(engines, rotation=45, ha='right')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}' if val < 100 else f'{val:.0f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_gs_render_comparison(
    renders: Dict[str, np.ndarray],
    ground_truth: np.ndarray,
    title: str = "Render Comparison",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Plot side-by-side render comparison.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    n_engines = len(renders)
    fig, axes = plt.subplots(2, n_engines + 1, figsize=(4 * (n_engines + 1), 8))
    
    # Ground truth
    axes[0, 0].imshow(ground_truth)
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Renders
    for i, (name, render) in enumerate(renders.items()):
        axes[0, i + 1].imshow(render)
        axes[0, i + 1].set_title(name)
        axes[0, i + 1].axis('off')
        
        # Error visualization
        error = np.mean(np.abs(render.astype(float) - ground_truth.astype(float)), axis=2)
        axes[1, i + 1].imshow(error, cmap='hot', vmin=0, vmax=50)
        axes[1, i + 1].set_title(f'{name} Error')
        axes[1, i + 1].axis('off')
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


# =============================================================================
# COMBINED / SUMMARY PLOTS
# =============================================================================

def plot_overall_summary(
    pose_results: List[Dict] = None,
    depth_results: List[Dict] = None,
    gs_results: List[Dict] = None,
    title: str = "AirSplatMap Benchmark Summary",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Create an overall summary figure with all benchmarks.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    # Count available results
    n_sections = sum([
        pose_results is not None and len(pose_results) > 0,
        depth_results is not None and len(depth_results) > 0,
        gs_results is not None and len(gs_results) > 0,
    ])
    
    if n_sections == 0:
        return None
    
    fig, axes = plt.subplots(n_sections, 2, figsize=(14, 5 * n_sections))
    
    if n_sections == 1:
        axes = axes.reshape(1, -1)
    
    row = 0
    
    # Pose results
    if pose_results and len(pose_results) > 0:
        methods = sorted(set(r['method'] for r in pose_results))
        
        # ATE bar
        ate_values = [np.mean([r['ate_rmse'] for r in pose_results if r['method'] == m]) for m in methods]
        colors = [get_color(m) for m in methods]
        axes[row, 0].bar(methods, ate_values, color=colors, edgecolor='black')
        axes[row, 0].set_ylabel('ATE RMSE (m)')
        axes[row, 0].set_title('Pose: Trajectory Error')
        axes[row, 0].set_xticklabels(methods, rotation=45, ha='right')
        
        # FPS bar
        fps_values = [np.mean([r['fps'] for r in pose_results if r['method'] == m]) for m in methods]
        axes[row, 1].bar(methods, fps_values, color=colors, edgecolor='black')
        axes[row, 1].set_ylabel('FPS')
        axes[row, 1].set_title('Pose: Speed')
        axes[row, 1].set_xticklabels(methods, rotation=45, ha='right')
        
        row += 1
    
    # Depth results
    if depth_results and len(depth_results) > 0:
        methods = sorted(set(r['method'] for r in depth_results))
        
        # AbsRel bar
        absrel_values = [np.mean([r.get('abs_rel', 0) for r in depth_results if r['method'] == m]) for m in methods]
        colors = [get_color(m) for m in methods]
        axes[row, 0].bar(methods, absrel_values, color=colors, edgecolor='black')
        axes[row, 0].set_ylabel('Abs Rel')
        axes[row, 0].set_title('Depth: Accuracy')
        axes[row, 0].set_xticklabels(methods, rotation=45, ha='right')
        
        # FPS bar
        fps_values = [np.mean([r.get('fps', 0) for r in depth_results if r['method'] == m]) for m in methods]
        axes[row, 1].bar(methods, fps_values, color=colors, edgecolor='black')
        axes[row, 1].set_ylabel('FPS')
        axes[row, 1].set_title('Depth: Speed')
        axes[row, 1].set_xticklabels(methods, rotation=45, ha='right')
        
        row += 1
    
    # GS results
    if gs_results and len(gs_results) > 0:
        engines = sorted(set(r['engine'] for r in gs_results))
        
        # PSNR bar
        psnr_values = [np.mean([r.get('psnr', 0) for r in gs_results if r['engine'] == e]) for e in engines]
        colors = [get_color(e) for e in engines]
        axes[row, 0].bar(engines, psnr_values, color=colors, edgecolor='black')
        axes[row, 0].set_ylabel('PSNR (dB)')
        axes[row, 0].set_title('3DGS: Quality')
        axes[row, 0].set_xticklabels(engines, rotation=45, ha='right')
        
        # FPS bar
        fps_values = [np.mean([r.get('fps', 0) for r in gs_results if r['engine'] == e]) for e in engines]
        axes[row, 1].bar(engines, fps_values, color=colors, edgecolor='black')
        axes[row, 1].set_ylabel('FPS')
        axes[row, 1].set_title('3DGS: Render Speed')
        axes[row, 1].set_xticklabels(engines, rotation=45, ha='right')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def generate_html_report(
    pose_results: List[Dict] = None,
    depth_results: List[Dict] = None,
    gs_results: List[Dict] = None,
    output_path: Path = None,
    title: str = "AirSplatMap Benchmark Report",
):
    """
    Generate interactive HTML benchmark report.
    """
    if output_path is None:
        output_path = Path("benchmarks/results/report.html")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 20px; background: #0d1117; color: #e6edf3; }}
        h1 {{ color: #58a6ff; border-bottom: 1px solid #30363d; padding-bottom: 10px; }}
        h2 {{ color: #7ee787; margin-top: 30px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin: 20px 0; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 20px; }}
        .card h3 {{ margin-top: 0; color: #e6edf3; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #30363d; }}
        th {{ background: #21262d; color: #7ee787; }}
        tr:hover {{ background: #21262d; }}
        .best {{ color: #3fb950; font-weight: bold; }}
        .chart-container {{ height: 300px; }}
        .timestamp {{ color: #7d8590; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 {title}</h1>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""
    
    # Pose results section
    if pose_results and len(pose_results) > 0:
        html += """
        <h2>📍 Pose Estimation</h2>
        <div class="grid">
            <div class="card">
                <h3>Results Table</h3>
                <table>
                    <tr><th>Method</th><th>ATE RMSE</th><th>RPE Trans</th><th>FPS</th><th>Inliers</th></tr>
"""
        # Find best values
        best_ate = min(r['ate_rmse'] for r in pose_results)
        best_fps = max(r['fps'] for r in pose_results)
        
        for r in sorted(pose_results, key=lambda x: x['ate_rmse']):
            ate_class = 'best' if r['ate_rmse'] == best_ate else ''
            fps_class = 'best' if r['fps'] == best_fps else ''
            html += f"""
                    <tr>
                        <td>{r['method']}</td>
                        <td class="{ate_class}">{r['ate_rmse']:.4f}m</td>
                        <td>{r['rpe_trans_rmse']:.4f}m</td>
                        <td class="{fps_class}">{r['fps']:.1f}</td>
                        <td>{r['avg_inliers']:.0f}</td>
                    </tr>
"""
        html += """
                </table>
            </div>
            <div class="card">
                <h3>ATE Comparison</h3>
                <div class="chart-container"><canvas id="poseChart"></canvas></div>
            </div>
        </div>
"""
    
    # Depth results section
    if depth_results and len(depth_results) > 0:
        html += """
        <h2>🎯 Depth Estimation</h2>
        <div class="grid">
            <div class="card">
                <h3>Results Table</h3>
                <table>
                    <tr><th>Method</th><th>Abs Rel</th><th>RMSE</th><th>δ&lt;1.25</th><th>FPS</th></tr>
"""
        for r in sorted(depth_results, key=lambda x: x.get('abs_rel', 999)):
            html += f"""
                    <tr>
                        <td>{r['method']}</td>
                        <td>{r.get('abs_rel', 0):.4f}</td>
                        <td>{r.get('rmse', 0):.4f}m</td>
                        <td>{r.get('delta1', 0):.3f}</td>
                        <td>{r.get('fps', 0):.1f}</td>
                    </tr>
"""
        html += """
                </table>
            </div>
            <div class="card">
                <h3>Accuracy Comparison</h3>
                <div class="chart-container"><canvas id="depthChart"></canvas></div>
            </div>
        </div>
"""
    
    # GS results section
    if gs_results and len(gs_results) > 0:
        html += """
        <h2>✨ Gaussian Splatting</h2>
        <div class="grid">
            <div class="card">
                <h3>Results Table</h3>
                <table>
                    <tr><th>Engine</th><th>PSNR</th><th>SSIM</th><th>Time</th><th>#Gaussians</th></tr>
"""
        for r in sorted(gs_results, key=lambda x: -x.get('psnr', 0)):
            html += f"""
                    <tr>
                        <td>{r['engine']}</td>
                        <td>{r.get('psnr', 0):.2f} dB</td>
                        <td>{r.get('ssim', 0):.4f}</td>
                        <td>{r.get('train_time', 0):.1f}s</td>
                        <td>{r.get('gaussians', 0):,}</td>
                    </tr>
"""
        html += """
                </table>
            </div>
            <div class="card">
                <h3>Quality Comparison</h3>
                <div class="chart-container"><canvas id="gsChart"></canvas></div>
            </div>
        </div>
"""
    
    # Add Chart.js scripts
    html += """
    <script>
"""
    
    if pose_results and len(pose_results) > 0:
        methods = sorted(set(r['method'] for r in pose_results))
        ate_values = [np.mean([r['ate_rmse'] for r in pose_results if r['method'] == m]) for m in methods]
        html += f"""
        new Chart(document.getElementById('poseChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(methods)},
                datasets: [{{ label: 'ATE RMSE (m)', data: {json.dumps(ate_values)}, backgroundColor: '#58a6ff' }}]
            }},
            options: {{ responsive: true, maintainAspectRatio: false }}
        }});
"""
    
    if depth_results and len(depth_results) > 0:
        methods = sorted(set(r['method'] for r in depth_results))
        absrel_values = [np.mean([r.get('abs_rel', 0) for r in depth_results if r['method'] == m]) for m in methods]
        html += f"""
        new Chart(document.getElementById('depthChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(methods)},
                datasets: [{{ label: 'Abs Rel', data: {json.dumps(absrel_values)}, backgroundColor: '#3fb950' }}]
            }},
            options: {{ responsive: true, maintainAspectRatio: false }}
        }});
"""
    
    if gs_results and len(gs_results) > 0:
        engines = sorted(set(r['engine'] for r in gs_results))
        psnr_values = [np.mean([r.get('psnr', 0) for r in gs_results if r['engine'] == e]) for e in engines]
        html += f"""
        new Chart(document.getElementById('gsChart'), {{
            type: 'bar',
            data: {{
                labels: {json.dumps(engines)},
                datasets: [{{ label: 'PSNR (dB)', data: {json.dumps(psnr_values)}, backgroundColor: '#d29922' }}]
            }},
            options: {{ responsive: true, maintainAspectRatio: false }}
        }});
"""
    
    html += """
    </script>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"HTML report saved to: {output_path}")
    return output_path
