"""
Visualization Utilities for Benchmarks
=======================================

Common plotting functions for benchmark visualization.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import warnings
from datetime import datetime

# Suppress matplotlib warnings about tick labels
warnings.filterwarnings('ignore', message='.*set_ticklabels.*')

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


# Color scheme for consistency - vibrant colors for better visibility
COLORS = {
    # Pose methods
    'orb': '#1f77b4',        # Blue
    'sift': '#ff7f0e',       # Orange
    'robust_flow': '#2ca02c', # Green
    'flow': '#d62728',       # Red
    'keyframe': '#9467bd',   # Purple
    'loftr': '#8c564b',      # Brown
    'superpoint': '#e377c2', # Pink
    
    # GS engines - vibrant distinct colors
    'graphdeco': '#e74c3c',  # Vibrant Red
    'gsplat': '#3498db',     # Vibrant Blue
    'splatam': '#2ecc71',    # Vibrant Green
    'monogs': '#9b59b6',     # Vibrant Purple
    'gslam': '#f39c12',      # Vibrant Orange
    'photoslam': '#1abc9c',  # Vibrant Teal
    'da3gs': '#e91e63',      # Vibrant Pink
    'da3': '#e91e63',        # Vibrant Pink
    
    # Depth methods - distinct vibrant colors
    'depth_anything_v3': '#e74c3c',    # Vibrant Red
    'depth_anything_v2': '#3498db',    # Vibrant Blue
    'depth_anything': '#3498db',       # Vibrant Blue
    'dav3': '#e74c3c',                 # Vibrant Red
    'dav2': '#3498db',                 # Vibrant Blue
    'midas': '#2ecc71',                # Vibrant Green
    'midas_small': '#2ecc71',          # Vibrant Green
    'midas_large': '#27ae60',          # Darker Green
    'zoedepth': '#9b59b6',             # Vibrant Purple
    'metric3d': '#f39c12',             # Vibrant Orange
    'stereo': '#1abc9c',               # Vibrant Teal
    'depth_pro': '#e91e63',            # Vibrant Pink
}

# Fallback color palette for unknown methods
FALLBACK_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', 
    '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b',
]
_fallback_idx = 0

def get_color(name: str) -> str:
    """Get consistent color for a method/engine."""
    global _fallback_idx
    name_lower = name.lower()
    if name_lower in COLORS:
        return COLORS[name_lower]
    # Use fallback colors for unknown methods
    color = FALLBACK_COLORS[_fallback_idx % len(FALLBACK_COLORS)]
    _fallback_idx += 1
    COLORS[name_lower] = color  # Remember for consistency
    return color


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


def save_figure(fig, path: Path, formats: List[str] = ['png']):
    """Save figure to PNG format."""
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
        ax.set_xticks(range(len(methods)))
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
# LATENCY PLOTS
# =============================================================================

def plot_latency_comparison(
    results: List[Dict[str, Any]],
    benchmark_type: str = 'pose',  # 'pose', 'depth', or 'gs'
    title: str = "Latency Comparison",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Plot latency comparison bar chart with p95/p99 markers.
    
    Args:
        results: List of benchmark result dicts with latency metrics
        benchmark_type: Type of benchmark ('pose', 'depth', 'gs')
        title: Plot title
        output_path: Path to save figure
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Determine method/engine key
    if benchmark_type == 'gs':
        name_key = 'engine'
    else:
        name_key = 'method'
    
    names = sorted(set(r.get(name_key, 'unknown') for r in results))
    
    # Left: Average latency with min/max whiskers
    ax1 = axes[0]
    avg_latencies = []
    min_latencies = []
    max_latencies = []
    colors = []
    
    for name in names:
        name_results = [r for r in results if r.get(name_key) == name]
        if name_results:
            avg = np.mean([r.get('avg_latency_ms', 0) for r in name_results])
            min_val = np.mean([r.get('min_latency_ms', 0) for r in name_results])
            max_val = np.mean([r.get('max_latency_ms', 0) for r in name_results])
            avg_latencies.append(avg)
            min_latencies.append(min_val)
            max_latencies.append(max_val)
            colors.append(get_color(name))
    
    x = np.arange(len(names))
    bars = ax1.bar(x, avg_latencies, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add error bars for min/max
    yerr_lower = [avg - min_val for avg, min_val in zip(avg_latencies, min_latencies)]
    yerr_upper = [max_val - avg for avg, max_val in zip(avg_latencies, max_latencies)]
    ax1.errorbar(x, avg_latencies, yerr=[yerr_lower, yerr_upper], fmt='none', color='black', capsize=5)
    
    ax1.set_ylabel('Latency (ms)')
    ax1.set_title('Average Latency (with min/max range)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars, avg_latencies):
        height = bar.get_height()
        ax1.annotate(f'{val:.1f}ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    # Right: FPS comparison
    ax2 = axes[1]
    fps_values = []
    
    for name in names:
        name_results = [r for r in results if r.get(name_key) == name]
        if name_results:
            fps = np.mean([r.get('fps', 0) for r in name_results])
            fps_values.append(fps)
    
    bars2 = ax2.bar(x, fps_values, color=colors, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    # Add real-time threshold line at 30 FPS
    ax2.axhline(y=30, color='red', linestyle='--', linewidth=1.5, label='30 FPS (real-time)')
    ax2.axhline(y=60, color='orange', linestyle='--', linewidth=1.5, label='60 FPS')
    
    ax2.set_ylabel('FPS')
    ax2.set_title('Processing Speed')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, val in zip(bars2, fps_values):
        height = bar.get_height()
        ax2.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_latency_percentiles(
    results: List[Dict[str, Any]],
    benchmark_type: str = 'pose',
    title: str = "Latency Percentiles",
    output_path: Optional[Path] = None,
) -> Optional[plt.Figure]:
    """
    Plot latency percentile comparison (p50, p95, p99).
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Determine method/engine key
    if benchmark_type == 'gs':
        name_key = 'engine'
    else:
        name_key = 'method'
    
    names = sorted(set(r.get(name_key, 'unknown') for r in results))
    x = np.arange(len(names))
    width = 0.25
    
    # Collect p50 (avg), p95, p99
    p50_values = []
    p95_values = []
    p99_values = []
    
    for name in names:
        name_results = [r for r in results if r.get(name_key) == name]
        if name_results:
            p50_values.append(np.mean([r.get('avg_latency_ms', 0) for r in name_results]))
            p95_values.append(np.mean([r.get('p95_latency_ms', 0) for r in name_results]))
            p99_values.append(np.mean([r.get('p99_latency_ms', 0) for r in name_results]))
    
    bars1 = ax.bar(x - width, p50_values, width, label='p50 (avg)', color='#2ca02c', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, p95_values, width, label='p95', color='#ff7f0e', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, p99_values, width, label='p99', color='#d62728', edgecolor='black', linewidth=0.5)
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
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


def plot_depth_comparison_grid(
    rgb_images: List[np.ndarray],
    gt_depths: List[np.ndarray],
    pred_depths: Dict[str, List[np.ndarray]],
    output_path: Path = None,
    title: str = "Depth Estimation Comparison",
    num_samples: int = 3,
):
    """
    Create a grid comparing RGB, GT depth, and predicted depths from multiple methods.
    
    Args:
        rgb_images: List of RGB images
        gt_depths: List of ground truth depth maps
        pred_depths: Dict mapping method names to list of predicted depths
        output_path: Where to save the figure
        title: Plot title
        num_samples: Number of sample frames to show
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    methods = list(pred_depths.keys())
    n_methods = len(methods)
    n_cols = 2 + n_methods  # RGB + GT + predictions
    
    # Select sample indices
    n_frames = len(rgb_images)
    indices = np.linspace(0, n_frames - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(num_samples, n_cols, figsize=(4 * n_cols, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for row, idx in enumerate(indices):
        # RGB image
        axes[row, 0].imshow(rgb_images[idx])
        axes[row, 0].set_title('RGB' if row == 0 else '')
        axes[row, 0].axis('off')
        
        # Ground truth depth
        gt = gt_depths[idx]
        vmin, vmax = np.nanpercentile(gt[gt > 0], [5, 95]) if np.any(gt > 0) else (0, 1)
        im = axes[row, 1].imshow(gt, cmap='turbo', vmin=vmin, vmax=vmax)
        axes[row, 1].set_title('Ground Truth' if row == 0 else '')
        axes[row, 1].axis('off')
        
        # Predicted depths
        for col, method in enumerate(methods, start=2):
            pred = pred_depths[method][idx]
            # Scale prediction to match GT range
            if np.any(pred > 0) and np.any(gt > 0):
                valid = (pred > 0) & (gt > 0)
                if valid.sum() > 100:
                    scale = np.median(gt[valid]) / np.median(pred[valid])
                    pred = pred * scale
            
            axes[row, col].imshow(pred, cmap='turbo', vmin=vmin, vmax=vmax)
            axes[row, col].set_title(method if row == 0 else '')
            axes[row, col].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Depth (m)')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_depth_error_maps(
    gt_depths: List[np.ndarray],
    pred_depths: Dict[str, List[np.ndarray]],
    output_path: Path = None,
    title: str = "Depth Error Maps",
    num_samples: int = 2,
):
    """
    Create error maps showing where each method makes mistakes.
    
    Args:
        gt_depths: List of ground truth depth maps
        pred_depths: Dict mapping method names to list of predicted depths
        output_path: Where to save the figure
        title: Plot title
        num_samples: Number of sample frames to show
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    methods = list(pred_depths.keys())
    n_methods = len(methods)
    
    # Select sample indices
    n_frames = len(gt_depths)
    indices = np.linspace(0, n_frames - 1, num_samples, dtype=int)
    
    fig, axes = plt.subplots(num_samples, n_methods, figsize=(4 * n_methods, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if n_methods == 1:
        axes = axes.reshape(-1, 1)
    
    for row, idx in enumerate(indices):
        gt = gt_depths[idx]
        
        for col, method in enumerate(methods):
            pred = pred_depths[method][idx]
            
            # Compute relative error
            valid = (pred > 0.1) & (gt > 0.1)
            if valid.sum() > 100:
                # Scale prediction to match GT
                scale = np.median(gt[valid]) / np.median(pred[valid])
                pred_scaled = pred * scale
                
                # Relative error
                error = np.abs(pred_scaled - gt) / gt
                error[~valid] = np.nan
            else:
                error = np.zeros_like(gt)
            
            im = axes[row, col].imshow(error, cmap='hot', vmin=0, vmax=0.5)
            axes[row, col].set_title(f'{method}' if row == 0 else '')
            axes[row, col].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label='Relative Error')
    
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_trajectory_3d(
    gt_poses: np.ndarray,
    estimated_poses: Dict[str, np.ndarray] = None,
    output_path: Path = None,
    title: str = "3D Trajectory Comparison",
):
    """
    Create 3D plot comparing ground truth and estimated trajectories.
    
    Args:
        gt_poses: Ground truth poses (N, 4, 4)
        estimated_poses: Dict mapping method names to estimated poses (N, 4, 4)
        output_path: Where to save the figure
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract translations from poses
    gt_trans = gt_poses[:, :3, 3]
    
    # Plot ground truth
    ax.plot(gt_trans[:, 0], gt_trans[:, 1], gt_trans[:, 2], 
            'k-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax.scatter(gt_trans[0, 0], gt_trans[0, 1], gt_trans[0, 2], 
               c='green', s=100, marker='o', label='Start')
    ax.scatter(gt_trans[-1, 0], gt_trans[-1, 1], gt_trans[-1, 2], 
               c='red', s=100, marker='s', label='End')
    
    # Plot estimated trajectories
    if estimated_poses:
        for method, poses in estimated_poses.items():
            if poses is None or len(poses) == 0:
                continue
            trans = poses[:, :3, 3]
            color = get_color(method)
            ax.plot(trans[:, 0], trans[:, 1], trans[:, 2], 
                    color=color, linewidth=1.5, label=method, alpha=0.7)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    
    # Equal aspect ratio
    max_range = np.array([gt_trans[:, 0].max() - gt_trans[:, 0].min(),
                          gt_trans[:, 1].max() - gt_trans[:, 1].min(),
                          gt_trans[:, 2].max() - gt_trans[:, 2].min()]).max() / 2.0
    mid_x = (gt_trans[:, 0].max() + gt_trans[:, 0].min()) * 0.5
    mid_y = (gt_trans[:, 1].max() + gt_trans[:, 1].min()) * 0.5
    mid_z = (gt_trans[:, 2].max() + gt_trans[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_trajectory_2d_views(
    gt_poses: np.ndarray,
    estimated_poses: Dict[str, np.ndarray] = None,
    output_path: Path = None,
    title: str = "Trajectory 2D Views",
):
    """
    Create 2D projections of trajectories (XY, XZ, YZ views).
    
    Args:
        gt_poses: Ground truth poses (N, 4, 4)
        estimated_poses: Dict mapping method names to estimated poses (N, 4, 4)
        output_path: Where to save the figure
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Extract translations
    gt_trans = gt_poses[:, :3, 3]
    
    views = [
        (0, 1, 'XY View (Top-down)', 'X (m)', 'Y (m)'),
        (0, 2, 'XZ View (Side)', 'X (m)', 'Z (m)'),
        (1, 2, 'YZ View (Front)', 'Y (m)', 'Z (m)'),
    ]
    
    for ax, (i, j, view_title, xlabel, ylabel) in zip(axes, views):
        # Ground truth
        ax.plot(gt_trans[:, i], gt_trans[:, j], 'k-', linewidth=2, 
                label='Ground Truth', alpha=0.8)
        ax.scatter(gt_trans[0, i], gt_trans[0, j], c='green', s=100, 
                   marker='o', zorder=5)
        ax.scatter(gt_trans[-1, i], gt_trans[-1, j], c='red', s=100, 
                   marker='s', zorder=5)
        
        # Estimated trajectories
        if estimated_poses:
            for method, poses in estimated_poses.items():
                if poses is None or len(poses) == 0:
                    continue
                trans = poses[:, :3, 3]
                color = get_color(method)
                ax.plot(trans[:, i], trans[:, j], color=color, 
                        linewidth=1.5, label=method, alpha=0.7)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(view_title)
        ax.axis('equal')
        ax.grid(True, alpha=0.3)
    
    axes[0].legend(loc='upper left')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_ate_over_trajectory(
    gt_poses: np.ndarray,
    estimated_poses: Dict[str, np.ndarray],
    output_path: Path = None,
    title: str = "Position Error Along Trajectory",
):
    """
    Plot ATE error at each point along the trajectory.
    
    Args:
        gt_poses: Ground truth poses (N, 4, 4)
        estimated_poses: Dict mapping method names to estimated poses (N, 4, 4)
        output_path: Where to save the figure
        title: Plot title
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    gt_trans = gt_poses[:, :3, 3]
    
    # Compute cumulative distance along GT trajectory
    distances = np.zeros(len(gt_trans))
    for i in range(1, len(gt_trans)):
        distances[i] = distances[i-1] + np.linalg.norm(gt_trans[i] - gt_trans[i-1])
    
    for method, poses in estimated_poses.items():
        if poses is None or len(poses) == 0:
            continue
        
        est_trans = poses[:, :3, 3]
        
        # Align trajectories (simple translation alignment)
        offset = gt_trans[0] - est_trans[0]
        est_trans_aligned = est_trans + offset
        
        # Compute per-frame error
        errors = np.linalg.norm(est_trans_aligned - gt_trans, axis=1)
        
        color = get_color(method)
        ax.plot(distances, errors, color=color, linewidth=1.5, label=method, alpha=0.8)
    
    ax.set_xlabel('Distance Along Trajectory (m)')
    ax.set_ylabel('Position Error (m)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, distances[-1])
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_depth_scatter(
    gt_depth: np.ndarray,
    pred_depth: np.ndarray,
    method_name: str = "Prediction",
    output_path: Path = None,
    title: str = None,
    max_points: int = 10000,
):
    """
    Create scatter plot of predicted vs ground truth depth values.
    
    Args:
        gt_depth: Ground truth depth (H, W)
        pred_depth: Predicted depth (H, W)
        method_name: Name of the prediction method
        output_path: Where to save the figure
        title: Plot title
        max_points: Maximum number of points to plot
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    # Get valid points
    valid = (gt_depth > 0.1) & (pred_depth > 0.1) & np.isfinite(gt_depth) & np.isfinite(pred_depth)
    gt_valid = gt_depth[valid].flatten()
    pred_valid = pred_depth[valid].flatten()
    
    if len(gt_valid) == 0:
        return None
    
    # Scale prediction to match GT
    scale = np.median(gt_valid) / np.median(pred_valid)
    pred_valid = pred_valid * scale
    
    # Subsample if too many points
    if len(gt_valid) > max_points:
        idx = np.random.choice(len(gt_valid), max_points, replace=False)
        gt_valid = gt_valid[idx]
        pred_valid = pred_valid[idx]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 2D histogram for density visualization
    h = ax.hist2d(gt_valid, pred_valid, bins=100, cmap='Blues', 
                   norm=plt.matplotlib.colors.LogNorm())
    plt.colorbar(h[3], ax=ax, label='Count')
    
    # Perfect prediction line
    max_val = max(gt_valid.max(), pred_valid.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect')
    
    # ±10% and ±25% error bands
    ax.fill_between([0, max_val], [0, max_val * 0.75], [0, max_val * 1.25], 
                    alpha=0.1, color='green', label='±25%')
    ax.fill_between([0, max_val], [0, max_val * 0.9], [0, max_val * 1.1], 
                    alpha=0.2, color='green', label='±10%')
    
    ax.set_xlabel('Ground Truth Depth (m)')
    ax.set_ylabel(f'{method_name} Depth (m)')
    ax.set_title(title or f'{method_name} vs Ground Truth')
    ax.legend(loc='upper left')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal')
    
    # Add statistics
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    ax.text(0.95, 0.05, f'AbsRel: {abs_rel:.4f}\nRMSE: {rmse:.4f}m', 
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig


def plot_method_radar(
    results: List[Dict],
    metrics: List[str],
    method_key: str = 'method',
    output_path: Path = None,
    title: str = "Method Comparison Radar",
    lower_is_better: List[bool] = None,
):
    """
    Create radar/spider chart comparing methods across multiple metrics.
    
    Args:
        results: List of result dictionaries
        metrics: List of metric names to compare
        method_key: Key for method name in results ('method' or 'engine')
        output_path: Where to save the figure
        title: Plot title
        lower_is_better: List of bools indicating if lower is better for each metric
    """
    if not HAS_MATPLOTLIB:
        return None
    
    setup_plot_style()
    
    # Get unique methods
    methods = sorted(set(r[method_key] for r in results))
    n_metrics = len(metrics)
    
    if lower_is_better is None:
        lower_is_better = [True] * n_metrics
    
    # Compute average values for each method
    values = {}
    for method in methods:
        method_results = [r for r in results if r[method_key] == method]
        values[method] = []
        for metric in metrics:
            vals = [r.get(metric, 0) for r in method_results]
            values[method].append(np.mean(vals) if vals else 0)
    
    # Normalize values (0-1 scale, higher is better)
    normalized = {}
    for method in methods:
        normalized[method] = []
        for i, (val, lib) in enumerate(zip(values[method], lower_is_better)):
            all_vals = [values[m][i] for m in methods]
            min_val, max_val = min(all_vals), max(all_vals)
            if max_val - min_val > 0:
                norm_val = (val - min_val) / (max_val - min_val)
                if lib:  # Lower is better, invert
                    norm_val = 1 - norm_val
            else:
                norm_val = 0.5
            normalized[method].append(norm_val)
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for method in methods:
        vals = normalized[method] + normalized[method][:1]
        color = get_color(method)
        ax.plot(angles, vals, color=color, linewidth=2, label=method)
        ax.fill(angles, vals, color=color, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()
    
    if output_path:
        save_figure(fig, output_path)
    
    return fig
