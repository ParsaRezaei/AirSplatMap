"""
Gaussian Splatting Visualizations (Paper-Level Quality)
========================================================

Comprehensive visualization tools for 3D Gaussian Splatting analysis:
- Novel view rendering comparisons
- Gaussian statistics (radii, opacity distributions)
- Training curves (loss, PSNR over iterations)
- Camera path visualization
- Per-frame quality metrics
- Splat consistency analysis
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import FancyBboxPatch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def setup_style():
    """Setup publication-quality plot style."""
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'sans-serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 18,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


# =============================================================================
# Novel View Rendering Comparisons
# =============================================================================

def plot_novel_view_comparison(
    gt_images: List[np.ndarray],
    rendered_images: Dict[str, List[np.ndarray]],
    output_path: Path,
    view_indices: List[int] = None,
    metrics: Dict[str, Dict[str, List[float]]] = None,
):
    """
    Side-by-side comparison of rendered novel views vs ground truth.
    
    Most important visualization for 3DGS papers.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    methods = list(rendered_images.keys())
    n_methods = len(methods)
    
    if view_indices is None:
        view_indices = list(range(min(5, len(gt_images))))
    
    n_views = len(view_indices)
    
    # Create figure: rows = views, cols = GT + methods + error maps
    fig, axes = plt.subplots(n_views, n_methods + 2, figsize=(4 * (n_methods + 2), 4 * n_views))
    
    if n_views == 1:
        axes = axes.reshape(1, -1)
    
    for row, view_idx in enumerate(view_indices):
        gt = gt_images[view_idx]
        
        # GT image
        axes[row, 0].imshow(gt)
        if row == 0:
            axes[row, 0].set_title('Ground Truth', fontweight='bold')
        axes[row, 0].axis('off')
        axes[row, 0].set_ylabel(f'View {view_idx}', fontsize=12, fontweight='bold')
        
        # Each method's rendering
        for col, method in enumerate(methods):
            rendered = rendered_images[method][view_idx] if view_idx < len(rendered_images[method]) else np.zeros_like(gt)
            
            axes[row, col + 1].imshow(rendered)
            if row == 0:
                axes[row, col + 1].set_title(method, fontweight='bold')
            axes[row, col + 1].axis('off')
            
            # Add metrics text
            if metrics and method in metrics:
                psnr = metrics[method].get('psnr', [0])[view_idx] if view_idx < len(metrics[method].get('psnr', [])) else 0
                ssim = metrics[method].get('ssim', [0])[view_idx] if view_idx < len(metrics[method].get('ssim', [])) else 0
                axes[row, col + 1].text(5, 20, f'PSNR: {psnr:.1f}\nSSIM: {ssim:.3f}',
                                        fontsize=9, color='white', backgroundcolor='black', alpha=0.7)
        
        # Error map for best method
        if methods:
            best_rendered = rendered_images[methods[0]][view_idx] if view_idx < len(rendered_images[methods[0]]) else np.zeros_like(gt)
            error = np.mean(np.abs(gt.astype(float) - best_rendered.astype(float)), axis=2)
            im = axes[row, -1].imshow(error, cmap='hot', vmin=0, vmax=50)
            if row == 0:
                axes[row, -1].set_title('Error Map', fontweight='bold')
            axes[row, -1].axis('off')
    
    plt.suptitle('Novel View Synthesis Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_render_quality_metrics(
    results: List[Dict],
    output_path: Path,
):
    """
    Bar chart comparing PSNR, SSIM, LPIPS across GS engines.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    engines = [r['engine'] for r in results]
    psnrs = [r.get('psnr', 0) for r in results]
    ssims = [r.get('ssim', 0) for r in results]
    lpips = [r.get('lpips', 0) for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(engines)))
    
    # PSNR (higher is better)
    sorted_idx = np.argsort(psnrs)[::-1]
    bars = axes[0].barh([engines[i] for i in sorted_idx], [psnrs[i] for i in sorted_idx],
                        color=[colors[i] for i in sorted_idx], edgecolor='black')
    axes[0].set_xlabel('PSNR (dB) ↑')
    axes[0].set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
    bars[0].set_edgecolor('gold')
    bars[0].set_linewidth(3)
    for bar, val in zip(bars, [psnrs[i] for i in sorted_idx]):
        axes[0].text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center')
    
    # SSIM (higher is better)
    sorted_idx = np.argsort(ssims)[::-1]
    bars = axes[1].barh([engines[i] for i in sorted_idx], [ssims[i] for i in sorted_idx],
                        color=[colors[i] for i in sorted_idx], edgecolor='black')
    axes[1].set_xlabel('SSIM ↑')
    axes[1].set_title('Structural Similarity', fontweight='bold')
    bars[0].set_edgecolor('gold')
    bars[0].set_linewidth(3)
    for bar, val in zip(bars, [ssims[i] for i in sorted_idx]):
        axes[1].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')
    
    # LPIPS (lower is better)
    sorted_idx = np.argsort(lpips)
    bars = axes[2].barh([engines[i] for i in sorted_idx], [lpips[i] for i in sorted_idx],
                        color=[colors[i] for i in sorted_idx], edgecolor='black')
    axes[2].set_xlabel('LPIPS ↓')
    axes[2].set_title('Perceptual Similarity', fontweight='bold')
    if lpips[sorted_idx[0]] > 0:
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(3)
    for bar, val in zip(bars, [lpips[i] for i in sorted_idx]):
        axes[2].text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')
    
    plt.suptitle('Gaussian Splatting Quality Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Gaussian Statistics
# =============================================================================

def plot_gaussian_statistics(
    results: List[Dict],
    output_path: Path,
):
    """
    Visualize Gaussian statistics: count, radii distribution (if available).
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    engines = [r['engine'] for r in results]
    n_gaussians = [r.get('final_gaussians', 0) for r in results]
    fps = [r.get('fps', 0) for r in results]
    memory = [r.get('peak_memory_mb', 0) for r in results]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(engines)))
    
    # Number of Gaussians
    bars = axes[0].bar(engines, n_gaussians, color=colors, edgecolor='black')
    axes[0].set_ylabel('Number of Gaussians')
    axes[0].set_title('Final Gaussian Count', fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, n_gaussians):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                    f'{val:,}', ha='center', va='bottom', fontsize=10)
    
    # Processing FPS
    bars = axes[1].bar(engines, fps, color=colors, edgecolor='black')
    axes[1].set_ylabel('Frames per Second')
    axes[1].set_title('Processing Speed', fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, fps):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Memory usage
    if any(m > 0 for m in memory):
        bars = axes[2].bar(engines, memory, color=colors, edgecolor='black')
        axes[2].set_ylabel('Peak Memory (MB)')
        axes[2].set_title('GPU Memory Usage', fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        for bar, val in zip(bars, memory):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                        f'{val:.0f}', ha='center', va='bottom', fontsize=10)
    else:
        axes[2].text(0.5, 0.5, 'Memory data\nnot available', ha='center', va='center',
                    transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title('GPU Memory Usage', fontweight='bold')
    
    plt.suptitle('Gaussian Splatting Efficiency Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Training Curves
# =============================================================================

def plot_training_curves(
    results: List[Dict],
    output_path: Path,
):
    """
    Plot training curves: loss and PSNR over iterations.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    
    has_loss = False
    has_psnr = False
    
    for i, r in enumerate(results):
        engine = r['engine']
        color = colors[i]
        
        # Loss curve
        if 'loss_history' in r and len(r['loss_history']) > 0:
            axes[0].plot(r['loss_history'], label=engine, color=color, linewidth=2)
            has_loss = True
        
        # PSNR curve
        if 'psnr_history' in r and len(r['psnr_history']) > 0:
            axes[1].plot(r['psnr_history'], label=engine, color=color, linewidth=2)
            has_psnr = True
    
    if has_loss:
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Loss', fontweight='bold')
        axes[0].legend()
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'Loss history\nnot available', ha='center', va='center',
                    transform=axes[0].transAxes, fontsize=14)
    
    if has_psnr:
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('PSNR (dB)')
        axes[1].set_title('PSNR Over Training', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'PSNR history\nnot available', ha='center', va='center',
                    transform=axes[1].transAxes, fontsize=14)
    
    plt.suptitle('Training Progress', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Camera Path Visualization
# =============================================================================

def plot_camera_path_3d(
    poses: np.ndarray,
    output_path: Path,
    title: str = "Camera Trajectory",
    show_frustums: bool = True,
    frustum_scale: float = 0.1,
):
    """
    Visualize camera path in 3D with optional frustums.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract positions
    positions = poses[:, :3, 3]
    
    # Plot trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
            'b-', linewidth=2, label='Camera Path')
    
    # Plot start and end
    ax.scatter(*positions[0], color='green', s=100, marker='o', label='Start')
    ax.scatter(*positions[-1], color='red', s=100, marker='s', label='End')
    
    # Plot camera frustums at intervals
    if show_frustums:
        n_frustums = min(20, len(poses))
        indices = np.linspace(0, len(poses)-1, n_frustums, dtype=int)
        
        for idx in indices:
            pose = poses[idx]
            pos = pose[:3, 3]
            R = pose[:3, :3]
            
            # Camera frustum corners
            corners = np.array([
                [0, 0, 0],
                [-1, -0.75, 1],
                [1, -0.75, 1],
                [1, 0.75, 1],
                [-1, 0.75, 1],
            ]) * frustum_scale
            
            # Transform corners
            corners_world = (R @ corners.T).T + pos
            
            # Draw frustum edges
            for i in range(1, 5):
                ax.plot3D(*zip(corners_world[0], corners_world[i]), 'gray', alpha=0.5, linewidth=0.5)
            for i in range(1, 5):
                j = i % 4 + 1
                ax.plot3D(*zip(corners_world[i], corners_world[j]), 'gray', alpha=0.5, linewidth=0.5)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.array([positions[:, 0].max() - positions[:, 0].min(),
                          positions[:, 1].max() - positions[:, 1].min(),
                          positions[:, 2].max() - positions[:, 2].min()]).max() / 2.0
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Pipeline Comparison (GT vs Estimated)
# =============================================================================

def plot_pipeline_comparison(
    results: List[Dict],
    output_path: Path,
):
    """
    Compare GS quality with different pose/depth source combinations.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    configs = [r['name'] for r in results]
    psnrs = [r.get('psnr', 0) for r in results]
    ssims = [r.get('ssim', 0) for r in results]
    
    # If too many results, show only top N and bottom N
    MAX_DISPLAY = 30
    if len(configs) > MAX_DISPLAY:
        # Sort by PSNR and take top and bottom
        sorted_indices = np.argsort(psnrs)[::-1]
        top_n = MAX_DISPLAY // 2
        selected_indices = list(sorted_indices[:top_n]) + list(sorted_indices[-top_n:])
        selected_indices = sorted(set(selected_indices), key=lambda x: psnrs[x], reverse=True)
        
        configs = [configs[i] for i in selected_indices]
        psnrs = [psnrs[i] for i in selected_indices]
        ssims = [ssims[i] for i in selected_indices]
    
    # Shorten config names for display
    def shorten_name(name):
        # Replace common patterns
        name = name.replace('_pose_', ' → ')
        name = name.replace('_depth', '')
        name = name.replace('depth_anything_v', 'DAv')
        name = name.replace('depth_pro_lite', 'DPL')
        name = name.replace('depth_pro', 'DP')
        name = name.replace('robust_flow', 'rflow')
        name = name.replace('superpoint', 'SP')
        name = name.replace('lightglue', 'LG')
        name = name.replace('keyframe', 'kf')
        return name
    
    short_configs = [shorten_name(c) for c in configs]
    
    # Color by configuration type
    def get_config_color(name):
        if 'GT_pose_GT_depth' in name or 'GT → GT' in name:
            return '#2ecc71'  # Green - best case
        elif 'GT_pose' in name or 'GT → ' in name:
            return '#3498db'  # Blue - GT pose
        elif 'GT_depth' in name or '→ GT' in name:
            return '#e74c3c'  # Red - GT depth
        else:
            return '#9b59b6'  # Purple - fully estimated
    
    colors = [get_config_color(c) for c in configs]
    
    # Dynamic figure size based on number of configs
    height = max(8, len(configs) * 0.35)
    fig, axes = plt.subplots(1, 2, figsize=(18, height))
    
    # PSNR comparison
    sorted_idx = np.argsort(psnrs)[::-1]
    y_pos = np.arange(len(configs))
    
    bars = axes[0].barh(y_pos, [psnrs[i] for i in sorted_idx], 
                        color=[colors[i] for i in sorted_idx], edgecolor='black', height=0.7)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels([short_configs[i] for i in sorted_idx], fontsize=8)
    axes[0].set_xlabel('PSNR (dB)')
    axes[0].set_title('Reconstruction Quality by Pipeline', fontweight='bold')
    axes[0].invert_yaxis()
    
    for bar, val in zip(bars, [psnrs[i] for i in sorted_idx]):
        if val > 0:
            axes[0].text(val + 0.2, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=7)
    
    # SSIM comparison
    sorted_idx = np.argsort(ssims)[::-1]
    bars = axes[1].barh(y_pos, [ssims[i] for i in sorted_idx],
                        color=[colors[i] for i in sorted_idx], edgecolor='black', height=0.7)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([short_configs[i] for i in sorted_idx], fontsize=8)
    axes[1].set_xlabel('SSIM')
    axes[1].set_title('Structural Similarity by Pipeline', fontweight='bold')
    axes[1].invert_yaxis()
    
    for bar, val in zip(bars, [ssims[i] for i in sorted_idx]):
        if val > 0:
            axes[1].text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontsize=7)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='GT Pose + GT Depth'),
        Patch(facecolor='#3498db', label='GT Pose + Est. Depth'),
        Patch(facecolor='#e74c3c', label='Est. Pose + GT Depth'),
        Patch(facecolor='#9b59b6', label='Est. Pose + Est. Depth'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.02))
    
    plt.suptitle('Impact of Pose/Depth Estimation on 3DGS Quality', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Comprehensive Summary
# =============================================================================

def plot_gs_comprehensive_summary(
    results: List[Dict],
    output_path: Path,
):
    """
    Create comprehensive summary figure for GS benchmark.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    engines = [r['engine'] for r in results]
    n_engines = len(engines)
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_engines))
    
    # 1. PSNR bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    psnrs = [r.get('psnr', 0) for r in results]
    bars = ax1.bar(engines, psnrs, color=colors, edgecolor='black')
    ax1.set_ylabel('PSNR (dB)')
    ax1.set_title('Peak Signal-to-Noise Ratio', fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. SSIM bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    ssims = [r.get('ssim', 0) for r in results]
    bars = ax2.bar(engines, ssims, color=colors, edgecolor='black')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Structural Similarity', fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Gaussian count
    ax3 = fig.add_subplot(gs[0, 2])
    n_gaussians = [r.get('final_gaussians', 0) for r in results]
    bars = ax3.bar(engines, n_gaussians, color=colors, edgecolor='black')
    ax3.set_ylabel('Count')
    ax3.set_title('Number of Gaussians', fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. FPS comparison
    ax4 = fig.add_subplot(gs[1, 0])
    fps = [r.get('fps', 0) for r in results]
    bars = ax4.bar(engines, fps, color=colors, edgecolor='black')
    ax4.set_ylabel('FPS')
    ax4.set_title('Processing Speed', fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Quality vs Speed scatter
    ax5 = fig.add_subplot(gs[1, 1])
    for i, r in enumerate(results):
        ax5.scatter(r.get('fps', 0), r.get('psnr', 0), s=200, c=[colors[i]], 
                   edgecolors='black', label=r['engine'])
    ax5.set_xlabel('FPS')
    ax5.set_ylabel('PSNR (dB)')
    ax5.set_title('Quality vs Speed Trade-off', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. LPIPS (if available)
    ax6 = fig.add_subplot(gs[1, 2])
    lpips = [r.get('lpips', 0) for r in results]
    if any(l > 0 for l in lpips):
        bars = ax6.bar(engines, lpips, color=colors, edgecolor='black')
        ax6.set_ylabel('LPIPS')
        ax6.set_title('Perceptual Similarity (lower=better)', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'LPIPS not available', ha='center', va='center', fontsize=12)
    ax6.tick_params(axis='x', rotation=45)
    
    # 7-9. Summary table
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Engine', 'PSNR (dB)', 'SSIM', 'LPIPS', 'Gaussians', 'FPS']
    for r in results:
        table_data.append([
            r['engine'],
            f"{r.get('psnr', 0):.2f}",
            f"{r.get('ssim', 0):.4f}",
            f"{r.get('lpips', 0):.4f}" if r.get('lpips', 0) > 0 else 'N/A',
            f"{r.get('final_gaussians', 0):,}",
            f"{r.get('fps', 0):.1f}",
        ])
    
    table = ax7.table(cellText=table_data, colLabels=headers, loc='center',
                      cellLoc='center', colColours=['#e8e8e8'] * len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    plt.suptitle('Gaussian Splatting Benchmark Summary', fontsize=18, fontweight='bold')
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Main Generator
# =============================================================================

def generate_all_gs_plots(
    results: List[Dict],
    pipeline_results: List[Dict],
    output_dir: Path,
    sample_data: Optional[Dict] = None,
):
    """
    Generate all Gaussian splatting visualizations.
    
    Args:
        results: List of GS benchmark results
        pipeline_results: List of pipeline (GT vs estimated) results
        output_dir: Output directory
        sample_data: Optional rendered images for qualitative comparison
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available, skipping GS plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating GS plots in {output_dir}")
    
    # Quality metrics
    try:
        plot_render_quality_metrics(results, output_dir / 'quality_metrics.png')
        logger.info("  ✓ quality_metrics.png")
    except Exception as e:
        logger.warning(f"  ✗ quality_metrics failed: {e}")
    
    # Gaussian statistics
    try:
        plot_gaussian_statistics(results, output_dir / 'gaussian_statistics.png')
        logger.info("  ✓ gaussian_statistics.png")
    except Exception as e:
        logger.warning(f"  ✗ gaussian_statistics failed: {e}")
    
    # Training curves
    try:
        plot_training_curves(results, output_dir / 'training_curves.png')
        logger.info("  ✓ training_curves.png")
    except Exception as e:
        logger.warning(f"  ✗ training_curves failed: {e}")
    
    # Comprehensive summary
    try:
        plot_gs_comprehensive_summary(results, output_dir / 'comprehensive_summary.png')
        logger.info("  ✓ comprehensive_summary.png")
    except Exception as e:
        logger.warning(f"  ✗ comprehensive_summary failed: {e}")
    
    # Pipeline comparison
    if pipeline_results:
        try:
            plot_pipeline_comparison(pipeline_results, output_dir / 'pipeline_comparison.png')
            logger.info("  ✓ pipeline_comparison.png")
        except Exception as e:
            logger.warning(f"  ✗ pipeline_comparison failed: {e}")
    
    # Novel view comparison (if sample data available)
    if sample_data and 'gt_images' in sample_data and 'rendered_images' in sample_data:
        try:
            plot_novel_view_comparison(
                sample_data['gt_images'],
                sample_data['rendered_images'],
                output_dir / 'novel_view_comparison.png',
            )
            logger.info("  ✓ novel_view_comparison.png")
        except Exception as e:
            logger.warning(f"  ✗ novel_view_comparison failed: {e}")
