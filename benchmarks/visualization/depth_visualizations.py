"""
Advanced Depth Estimation Visualizations
=========================================

Comprehensive visualization tools for depth estimation analysis:
- Depth error heatmaps (absolute and relative)
- Log-scale depth difference
- Depth histograms and distributions
- Depth edge accuracy
- 3D point cloud visualization
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def plot_depth_error_heatmap(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    rgb: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "Depth Error Analysis",
    method_name: str = "Predicted",
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Optional[plt.Figure]:
    """
    Create depth error heatmap showing where model fails.
    
    Shows:
    - RGB image (if provided)
    - GT depth
    - Predicted depth  
    - Absolute error heatmap
    - Relative error heatmap
    
    Args:
        pred_depth: Predicted depth (H, W)
        gt_depth: Ground truth depth (H, W)
        rgb: Optional RGB image (H, W, 3)
        output_path: Where to save figure
        title: Plot title
        method_name: Name of prediction method
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
    """
    if not HAS_MATPLOTLIB:
        return None
    
    # Create valid mask
    valid = (gt_depth > min_depth) & (gt_depth < max_depth) & (pred_depth > 0)
    
    # Scale prediction to match GT (median scaling)
    if valid.sum() > 100:
        scale = np.median(gt_depth[valid]) / (np.median(pred_depth[valid]) + 1e-8)
        pred_scaled = pred_depth * scale
    else:
        pred_scaled = pred_depth
        scale = 1.0
    
    # Compute errors
    abs_error = np.abs(pred_scaled - gt_depth)
    abs_error[~valid] = 0
    
    rel_error = abs_error / (gt_depth + 1e-8)
    rel_error[~valid] = 0
    
    # Create figure
    n_cols = 5 if rgb is not None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    
    col = 0
    
    # RGB image
    if rgb is not None:
        axes[col].imshow(rgb)
        axes[col].set_title("RGB Input")
        axes[col].axis('off')
        col += 1
    
    # GT depth
    im1 = axes[col].imshow(gt_depth, cmap='turbo', vmin=min_depth, vmax=max_depth)
    axes[col].set_title("Ground Truth Depth")
    axes[col].axis('off')
    plt.colorbar(im1, ax=axes[col], fraction=0.046, pad=0.04, label='m')
    col += 1
    
    # Predicted depth (scaled)
    im2 = axes[col].imshow(pred_scaled, cmap='turbo', vmin=min_depth, vmax=max_depth)
    axes[col].set_title(f"{method_name} (scale={scale:.2f})")
    axes[col].axis('off')
    plt.colorbar(im2, ax=axes[col], fraction=0.046, pad=0.04, label='m')
    col += 1
    
    # Absolute error heatmap
    error_max = np.percentile(abs_error[valid], 95) if valid.sum() > 0 else 1.0
    im3 = axes[col].imshow(abs_error, cmap='magma', vmin=0, vmax=error_max)
    axes[col].set_title("Absolute Error")
    axes[col].axis('off')
    plt.colorbar(im3, ax=axes[col], fraction=0.046, pad=0.04, label='m')
    col += 1
    
    # Relative error heatmap
    im4 = axes[col].imshow(rel_error, cmap='magma', vmin=0, vmax=0.5)
    axes[col].set_title("Relative Error")
    axes[col].axis('off')
    plt.colorbar(im4, ax=axes[col], fraction=0.046, pad=0.04, label='ratio')
    
    # Add statistics
    if valid.sum() > 0:
        abs_rel = np.mean(rel_error[valid])
        rmse = np.sqrt(np.mean(abs_error[valid] ** 2))
        fig.suptitle(f"{title}\nAbsRel: {abs_rel:.4f} | RMSE: {rmse:.4f}m | Scale: {scale:.3f}", 
                     fontsize=12, y=1.02)
    else:
        fig.suptitle(title, fontsize=12, y=1.02)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Saved depth error heatmap to {output_path}")
    
    return fig


def plot_log_depth_difference(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Log-Scale Depth Difference",
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Optional[plt.Figure]:
    """
    Plot log-scale depth difference: log(pred) - log(gt).
    
    Highlights structural error instead of raw meter differences.
    More meaningful for scale-consistent 3DGS reconstruction.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    valid = (gt_depth > min_depth) & (gt_depth < max_depth) & (pred_depth > min_depth)
    
    # Scale prediction
    if valid.sum() > 100:
        scale = np.median(gt_depth[valid]) / (np.median(pred_depth[valid]) + 1e-8)
        pred_scaled = pred_depth * scale
    else:
        pred_scaled = pred_depth
    
    # Log difference
    log_diff = np.zeros_like(gt_depth)
    log_diff[valid] = np.log(pred_scaled[valid] + 1e-8) - np.log(gt_depth[valid] + 1e-8)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Log GT depth
    log_gt = np.log(gt_depth + 1e-8)
    log_gt[~valid] = np.nan
    im1 = axes[0].imshow(log_gt, cmap='viridis')
    axes[0].set_title("log(GT Depth)")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    # Log predicted depth
    log_pred = np.log(pred_scaled + 1e-8)
    log_pred[~valid] = np.nan
    im2 = axes[1].imshow(log_pred, cmap='viridis')
    axes[1].set_title("log(Predicted Depth)")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Log difference (symmetric colormap)
    vmax = np.percentile(np.abs(log_diff[valid]), 95) if valid.sum() > 0 else 1.0
    im3 = axes[2].imshow(log_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    axes[2].set_title("log(Pred) - log(GT)")
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Statistics
    if valid.sum() > 0:
        si_log = np.sqrt(np.mean(log_diff[valid] ** 2) - np.mean(log_diff[valid]) ** 2)
        fig.suptitle(f"{title}\nScale-Invariant Log Error: {si_log:.4f}", fontsize=12)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_depth_histograms(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Depth Distribution Analysis",
    method_name: str = "Predicted",
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    n_bins: int = 100,
) -> Optional[plt.Figure]:
    """
    Plot depth histograms showing distribution bias.
    
    Shows:
    - Histogram of GT depth
    - Histogram of predicted depth
    - Histogram of (GT - predicted) error
    - Cumulative distribution comparison
    """
    if not HAS_MATPLOTLIB:
        return None
    
    valid = (gt_depth > min_depth) & (gt_depth < max_depth) & (pred_depth > 0)
    
    gt_valid = gt_depth[valid].flatten()
    pred_valid = pred_depth[valid].flatten()
    
    # Scale prediction
    scale = np.median(gt_valid) / (np.median(pred_valid) + 1e-8)
    pred_scaled = pred_valid * scale
    
    error = gt_valid - pred_scaled
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Depth histograms
    bins = np.linspace(min_depth, max_depth, n_bins)
    axes[0, 0].hist(gt_valid, bins=bins, alpha=0.7, label='Ground Truth', color='#3498db')
    axes[0, 0].hist(pred_scaled, bins=bins, alpha=0.7, label=f'{method_name} (scaled)', color='#e74c3c')
    axes[0, 0].set_xlabel('Depth (m)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Depth Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_xlim(min_depth, max_depth)
    
    # Error histogram
    error_range = np.percentile(np.abs(error), 99)
    error_bins = np.linspace(-error_range, error_range, n_bins)
    axes[0, 1].hist(error, bins=error_bins, alpha=0.7, color='#9b59b6')
    axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
    axes[0, 1].axvline(np.mean(error), color='orange', linestyle='-', linewidth=2, 
                       label=f'Mean: {np.mean(error):.3f}m')
    axes[0, 1].set_xlabel('Error (GT - Pred) [m]')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    
    # Cumulative distribution
    gt_sorted = np.sort(gt_valid)
    pred_sorted = np.sort(pred_scaled)
    axes[1, 0].plot(gt_sorted, np.linspace(0, 1, len(gt_sorted)), 
                    label='Ground Truth', linewidth=2, color='#3498db')
    axes[1, 0].plot(pred_sorted, np.linspace(0, 1, len(pred_sorted)), 
                    label=f'{method_name}', linewidth=2, color='#e74c3c')
    axes[1, 0].set_xlabel('Depth (m)')
    axes[1, 0].set_ylabel('Cumulative Probability')
    axes[1, 0].set_title('Cumulative Distribution')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(min_depth, max_depth)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Q-Q plot
    n_quantiles = min(1000, len(gt_valid))
    quantiles = np.linspace(0, 1, n_quantiles)
    gt_quantiles = np.quantile(gt_valid, quantiles)
    pred_quantiles = np.quantile(pred_scaled, quantiles)
    
    axes[1, 1].scatter(gt_quantiles, pred_quantiles, alpha=0.5, s=10, color='#2ecc71')
    axes[1, 1].plot([min_depth, max_depth], [min_depth, max_depth], 'r--', 
                    linewidth=2, label='Perfect match')
    axes[1, 1].set_xlabel('GT Depth Quantiles (m)')
    axes[1, 1].set_ylabel(f'{method_name} Depth Quantiles (m)')
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(min_depth, max_depth)
    axes[1, 1].set_ylim(min_depth, max_depth)
    axes[1, 1].set_aspect('equal')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Statistics text
    stats_text = (f"Statistics:\n"
                  f"GT Mean: {np.mean(gt_valid):.3f}m\n"
                  f"Pred Mean: {np.mean(pred_scaled):.3f}m\n"
                  f"Error Mean: {np.mean(error):.3f}m\n"
                  f"Error Std: {np.std(error):.3f}m\n"
                  f"Scale: {scale:.3f}")
    fig.text(0.02, 0.02, stats_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_depth_edge_accuracy(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Depth Edge Accuracy",
    edge_threshold: float = 0.1,
) -> Optional[plt.Figure]:
    """
    Compare edges in predicted vs GT depth.
    
    Critical for avoiding "melty" splats near boundaries.
    Shows edge detection results and precision/recall.
    """
    if not HAS_MATPLOTLIB or not HAS_CV2:
        return None
    
    # Normalize depths for edge detection
    gt_norm = (gt_depth - gt_depth.min()) / (gt_depth.max() - gt_depth.min() + 1e-8)
    pred_norm = (pred_depth - pred_depth.min()) / (pred_depth.max() - pred_depth.min() + 1e-8)
    
    gt_uint8 = (gt_norm * 255).astype(np.uint8)
    pred_uint8 = (pred_norm * 255).astype(np.uint8)
    
    # Compute edges using Canny
    gt_edges = cv2.Canny(gt_uint8, 50, 150)
    pred_edges = cv2.Canny(pred_uint8, 50, 150)
    
    # Compute Sobel gradients for continuous comparison
    gt_sobel_x = cv2.Sobel(gt_norm.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gt_sobel_y = cv2.Sobel(gt_norm.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    gt_grad = np.sqrt(gt_sobel_x**2 + gt_sobel_y**2)
    
    pred_sobel_x = cv2.Sobel(pred_norm.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    pred_sobel_y = cv2.Sobel(pred_norm.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    pred_grad = np.sqrt(pred_sobel_x**2 + pred_sobel_y**2)
    
    # Compute precision/recall for edges
    gt_edge_mask = gt_edges > 0
    pred_edge_mask = pred_edges > 0
    
    # Dilate GT edges slightly for tolerance
    kernel = np.ones((3, 3), np.uint8)
    gt_dilated = cv2.dilate(gt_edges, kernel, iterations=1) > 0
    pred_dilated = cv2.dilate(pred_edges, kernel, iterations=1) > 0
    
    tp = np.sum(pred_edge_mask & gt_dilated)
    fp = np.sum(pred_edge_mask & ~gt_dilated)
    fn = np.sum(gt_edge_mask & ~pred_dilated)
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # GT edges
    axes[0, 0].imshow(gt_edges, cmap='gray')
    axes[0, 0].set_title('GT Depth Edges')
    axes[0, 0].axis('off')
    
    # Predicted edges
    axes[0, 1].imshow(pred_edges, cmap='gray')
    axes[0, 1].set_title('Predicted Depth Edges')
    axes[0, 1].axis('off')
    
    # Edge comparison (RGB: R=GT only, G=both, B=Pred only)
    edge_comparison = np.zeros((*gt_edges.shape, 3), dtype=np.uint8)
    edge_comparison[gt_edge_mask & ~pred_dilated, 0] = 255  # Red: GT only (missed)
    edge_comparison[gt_dilated & pred_edge_mask, 1] = 255   # Green: matched
    edge_comparison[pred_edge_mask & ~gt_dilated, 2] = 255  # Blue: Pred only (false)
    axes[0, 2].imshow(edge_comparison)
    axes[0, 2].set_title('Edge Comparison\n(R=Missed, G=Match, B=False)')
    axes[0, 2].axis('off')
    
    # GT gradient magnitude
    im1 = axes[1, 0].imshow(gt_grad, cmap='hot')
    axes[1, 0].set_title('GT Gradient Magnitude')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Predicted gradient magnitude
    im2 = axes[1, 1].imshow(pred_grad, cmap='hot')
    axes[1, 1].set_title('Predicted Gradient Magnitude')
    axes[1, 1].axis('off')
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # Gradient difference
    grad_diff = np.abs(pred_grad - gt_grad)
    im3 = axes[1, 2].imshow(grad_diff, cmap='magma')
    axes[1, 2].set_title('Gradient Difference')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    # Add metrics
    metrics_text = (f"Edge Metrics:\n"
                    f"Precision: {precision:.3f}\n"
                    f"Recall: {recall:.3f}\n"
                    f"F1 Score: {f1:.3f}\n"
                    f"Grad Corr: {np.corrcoef(gt_grad.flatten(), pred_grad.flatten())[0,1]:.3f}")
    fig.text(0.02, 0.02, metrics_text, fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_depth_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Dict[str, float],
    pose: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "3D Point Cloud from Depth",
    max_points: int = 50000,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> Optional[plt.Figure]:
    """
    Create 3D point cloud visualization from depth + pose.
    
    Args:
        depth: Depth map (H, W)
        rgb: RGB image (H, W, 3)
        intrinsics: Camera intrinsics dict with fx, fy, cx, cy
        pose: Optional 4x4 camera pose matrix
        output_path: Where to save figure
        title: Plot title
        max_points: Maximum points to visualize
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
    """
    if not HAS_MATPLOTLIB:
        return None
    
    H, W = depth.shape
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # Valid mask
    valid = (depth > min_depth) & (depth < max_depth)
    
    # Unproject to 3D
    z = depth[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    
    points = np.stack([x, y, z], axis=-1)
    colors = rgb[valid] / 255.0 if rgb.max() > 1 else rgb[valid]
    
    # Apply pose if provided
    if pose is not None:
        R = pose[:3, :3]
        t = pose[:3, 3]
        points = points @ R.T + t
    
    # Subsample if too many points
    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        colors = colors[idx]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
               c=colors, s=1, alpha=0.6)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.max([
        points[:, 0].max() - points[:, 0].min(),
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()
    ]) / 2.0
    
    mid_x = (points[:, 0].max() + points[:, 0].min()) / 2
    mid_y = (points[:, 1].max() + points[:, 1].min()) / 2
    mid_z = (points[:, 2].max() + points[:, 2].min()) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_depth_comparison_grid(
    samples: List[Dict],
    output_path: Optional[Path] = None,
    title: str = "Depth Method Comparison",
    max_samples: int = 4,
) -> Optional[plt.Figure]:
    """
    Create grid comparison of multiple depth methods.
    
    Args:
        samples: List of dicts with keys: 'rgb', 'gt_depth', 'predictions' (dict of method->depth)
        output_path: Where to save
        title: Plot title
        max_samples: Max number of samples to show
    """
    if not HAS_MATPLOTLIB:
        return None
    
    samples = samples[:max_samples]
    n_samples = len(samples)
    
    if n_samples == 0:
        return None
    
    # Get all methods from first sample
    methods = list(samples[0].get('predictions', {}).keys())
    n_cols = 2 + len(methods)  # RGB + GT + methods
    
    fig, axes = plt.subplots(n_samples, n_cols, figsize=(3 * n_cols, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    
    for i, sample in enumerate(samples):
        rgb = sample.get('rgb')
        gt = sample.get('gt_depth')
        predictions = sample.get('predictions', {})
        
        col = 0
        
        # RGB
        if rgb is not None:
            axes[i, col].imshow(rgb)
            if i == 0:
                axes[i, col].set_title('RGB')
            axes[i, col].axis('off')
        col += 1
        
        # GT depth
        if gt is not None:
            vmin, vmax = np.percentile(gt[gt > 0], [5, 95]) if (gt > 0).any() else (0, 1)
            im = axes[i, col].imshow(gt, cmap='turbo', vmin=vmin, vmax=vmax)
            if i == 0:
                axes[i, col].set_title('Ground Truth')
            axes[i, col].axis('off')
        col += 1
        
        # Method predictions
        for method in methods:
            pred = predictions.get(method)
            if pred is not None:
                # Scale to GT
                valid = (gt > 0.1) & (pred > 0)
                if valid.sum() > 100:
                    scale = np.median(gt[valid]) / (np.median(pred[valid]) + 1e-8)
                    pred = pred * scale
                
                axes[i, col].imshow(pred, cmap='turbo', vmin=vmin, vmax=vmax)
                
                # Compute error
                if gt is not None and valid.sum() > 0:
                    abs_rel = np.mean(np.abs(pred[valid] - gt[valid]) / gt[valid])
                    axes[i, col].text(0.02, 0.98, f'AbsRel:{abs_rel:.3f}', 
                                      transform=axes[i, col].transAxes,
                                      fontsize=8, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
            if i == 0:
                axes[i, col].set_title(method)
            axes[i, col].axis('off')
            col += 1
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig
