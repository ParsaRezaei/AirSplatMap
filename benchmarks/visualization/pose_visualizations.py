"""
Advanced Pose Estimation Visualizations
========================================

Comprehensive visualization tools for pose/trajectory analysis:
- 2D/3D trajectory plots (XY, XZ, 3D)
- ATE curve over time
- RPE analysis (translational and rotational)
- Drift vs distance traveled
- Roll/Pitch/Yaw orientation plots
- Trajectory alignment visualization
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    
    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


def align_trajectories_umeyama(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Align estimated trajectory to ground truth using Umeyama alignment.
    
    Args:
        estimated: Estimated poses (N, 4, 4)
        ground_truth: Ground truth poses (N, 4, 4)
        
    Returns:
        aligned: Aligned estimated poses
        scale: Scale factor
        R: Rotation matrix
        t: Translation vector
    """
    # Extract translations
    est_trans = estimated[:, :3, 3]
    gt_trans = ground_truth[:, :3, 3]
    
    # Center the trajectories
    est_mean = est_trans.mean(axis=0)
    gt_mean = gt_trans.mean(axis=0)
    
    est_centered = est_trans - est_mean
    gt_centered = gt_trans - gt_mean
    
    # Compute scale
    est_var = np.sum(est_centered ** 2)
    gt_var = np.sum(gt_centered ** 2)
    scale = np.sqrt(gt_var / (est_var + 1e-8))
    
    # Compute rotation using SVD
    H = est_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = gt_mean - scale * R @ est_mean
    
    # Apply transformation
    aligned = estimated.copy()
    for i in range(len(estimated)):
        aligned[i, :3, 3] = scale * R @ estimated[i, :3, 3] + t
        aligned[i, :3, :3] = R @ estimated[i, :3, :3]
    
    return aligned, scale, R, t


def plot_trajectory_2d(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Trajectory Comparison",
    method_name: str = "Estimated",
    align: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot 2D trajectory comparison (XY and XZ planes).
    
    Args:
        estimated: Estimated poses (N, 4, 4)
        ground_truth: Ground truth poses (N, 4, 4)
        output_path: Where to save
        title: Plot title
        method_name: Name of estimation method
        align: Whether to align trajectories
    """
    if not HAS_MATPLOTLIB:
        return None
    
    if align and len(estimated) > 3:
        estimated, scale, _, _ = align_trajectories_umeyama(estimated, ground_truth)
        title += f" (aligned, scale={scale:.3f})"
    
    # Extract translations
    est_trans = estimated[:, :3, 3]
    gt_trans = ground_truth[:, :3, 3]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # XY plane
    axes[0].plot(gt_trans[:, 0], gt_trans[:, 1], 'b-', linewidth=2, label='Ground Truth')
    axes[0].plot(est_trans[:, 0], est_trans[:, 1], 'r-', linewidth=2, label=method_name, alpha=0.8)
    axes[0].scatter(gt_trans[0, 0], gt_trans[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    axes[0].scatter(gt_trans[-1, 0], gt_trans[-1, 1], c='red', s=100, marker='x', label='End', zorder=5)
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Y (m)')
    axes[0].set_title('Top View (XY Plane)')
    axes[0].legend()
    axes[0].set_aspect('equal')
    axes[0].grid(True, alpha=0.3)
    
    # XZ plane
    axes[1].plot(gt_trans[:, 0], gt_trans[:, 2], 'b-', linewidth=2, label='Ground Truth')
    axes[1].plot(est_trans[:, 0], est_trans[:, 2], 'r-', linewidth=2, label=method_name, alpha=0.8)
    axes[1].scatter(gt_trans[0, 0], gt_trans[0, 2], c='green', s=100, marker='o', label='Start', zorder=5)
    axes[1].scatter(gt_trans[-1, 0], gt_trans[-1, 2], c='red', s=100, marker='x', label='End', zorder=5)
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Z (m)')
    axes[1].set_title('Side View (XZ Plane)')
    axes[1].legend()
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_trajectory_3d(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "3D Trajectory Comparison",
    method_name: str = "Estimated",
    align: bool = True,
    show_orientation: bool = False,
    orientation_skip: int = 10,
) -> Optional[plt.Figure]:
    """
    Plot 3D trajectory with optional orientation axes.
    
    Args:
        estimated: Estimated poses (N, 4, 4)
        ground_truth: Ground truth poses (N, 4, 4)
        output_path: Where to save
        title: Plot title
        method_name: Name of estimation method
        align: Whether to align trajectories
        show_orientation: Whether to show camera orientation axes
        orientation_skip: Skip factor for orientation axes
    """
    if not HAS_MATPLOTLIB:
        return None
    
    if align and len(estimated) > 3:
        estimated, scale, _, _ = align_trajectories_umeyama(estimated, ground_truth)
        title += f" (aligned, scale={scale:.3f})"
    
    est_trans = estimated[:, :3, 3]
    gt_trans = ground_truth[:, :3, 3]
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    ax.plot(gt_trans[:, 0], gt_trans[:, 1], gt_trans[:, 2], 
            'b-', linewidth=2, label='Ground Truth')
    ax.plot(est_trans[:, 0], est_trans[:, 1], est_trans[:, 2], 
            'r-', linewidth=2, label=method_name, alpha=0.8)
    
    # Start/End markers
    ax.scatter(*gt_trans[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(*gt_trans[-1], c='red', s=100, marker='x', label='End')
    
    # Show orientation axes
    if show_orientation:
        axis_length = np.linalg.norm(gt_trans.max(axis=0) - gt_trans.min(axis=0)) * 0.05
        for i in range(0, len(estimated), orientation_skip):
            pos = est_trans[i]
            R = estimated[i, :3, :3]
            # X axis (red), Y axis (green), Z axis (blue)
            ax.quiver(*pos, *(R[:, 0] * axis_length), color='red', alpha=0.5, arrow_length_ratio=0.3)
            ax.quiver(*pos, *(R[:, 1] * axis_length), color='green', alpha=0.5, arrow_length_ratio=0.3)
            ax.quiver(*pos, *(R[:, 2] * axis_length), color='blue', alpha=0.5, arrow_length_ratio=0.3)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(title)
    ax.legend()
    
    # Equal aspect ratio
    max_range = np.max([
        gt_trans[:, 0].max() - gt_trans[:, 0].min(),
        gt_trans[:, 1].max() - gt_trans[:, 1].min(),
        gt_trans[:, 2].max() - gt_trans[:, 2].min()
    ]) / 2.0
    
    mid = gt_trans.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_ate_over_time(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    title: str = "Absolute Trajectory Error Over Time",
    method_name: str = "Estimated",
    align: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot ATE per timestamp to show drift and failure points.
    
    Shows where estimation fails (textureless scenes, motion blur, turns).
    """
    if not HAS_MATPLOTLIB:
        return None
    
    if align and len(estimated) > 3:
        estimated, scale, _, _ = align_trajectories_umeyama(estimated, ground_truth)
    
    # Compute per-frame ATE
    est_trans = estimated[:, :3, 3]
    gt_trans = ground_truth[:, :3, 3]
    ate_per_frame = np.linalg.norm(est_trans - gt_trans, axis=1)
    
    if timestamps is None:
        timestamps = np.arange(len(ate_per_frame))
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={'height_ratios': [2, 1]})
    
    # ATE over time
    axes[0].fill_between(timestamps, 0, ate_per_frame, alpha=0.3, color='#e74c3c')
    axes[0].plot(timestamps, ate_per_frame, 'r-', linewidth=1.5, label=f'{method_name} ATE')
    axes[0].axhline(np.mean(ate_per_frame), color='orange', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(ate_per_frame):.4f}m')
    axes[0].axhline(np.median(ate_per_frame), color='green', linestyle=':', 
                    linewidth=2, label=f'Median: {np.median(ate_per_frame):.4f}m')
    
    # Mark max error
    max_idx = np.argmax(ate_per_frame)
    axes[0].scatter(timestamps[max_idx], ate_per_frame[max_idx], c='red', s=100, 
                    marker='v', zorder=5, label=f'Max: {ate_per_frame[max_idx]:.4f}m')
    
    axes[0].set_xlabel('Frame / Time')
    axes[0].set_ylabel('ATE (m)')
    axes[0].set_title(title)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(timestamps[0], timestamps[-1])
    
    # Cumulative ATE (drift)
    cumulative_ate = np.cumsum(ate_per_frame) / np.arange(1, len(ate_per_frame) + 1)
    axes[1].plot(timestamps, cumulative_ate, 'b-', linewidth=2, label='Cumulative Mean ATE')
    axes[1].fill_between(timestamps, 0, cumulative_ate, alpha=0.2, color='#3498db')
    axes[1].set_xlabel('Frame / Time')
    axes[1].set_ylabel('Cumulative Mean ATE (m)')
    axes[1].set_title('Drift Accumulation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(timestamps[0], timestamps[-1])
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_rpe_analysis(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Relative Pose Error Analysis",
    method_name: str = "Estimated",
) -> Optional[plt.Figure]:
    """
    Plot RPE (translational and rotational) over sequence.
    
    Detects jitter, scale drift, and frame-to-frame instability.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    n = len(estimated)
    
    # Compute relative poses
    rpe_trans = []
    rpe_rot = []
    
    for i in range(1, n):
        # Relative GT pose
        gt_rel = np.linalg.inv(ground_truth[i-1]) @ ground_truth[i]
        # Relative estimated pose
        est_rel = np.linalg.inv(estimated[i-1]) @ estimated[i]
        
        # RPE
        error = np.linalg.inv(gt_rel) @ est_rel
        
        # Translation error
        rpe_trans.append(np.linalg.norm(error[:3, 3]))
        
        # Rotation error (angle)
        R_err = error[:3, :3]
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1, 1))
        rpe_rot.append(np.degrees(angle))
    
    rpe_trans = np.array(rpe_trans)
    rpe_rot = np.array(rpe_rot)
    frames = np.arange(1, n)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Translation RPE over time
    axes[0, 0].fill_between(frames, 0, rpe_trans, alpha=0.3, color='#e74c3c')
    axes[0, 0].plot(frames, rpe_trans, 'r-', linewidth=1, label=f'{method_name}')
    axes[0, 0].axhline(np.mean(rpe_trans), color='orange', linestyle='--', 
                       label=f'Mean: {np.mean(rpe_trans):.4f}m')
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('RPE Translation (m)')
    axes[0, 0].set_title('Translational RPE per Frame')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Rotation RPE over time
    axes[0, 1].fill_between(frames, 0, rpe_rot, alpha=0.3, color='#3498db')
    axes[0, 1].plot(frames, rpe_rot, 'b-', linewidth=1, label=f'{method_name}')
    axes[0, 1].axhline(np.mean(rpe_rot), color='orange', linestyle='--', 
                       label=f'Mean: {np.mean(rpe_rot):.2f}°')
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('RPE Rotation (deg)')
    axes[0, 1].set_title('Rotational RPE per Frame')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Translation RPE histogram
    axes[1, 0].hist(rpe_trans, bins=50, alpha=0.7, color='#e74c3c', edgecolor='black')
    axes[1, 0].axvline(np.mean(rpe_trans), color='orange', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(rpe_trans):.4f}m')
    axes[1, 0].axvline(np.median(rpe_trans), color='green', linestyle=':', linewidth=2,
                       label=f'Median: {np.median(rpe_trans):.4f}m')
    axes[1, 0].set_xlabel('RPE Translation (m)')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Translation RPE Distribution')
    axes[1, 0].legend()
    
    # Rotation RPE histogram
    axes[1, 1].hist(rpe_rot, bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    axes[1, 1].axvline(np.mean(rpe_rot), color='orange', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(rpe_rot):.2f}°')
    axes[1, 1].axvline(np.median(rpe_rot), color='green', linestyle=':', linewidth=2,
                       label=f'Median: {np.median(rpe_rot):.2f}°')
    axes[1, 1].set_xlabel('RPE Rotation (deg)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Rotation RPE Distribution')
    axes[1, 1].legend()
    
    # Stats text
    stats_text = (f"RPE Statistics:\n"
                  f"Trans RMSE: {np.sqrt(np.mean(rpe_trans**2)):.4f}m\n"
                  f"Trans Mean: {np.mean(rpe_trans):.4f}m\n"
                  f"Rot RMSE: {np.sqrt(np.mean(rpe_rot**2)):.2f}°\n"
                  f"Rot Mean: {np.mean(rpe_rot):.2f}°")
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


def plot_drift_vs_distance(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Drift vs Distance Traveled",
    method_name: str = "Estimated",
    align: bool = True,
) -> Optional[plt.Figure]:
    """
    Plot cumulative drift vs distance traveled.
    
    Reveals whether drift is linear, quadratic, or catastrophic.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    if align and len(estimated) > 3:
        estimated, _, _, _ = align_trajectories_umeyama(estimated, ground_truth)
    
    est_trans = estimated[:, :3, 3]
    gt_trans = ground_truth[:, :3, 3]
    
    # Compute distance traveled (GT)
    distances = np.cumsum(np.linalg.norm(np.diff(gt_trans, axis=0), axis=1))
    distances = np.concatenate([[0], distances])
    
    # Compute cumulative drift (ATE)
    ate = np.linalg.norm(est_trans - gt_trans, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Drift vs distance
    axes[0].scatter(distances, ate, c=np.arange(len(ate)), cmap='viridis', 
                    s=20, alpha=0.7)
    axes[0].plot(distances, ate, 'r-', alpha=0.3, linewidth=1)
    
    # Fit linear trend
    if len(distances) > 10:
        coeffs = np.polyfit(distances, ate, 1)
        trend_line = np.polyval(coeffs, distances)
        axes[0].plot(distances, trend_line, 'b--', linewidth=2, 
                     label=f'Linear trend: {coeffs[0]*100:.2f}% drift/m')
    
    axes[0].set_xlabel('Distance Traveled (m)')
    axes[0].set_ylabel('Absolute Trajectory Error (m)')
    axes[0].set_title('Drift vs Distance')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Drift percentage vs distance
    drift_percent = (ate / (distances + 1e-8)) * 100
    drift_percent[0] = 0  # Avoid division by zero
    
    axes[1].scatter(distances[1:], drift_percent[1:], c=np.arange(len(ate))[1:], 
                    cmap='viridis', s=20, alpha=0.7)
    axes[1].axhline(np.mean(drift_percent[1:]), color='red', linestyle='--', 
                    linewidth=2, label=f'Mean: {np.mean(drift_percent[1:]):.2f}%')
    axes[1].set_xlabel('Distance Traveled (m)')
    axes[1].set_ylabel('Drift (% of distance)')
    axes[1].set_title('Relative Drift')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Add colorbar for time
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(0, len(ate)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axes, shrink=0.5)
    cbar.set_label('Frame')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_orientation_over_time(
    estimated: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Orientation Analysis (Roll/Pitch/Yaw)",
    method_name: str = "Estimated",
) -> Optional[plt.Figure]:
    """
    Plot roll/pitch/yaw separately over time.
    
    Helps diagnose pose flips and IMU fusion issues.
    """
    if not HAS_MATPLOTLIB:
        return None
    
    n = len(estimated)
    
    # Extract Euler angles
    est_euler = np.array([rotation_matrix_to_euler(estimated[i, :3, :3]) for i in range(n)])
    gt_euler = np.array([rotation_matrix_to_euler(ground_truth[i, :3, :3]) for i in range(n)])
    
    frames = np.arange(n)
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    labels = ['Roll', 'Pitch', 'Yaw']
    colors = ['#e74c3c', '#2ecc71', '#3498db']
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        # Angle over time
        axes[i, 0].plot(frames, gt_euler[:, i], 'b-', linewidth=2, label='Ground Truth', alpha=0.8)
        axes[i, 0].plot(frames, est_euler[:, i], color=color, linewidth=2, 
                        label=method_name, alpha=0.8)
        axes[i, 0].set_xlabel('Frame')
        axes[i, 0].set_ylabel(f'{label} (deg)')
        axes[i, 0].set_title(f'{label} Over Time')
        axes[i, 0].legend()
        axes[i, 0].grid(True, alpha=0.3)
        
        # Angle error
        error = est_euler[:, i] - gt_euler[:, i]
        # Wrap angle error to [-180, 180]
        error = np.mod(error + 180, 360) - 180
        
        axes[i, 1].fill_between(frames, 0, error, alpha=0.3, color=color)
        axes[i, 1].plot(frames, error, color=color, linewidth=1)
        axes[i, 1].axhline(np.mean(error), color='orange', linestyle='--', 
                          label=f'Mean: {np.mean(error):.2f}°')
        axes[i, 1].axhline(0, color='black', linestyle='-', linewidth=0.5)
        axes[i, 1].set_xlabel('Frame')
        axes[i, 1].set_ylabel(f'{label} Error (deg)')
        axes[i, 1].set_title(f'{label} Error')
        axes[i, 1].legend()
        axes[i, 1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_trajectory_comparison_multi(
    trajectories: Dict[str, np.ndarray],
    ground_truth: np.ndarray,
    output_path: Optional[Path] = None,
    title: str = "Multi-Method Trajectory Comparison",
    align: bool = True,
) -> Optional[plt.Figure]:
    """
    Compare multiple estimation methods in one plot.
    
    Args:
        trajectories: Dict of method_name -> poses (N, 4, 4)
        ground_truth: Ground truth poses (N, 4, 4)
        output_path: Where to save
        title: Plot title
        align: Whether to align each trajectory
    """
    if not HAS_MATPLOTLIB:
        return None
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    fig = plt.figure(figsize=(16, 6))
    
    # 2D XY
    ax1 = fig.add_subplot(131)
    gt_trans = ground_truth[:, :3, 3]
    ax1.plot(gt_trans[:, 0], gt_trans[:, 1], 'k-', linewidth=3, label='Ground Truth')
    
    for (name, traj), color in zip(trajectories.items(), colors):
        if align and len(traj) > 3:
            traj, scale, _, _ = align_trajectories_umeyama(traj, ground_truth)
            name = f"{name} (s={scale:.2f})"
        trans = traj[:, :3, 3]
        ax1.plot(trans[:, 0], trans[:, 1], '-', color=color, linewidth=1.5, 
                 label=name, alpha=0.8)
    
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Top View (XY)')
    ax1.legend(fontsize=8)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2D XZ
    ax2 = fig.add_subplot(132)
    ax2.plot(gt_trans[:, 0], gt_trans[:, 2], 'k-', linewidth=3, label='Ground Truth')
    
    for (name, traj), color in zip(trajectories.items(), colors):
        if align and len(traj) > 3:
            traj, _, _, _ = align_trajectories_umeyama(traj, ground_truth)
        trans = traj[:, :3, 3]
        ax2.plot(trans[:, 0], trans[:, 2], '-', color=color, linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.set_title('Side View (XZ)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # 3D
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(gt_trans[:, 0], gt_trans[:, 1], gt_trans[:, 2], 'k-', linewidth=3)
    
    for (name, traj), color in zip(trajectories.items(), colors):
        if align and len(traj) > 3:
            traj, _, _, _ = align_trajectories_umeyama(traj, ground_truth)
        trans = traj[:, :3, 3]
        ax3.plot(trans[:, 0], trans[:, 1], trans[:, 2], '-', color=color, 
                 linewidth=1.5, alpha=0.8)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D View')
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig


def plot_pose_error_summary(
    results: List[Dict],
    output_path: Optional[Path] = None,
    title: str = "Pose Estimation Error Summary",
) -> Optional[plt.Figure]:
    """
    Create comprehensive error summary with multiple metrics.
    """
    if not HAS_MATPLOTLIB or not results:
        return None
    
    # Group by method
    methods = sorted(set(r['method'] for r in results))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # ATE comparison by method
    ate_data = {m: [r['ate_rmse'] for r in results if r['method'] == m] for m in methods}
    positions = np.arange(len(methods))
    
    bp = axes[0, 0].boxplot([ate_data[m] for m in methods], positions=positions,
                            patch_artist=True)
    for patch, m in zip(bp['boxes'], methods):
        from .plot_utils import get_color
        patch.set_facecolor(get_color(m))
        patch.set_alpha(0.7)
    axes[0, 0].set_xticks(positions)
    axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 0].set_ylabel('ATE RMSE (m)')
    axes[0, 0].set_title('ATE by Method')
    axes[0, 0].grid(True, alpha=0.3)
    
    # RPE Translation
    rpe_trans_data = {m: [r['rpe_trans_rmse'] for r in results if r['method'] == m] for m in methods}
    bp = axes[0, 1].boxplot([rpe_trans_data[m] for m in methods], positions=positions,
                            patch_artist=True)
    for patch, m in zip(bp['boxes'], methods):
        from .plot_utils import get_color
        patch.set_facecolor(get_color(m))
        patch.set_alpha(0.7)
    axes[0, 1].set_xticks(positions)
    axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 1].set_ylabel('RPE Trans RMSE (m)')
    axes[0, 1].set_title('RPE Translation by Method')
    axes[0, 1].grid(True, alpha=0.3)
    
    # RPE Rotation
    rpe_rot_data = {m: [r['rpe_rot_rmse'] for r in results if r['method'] == m] for m in methods}
    bp = axes[0, 2].boxplot([rpe_rot_data[m] for m in methods], positions=positions,
                            patch_artist=True)
    for patch, m in zip(bp['boxes'], methods):
        from .plot_utils import get_color
        patch.set_facecolor(get_color(m))
        patch.set_alpha(0.7)
    axes[0, 2].set_xticks(positions)
    axes[0, 2].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 2].set_ylabel('RPE Rot RMSE (deg)')
    axes[0, 2].set_title('RPE Rotation by Method')
    axes[0, 2].grid(True, alpha=0.3)
    
    # FPS comparison
    fps_data = {m: [r['fps'] for r in results if r['method'] == m] for m in methods}
    means = [np.mean(fps_data[m]) for m in methods]
    bars = axes[1, 0].bar(positions, means, color=[get_color(m) for m in methods], alpha=0.7)
    axes[1, 0].set_xticks(positions)
    axes[1, 0].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 0].set_ylabel('FPS')
    axes[1, 0].set_title('Speed (FPS)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # ATE vs FPS scatter
    for m in methods:
        m_results = [r for r in results if r['method'] == m]
        ates = [r['ate_rmse'] for r in m_results]
        fpss = [r['fps'] for r in m_results]
        from .plot_utils import get_color
        axes[1, 1].scatter(fpss, ates, c=get_color(m), label=m, s=100, alpha=0.7)
    axes[1, 1].set_xlabel('FPS')
    axes[1, 1].set_ylabel('ATE RMSE (m)')
    axes[1, 1].set_title('Accuracy vs Speed')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Lost frames
    lost_data = {m: [r.get('lost_frames', 0) for r in results if r['method'] == m] for m in methods}
    means = [np.mean(lost_data[m]) for m in methods]
    axes[1, 2].bar(positions, means, color=[get_color(m) for m in methods], alpha=0.7)
    axes[1, 2].set_xticks(positions)
    axes[1, 2].set_xticklabels(methods, rotation=45, ha='right')
    axes[1, 2].set_ylabel('Lost Frames')
    axes[1, 2].set_title('Tracking Robustness')
    axes[1, 2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    return fig
