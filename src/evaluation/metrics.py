"""
Image and trajectory quality metrics for 3DGS evaluation.

Provides standard metrics used in NeRF/3DGS papers:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- ATE (Absolute Trajectory Error)
- RPE (Relative Pose Error)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
    _lpips_model = None
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from evo.core import trajectory, metrics, sync, geometry
    from evo.core.trajectory import PoseTrajectory3D
    import evo.main_ape as main_ape
    import evo.main_rpe as main_rpe
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False


def compute_psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1: First image (H, W, C) or (H, W)
        img2: Second image (same shape as img1)
        max_val: Maximum possible pixel value (1.0 for float, 255 for uint8)
        
    Returns:
        PSNR value in dB (higher is better, typically 20-40 for good reconstructions)
    """
    if SKIMAGE_AVAILABLE:
        return float(psnr_fn(img1, img2, data_range=max_val))
    
    # Fallback implementation
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(max_val ** 2 / mse))


def compute_ssim(
    img1: np.ndarray, 
    img2: np.ndarray,
    multichannel: bool = True,
    data_range: float = 1.0
) -> float:
    """
    Compute Structural Similarity Index between two images.
    
    Args:
        img1: First image (H, W, C) or (H, W)
        img2: Second image (same shape as img1)
        multichannel: Whether images have multiple channels
        data_range: The data range of the images
        
    Returns:
        SSIM value in [0, 1] (higher is better)
    """
    if not SKIMAGE_AVAILABLE:
        logger.warning("skimage not available for SSIM computation")
        return 0.0
    
    # Handle different skimage versions
    try:
        # Newer versions use channel_axis
        if len(img1.shape) == 3:
            return float(ssim_fn(img1, img2, data_range=data_range, channel_axis=2))
        else:
            return float(ssim_fn(img1, img2, data_range=data_range))
    except TypeError:
        # Older versions use multichannel
        return float(ssim_fn(img1, img2, data_range=data_range, multichannel=multichannel))


def compute_lpips(
    img1: np.ndarray,
    img2: np.ndarray,
    net: str = 'alex',
    device: str = 'cuda'
) -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).
    
    Args:
        img1: First image (H, W, C) in [0, 1] range
        img2: Second image (same shape as img1)
        net: Network to use ('alex', 'vgg', 'squeeze')
        device: Device to run on ('cuda', 'mps', 'cpu', or 'auto')
        
    Returns:
        LPIPS value (lower is better, typically 0.0-0.5 for good reconstructions)
    """
    if not LPIPS_AVAILABLE or not TORCH_AVAILABLE:
        logger.warning("lpips or torch not available")
        return 0.0
    
    global _lpips_model
    
    # Determine actual device to use
    actual_device = device
    if device == 'cuda' and not torch.cuda.is_available():
        # Try MPS if CUDA not available (macOS)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            actual_device = 'mps'
        else:
            actual_device = 'cpu'
    elif device == 'auto':
        if torch.cuda.is_available():
            actual_device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            actual_device = 'mps'
        else:
            actual_device = 'cpu'
    
    # Lazy load model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net=net, verbose=False)
        if actual_device != 'cpu':
            _lpips_model = _lpips_model.to(actual_device)
    
    # Convert to tensor: (H, W, C) -> (1, C, H, W)
    if img1.max() > 1.0:
        img1 = img1 / 255.0
        img2 = img2 / 255.0
    
    # Convert to [-1, 1] range as expected by LPIPS
    img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
    img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() * 2 - 1
    
    if actual_device != 'cpu':
        img1_t = img1_t.to(actual_device)
        img2_t = img2_t.to(actual_device)
    
    with torch.no_grad():
        score = _lpips_model(img1_t, img2_t)
    
    return float(score.item())


def compute_image_metrics(
    rendered: np.ndarray,
    ground_truth: np.ndarray,
    compute_lpips_metric: bool = True
) -> Dict[str, float]:
    """
    Compute all image quality metrics between rendered and ground truth images.
    
    Args:
        rendered: Rendered image (H, W, C) uint8 or float in [0, 1]
        ground_truth: Ground truth image (same shape as rendered)
        compute_lpips_metric: Whether to compute LPIPS (slower, requires GPU)
        
    Returns:
        Dictionary with 'psnr', 'ssim', 'lpips' values
    """
    # Normalize to [0, 1] float
    if rendered.dtype == np.uint8:
        rendered = rendered.astype(np.float32) / 255.0
    if ground_truth.dtype == np.uint8:
        ground_truth = ground_truth.astype(np.float32) / 255.0
    
    metrics = {
        'psnr': compute_psnr(rendered, ground_truth, max_val=1.0),
        'ssim': compute_ssim(rendered, ground_truth, data_range=1.0),
    }
    
    if compute_lpips_metric:
        metrics['lpips'] = compute_lpips(rendered, ground_truth)
    
    return metrics


def compute_ate(
    estimated_poses: np.ndarray,
    ground_truth_poses: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    align: bool = True
) -> Dict[str, float]:
    """
    Compute Absolute Trajectory Error (ATE).
    
    Args:
        estimated_poses: Nx4x4 array of estimated camera-to-world transforms
        ground_truth_poses: Nx4x4 array of ground truth transforms
        timestamps: Optional Nx1 array of timestamps for alignment
        align: Whether to align trajectories using Umeyama alignment
        
    Returns:
        Dictionary with 'ate_rmse', 'ate_mean', 'ate_median', 'ate_std', 'ate_max'
    """
    # Helper function for fallback computation
    def compute_ate_fallback(est_poses, gt_poses, do_align):
        est_pos = est_poses[:, :3, 3].copy()
        gt_pos = gt_poses[:, :3, 3].copy()
        
        if do_align:
            # Umeyama alignment fallback
            try:
                from scipy.spatial.transform import Rotation
                
                # Center the points
                est_mean = est_pos.mean(axis=0)
                gt_mean = gt_pos.mean(axis=0)
                est_centered = est_pos - est_mean
                gt_centered = gt_pos - gt_mean
                
                # Compute optimal rotation using SVD
                H = est_centered.T @ gt_centered
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                
                # Handle reflection case
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                
                # Apply transformation
                est_pos = (R @ est_centered.T).T + gt_mean
            except Exception:
                # Simple centroid alignment as last resort
                est_pos = est_pos - est_pos.mean(axis=0) + gt_pos.mean(axis=0)
        
        errors = np.linalg.norm(est_pos - gt_pos, axis=1)
        return {
            'ate_rmse': float(np.sqrt(np.mean(errors ** 2))),
            'ate_mean': float(np.mean(errors)),
            'ate_median': float(np.median(errors)),
            'ate_std': float(np.std(errors)),
            'ate_max': float(np.max(errors)),
        }
    
    if not EVO_AVAILABLE:
        return compute_ate_fallback(estimated_poses, ground_truth_poses, align)
    
    # Try evo library for proper trajectory evaluation
    try:
        if timestamps is None:
            timestamps = np.arange(len(estimated_poses), dtype=np.float64)
        
        # Convert to evo trajectories
        est_traj = poses_to_trajectory(estimated_poses, timestamps)
        gt_traj = poses_to_trajectory(ground_truth_poses, timestamps)
        
        # Synchronize trajectories
        gt_traj, est_traj = sync.associate_trajectories(gt_traj, est_traj)
        
        # Align if requested using Umeyama alignment
        if align:
            try:
                r, t, s = geometry.umeyama_alignment(
                    est_traj.positions_xyz.T,
                    gt_traj.positions_xyz.T,
                    with_scale=False
                )
                # Apply transformation to estimated trajectory
                transform_matrix = np.eye(4)
                transform_matrix[:3, :3] = r
                transform_matrix[:3, 3] = t
                est_traj.transform(transform_matrix)
            except Exception:
                # If umeyama fails, continue without alignment
                pass
        
        # Compute APE (Absolute Pose Error)
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data((gt_traj, est_traj))
        
        stats = ape_metric.get_all_statistics()
        
        return {
            'ate_rmse': float(stats['rmse']),
            'ate_mean': float(stats['mean']),
            'ate_median': float(stats['median']),
            'ate_std': float(stats['std']),
            'ate_max': float(stats['max']),
        }
    except Exception as e:
        # Fallback to simple computation
        logger.warning(f"evo ATE computation failed: {e}, using fallback")
        return compute_ate_fallback(estimated_poses, ground_truth_poses, align)


def compute_rpe(
    estimated_poses: np.ndarray,
    ground_truth_poses: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    delta: float = 1.0,
    delta_unit: str = 'f'  # 'f' for frames, 'm' for meters, 's' for seconds
) -> Dict[str, float]:
    """
    Compute Relative Pose Error (RPE).
    
    Args:
        estimated_poses: Nx4x4 array of estimated camera-to-world transforms
        ground_truth_poses: Nx4x4 array of ground truth transforms
        timestamps: Optional Nx1 array of timestamps
        delta: The delta for relative poses
        delta_unit: Unit for delta ('f' frames, 'm' meters, 's' seconds)
        
    Returns:
        Dictionary with 'rpe_trans_rmse', 'rpe_rot_rmse' etc.
    """
    if not EVO_AVAILABLE:
        # Simplified fallback
        n = len(estimated_poses)
        trans_errors = []
        rot_errors = []
        
        delta_int = int(delta)
        for i in range(n - delta_int):
            # Relative pose in estimated
            est_rel = np.linalg.inv(estimated_poses[i]) @ estimated_poses[i + delta_int]
            gt_rel = np.linalg.inv(ground_truth_poses[i]) @ ground_truth_poses[i + delta_int]
            
            # Error
            error = np.linalg.inv(gt_rel) @ est_rel
            trans_errors.append(np.linalg.norm(error[:3, 3]))
            rot_errors.append(np.arccos(np.clip((np.trace(error[:3, :3]) - 1) / 2, -1, 1)))
        
        trans_errors = np.array(trans_errors)
        rot_errors = np.array(rot_errors)
        
        return {
            'rpe_trans_rmse': float(np.sqrt(np.mean(trans_errors ** 2))),
            'rpe_trans_mean': float(np.mean(trans_errors)),
            'rpe_rot_rmse': float(np.sqrt(np.mean(rot_errors ** 2))),
            'rpe_rot_mean': float(np.mean(rot_errors)),
        }
    
    # Use evo for proper RPE computation
    if timestamps is None:
        timestamps = np.arange(len(estimated_poses), dtype=np.float64)
    
    est_traj = poses_to_trajectory(estimated_poses, timestamps)
    gt_traj = poses_to_trajectory(ground_truth_poses, timestamps)
    
    gt_traj, est_traj = sync.associate_trajectories(gt_traj, est_traj)
    
    # Translation RPE
    rpe_trans = metrics.RPE(
        metrics.PoseRelation.translation_part,
        delta=delta,
        delta_unit=metrics.Unit.frames if delta_unit == 'f' else metrics.Unit.meters
    )
    rpe_trans.process_data((gt_traj, est_traj))
    trans_stats = rpe_trans.get_all_statistics()
    
    # Rotation RPE
    rpe_rot = metrics.RPE(
        metrics.PoseRelation.rotation_angle_deg,
        delta=delta,
        delta_unit=metrics.Unit.frames if delta_unit == 'f' else metrics.Unit.meters
    )
    rpe_rot.process_data((gt_traj, est_traj))
    rot_stats = rpe_rot.get_all_statistics()
    
    return {
        'rpe_trans_rmse': float(trans_stats['rmse']),
        'rpe_trans_mean': float(trans_stats['mean']),
        'rpe_trans_std': float(trans_stats['std']),
        'rpe_rot_rmse': float(rot_stats['rmse']),
        'rpe_rot_mean': float(rot_stats['mean']),
        'rpe_rot_std': float(rot_stats['std']),
    }


def compute_trajectory_metrics(
    estimated_poses: np.ndarray,
    ground_truth_poses: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    align: bool = True
) -> Dict[str, float]:
    """
    Compute all trajectory metrics (ATE and RPE).
    
    Args:
        estimated_poses: Nx4x4 estimated poses
        ground_truth_poses: Nx4x4 ground truth poses  
        timestamps: Optional timestamps
        align: Whether to align trajectories
        
    Returns:
        Dictionary with all trajectory metrics
    """
    metrics = {}
    
    # ATE
    ate = compute_ate(estimated_poses, ground_truth_poses, timestamps, align)
    metrics.update(ate)
    
    # RPE
    rpe = compute_rpe(estimated_poses, ground_truth_poses, timestamps)
    metrics.update(rpe)
    
    return metrics


def poses_to_trajectory(poses: np.ndarray, timestamps: np.ndarray):
    """Convert Nx4x4 poses to evo PoseTrajectory3D."""
    if not EVO_AVAILABLE:
        return None
    
    from scipy.spatial.transform import Rotation
    
    positions = poses[:, :3, 3]
    quaternions = []
    for pose in poses:
        r = Rotation.from_matrix(pose[:3, :3])
        q = r.as_quat()  # xyzw
        quaternions.append([q[3], q[0], q[1], q[2]])  # wxyz for evo
    
    quaternions = np.array(quaternions)
    
    return PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=quaternions,
        timestamps=timestamps
    )


def load_metrics_from_json(filepath: Union[str, Path]) -> Dict[str, float]:
    """Load metrics from a JSON file."""
    with open(filepath) as f:
        return json.load(f)


def save_metrics_to_json(metrics: Dict[str, float], filepath: Union[str, Path]) -> None:
    """Save metrics to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)
