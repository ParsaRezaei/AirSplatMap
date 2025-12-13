"""
GSplat Engine - Nerfstudio's Optimized Gaussian Splatting Backend
==================================================================

This engine wraps the gsplat library (https://github.com/nerfstudio-project/gsplat)
which provides optimized CUDA kernels for 3D Gaussian Splatting.

gsplat advantages over original 3DGS:
- Up to 4x less GPU memory
- 15% faster training  
- Clean Python API
- Supports depth loss, camera optimization, MCMC densification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import json
import logging

from .base import BaseGSEngine

logger = logging.getLogger(__name__)


class GSplatEngine(BaseGSEngine):
    """
    3D Gaussian Splatting engine using gsplat library.
    
    This provides a cleaner, more memory-efficient implementation
    compared to the original GRAPHDECO code.
    """
    
    def __init__(self, device: str = "cuda:0"):
        """
        Initialize the GSplat engine.
        
        Args:
            device: CUDA device to use (default: "cuda:0")
        """
        # Ensure CUDA is available - gsplat requires GPU
        if not torch.cuda.is_available():
            raise RuntimeError("GSplat requires CUDA but no GPU is available")
        
        # Use specified device or default to first GPU
        if device == "cuda" or device == "cuda:0":
            self.device = torch.device("cuda:0")
        elif device.startswith("cuda:"):
            device_id = int(device.split(":")[1])
            if device_id >= torch.cuda.device_count():
                logger.warning(f"Device {device} not available, using cuda:0")
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device(device)
        else:
            logger.warning(f"GSplat requires CUDA, ignoring device={device}")
            self.device = torch.device("cuda:0")
        
        self._initialized = False
        self._intrinsics = None
        self._config = None
        
        # Gaussian parameters (learnable)
        self._means = None          # (N, 3) positions
        self._scales = None         # (N, 3) log-scales  
        self._quats = None          # (N, 4) quaternions for rotation
        self._opacities = None      # (N,) logit-opacities
        self._sh_coeffs = None      # (N, K, 3) spherical harmonics
        
        # Training state
        self._optimizer = None
        self._iteration = 0
        self._frames = []  # List of registered frames
        
        # Background color
        self._bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device)
        
        logger.info(f"GSplat engine initialized on {self.device}")
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def is_available(self) -> bool:
        """GSplat is always available once imported."""
        return True
    
    def initialize_scene(
        self,
        intrinsics: Dict[str, float],
        config: Dict[str, Any]
    ) -> None:
        """Initialize the scene with camera intrinsics and config."""
        
        if self._initialized:
            raise RuntimeError("Scene already initialized. Call reset() first.")
        
        # Validate intrinsics
        required_keys = ['fx', 'fy', 'cx', 'cy', 'width', 'height']
        for key in required_keys:
            if key not in intrinsics:
                raise ValueError(f"Missing required intrinsic: {key}")
        
        self._intrinsics = intrinsics.copy()
        
        # Default config
        self._config = {
            'sh_degree': config.get('sh_degree', 3),
            'position_lr': config.get('position_lr_init', 0.00016),
            'scale_lr': config.get('scaling_lr', 0.005),
            'rotation_lr': config.get('rotation_lr', 0.001),
            'opacity_lr': config.get('opacity_lr', 0.05),
            'sh_lr': config.get('feature_lr', 0.0025),
            'densify_grad_threshold': config.get('densify_grad_threshold', 0.0002),
            'densification_interval': config.get('densification_interval', 100),
            'opacity_reset_interval': config.get('opacity_reset_interval', 3000),
            'densify_from_iter': config.get('densify_from_iter', 500),
            'densify_until_iter': config.get('densify_until_iter', 15000),
            'max_gaussians': config.get('max_gaussians', 200000),
            'percent_dense': config.get('percent_dense', 0.01),
            'lambda_dssim': config.get('lambda_dssim', 0.2),
        }
        self._config.update(config)
        
        # Initialize empty Gaussians (will be populated from first frame with depth)
        self._means = torch.empty(0, 3, device=self.device, dtype=torch.float32)
        self._scales = torch.empty(0, 3, device=self.device, dtype=torch.float32)
        self._quats = torch.empty(0, 4, device=self.device, dtype=torch.float32)
        self._opacities = torch.empty(0, device=self.device, dtype=torch.float32)
        
        # SH coefficients: (N, (sh_degree+1)^2, 3)
        sh_dim = (self._config['sh_degree'] + 1) ** 2
        self._sh_coeffs = torch.empty(0, sh_dim, 3, device=self.device, dtype=torch.float32)
        
        # Gradient accumulators for densification
        self._xyz_gradient_accum = torch.empty(0, device=self.device)
        self._denom = torch.empty(0, device=self.device)
        
        self._initialized = True
        logger.info(f"GSplat scene initialized: SH degree={self._config['sh_degree']}, "
                   f"image size={intrinsics['width']}x{intrinsics['height']}")
    
    def _init_gaussians_from_points(self, points: np.ndarray, colors: np.ndarray) -> None:
        """Initialize Gaussians from a point cloud."""
        
        n_points = len(points)
        if n_points == 0:
            return
            
        logger.info(f"Initializing {n_points:,} Gaussians from point cloud")
        
        # Positions
        self._means = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        # Compute initial scales based on local point density
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=4)  # Find 3 nearest neighbors
        avg_dist = np.mean(dists[:, 1:], axis=1)  # Exclude self
        avg_dist = np.clip(avg_dist, 0.001, 0.1)
        
        # Log-scales (3 per Gaussian, isotropic initially)
        scales = np.log(avg_dist[:, None] * np.ones((n_points, 3)))
        self._scales = torch.tensor(scales, dtype=torch.float32, device=self.device)
        
        # Quaternions (identity rotation)
        quats = np.zeros((n_points, 4))
        quats[:, 0] = 1.0  # w=1, x=y=z=0
        self._quats = torch.tensor(quats, dtype=torch.float32, device=self.device)
        
        # Opacities (logit space, start at ~0.5)
        self._opacities = torch.zeros(n_points, dtype=torch.float32, device=self.device)
        
        # SH coefficients from colors
        sh_dim = (self._config['sh_degree'] + 1) ** 2
        sh_coeffs = np.zeros((n_points, sh_dim, 3))
        # DC component (first SH basis function)
        C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))
        sh_coeffs[:, 0, :] = (colors / 255.0 - 0.5) / C0
        self._sh_coeffs = torch.tensor(sh_coeffs, dtype=torch.float32, device=self.device)
        
        # Gradient accumulators
        self._xyz_gradient_accum = torch.zeros(n_points, device=self.device)
        self._denom = torch.zeros(n_points, device=self.device)
        
        # Setup optimizer
        self._setup_optimizer()
    
    def _setup_optimizer(self) -> None:
        """Setup the optimizer for Gaussian parameters."""
        
        # Make parameters require gradients
        self._means.requires_grad_(True)
        self._scales.requires_grad_(True)
        self._quats.requires_grad_(True)
        self._opacities.requires_grad_(True)
        self._sh_coeffs.requires_grad_(True)
        
        # Create optimizer with different learning rates per parameter group
        params = [
            {'params': [self._means], 'lr': self._config['position_lr'], 'name': 'xyz'},
            {'params': [self._scales], 'lr': self._config['scale_lr'], 'name': 'scaling'},
            {'params': [self._quats], 'lr': self._config['rotation_lr'], 'name': 'rotation'},
            {'params': [self._opacities], 'lr': self._config['opacity_lr'], 'name': 'opacity'},
            {'params': [self._sh_coeffs], 'lr': self._config['sh_lr'], 'name': 'sh'},
        ]
        
        self._optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
    
    def add_frame(
        self,
        frame_id: int,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        pose_world_cam: np.ndarray
    ) -> None:
        """Add a new frame to the scene."""
        
        if not self._initialized:
            raise RuntimeError("Scene not initialized. Call initialize_scene() first.")
        
        # Convert to tensors
        if rgb.dtype == np.uint8:
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32, device=self.device) / 255.0
        else:
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32, device=self.device)
        
        pose_tensor = torch.tensor(pose_world_cam, dtype=torch.float32, device=self.device)
        
        depth_tensor = None
        if depth is not None:
            depth_tensor = torch.tensor(depth, dtype=torch.float32, device=self.device)
        
        # Store frame
        frame_data = {
            'id': frame_id,
            'rgb': rgb_tensor,
            'depth': depth_tensor,
            'pose': pose_tensor,
        }
        self._frames.append(frame_data)
        
        # Initialize Gaussians from first frame with depth
        if len(self._means) == 0 and depth is not None:
            points, colors = self._unproject_depth(rgb, depth, pose_world_cam)
            if len(points) > 0:
                self._init_gaussians_from_points(points, colors)
        
        # Optionally add new Gaussians from this frame
        elif depth is not None and self._config.get('add_gaussians_per_frame', True):
            if frame_id % self._config.get('add_gaussians_every', 5) == 0:
                self._add_gaussians_from_frame(rgb, depth, pose_world_cam)
    
    def _unproject_depth(
        self, 
        rgb: np.ndarray, 
        depth: np.ndarray, 
        pose: np.ndarray,
        subsample: int = 4
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject depth map to 3D points."""
        
        fx = self._intrinsics['fx']
        fy = self._intrinsics['fy']
        cx = self._intrinsics['cx']
        cy = self._intrinsics['cy']
        
        h, w = depth.shape
        
        # Create pixel grid (subsampled)
        u = np.arange(0, w, subsample)
        v = np.arange(0, h, subsample)
        u, v = np.meshgrid(u, v)
        u = u.flatten()
        v = v.flatten()
        
        # Get depths at these pixels
        d = depth[v, u]
        
        # Filter valid depths
        valid = (d > 0.1) & (d < 10.0) & np.isfinite(d)
        u = u[valid]
        v = v[valid]
        d = d[valid]
        
        if len(d) == 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        
        # Unproject to camera frame
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        z = d
        
        points_cam = np.stack([x, y, z], axis=1)
        
        # Transform to world frame
        R = pose[:3, :3]
        t = pose[:3, 3]
        points_world = (R @ points_cam.T).T + t
        
        # Get colors
        colors = rgb[v, u]
        
        return points_world, colors
    
    def _add_gaussians_from_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray
    ) -> None:
        """Add new Gaussians from a frame (for incremental mapping)."""
        
        subsample = self._config.get('add_gaussians_subsample', 8)
        points, colors = self._unproject_depth(rgb, depth, pose, subsample)
        
        if len(points) == 0:
            return
        
        # Check max Gaussians limit
        current_n = len(self._means)
        max_n = self._config['max_gaussians']
        if current_n >= max_n:
            return
        
        # Limit new points
        n_new = min(len(points), max_n - current_n)
        if n_new < len(points):
            indices = np.random.choice(len(points), n_new, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        # Create new Gaussian parameters
        new_means = torch.tensor(points, dtype=torch.float32, device=self.device)
        
        # Initial scales
        init_scale = np.log(0.01)
        new_scales = torch.full((n_new, 3), init_scale, dtype=torch.float32, device=self.device)
        
        # Identity rotations
        new_quats = torch.zeros((n_new, 4), dtype=torch.float32, device=self.device)
        new_quats[:, 0] = 1.0
        
        # Initial opacities
        new_opacities = torch.zeros(n_new, dtype=torch.float32, device=self.device)
        
        # SH coefficients
        sh_dim = (self._config['sh_degree'] + 1) ** 2
        new_sh = torch.zeros((n_new, sh_dim, 3), dtype=torch.float32, device=self.device)
        C0 = 0.28209479177387814
        new_sh[:, 0, :] = torch.tensor((colors / 255.0 - 0.5) / C0, dtype=torch.float32, device=self.device)
        
        # Concatenate with existing
        self._means = nn.Parameter(torch.cat([self._means.data, new_means], dim=0))
        self._scales = nn.Parameter(torch.cat([self._scales.data, new_scales], dim=0))
        self._quats = nn.Parameter(torch.cat([self._quats.data, new_quats], dim=0))
        self._opacities = nn.Parameter(torch.cat([self._opacities.data, new_opacities], dim=0))
        self._sh_coeffs = nn.Parameter(torch.cat([self._sh_coeffs.data, new_sh], dim=0))
        
        # Extend gradient accumulators
        self._xyz_gradient_accum = torch.cat([
            self._xyz_gradient_accum,
            torch.zeros(n_new, device=self.device)
        ])
        self._denom = torch.cat([
            self._denom,
            torch.zeros(n_new, device=self.device)
        ])
        
        # Recreate optimizer with new parameters
        self._setup_optimizer()
    
    def optimize_step(self, n_steps: int = 1) -> Dict[str, float]:
        """Run optimization steps."""
        
        if not self._initialized:
            raise RuntimeError("Scene not initialized")
        
        if len(self._frames) == 0:
            return {'loss': 0.0, 'num_gaussians': 0}
        
        if len(self._means) == 0:
            return {'loss': 0.0, 'num_gaussians': 0, 'psnr': 0.0}
        
        try:
            import gsplat
        except ImportError:
            raise ImportError("gsplat not installed. Run: pip install gsplat")
        
        total_loss = 0.0
        
        for step in range(n_steps):
            # Sample a random frame
            frame = self._frames[np.random.randint(len(self._frames))]
            
            # Render
            rendered = self._render_frame(frame['pose'], gsplat)
            
            # Compute loss
            gt = frame['rgb']
            l1_loss = torch.abs(rendered - gt).mean()
            
            # SSIM loss (simplified)
            ssim_loss = 1.0 - self._ssim(rendered, gt)
            
            lambda_dssim = self._config['lambda_dssim']
            loss = (1 - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss
            
            # Backward pass
            self._optimizer.zero_grad()
            loss.backward()
            
            # Accumulate gradients for densification
            if self._means.grad is not None:
                grad_norm = self._means.grad.norm(dim=1)
                self._xyz_gradient_accum += grad_norm
                self._denom += 1
            
            # Step optimizer
            self._optimizer.step()
            
            # Clamp opacities
            with torch.no_grad():
                self._opacities.clamp_(-10, 10)
                # Normalize quaternions
                self._quats.data = self._quats.data / (self._quats.data.norm(dim=1, keepdim=True) + 1e-8)
            
            total_loss += loss.item()
            self._iteration += 1
            
            # Densification
            if self._iteration % self._config['densification_interval'] == 0:
                if self._config['densify_from_iter'] <= self._iteration <= self._config['densify_until_iter']:
                    self._densify_and_prune()
        
        avg_loss = total_loss / n_steps
        
        # Compute PSNR
        with torch.no_grad():
            mse = ((rendered - gt) ** 2).mean().item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-10)) if mse > 0 else 40.0
        
        return {
            'loss': avg_loss,
            'psnr': psnr,
            'num_gaussians': len(self._means),
            'iteration': self._iteration,
        }
    
    def _render_frame(self, pose: torch.Tensor, gsplat) -> torch.Tensor:
        """Render a frame using gsplat."""
        
        width = int(self._intrinsics['width'])
        height = int(self._intrinsics['height'])
        fx = self._intrinsics['fx']
        fy = self._intrinsics['fy']
        cx = self._intrinsics['cx']
        cy = self._intrinsics['cy']
        
        # Camera matrices
        # gsplat expects world-to-camera transform
        viewmat = torch.linalg.inv(pose)  # (4, 4)
        
        # Get Gaussian parameters
        means = self._means  # (N, 3)
        scales = torch.exp(self._scales)  # (N, 3)
        quats = self._quats  # (N, 4)
        opacities = torch.sigmoid(self._opacities)  # (N,)
        
        # Compute colors from SH
        colors = self._sh_to_rgb(self._sh_coeffs, means, pose[:3, 3])
        
        try:
            # Ensure CUDA is synchronized before rasterization
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            
            # gsplat 0.1.x API uses project_gaussians + rasterize_gaussians
            # This is the prebuilt Windows-compatible version
            
            # Tile/block size for rasterization
            BLOCK_X, BLOCK_Y = 16, 16
            tile_bounds = (
                (width + BLOCK_X - 1) // BLOCK_X,
                (height + BLOCK_Y - 1) // BLOCK_Y,
                1
            )
            block = (BLOCK_X, BLOCK_Y, 1)
            img_size = (width, height)
            
            # Near/far planes
            near_plane = 0.01
            far_plane = 100.0
            
            # Compute 2D projection using gsplat.project_gaussians
            # viewmat is world-to-camera (4x4)
            xys, depths, radii, conics, compensation, num_tiles_hit, cov3d = gsplat.project_gaussians(
                means3d=means,
                scales=scales,
                glob_scale=1.0,
                quats=quats,
                viewmat=viewmat,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                img_height=height,
                img_width=width,
                block_width=BLOCK_X,
                clip_thresh=0.01,
            )
            
            # Check if any Gaussians are visible
            visible_mask = radii > 0
            if visible_mask.sum() == 0:
                return torch.zeros(height, width, 3, device=self.device)
            
            # Rasterize using gsplat.rasterize_gaussians
            out_img, out_alpha = gsplat.rasterize_gaussians(
                xys=xys,
                depths=depths,
                radii=radii,
                conics=conics,
                num_tiles_hit=num_tiles_hit,
                colors=colors,
                opacity=opacities,
                img_height=height,
                img_width=width,
                block_width=BLOCK_X,
                background=self._bg_color,
                return_alpha=True,
            )
            
            # out_img is (H, W, 3)
            return out_img
            
        except Exception as e:
            # Convert exception to string safely to avoid codec errors
            try:
                err_msg = str(e)
            except:
                err_msg = "Unknown error (could not decode error message)"
            
            # Only log first occurrence to avoid spam
            if not hasattr(self, '_rasterization_error_logged'):
                self._rasterization_error_logged = True
                logger.warning(f"gsplat rasterization failed: {err_msg}, using fallback")
            
            # Mark that we're using fallback (for benchmark results)
            self._using_fallback = True
            
            # Fallback - returns zeros (black) which gives bad metrics
            return torch.zeros(height, width, 3, device=self.device)
    
    def _sh_to_rgb(self, sh: torch.Tensor, positions: torch.Tensor, camera_pos: torch.Tensor) -> torch.Tensor:
        """Convert SH coefficients to RGB colors given view direction."""
        
        # View direction (from Gaussian to camera)
        dirs = camera_pos.unsqueeze(0) - positions  # (N, 3)
        dirs = dirs / (dirs.norm(dim=1, keepdim=True) + 1e-8)
        
        # For now, just use DC component (degree 0)
        C0 = 0.28209479177387814
        colors = sh[:, 0, :] * C0 + 0.5  # (N, 3)
        
        return torch.clamp(colors, 0, 1)
    
    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Simplified SSIM computation."""
        # Simple structural similarity approximation
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1_sq = ((img1 - mu1) ** 2).mean()
        sigma2_sq = ((img2 - mu2) ** 2).mean()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim
    
    def _densify_and_prune(self) -> None:
        """Densify and prune Gaussians based on gradients."""
        
        if len(self._means) == 0:
            return
        
        # Compute average gradient
        avg_grad = self._xyz_gradient_accum / (self._denom + 1e-8)
        
        # Find Gaussians to split/clone
        grad_threshold = self._config['densify_grad_threshold']
        high_grad_mask = avg_grad > grad_threshold
        
        # Prune low opacity
        opacity_threshold = 0.005
        opacities = torch.sigmoid(self._opacities)
        low_opacity_mask = opacities < opacity_threshold
        
        # Prune
        keep_mask = ~low_opacity_mask
        if keep_mask.sum() < len(self._means):
            self._prune_gaussians(keep_mask)
        
        # Reset accumulators
        self._xyz_gradient_accum = torch.zeros(len(self._means), device=self.device)
        self._denom = torch.zeros(len(self._means), device=self.device)
    
    def _prune_gaussians(self, keep_mask: torch.Tensor) -> None:
        """Remove Gaussians based on mask."""
        
        self._means = nn.Parameter(self._means.data[keep_mask])
        self._scales = nn.Parameter(self._scales.data[keep_mask])
        self._quats = nn.Parameter(self._quats.data[keep_mask])
        self._opacities = nn.Parameter(self._opacities.data[keep_mask])
        self._sh_coeffs = nn.Parameter(self._sh_coeffs.data[keep_mask])
        
        self._xyz_gradient_accum = self._xyz_gradient_accum[keep_mask]
        self._denom = self._denom[keep_mask]
        
        self._setup_optimizer()
    
    def render_view(
        self,
        pose_world_cam: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Render a view from the current Gaussian scene."""
        
        if not self._initialized:
            raise RuntimeError("Scene not initialized")
        
        if len(self._means) == 0:
            return np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        try:
            import gsplat
        except ImportError:
            return np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        pose_tensor = torch.tensor(pose_world_cam, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            rendered = self._render_frame(pose_tensor, gsplat)
            rendered_np = (rendered.cpu().numpy() * 255).astype(np.uint8)
        
        return rendered_np
    
    def save_state(self, path: str) -> None:
        """Save the engine state."""
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if len(self._means) == 0:
            logger.warning("No Gaussians to save")
            return
        
        # Save as PLY (standard 3DGS format)
        ply_path = path / "point_cloud.ply"
        self._save_ply(ply_path)
        
        # Save checkpoint
        checkpoint = {
            'means': self._means.data.cpu(),
            'scales': self._scales.data.cpu(),
            'quats': self._quats.data.cpu(),
            'opacities': self._opacities.data.cpu(),
            'sh_coeffs': self._sh_coeffs.data.cpu(),
            'iteration': self._iteration,
        }
        torch.save(checkpoint, path / "checkpoint.pth")
        
        # Save config
        config = {
            'intrinsics': self._intrinsics,
            'config': self._config,
            'num_gaussians': len(self._means),
            'engine': 'gsplat',
        }
        with open(path / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved {len(self._means):,} Gaussians to {path}")
    
    def _save_ply(self, path: Path) -> None:
        """Save Gaussians as PLY file."""
        
        means = self._means.data.cpu().numpy()
        scales = self._scales.data.cpu().numpy()
        quats = self._quats.data.cpu().numpy()
        opacities = self._opacities.data.cpu().numpy()
        sh_coeffs = self._sh_coeffs.data.cpu().numpy()
        
        n = len(means)
        sh_dim = sh_coeffs.shape[1]
        
        # Build PLY header
        header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float nx
property float ny
property float nz
"""
        # SH coefficients
        for i in range(sh_dim):
            for c in ['_r', '_g', '_b']:
                if i == 0:
                    header += f"property float f_dc{c[1:]}\n"
                else:
                    header += f"property float f_rest_{i-1}{c}\n"
        
        header += """property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
        
        # Build vertex data
        import struct
        vertex_data = []
        for i in range(n):
            # Position
            vertex_data.extend(means[i])
            # Normal (dummy)
            vertex_data.extend([0, 0, 0])
            # SH coefficients (flattened: RGB interleaved)
            for j in range(sh_dim):
                vertex_data.extend(sh_coeffs[i, j])
            # Opacity (logit)
            vertex_data.append(opacities[i])
            # Scales (log)
            vertex_data.extend(scales[i])
            # Quaternion
            vertex_data.extend(quats[i])
        
        # Write binary
        with open(path, 'wb') as f:
            f.write(header.encode())
            f.write(struct.pack(f'{len(vertex_data)}f', *vertex_data))
    
    def load_state(self, path: str) -> None:
        """Load engine state from disk."""
        
        path = Path(path)
        
        checkpoint_path = path / "checkpoint.pth"
        config_path = path / "config.json"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load config
        if config_path.exists():
            with open(config_path) as f:
                saved_config = json.load(f)
            self._intrinsics = saved_config.get('intrinsics')
            self._config = saved_config.get('config', {})
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self._means = nn.Parameter(checkpoint['means'].to(self.device))
        self._scales = nn.Parameter(checkpoint['scales'].to(self.device))
        self._quats = nn.Parameter(checkpoint['quats'].to(self.device))
        self._opacities = nn.Parameter(checkpoint['opacities'].to(self.device))
        self._sh_coeffs = nn.Parameter(checkpoint['sh_coeffs'].to(self.device))
        self._iteration = checkpoint.get('iteration', 0)
        
        # Initialize accumulators
        n = len(self._means)
        self._xyz_gradient_accum = torch.zeros(n, device=self.device)
        self._denom = torch.zeros(n, device=self.device)
        
        self._setup_optimizer()
        self._initialized = True
        
        logger.info(f"Loaded {n:,} Gaussians from {path}")
    
    def reset(self) -> None:
        """Reset the engine to uninitialized state."""
        
        self._initialized = False
        self._intrinsics = None
        self._config = None
        self._means = None
        self._scales = None
        self._quats = None
        self._opacities = None
        self._sh_coeffs = None
        self._optimizer = None
        self._iteration = 0
        self._frames = []
        self._xyz_gradient_accum = None
        self._denom = None
    
    def get_num_gaussians(self) -> int:
        if self._means is None:
            return 0
        return len(self._means)
    
    def get_point_cloud(self) -> Optional[np.ndarray]:
        if self._means is None or len(self._means) == 0:
            return None
        return self._means.data.cpu().numpy()
    
    def get_gaussian_colors(self) -> Optional[np.ndarray]:
        """Get RGB colors for each Gaussian (from SH DC component)."""
        if self._sh_coeffs is None or len(self._sh_coeffs) == 0:
            return None
        C0 = 0.28209479177387814
        colors = self._sh_coeffs[:, 0, :].data * C0 + 0.5
        return torch.clamp(colors, 0, 1).cpu().numpy()
    
    def get_gaussian_scales(self) -> Optional[np.ndarray]:
        """Get scales for each Gaussian."""
        if self._scales is None or len(self._scales) == 0:
            return None
        return torch.exp(self._scales).data.cpu().numpy()
    
    def get_gaussian_opacities(self) -> Optional[np.ndarray]:
        """Get opacities for each Gaussian."""
        if self._opacities is None or len(self._opacities) == 0:
            return None
        return torch.sigmoid(self._opacities).data.cpu().numpy().flatten()
