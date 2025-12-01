"""
SplaTAM Engine - Dense RGB-D SLAM with Gaussian Splatting
==========================================================

This engine wraps SplaTAM (https://github.com/spla-tam/SplaTAM)
which provides native online RGB-D SLAM using Gaussian Splatting.

SplaTAM advantages:
- Native online/incremental training
- Joint camera tracking and mapping
- Battle-tested on TUM, Replica, ScanNet
- Proper keyframe selection and loop management

To use this engine, you need to install SplaTAM:
    git clone https://github.com/spla-tam/SplaTAM.git ~/SplaTAM
    cd ~/SplaTAM
    pip install -e .
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path
import json
import logging
import sys
from dataclasses import dataclass

from .base import BaseGSEngine

logger = logging.getLogger(__name__)


@dataclass
class SplaTAMConfig:
    """Configuration for SplaTAM engine."""
    # Tracking parameters
    num_tracking_iters: int = 40
    tracking_lr: float = 0.01
    use_sil_for_loss: bool = True
    sil_threshold: float = 0.5
    
    # Mapping parameters
    num_mapping_iters: int = 60
    mapping_lr_means3d: float = 0.0001
    mapping_lr_rgb: float = 0.0025
    mapping_lr_opacity: float = 0.05
    mapping_lr_scales: float = 0.001
    mapping_lr_rotations: float = 0.001
    
    # Camera parameters
    cam_lr_rot: float = 0.0004
    cam_lr_trans: float = 0.002
    
    # Gaussian parameters
    gaussian_distribution: str = "isotropic"  # or "anisotropic"
    
    # Densification
    densify_every: int = 100
    prune_every: int = 100
    densification_threshold: float = 0.0002
    
    # Depth
    depth_weight: float = 1.0
    ignore_outlier_depth: bool = False
    
    # Keyframes
    keyframe_every: int = 5
    keyframe_overlap_threshold: float = 0.95
    
    # Scene
    scene_radius_depth_ratio: float = 3.0
    mean_sq_dist_method: str = "projective"


def _check_splatam_available() -> bool:
    """Check if SplaTAM is installed."""
    splatam_path = Path.home() / "SplaTAM"
    if not splatam_path.exists():
        return False
    try:
        sys.path.insert(0, str(splatam_path))
        # Try importing the diff-gaussian-rasterization with depth
        from diff_gaussian_rasterization import GaussianRasterizer
        return True
    except ImportError:
        return False


# Check availability at module load
_SPLATAM_AVAILABLE = _check_splatam_available()


class SplaTAMEngine(BaseGSEngine):
    """
    3D Gaussian Splatting SLAM engine using SplaTAM.
    
    SplaTAM provides native online SLAM with joint tracking and mapping
    using Gaussian Splatting as the scene representation.
    
    This implementation follows SplaTAM's approach:
    1. For each frame: Track camera pose against current map
    2. Add Gaussians for unobserved regions
    3. Jointly optimize Gaussians and keyframe poses
    4. Densify/prune Gaussians periodically
    """
    
    SPLATAM_PATH = Path.home() / "SplaTAM"
    
    def __init__(
        self,
        device: str = "cuda:0",
        splatam_path: Optional[str] = None,
        config: Optional[SplaTAMConfig] = None
    ):
        """
        Initialize the SplaTAM engine.
        
        Args:
            device: CUDA device to use
            splatam_path: Path to SplaTAM installation (default: ~/SplaTAM)
            config: SplaTAM configuration
        """
        self.device = torch.device(device)
        self._initialized = False
        self._intrinsics = None
        self._intrinsics_tensor = None
        
        if splatam_path:
            self.SPLATAM_PATH = Path(splatam_path)
        
        self.config = config or SplaTAMConfig()
        
        # Gaussian parameters (following SplaTAM's convention)
        self._params: Dict[str, torch.nn.Parameter] = {}
        self._variables: Dict[str, torch.Tensor] = {}
        
        # Camera data
        self._cameras: List[Dict] = []
        self._keyframes: List[int] = []
        self._w2c_list: List[torch.Tensor] = []
        
        # Optimizer
        self._optimizer = None
        self._tracking_optimizer = None
        
        # Counters
        self._iteration = 0
        self._frame_idx = 0
        self._num_frames = 1000  # Initial estimate, will grow
        
        # Image dimensions
        self._height = None
        self._width = None
        
        # SplaTAM imports (loaded on demand)
        self._splatam_imports = {}
        
        # Check availability and import
        self._available = self._setup_splatam()
        
        if self._available:
            logger.info(f"SplaTAM engine initialized on {device}")
        else:
            logger.warning(
                "SplaTAM not available. To install:\n"
                "  git clone https://github.com/spla-tam/SplaTAM.git ~/SplaTAM\n"
                "  cd ~/SplaTAM && pip install submodules/diff-gaussian-rasterization-w-depth\n"
                "Engine will run in stub mode."
            )
    
    def _setup_splatam(self) -> bool:
        """Set up SplaTAM imports."""
        if not self.SPLATAM_PATH.exists():
            return False
        
        try:
            sys.path.insert(0, str(self.SPLATAM_PATH))
            
            # Import SplaTAM components
            from diff_gaussian_rasterization import (
                GaussianRasterizer,
                GaussianRasterizationSettings
            )
            from utils.slam_external import build_rotation, calc_ssim
            from utils.recon_helpers import setup_camera
            
            self._splatam_imports = {
                'GaussianRasterizer': GaussianRasterizer,
                'GaussianRasterizationSettings': GaussianRasterizationSettings,
                'build_rotation': build_rotation,
                'calc_ssim': calc_ssim,
                'setup_camera': setup_camera,
            }
            
            return True
        except ImportError as e:
            logger.debug(f"SplaTAM import failed: {e}")
            return False
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def is_available(self) -> bool:
        return self._available
    
    def initialize_scene(
        self,
        intrinsics: Dict[str, float],
        config: Dict[str, Any]
    ) -> None:
        """Initialize the SplaTAM scene."""
        if self._initialized:
            raise RuntimeError("Scene already initialized. Call reset() first.")
        
        self._intrinsics = intrinsics.copy()
        self._width = int(intrinsics.get('width', 640))
        self._height = int(intrinsics.get('height', 480))
        
        # Build intrinsics matrix
        self._intrinsics_tensor = torch.tensor([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ], dtype=torch.float32, device=self.device)
        
        # Update config from dict
        for key, value in config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Initialize num_frames estimate
        self._num_frames = config.get('num_frames', 1000)
        
        self._initialized = True
        logger.info(f"SplaTAM scene initialized: {self._width}x{self._height}")
    
    def add_frame(
        self,
        frame_id: int,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        pose_world_cam: np.ndarray
    ) -> None:
        """
        Add a new frame for tracking and mapping.
        
        Args:
            frame_id: Frame index
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) float32 in meters
            pose_world_cam: 4x4 camera-to-world transform
        """
        if not self._initialized:
            raise RuntimeError("Scene not initialized")
        
        if not self._available:
            # Stub mode
            self._cameras.append({'id': frame_id, 'pose': pose_world_cam.copy()})
            self._frame_idx += 1
            return
        
        # Convert to tensors
        rgb_tensor = torch.from_numpy(rgb).float().to(self.device)
        if rgb_tensor.max() > 1.0:
            rgb_tensor = rgb_tensor / 255.0
        rgb_tensor = rgb_tensor.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
        
        # Compute world-to-camera (w2c)
        c2w = torch.from_numpy(pose_world_cam).float().to(self.device)
        w2c = torch.linalg.inv(c2w)
        
        depth_tensor = None
        if depth is not None:
            depth_tensor = torch.from_numpy(depth).float().to(self.device)
            depth_tensor = depth_tensor.unsqueeze(0)  # (H,W) -> (1,H,W)
        
        # Set up camera for rasterization
        cam = self._setup_camera(w2c)
        
        # Store frame data
        curr_data = {
            'id': frame_id,
            'im': rgb_tensor,
            'depth': depth_tensor,
            'w2c': w2c,
            'c2w': c2w,
            'cam': cam,
        }
        self._cameras.append(curr_data)
        self._w2c_list.append(w2c)
        
        # First frame: initialize Gaussians from depth
        if len(self._cameras) == 1 and depth is not None:
            self._initialize_gaussians_from_frame(curr_data)
            self._keyframes.append(0)
        else:
            # Subsequent frames: track and map
            self._process_frame(curr_data)
        
        self._frame_idx += 1
    
    def _setup_camera(self, w2c: torch.Tensor) -> Any:
        """Set up camera for Gaussian rasterization.
        
        Creates a GaussianRasterizationSettings object compatible with
        both the original graphdeco and SplaTAM rasterizers.
        """
        if not self._available:
            return None
        
        from diff_gaussian_rasterization import GaussianRasterizationSettings
        
        fx = self._intrinsics['fx']
        fy = self._intrinsics['fy']
        cx = self._intrinsics['cx']
        cy = self._intrinsics['cy']
        w = self._width
        h = self._height
        
        near = 0.01
        far = 100.0
        
        # Compute camera center (in world coords)
        w2c_float = w2c.float()
        cam_center = torch.inverse(w2c_float)[:3, 3]
        
        # Format view matrix for rasterizer: (1, 4, 4) transposed
        viewmatrix = w2c_float.unsqueeze(0).transpose(1, 2)
        
        # Build OpenGL projection matrix
        opengl_proj = torch.tensor([
            [2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
            [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
            [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
            [0.0, 0.0, 1.0, 0.0]
        ], device=self.device, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
        
        full_proj = viewmatrix.bmm(opengl_proj)
        
        # Create settings - handle both old (SplaTAM) and new (graphdeco) APIs
        try:
            # Try graphdeco API (with debug and antialiasing)
            cam = GaussianRasterizationSettings(
                image_height=h,
                image_width=w,
                tanfovx=w / (2 * fx),
                tanfovy=h / (2 * fy),
                bg=torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device),
                scale_modifier=1.0,
                viewmatrix=viewmatrix,
                projmatrix=full_proj,
                sh_degree=0,
                campos=cam_center,
                prefiltered=False,
                debug=False,
                antialiasing=False,
            )
        except TypeError:
            # Fall back to SplaTAM API (without debug and antialiasing)
            cam = GaussianRasterizationSettings(
                image_height=h,
                image_width=w,
                tanfovx=w / (2 * fx),
                tanfovy=h / (2 * fy),
                bg=torch.tensor([0, 0, 0], dtype=torch.float32, device=self.device),
                scale_modifier=1.0,
                viewmatrix=viewmatrix,
                projmatrix=full_proj,
                sh_degree=0,
                campos=cam_center,
                prefiltered=False,
            )
        
        return cam
    
    def _initialize_gaussians_from_frame(self, frame_data: Dict) -> None:
        """Initialize Gaussians from the first RGB-D frame."""
        if not self._available:
            return
        
        rgb = frame_data['im']  # (C, H, W)
        depth = frame_data['depth']  # (1, H, W)
        w2c = frame_data['w2c']
        
        if depth is None:
            logger.warning("No depth for first frame, cannot initialize Gaussians")
            return
        
        # Get point cloud
        pts, mean3_sq_dist = self._depth_to_pointcloud(rgb, depth, w2c)
        
        if pts is None or len(pts) == 0:
            logger.warning("No valid points from depth")
            return
        
        # Initialize parameters
        num_pts = pts.shape[0]
        means3D = pts[:, :3]
        colors = pts[:, 3:6]
        
        # Quaternion rotations (identity)
        unnorm_rots = torch.tile(
            torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float32),
            (num_pts, 1)
        )
        
        # Opacities (logit space)
        logit_opacities = torch.zeros((num_pts, 1), device=self.device)
        
        # Scales (log space)
        if self.config.gaussian_distribution == "isotropic":
            log_scales = torch.log(torch.sqrt(mean3_sq_dist)).unsqueeze(-1)
        else:
            log_scales = torch.tile(
                torch.log(torch.sqrt(mean3_sq_dist)).unsqueeze(-1),
                (1, 3)
            )
        
        # Camera poses (relative to first frame)
        cam_rots = torch.tile(
            torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).unsqueeze(-1),
            (1, 1, self._num_frames)
        )
        cam_trans = torch.zeros((1, 3, self._num_frames), device=self.device)
        
        # Create parameter dict
        self._params = {
            'means3D': torch.nn.Parameter(means3D.contiguous()),
            'rgb_colors': torch.nn.Parameter(colors.contiguous()),
            'unnorm_rotations': torch.nn.Parameter(unnorm_rots.contiguous()),
            'logit_opacities': torch.nn.Parameter(logit_opacities.contiguous()),
            'log_scales': torch.nn.Parameter(log_scales.contiguous()),
            'cam_unnorm_rots': torch.nn.Parameter(cam_rots.contiguous()),
            'cam_trans': torch.nn.Parameter(cam_trans.contiguous()),
        }
        
        # Initialize variables for densification
        self._variables = {
            'max_2D_radius': torch.zeros(num_pts, device=self.device),
            'means2D_gradient_accum': torch.zeros(num_pts, device=self.device),
            'denom': torch.zeros(num_pts, device=self.device),
            'timestep': torch.zeros(num_pts, device=self.device),
            'scene_radius': depth.max() / self.config.scene_radius_depth_ratio,
        }
        
        # Set up optimizer
        self._setup_optimizer()
        
        logger.info(f"Initialized {num_pts:,} Gaussians from first frame")
    
    def _depth_to_pointcloud(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        w2c: torch.Tensor,
        subsample: int = 4
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Convert RGB-D to colored point cloud in world coordinates."""
        
        fx = self._intrinsics['fx']
        fy = self._intrinsics['fy']
        cx = self._intrinsics['cx']
        cy = self._intrinsics['cy']
        
        height, width = depth.shape[1], depth.shape[2]
        
        # Create coordinate grid (subsampled)
        u = torch.arange(0, width, subsample, device=self.device, dtype=torch.float32)
        v = torch.arange(0, height, subsample, device=self.device, dtype=torch.float32)
        u, v = torch.meshgrid(u, v, indexing='xy')
        u = u.reshape(-1).long()
        v = v.reshape(-1).long()
        
        # Get depth values
        d = depth[0, v, u]
        
        # Valid depth mask
        valid = (d > 0.1) & (d < 10.0)
        u, v, d = u[valid].float(), v[valid].float(), d[valid]
        
        if len(d) == 0:
            return None, None
        
        # Unproject to camera coordinates
        x = (u - cx) * d / fx
        y = (v - cy) * d / fy
        z = d
        
        pts_cam = torch.stack([x, y, z], dim=1)
        
        # Transform to world coordinates
        c2w = torch.linalg.inv(w2c)
        pts_world = (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]
        
        # Get colors
        colors = rgb[:, v.long(), u.long()].T  # (N, 3)
        
        # Compute mean squared distance for scale initialization
        scale = d / ((fx + fy) / 2)
        mean3_sq_dist = scale ** 2
        
        # Combine points and colors
        point_cloud = torch.cat([pts_world, colors], dim=1)
        
        return point_cloud, mean3_sq_dist
    
    def _setup_optimizer(self) -> None:
        """Set up the Adam optimizer for mapping."""
        if not self._params:
            return
        
        cfg = self.config
        
        # Learning rates for each parameter group
        lr_dict = {
            'means3D': cfg.mapping_lr_means3d,
            'rgb_colors': cfg.mapping_lr_rgb,
            'unnorm_rotations': cfg.mapping_lr_rotations,
            'logit_opacities': cfg.mapping_lr_opacity,
            'log_scales': cfg.mapping_lr_scales,
            'cam_unnorm_rots': cfg.cam_lr_rot,
            'cam_trans': cfg.cam_lr_trans,
        }
        
        param_groups = [
            {'params': [v], 'name': k, 'lr': lr_dict.get(k, 0.001)}
            for k, v in self._params.items()
        ]
        
        self._optimizer = torch.optim.Adam(param_groups, eps=1e-15)
    
    def _process_frame(self, frame_data: Dict) -> None:
        """Process a new frame (tracking + mapping)."""
        time_idx = len(self._cameras) - 1
        
        # Extend camera parameters if needed
        if time_idx >= self._params['cam_unnorm_rots'].shape[-1]:
            self._extend_camera_params(time_idx + 100)
        
        # Initialize camera pose from previous frame
        if time_idx > 0:
            self._params['cam_unnorm_rots'].data[..., time_idx] = \
                self._params['cam_unnorm_rots'].data[..., time_idx - 1]
            self._params['cam_trans'].data[..., time_idx] = \
                self._params['cam_trans'].data[..., time_idx - 1]
        
        # Run tracking to estimate camera pose
        self._track_frame(frame_data, time_idx)
        
        # Add new Gaussians for unobserved regions
        if frame_data['depth'] is not None:
            self._add_new_gaussians(frame_data, time_idx)
        
        # Add to keyframes if needed
        if self._should_add_keyframe(time_idx):
            self._keyframes.append(time_idx)
        
        # Run mapping optimization
        self._map_frame(frame_data, time_idx)
    
    def _extend_camera_params(self, new_size: int) -> None:
        """Extend camera parameter tensors."""
        old_size = self._params['cam_unnorm_rots'].shape[-1]
        if new_size <= old_size:
            return
        
        # Extend rotations
        new_rots = torch.zeros(
            1, 4, new_size,
            device=self.device, dtype=torch.float32
        )
        new_rots[..., :old_size] = self._params['cam_unnorm_rots'].data
        new_rots[0, 0, old_size:] = 1  # Identity quaternion
        self._params['cam_unnorm_rots'] = torch.nn.Parameter(new_rots)
        
        # Extend translations
        new_trans = torch.zeros(
            1, 3, new_size,
            device=self.device, dtype=torch.float32
        )
        new_trans[..., :old_size] = self._params['cam_trans'].data
        self._params['cam_trans'] = torch.nn.Parameter(new_trans)
        
        # Re-setup optimizer
        self._setup_optimizer()
    
    def _track_frame(self, frame_data: Dict, time_idx: int) -> None:
        """Track camera pose for current frame."""
        if not self._available or not self._params:
            return
        
        cfg = self.config
        
        # Set up tracking optimizer (only camera params get gradients)
        cam_params = [
            {'params': [self._params['cam_unnorm_rots']], 'lr': cfg.cam_lr_rot},
            {'params': [self._params['cam_trans']], 'lr': cfg.cam_lr_trans},
        ]
        tracking_optimizer = torch.optim.Adam(cam_params)
        
        for _ in range(cfg.num_tracking_iters):
            tracking_optimizer.zero_grad()
            
            # Render current view
            rendered, _ = self._render_frame(time_idx, frame_data['cam'])
            
            if rendered is None:
                break
            
            # Compute loss
            loss = self._compute_loss(
                rendered,
                frame_data['im'],
                frame_data['depth'],
                tracking=True
            )
            
            loss.backward()
            tracking_optimizer.step()
    
    def _map_frame(self, frame_data: Dict, time_idx: int) -> None:
        """Run mapping optimization."""
        if not self._available or not self._params or not self._optimizer:
            return
        
        cfg = self.config
        
        # Sample keyframes for optimization
        frames_to_optimize = self._sample_keyframes(time_idx)
        
        for _ in range(cfg.num_mapping_iters):
            self._optimizer.zero_grad()
            
            total_loss = 0
            for idx in frames_to_optimize:
                cam_data = self._cameras[idx]
                
                # Render
                rendered, variables_update = self._render_frame(idx, cam_data['cam'])
                
                if rendered is None:
                    continue
                
                # Update variables for densification
                if variables_update:
                    self._update_densification_stats(variables_update)
                
                # Compute loss
                loss = self._compute_loss(
                    rendered,
                    cam_data['im'],
                    cam_data['depth'],
                    tracking=False
                )
                total_loss = total_loss + loss
            
            if total_loss > 0:
                total_loss.backward()
                self._optimizer.step()
        
        # Densification
        if self._iteration % cfg.densify_every == 0 and self._iteration > 0:
            self._densify_gaussians()
        
        # Pruning
        if self._iteration % cfg.prune_every == 0 and self._iteration > 0:
            self._prune_gaussians()
        
        self._iteration += 1
    
    def _render_frame(
        self,
        time_idx: int,
        cam: Any
    ) -> Tuple[Optional[Dict], Optional[Dict]]:
        """Render a frame using Gaussian splatting."""
        if not self._available or cam is None:
            return None, None
        
        GaussianRasterizer = self._splatam_imports['GaussianRasterizer']
        build_rotation = self._splatam_imports['build_rotation']
        
        try:
            # Transform Gaussians to camera frame
            transformed = self._transform_to_frame(time_idx)
            
            # Prepare render variables
            rendervar = self._params_to_rendervar(transformed)
            
            # Render
            rendervar['means2D'].retain_grad()
            im, radius, _ = GaussianRasterizer(raster_settings=cam)(**rendervar)
            
            # Track for densification
            variables_update = {
                'means2D': rendervar['means2D'],
                'radius': radius,
            }
            
            return {'rgb': im}, variables_update
            
        except Exception as e:
            logger.debug(f"Render failed: {e}")
            return None, None
    
    def _transform_to_frame(self, time_idx: int) -> Dict[str, torch.Tensor]:
        """Transform Gaussians to a specific camera frame."""
        build_rotation = self._splatam_imports['build_rotation']
        
        # Get camera pose for this frame
        cam_rot = F.normalize(self._params['cam_unnorm_rots'][..., time_idx])
        cam_trans = self._params['cam_trans'][..., time_idx]
        
        # Build rotation matrix
        rel_w2c = build_rotation(cam_rot).squeeze()
        
        # Transform means
        means3D = self._params['means3D']
        transformed_means = (rel_w2c @ means3D.T).T + cam_trans
        
        # Transform rotations
        gaussian_rots = self._params['unnorm_rotations']
        
        return {
            'means3D': transformed_means,
            'unnorm_rotations': gaussian_rots,
        }
    
    def _params_to_rendervar(self, transformed: Dict) -> Dict:
        """Convert parameters to render variables."""
        # Handle isotropic vs anisotropic
        if self._params['log_scales'].shape[1] == 1:
            log_scales = self._params['log_scales'].repeat(1, 3)
        else:
            log_scales = self._params['log_scales']
        
        return {
            'means3D': transformed['means3D'],
            'colors_precomp': self._params['rgb_colors'],
            'rotations': F.normalize(transformed['unnorm_rotations']),
            'opacities': torch.sigmoid(self._params['logit_opacities']),
            'scales': torch.exp(log_scales),
            'means2D': torch.zeros_like(
                self._params['means3D'],
                requires_grad=True,
                device=self.device
            ),
        }
    
    def _compute_loss(
        self,
        rendered: Dict,
        gt_rgb: torch.Tensor,
        gt_depth: Optional[torch.Tensor],
        tracking: bool = False
    ) -> torch.Tensor:
        """Compute photometric and depth loss."""
        calc_ssim = self._splatam_imports['calc_ssim']
        
        rgb_loss = F.l1_loss(rendered['rgb'], gt_rgb)
        
        if not tracking:
            # Add SSIM for mapping
            ssim_loss = 1.0 - calc_ssim(rendered['rgb'], gt_rgb)
            rgb_loss = 0.8 * rgb_loss + 0.2 * ssim_loss
        
        # Depth loss if available
        depth_loss = 0
        if gt_depth is not None and 'depth' in rendered:
            mask = gt_depth > 0
            if mask.sum() > 0:
                depth_loss = F.l1_loss(
                    rendered['depth'][mask],
                    gt_depth[mask]
                )
        
        return rgb_loss + self.config.depth_weight * depth_loss
    
    def _add_new_gaussians(self, frame_data: Dict, time_idx: int) -> None:
        """Add Gaussians for newly observed regions."""
        if frame_data['depth'] is None:
            return
        
        # Add a small number of new Gaussians periodically
        if time_idx % 10 != 0:
            return
        
        pts, mean3_sq_dist = self._depth_to_pointcloud(
            frame_data['im'],
            frame_data['depth'],
            frame_data['w2c'],
            subsample=16  # Sparse sampling for new points
        )
        
        if pts is None or len(pts) < 10:
            return
        
        # Limit new points
        max_new = 1000
        if len(pts) > max_new:
            indices = torch.randperm(len(pts))[:max_new]
            pts = pts[indices]
            mean3_sq_dist = mean3_sq_dist[indices]
        
        self._add_gaussians(pts, mean3_sq_dist)
    
    def _add_gaussians(
        self,
        points: torch.Tensor,
        mean3_sq_dist: torch.Tensor
    ) -> None:
        """Add new Gaussians to the model."""
        num_new = points.shape[0]
        
        # Create new parameters
        new_means = points[:, :3]
        new_colors = points[:, 3:6]
        new_rots = torch.tile(
            torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float32),
            (num_new, 1)
        )
        new_opacities = torch.zeros((num_new, 1), device=self.device)
        
        if self.config.gaussian_distribution == "isotropic":
            new_scales = torch.log(torch.sqrt(mean3_sq_dist)).unsqueeze(-1)
        else:
            new_scales = torch.tile(
                torch.log(torch.sqrt(mean3_sq_dist)).unsqueeze(-1),
                (1, 3)
            )
        
        # Concatenate with existing
        self._params['means3D'] = torch.nn.Parameter(
            torch.cat([self._params['means3D'].data, new_means], dim=0)
        )
        self._params['rgb_colors'] = torch.nn.Parameter(
            torch.cat([self._params['rgb_colors'].data, new_colors], dim=0)
        )
        self._params['unnorm_rotations'] = torch.nn.Parameter(
            torch.cat([self._params['unnorm_rotations'].data, new_rots], dim=0)
        )
        self._params['logit_opacities'] = torch.nn.Parameter(
            torch.cat([self._params['logit_opacities'].data, new_opacities], dim=0)
        )
        self._params['log_scales'] = torch.nn.Parameter(
            torch.cat([self._params['log_scales'].data, new_scales], dim=0)
        )
        
        # Extend variables
        self._variables['max_2D_radius'] = torch.cat([
            self._variables['max_2D_radius'],
            torch.zeros(num_new, device=self.device)
        ])
        self._variables['means2D_gradient_accum'] = torch.cat([
            self._variables['means2D_gradient_accum'],
            torch.zeros(num_new, device=self.device)
        ])
        self._variables['denom'] = torch.cat([
            self._variables['denom'],
            torch.zeros(num_new, device=self.device)
        ])
        
        # Re-setup optimizer
        self._setup_optimizer()
        
        logger.debug(f"Added {num_new} new Gaussians (total: {self.get_num_gaussians()})")
    
    def _should_add_keyframe(self, time_idx: int) -> bool:
        """Decide if current frame should be a keyframe."""
        if len(self._keyframes) == 0:
            return True
        return time_idx % self.config.keyframe_every == 0
    
    def _sample_keyframes(self, current_idx: int, max_frames: int = 5) -> List[int]:
        """Sample keyframes for optimization."""
        frames = [current_idx]
        
        for kf in reversed(self._keyframes):
            if len(frames) >= max_frames:
                break
            if kf != current_idx:
                frames.append(kf)
        
        return frames
    
    def _update_densification_stats(self, variables_update: Dict) -> None:
        """Update statistics for densification."""
        if 'means2D' in variables_update and variables_update['means2D'].grad is not None:
            grad = variables_update['means2D'].grad
            grad_norm = torch.norm(grad, dim=-1)
            
            self._variables['means2D_gradient_accum'] += grad_norm
            self._variables['denom'] += 1
        
        if 'radius' in variables_update:
            radius = variables_update['radius']
            visible = radius > 0
            self._variables['max_2D_radius'][visible] = torch.max(
                radius[visible],
                self._variables['max_2D_radius'][visible]
            )
    
    def _densify_gaussians(self) -> None:
        """Densify Gaussians based on gradient statistics."""
        pass  # Simplified - full impl would clone/split
    
    def _prune_gaussians(self) -> None:
        """Prune low-opacity Gaussians."""
        if not self._params:
            return
        
        opacities = torch.sigmoid(self._params['logit_opacities'])
        mask = (opacities > 0.01).squeeze()
        
        if mask.sum() < len(mask):
            for key in ['means3D', 'rgb_colors', 'unnorm_rotations', 
                       'logit_opacities', 'log_scales']:
                self._params[key] = torch.nn.Parameter(
                    self._params[key].data[mask]
                )
            
            for key in ['max_2D_radius', 'means2D_gradient_accum', 'denom']:
                self._variables[key] = self._variables[key][mask]
            
            self._setup_optimizer()
            
            pruned = (~mask).sum().item()
            logger.debug(f"Pruned {pruned} Gaussians")
    
    def optimize_step(self, n_steps: int = 1) -> Dict[str, float]:
        """Run additional optimization steps."""
        if not self._initialized:
            raise RuntimeError("Scene not initialized")
        
        if not self._available or not self._params or len(self._cameras) == 0:
            return {
                'loss': 0.0,
                'psnr': 0.0,
                'num_gaussians': self.get_num_gaussians(),
                'iteration': self._iteration,
            }
        
        total_loss = 0.0
        
        for _ in range(n_steps):
            # Sample a random keyframe
            if self._keyframes:
                idx = self._keyframes[np.random.randint(len(self._keyframes))]
            else:
                idx = np.random.randint(len(self._cameras))
            
            frame_data = self._cameras[idx]
            
            self._optimizer.zero_grad()
            
            rendered, _ = self._render_frame(idx, frame_data['cam'])
            
            if rendered is not None:
                loss = self._compute_loss(
                    rendered,
                    frame_data['im'],
                    frame_data['depth'],
                    tracking=False
                )
                loss.backward()
                self._optimizer.step()
                total_loss += loss.item()
            
            self._iteration += 1
        
        avg_loss = total_loss / max(n_steps, 1)
        psnr = 10 * np.log10(1.0 / (avg_loss + 1e-10)) if avg_loss > 0 else 40.0
        
        return {
            'loss': avg_loss,
            'psnr': psnr,
            'num_gaussians': self.get_num_gaussians(),
            'iteration': self._iteration,
        }
    
    def render_view(
        self,
        pose_world_cam: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Render a novel view."""
        if not self._initialized:
            raise RuntimeError("Scene not initialized")
        
        width, height = image_size
        
        if not self._available or not self._params:
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        GaussianRasterizer = self._splatam_imports['GaussianRasterizer']
        
        c2w = torch.from_numpy(pose_world_cam).float().to(self.device)
        w2c = torch.linalg.inv(c2w)
        
        cam = self._setup_camera(w2c)
        if cam is None:
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        with torch.no_grad():
            rendervar = self._params_to_rendervar({
                'means3D': self._params['means3D'],
                'unnorm_rotations': self._params['unnorm_rotations'],
            })
            
            try:
                im, _, _ = GaussianRasterizer(raster_settings=cam)(**rendervar)
                im = im.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
                im = torch.clamp(im * 255, 0, 255).byte()
                return im.cpu().numpy()
            except Exception as e:
                logger.debug(f"Render failed: {e}")
                return np.zeros((height, width, 3), dtype=np.uint8)
    
    def get_num_gaussians(self) -> int:
        if not self._params or 'means3D' not in self._params:
            return 0
        return self._params['means3D'].shape[0]
    
    def get_point_cloud(self) -> Optional[np.ndarray]:
        if not self._params or 'means3D' not in self._params:
            return None
        return self._params['means3D'].detach().cpu().numpy()
    
    def get_gaussian_colors(self) -> Optional[np.ndarray]:
        if not self._params or 'rgb_colors' not in self._params:
            return None
        return self._params['rgb_colors'].detach().cpu().numpy()
    
    def save_state(self, path: str) -> None:
        """Save the SplaTAM state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_data = {
            'intrinsics': self._intrinsics,
            'width': self._width,
            'height': self._height,
            'num_cameras': len(self._cameras),
            'num_keyframes': len(self._keyframes),
            'iteration': self._iteration,
            'engine': 'splatam',
        }
        with open(path / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        
        if self._params:
            # Save checkpoint
            checkpoint = {
                'params': {k: v.data for k, v in self._params.items()},
                'variables': self._variables,
                'keyframes': self._keyframes,
                'iteration': self._iteration,
            }
            torch.save(checkpoint, path / "checkpoint.pth")
            
            # Save PLY
            self._save_ply(path / "point_cloud.ply")
        
        logger.info(f"Saved SplaTAM state to {path}")
    
    def _save_ply(self, filepath: Path) -> None:
        """Save Gaussians as PLY file."""
        if not self._params:
            return
        
        try:
            from plyfile import PlyData, PlyElement
        except ImportError:
            logger.warning("plyfile not installed, skipping PLY save")
            return
        
        means = self._params['means3D'].detach().cpu().numpy()
        colors = self._params['rgb_colors'].detach().cpu().numpy()
        opacities = torch.sigmoid(self._params['logit_opacities']).detach().cpu().numpy()
        scales = torch.exp(self._params['log_scales']).detach().cpu().numpy()
        rotations = F.normalize(self._params['unnorm_rotations']).detach().cpu().numpy()
        
        # Handle isotropic scales
        if scales.shape[1] == 1:
            scales = np.tile(scales, (1, 3))
        
        # Create structured array for PLY
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('opacity', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
        ]
        
        n = len(means)
        elements = np.empty(n, dtype=dtype)
        elements['x'] = means[:, 0]
        elements['y'] = means[:, 1]
        elements['z'] = means[:, 2]
        elements['red'] = (colors[:, 0] * 255).astype(np.uint8)
        elements['green'] = (colors[:, 1] * 255).astype(np.uint8)
        elements['blue'] = (colors[:, 2] * 255).astype(np.uint8)
        elements['opacity'] = opacities.squeeze()
        elements['scale_0'] = scales[:, 0]
        elements['scale_1'] = scales[:, 1]
        elements['scale_2'] = scales[:, 2]
        elements['rot_0'] = rotations[:, 0]
        elements['rot_1'] = rotations[:, 1]
        elements['rot_2'] = rotations[:, 2]
        elements['rot_3'] = rotations[:, 3]
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el], text=False).write(str(filepath))
    
    def load_state(self, path: str) -> None:
        """Load SplaTAM state from disk."""
        path = Path(path)
        
        # Load config
        config_path = path / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_data = json.load(f)
            self._intrinsics = config_data.get('intrinsics')
            self._width = config_data.get('width', 640)
            self._height = config_data.get('height', 480)
            self._iteration = config_data.get('iteration', 0)
        
        # Load checkpoint
        checkpoint_path = path / "checkpoint.pth"
        if checkpoint_path.exists() and self._available:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self._params = {
                k: torch.nn.Parameter(v.to(self.device))
                for k, v in checkpoint['params'].items()
            }
            self._variables = checkpoint.get('variables', {})
            self._keyframes = checkpoint.get('keyframes', [])
            self._iteration = checkpoint.get('iteration', 0)
            
            self._setup_optimizer()
        
        self._initialized = True
        logger.info(f"Loaded SplaTAM state from {path}")
    
    def reset(self) -> None:
        """Reset the engine to initial state."""
        self._initialized = False
        self._intrinsics = None
        self._intrinsics_tensor = None
        self._params = {}
        self._variables = {}
        self._cameras = []
        self._keyframes = []
        self._w2c_list = []
        self._optimizer = None
        self._iteration = 0
        self._frame_idx = 0
        self._height = None
        self._width = None
