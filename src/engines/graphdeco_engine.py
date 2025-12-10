"""
Graphdeco 3D Gaussian Splatting Engine Implementation
====================================================

This module implements the BaseGSEngine interface using the Graphdeco
3DGS implementation (https://github.com/graphdeco-inria/gaussian-splatting).

The engine wraps Graphdeco's GaussianModel and rendering pipeline to provide
online/incremental Gaussian Splatting capabilities.

Key Adaptations:
- Instead of loading a full COLMAP dataset upfront, we dynamically add frames
- Optimization can be run in small steps rather than a full training loop
- Camera creation is done on-the-fly from provided poses and intrinsics

Assumptions:
- The Graphdeco gaussian-splatting repo is available at a known path
  (configured via GRAPHDECO_PATH environment variable or constructor arg)
- CUDA is available and the diff-gaussian-rasterization extension is compiled
"""

import os
import sys
import math
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass, field

# Setup DLL directories for Windows before importing CUDA modules
if sys.platform == 'win32':
    # Add CUDA toolkit DLLs
    cuda_paths = [
        os.path.join(os.environ.get('CUDA_PATH', ''), 'bin'),
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin',
    ]
    for cuda_bin in cuda_paths:
        if os.path.exists(cuda_bin):
            try:
                os.add_dll_directory(cuda_bin)
            except (OSError, AttributeError):
                pass
            break
    
    # Add PyTorch lib directory
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib):
            os.add_dll_directory(torch_lib)
    except (ImportError, OSError, AttributeError):
        pass

import numpy as np
import torch
import torch.nn as nn

from .base import BaseGSEngine

# Configure logging
logger = logging.getLogger(__name__)


def _find_graphdeco_path() -> Path:
    """
    Locate the Graphdeco gaussian-splatting repository.
    
    Search order:
    1. GRAPHDECO_PATH environment variable
    2. AirSplatMap/submodules/gaussian-splatting (git submodule)
    3. Workspace root gaussian-splatting folder (legacy, unless AIRSPLAT_USE_SUBMODULES=1)
    
    Set AIRSPLAT_USE_SUBMODULES=1 to force submodule-only mode.
    
    Returns:
        Path to the gaussian-splatting directory
        
    Raises:
        RuntimeError: If the path cannot be found
    """
    # Try environment variable first
    env_path = os.environ.get("GRAPHDECO_PATH")
    if env_path and os.path.isdir(env_path):
        return Path(env_path)
    
    # Check if we should use submodules only
    use_submodules_only = os.environ.get("AIRSPLAT_USE_SUBMODULES", "").lower() in ("1", "true", "yes")
    
    # Try relative paths from this file's location
    this_dir = Path(__file__).parent.resolve()
    
    # AirSplatMap root: src/engines -> src -> AirSplatMap
    airsplatmap_root = this_dir.parent.parent
    
    # Workspace root (parent of AirSplatMap)
    workspace_root = airsplatmap_root.parent
    
    # Primary: submodules directory (git submodule)
    submodule_path = airsplatmap_root / "submodules" / "gaussian-splatting"
    if submodule_path.is_dir() and (submodule_path / "scene").is_dir():
        return submodule_path
    
    if use_submodules_only:
        raise RuntimeError(
            "AIRSPLAT_USE_SUBMODULES=1 but submodule not found at: "
            f"{submodule_path}\nRun: git submodule update --init --recursive"
        )
    
    # Legacy fallbacks
    legacy_candidates = [
        workspace_root / "gaussian-splatting",
        Path.home() / "gaussian-splatting",
    ]
    
    for candidate in legacy_candidates:
        if candidate.is_dir() and (candidate / "scene").is_dir():
            logger.info(f"Using legacy path: {candidate} (set AIRSPLAT_USE_SUBMODULES=1 to disable)")
            return candidate
    
    raise RuntimeError(
        "Could not find Graphdeco gaussian-splatting repository. "
        "Please run: git submodule update --init --recursive\n"
        f"Searched: {[str(submodule_path)] + [str(c) for c in legacy_candidates]}"
    )


def _setup_graphdeco_imports(graphdeco_path: Path) -> None:
    """Add Graphdeco path to sys.path for imports."""
    path_str = str(graphdeco_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
        logger.info(f"Added Graphdeco path to sys.path: {path_str}")


@dataclass
class OnlineCamera:
    """
    Represents a camera for online Gaussian Splatting.
    
    This is a simplified camera representation that can be created from
    pose matrices and intrinsics without needing COLMAP data.
    """
    uid: int
    frame_id: int
    R: np.ndarray  # 3x3 rotation (world-to-camera)
    T: np.ndarray  # 3x1 translation (world-to-camera)
    FoVx: float
    FoVy: float
    image: torch.Tensor  # 3xHxW, normalized [0,1]
    image_width: int
    image_height: int
    image_name: str
    depth: Optional[torch.Tensor] = None  # HxW metric depth
    
    # Computed transforms (set by prepare_transforms)
    world_view_transform: Optional[torch.Tensor] = None
    projection_matrix: Optional[torch.Tensor] = None
    full_proj_transform: Optional[torch.Tensor] = None
    camera_center: Optional[torch.Tensor] = None
    
    def prepare_transforms(self, znear: float = 0.01, zfar: float = 100.0):
        """Compute view and projection transforms for rendering."""
        # Import Graphdeco utilities
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix
        
        # World-to-view transform
        self.world_view_transform = torch.tensor(
            getWorld2View2(self.R, self.T)
        ).transpose(0, 1).cuda().float()
        
        # Projection matrix
        self.projection_matrix = getProjectionMatrix(
            znear=znear, zfar=zfar, 
            fovX=self.FoVx, fovY=self.FoVy
        ).transpose(0, 1).cuda().float()
        
        # Full projection (view @ projection)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        
        # Camera center in world coordinates
        self.camera_center = self.world_view_transform.inverse()[3, :3]


class GraphdecoEngine(BaseGSEngine):
    """
    3DGS Engine implementation using the Graphdeco codebase.
    
    This engine wraps the original Graphdeco 3D Gaussian Splatting implementation
    to support online/incremental mapping. Key features:
    
    - Dynamic frame addition (not limited to fixed COLMAP datasets)
    - Step-wise optimization for real-time operation
    - Configurable densification and pruning strategies
    
    Configuration options (passed to initialize_scene):
        sh_degree (int): Spherical harmonics degree. Default: 3
        white_background (bool): Use white background. Default: False
        position_lr_init (float): Initial learning rate for positions. Default: 0.00016
        position_lr_final (float): Final learning rate for positions. Default: 0.0000016
        feature_lr (float): Learning rate for SH features. Default: 0.0025
        opacity_lr (float): Learning rate for opacity. Default: 0.025
        scaling_lr (float): Learning rate for scale. Default: 0.005
        rotation_lr (float): Learning rate for rotation. Default: 0.001
        densify_from_iter (int): Start densification after N iters. Default: 500
        densify_until_iter (int): Stop densification after N iters. Default: 15000
        densify_grad_threshold (float): Gradient threshold for densification. Default: 0.0002
        densification_interval (int): Densify every N iters. Default: 100
        opacity_reset_interval (int): Reset opacity every N iters. Default: 3000
        percent_dense (float): Percent of scene extent for densification. Default: 0.01
        lambda_dssim (float): Weight for SSIM loss. Default: 0.2
        initial_point_cloud (np.ndarray): Optional Nx3 initial points. Default: None
        initial_colors (np.ndarray): Optional Nx3 initial colors [0,1]. Default: None
        
    Example usage:
        engine = GraphdecoEngine()
        engine.initialize_scene(
            intrinsics={'fx': 525, 'fy': 525, 'cx': 320, 'cy': 240, 
                       'width': 640, 'height': 480},
            config={'sh_degree': 3, 'lambda_dssim': 0.2}
        )
        engine.add_frame(0, rgb_image, depth_image, pose_matrix)
        metrics = engine.optimize_step(n_steps=10)
        rendered = engine.render_view(pose_matrix, (640, 480))
    """
    
    def __init__(self, graphdeco_path: Optional[str] = None):
        """
        Initialize the Graphdeco engine.
        
        Args:
            graphdeco_path: Optional path to the gaussian-splatting repository.
                If not provided, will search standard locations.
        """
        # Track availability
        self._graphdeco_available = False
        
        # Find and setup Graphdeco imports
        try:
            if graphdeco_path is not None:
                self._graphdeco_path = Path(graphdeco_path)
            else:
                self._graphdeco_path = _find_graphdeco_path()
            
            _setup_graphdeco_imports(self._graphdeco_path)
            logger.info(f"Using Graphdeco from: {self._graphdeco_path}")
            
            # Import Graphdeco modules (after path setup)
            # These imports are deferred to avoid import errors if path isn't set
            self._import_graphdeco_modules()
            self._graphdeco_available = True
        except (RuntimeError, ImportError) as e:
            logger.warning(f"Graphdeco not available: {e}")
            self._graphdeco_path = None
        
        # Engine state
        self._initialized = False
        self._intrinsics: Optional[Dict[str, float]] = None
        self._config: Dict[str, Any] = {}
        self._gaussians: Optional[Any] = None  # GaussianModel
        self._cameras: List[OnlineCamera] = []
        self._background: Optional[torch.Tensor] = None
        self._iteration: int = 0
        self._spatial_lr_scale: float = 1.0
        
        # Optimization state
        self._cameras_extent: float = 1.0
        self._opt_params: Optional[Any] = None  # Namespace with opt params
        
    def _import_graphdeco_modules(self):
        """Import Graphdeco modules after path is configured."""
        try:
            # Core modules
            from scene.gaussian_model import GaussianModel
            from gaussian_renderer import render
            from utils.loss_utils import l1_loss, ssim
            from utils.sh_utils import RGB2SH
            from utils.graphics_utils import BasicPointCloud
            from utils.general_utils import get_expon_lr_func
            
            self._GaussianModel = GaussianModel
            self._render = render
            self._l1_loss = l1_loss
            self._ssim = ssim
            self._RGB2SH = RGB2SH
            self._BasicPointCloud = BasicPointCloud
            self._get_expon_lr_func = get_expon_lr_func
            
            # Check for fused SSIM
            try:
                from fused_ssim import fused_ssim
                self._fused_ssim = fused_ssim
                self._use_fused_ssim = True
            except ImportError:
                self._fused_ssim = None
                self._use_fused_ssim = False
                logger.info("Fused SSIM not available, using standard SSIM")
                
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import Graphdeco modules. Make sure the gaussian-splatting "
                f"repository is properly set up with compiled CUDA extensions. Error: {e}"
            )
    
    def initialize_scene(
        self,
        intrinsics: Dict[str, float],
        config: Dict[str, Any]
    ) -> None:
        """
        Initialize the Gaussian scene with camera intrinsics and configuration.
        
        See class docstring for available config options.
        """
        if self._initialized:
            raise RuntimeError("Scene already initialized. Call reset() first.")
        
        # Validate intrinsics
        required_keys = ['fx', 'fy', 'cx', 'cy', 'width', 'height']
        for key in required_keys:
            if key not in intrinsics:
                raise ValueError(f"Missing required intrinsic parameter: {key}")
        
        self._intrinsics = intrinsics.copy()
        self._config = self._get_default_config()
        self._config.update(config)
        
        # Initialize Gaussian model
        sh_degree = self._config.get('sh_degree', 3)
        optimizer_type = self._config.get('optimizer_type', 'default')
        self._gaussians = self._GaussianModel(sh_degree, optimizer_type)
        
        # Initialize exposure_mapping as empty dict (will be populated as frames are added)
        # This prevents AttributeError if initialization from depth fails
        if not hasattr(self._gaussians, 'exposure_mapping'):
            self._gaussians.exposure_mapping = {}
        
        # Setup background
        bg_color = [1, 1, 1] if self._config.get('white_background', False) else [0, 0, 0]
        self._background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # Initialize from point cloud if provided
        initial_pcd = self._config.get('initial_point_cloud')
        initial_colors = self._config.get('initial_colors')
        
        if initial_pcd is not None:
            self._init_from_point_cloud(initial_pcd, initial_colors)
        else:
            # Will initialize from first frame's depth
            logger.info("No initial point cloud provided. Will initialize from first frame with depth.")
        
        # Setup optimization parameters namespace
        self._setup_opt_params()
        
        self._iteration = 0
        self._cameras = []
        self._initialized = True
        
        logger.info(f"Scene initialized with intrinsics: {intrinsics}")
        logger.info(f"Config: {self._config}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration values."""
        return {
            'sh_degree': 3,
            'white_background': False,
            'optimizer_type': 'default',
            
            # Learning rates
            'position_lr_init': 0.00016,
            'position_lr_final': 0.0000016,
            'position_lr_delay_mult': 0.01,
            'position_lr_max_steps': 30000,
            'feature_lr': 0.0025,
            'opacity_lr': 0.025,
            'scaling_lr': 0.005,
            'rotation_lr': 0.001,
            
            # Densification
            'densify_from_iter': 500,
            'densify_until_iter': 15000,
            'densify_grad_threshold': 0.0002,
            'densification_interval': 100,
            'opacity_reset_interval': 3000,
            'percent_dense': 0.01,
            
            # Loss
            'lambda_dssim': 0.2,
            
            # Depth
            'depth_l1_weight_init': 0.0,
            'depth_l1_weight_final': 0.0,
        }
    
    def _setup_opt_params(self):
        """Setup optimization parameters namespace for GaussianModel."""
        from argparse import Namespace
        
        self._opt_params = Namespace(
            position_lr_init=self._config['position_lr_init'],
            position_lr_final=self._config['position_lr_final'],
            position_lr_delay_mult=self._config['position_lr_delay_mult'],
            position_lr_max_steps=self._config['position_lr_max_steps'],
            feature_lr=self._config['feature_lr'],
            opacity_lr=self._config['opacity_lr'],
            scaling_lr=self._config['scaling_lr'],
            rotation_lr=self._config['rotation_lr'],
            percent_dense=self._config['percent_dense'],
            densify_from_iter=self._config['densify_from_iter'],
            densify_until_iter=self._config['densify_until_iter'],
            densify_grad_threshold=self._config['densify_grad_threshold'],
            densification_interval=self._config['densification_interval'],
            opacity_reset_interval=self._config['opacity_reset_interval'],
            lambda_dssim=self._config['lambda_dssim'],
            iterations=self._config.get('max_iterations', 30000),
            exposure_lr_init=0.01,
            exposure_lr_final=0.001,
            exposure_lr_delay_steps=0,
            exposure_lr_delay_mult=0.0,
            depth_l1_weight_init=self._config.get('depth_l1_weight_init', 0.0),
            depth_l1_weight_final=self._config.get('depth_l1_weight_final', 0.0),
        )
        
        # Create pipeline params namespace
        self._pipe_params = Namespace(
            convert_SHs_python=False,
            compute_cov3D_python=False,
            debug=False,
            antialiasing=self._config.get('antialiasing', False),
        )
    
    def _init_from_point_cloud(
        self, 
        points: np.ndarray, 
        colors: Optional[np.ndarray] = None
    ):
        """Initialize Gaussians from a point cloud."""
        if colors is None:
            # Default to gray
            colors = np.ones_like(points) * 0.5
        
        # Ensure proper shape
        points = np.asarray(points, dtype=np.float32)
        colors = np.asarray(colors, dtype=np.float32)
        
        if points.ndim == 1:
            points = points.reshape(-1, 3)
        if colors.ndim == 1:
            colors = colors.reshape(-1, 3)
        
        # Create BasicPointCloud
        normals = np.zeros_like(points)
        pcd = self._BasicPointCloud(
            points=points,
            colors=colors,
            normals=normals
        )
        
        # Compute spatial scale from point cloud extent
        self._spatial_lr_scale = self._compute_spatial_scale(points)
        self._cameras_extent = self._spatial_lr_scale
        
        # Create dummy cam_infos for exposure mapping
        # (just need image_name for the exposure mapping dict)
        class DummyCamInfo:
            def __init__(self, name):
                self.image_name = name
        
        cam_infos = [DummyCamInfo(f"frame_{i}") for i in range(1000)]
        
        # Initialize gaussians from point cloud
        self._gaussians.create_from_pcd(pcd, cam_infos, self._spatial_lr_scale)
        self._gaussians.training_setup(self._opt_params)
        
        logger.info(f"Initialized {len(points)} Gaussians from point cloud")
    
    def _compute_spatial_scale(self, points: np.ndarray) -> float:
        """Compute spatial learning rate scale from point cloud extent."""
        if len(points) == 0:
            return 1.0
        
        # Compute bounding box diagonal
        min_pt = points.min(axis=0)
        max_pt = points.max(axis=0)
        diagonal = np.linalg.norm(max_pt - min_pt)
        
        # Scale factor (similar to Graphdeco's getNerfppNorm)
        return max(diagonal / 2.0, 1.0)
    
    def add_frame(
        self,
        frame_id: int,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        pose_world_cam: np.ndarray
    ) -> None:
        """
        Add a new frame observation to the scene.
        
        If this is the first frame and no initial point cloud was provided,
        will initialize Gaussians from the depth map (if available).
        """
        if not self._initialized:
            raise RuntimeError("Scene not initialized. Call initialize_scene() first.")
        
        # Validate inputs
        rgb = self._validate_rgb(rgb)
        pose_world_cam = self._validate_pose(pose_world_cam)
        
        # Convert pose from camera-to-world to Graphdeco's expected format
        # Graphdeco stores R as transpose of world-to-camera rotation (due to GLM in CUDA)
        pose_cam_world = np.linalg.inv(pose_world_cam)
        R = np.transpose(pose_cam_world[:3, :3])  # TRANSPOSED world-to-camera rotation
        T = pose_cam_world[:3, 3]   # World-to-camera translation
        
        # Compute FoV from intrinsics
        width = self._intrinsics['width']
        height = self._intrinsics['height']
        fx = self._intrinsics['fx']
        fy = self._intrinsics['fy']
        
        FoVx = 2 * math.atan(width / (2 * fx))
        FoVy = 2 * math.atan(height / (2 * fy))
        
        # Convert RGB to torch tensor (3xHxW, normalized)
        if rgb.dtype == np.uint8:
            rgb_float = rgb.astype(np.float32) / 255.0
        else:
            rgb_float = rgb.astype(np.float32)
        
        rgb_tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).cuda()
        
        # Handle depth
        depth_tensor = None
        if depth is not None:
            depth_tensor = torch.from_numpy(depth.astype(np.float32)).cuda()
        
        # Create camera
        camera = OnlineCamera(
            uid=len(self._cameras),
            frame_id=frame_id,
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            image=rgb_tensor,
            image_width=int(width),
            image_height=int(height),
            image_name=f"frame_{frame_id:06d}",
            depth=depth_tensor,
        )
        camera.prepare_transforms()
        
        self._cameras.append(camera)
        
        # Initialize Gaussians from first frame if needed
        if self._gaussians.get_xyz.shape[0] == 0:
            if depth is not None:
                self._init_gaussians_from_rgbd(rgb_float, depth, pose_world_cam)
            else:
                # No depth yet - defer initialization to when we get depth
                # This happens for TUM datasets where depth recording starts later
                logger.info(f"Frame {frame_id}: No depth available yet, deferring Gaussian initialization")
                # Still add camera but don't initialize Gaussians
                logger.debug(f"Added frame {frame_id} (no Gaussians yet), total cameras: {len(self._cameras)}")
                return  # Skip exposure mapping setup since no Gaussians yet
        elif depth is not None and self._config.get('add_gaussians_per_frame', True):
            # Add new Gaussians from depth on subsequent frames
            self._add_gaussians_from_depth(rgb_float, depth, pose_world_cam)
        
        # Add to exposure mapping if not present
        # Ensure exposure_mapping exists (may not if create_from_pcd wasn't called yet)
        if not hasattr(self._gaussians, 'exposure_mapping'):
            self._gaussians.exposure_mapping = {}
        if camera.image_name not in self._gaussians.exposure_mapping:
            idx = len(self._gaussians.exposure_mapping)
            self._gaussians.exposure_mapping[camera.image_name] = idx
            # Expand exposure tensor
            if hasattr(self._gaussians, '_exposure') and self._gaussians._exposure is not None:
                new_exposure = torch.eye(3, 4, device="cuda")[None]
                self._gaussians._exposure = nn.Parameter(
                    torch.cat([self._gaussians._exposure, new_exposure], dim=0).requires_grad_(True)
                )
        
        logger.debug(f"Added frame {frame_id}, total cameras: {len(self._cameras)}")
    
    def _validate_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """Validate and ensure RGB is HxWx3."""
        rgb = np.asarray(rgb)
        
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB must be HxWx3, got shape {rgb.shape}")
        
        expected_h = self._intrinsics['height']
        expected_w = self._intrinsics['width']
        
        if rgb.shape[0] != expected_h or rgb.shape[1] != expected_w:
            raise ValueError(
                f"RGB size {rgb.shape[:2]} doesn't match intrinsics ({expected_h}, {expected_w})"
            )
        
        return rgb
    
    def _validate_pose(self, pose: np.ndarray) -> np.ndarray:
        """Validate pose is 4x4 transformation matrix."""
        pose = np.asarray(pose, dtype=np.float64)
        
        if pose.shape != (4, 4):
            raise ValueError(f"Pose must be 4x4, got shape {pose.shape}")
        
        # Check last row is [0, 0, 0, 1]
        if not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError("Pose last row must be [0, 0, 0, 1]")
        
        return pose
    
    def _init_gaussians_from_rgbd(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose_world_cam: np.ndarray
    ):
        """Initialize Gaussians by back-projecting depth to 3D points."""
        logger.debug(f"_init_gaussians_from_rgbd: depth shape={depth.shape}, range=[{depth.min():.3f}, {depth.max():.3f}]")
        
        # Auto-detect if depth is in millimeters and convert to meters
        depth = depth.copy()
        if depth.max() > 100:  # Likely in millimeters
            logger.info(f"Depth appears to be in millimeters (max={depth.max():.1f}), converting to meters")
            depth = depth / 1000.0
        
        height, width = depth.shape
        fx = self._intrinsics['fx']
        fy = self._intrinsics['fy']
        cx = self._intrinsics['cx']
        cy = self._intrinsics['cy']
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Filter valid depth (now in meters)
        valid_mask = (depth > 0.01) & (depth < 10.0) & np.isfinite(depth)  # 1cm to 10m
        
        # Subsample for efficiency (take every Nth pixel)
        subsample = self._config.get('depth_subsample', 4)
        subsample_mask = np.zeros_like(valid_mask)
        subsample_mask[::subsample, ::subsample] = True
        valid_mask = valid_mask & subsample_mask
        
        n_valid = np.sum(valid_mask)
        if n_valid == 0:
            logger.warning(f"No valid depth points found in frame (depth range: {depth.min():.2f}-{depth.max():.2f})")
            return
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth[valid_mask]
        
        # Back-project to camera coordinates
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = (v_valid - cy) * z_valid / fy
        z_cam = z_valid
        
        # Stack as Nx3
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # Transform to world coordinates
        R = pose_world_cam[:3, :3]
        t = pose_world_cam[:3, 3]
        points_world = (R @ points_cam.T).T + t
        
        # Get colors
        colors = rgb[valid_mask]  # Nx3, already [0,1]
        
        logger.info(f"Initializing {len(points_world)} Gaussians from depth")
        
        self._init_from_point_cloud(points_world, colors)
    
    def _add_gaussians_from_depth(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose_world_cam: np.ndarray
    ):
        """Add new Gaussians from depth map on subsequent frames."""
        # Only add every N frames to avoid explosion
        add_every = self._config.get('add_gaussians_every', 10)
        if len(self._cameras) % add_every != 0:
            return
        
        # Don't add if we're at the Gaussian limit
        max_gaussians = self._config.get('max_gaussians', 500000)
        current_count = self._gaussians.get_xyz.shape[0]
        if current_count >= max_gaussians * 0.95:
            logger.debug(f"Skipping Gaussian addition (at limit {current_count}/{max_gaussians})")
            return
        
        # Auto-detect if depth is in millimeters and convert to meters
        depth = depth.copy()
        if depth.max() > 100:  # Likely in millimeters
            depth = depth / 1000.0
        
        height, width = depth.shape
        fx = self._intrinsics['fx']
        fy = self._intrinsics['fy']
        cx = self._intrinsics['cx']
        cy = self._intrinsics['cy']
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Filter valid depth (in meters)
        valid_mask = (depth > 0.01) & (depth < 10.0)
        
        # More aggressive subsampling for subsequent frames
        subsample = self._config.get('add_gaussians_subsample', 8)
        subsample_mask = np.zeros_like(valid_mask)
        subsample_mask[::subsample, ::subsample] = True
        valid_mask = valid_mask & subsample_mask
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth[valid_mask]
        
        if len(z_valid) == 0:
            return
        
        # Back-project to camera coordinates
        x_cam = (u_valid - cx) * z_valid / fx
        y_cam = (v_valid - cy) * z_valid / fy
        z_cam = z_valid
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # Transform to world coordinates
        R = pose_world_cam[:3, :3]
        t = pose_world_cam[:3, 3]
        points_world = (R @ points_cam.T).T + t
        
        # Get colors
        colors = rgb[valid_mask]
        
        # Add to existing Gaussians
        self._extend_gaussians(points_world, colors)
        
        logger.debug(f"Added {len(points_world)} Gaussians from frame depth, total: {self._gaussians.get_xyz.shape[0]}")
    
    def _extend_gaussians(self, points: np.ndarray, colors: np.ndarray):
        """Add new Gaussians to existing scene."""
        from simple_knn._C import distCUDA2
        
        n_new = points.shape[0]
        if n_new == 0:
            return
        
        points = torch.tensor(points, dtype=torch.float32, device="cuda")
        colors_sh = self._RGB2SH(torch.tensor(colors, dtype=torch.float32, device="cuda"))
        
        # Initialize features (DC component only for new points)
        features_dc = colors_sh.unsqueeze(1)  # N x 1 x 3
        features_rest = torch.zeros((n_new, (self._gaussians.max_sh_degree + 1) ** 2 - 1, 3), device="cuda")
        
        # Compute scales from nearest neighbor distances
        dist2 = torch.clamp_min(distCUDA2(points), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        
        # Initialize rotations as identity quaternion
        rots = torch.zeros((n_new, 4), device="cuda")
        rots[:, 0] = 1
        
        # Initialize opacities (low initial opacity)
        opacities = self._gaussians.inverse_opacity_activation(
            0.1 * torch.ones((n_new, 1), device="cuda")
        )
        
        # Create tensors dict for cat_tensors_to_optimizer
        d = {
            "xyz": points,
            "f_dc": features_dc,
            "f_rest": features_rest,
            "opacity": opacities,
            "scaling": scales,
            "rotation": rots,
        }
        
        # Extend optimizer and parameters using Graphdeco's method
        optimizable_tensors = self._gaussians.cat_tensors_to_optimizer(d)
        
        # Update internal pointers
        self._gaussians._xyz = optimizable_tensors["xyz"]
        self._gaussians._features_dc = optimizable_tensors["f_dc"]
        self._gaussians._features_rest = optimizable_tensors["f_rest"]
        self._gaussians._opacity = optimizable_tensors["opacity"]
        self._gaussians._scaling = optimizable_tensors["scaling"]
        self._gaussians._rotation = optimizable_tensors["rotation"]
        
        # Handle tmp_radii - might be None after densify_and_prune or not exist yet
        if hasattr(self._gaussians, 'tmp_radii') and self._gaussians.tmp_radii is not None:
            self._gaussians.tmp_radii = torch.cat([
                self._gaussians.tmp_radii,
                torch.zeros(n_new, device="cuda")
            ])
        elif hasattr(self._gaussians, 'tmp_radii'):
            # Initialize if None
            self._gaussians.tmp_radii = torch.zeros(self._gaussians.get_xyz.shape[0], device="cuda")
        
        # Reset gradient accumulation tensors to match new size
        self._gaussians.xyz_gradient_accum = torch.zeros((self._gaussians.get_xyz.shape[0], 1), device="cuda")
        self._gaussians.denom = torch.zeros((self._gaussians.get_xyz.shape[0], 1), device="cuda")
        self._gaussians.max_radii2D = torch.zeros((self._gaussians.get_xyz.shape[0]), device="cuda")
    
    def _init_gaussians_random(self, pose_world_cam: np.ndarray):
        """Initialize with random points in front of the camera."""
        # Generate random points in camera frustum
        n_points = self._config.get('initial_num_points', 1000)
        
        # Random points in normalized device coordinates
        width = self._intrinsics['width']
        height = self._intrinsics['height']
        fx = self._intrinsics['fx']
        fy = self._intrinsics['fy']
        cx = self._intrinsics['cx']
        cy = self._intrinsics['cy']
        
        # Random pixel coordinates and depths
        u = np.random.uniform(0, width, n_points)
        v = np.random.uniform(0, height, n_points)
        z = np.random.uniform(0.5, 5.0, n_points)  # 0.5 to 5 meters
        
        # Back-project
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        points_cam = np.stack([x_cam, y_cam, z], axis=1)
        
        # Transform to world
        R = pose_world_cam[:3, :3]
        t = pose_world_cam[:3, 3]
        points_world = (R @ points_cam.T).T + t
        
        # Random gray colors
        colors = np.random.uniform(0.3, 0.7, (n_points, 3))
        
        self._init_from_point_cloud(points_world, colors)
    
    def optimize_step(self, n_steps: int = 1) -> Dict[str, float]:
        """
        Run n_steps of optimization on the Gaussian scene.
        
        The optimization follows Graphdeco's training loop but in small increments.
        """
        if not self._initialized:
            raise RuntimeError("Scene not initialized. Call initialize_scene() first.")
        
        if len(self._cameras) == 0:
            raise RuntimeError("No cameras added. Call add_frame() first.")
        
        num_gaussians = self._gaussians.get_xyz.shape[0]
        if num_gaussians == 0:
            # No Gaussians yet (waiting for first frame with depth)
            return {
                'loss': 0.0,
                'l1_loss': 0.0,
                'ssim_loss': 0.0,
                'num_gaussians': 0,
                'waiting_for_depth': True,
            }
        
        metrics = {
            'loss': 0.0,
            'l1_loss': 0.0,
            'ssim_loss': 0.0,
            'num_gaussians': num_gaussians,
        }
        
        lambda_dssim = self._config['lambda_dssim']
        
        for _ in range(n_steps):
            self._iteration += 1
            
            # Update learning rate
            self._gaussians.update_learning_rate(self._iteration)
            
            # Every 1000 iterations, increase SH degree
            if self._iteration % 1000 == 0:
                self._gaussians.oneupSHdegree()
            
            # Pick a random camera (favor recent frames for online mapping)
            camera = self._sample_camera()
            
            # Render
            render_pkg = self._render(
                camera, 
                self._gaussians, 
                self._pipe_params, 
                self._background,
                use_trained_exp=False,
                separate_sh=False,
            )
            
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
            
            # Compute loss
            gt_image = camera.image.cuda()
            
            Ll1 = self._l1_loss(image, gt_image)
            
            if self._use_fused_ssim:
                ssim_value = self._fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = self._ssim(image, gt_image)
            
            loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim_value)
            
            # Backward pass
            loss.backward()
            
            with torch.no_grad():
                # Accumulate metrics
                metrics['loss'] += loss.item() / n_steps
                metrics['l1_loss'] += Ll1.item() / n_steps
                metrics['ssim_loss'] += (1.0 - ssim_value.item()) / n_steps
                
                # Densification (with Gaussian count limit)
                max_gaussians = self._config.get('max_gaussians', 500000)
                current_count = self._gaussians.get_xyz.shape[0]
                
                if self._iteration < self._config['densify_until_iter'] and current_count < max_gaussians:
                    # Track max radii for pruning
                    self._gaussians.max_radii2D[visibility_filter] = torch.max(
                        self._gaussians.max_radii2D[visibility_filter],
                        radii[visibility_filter]
                    )
                    self._gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                    
                    if (self._iteration > self._config['densify_from_iter'] and 
                        self._iteration % self._config['densification_interval'] == 0):
                        
                        size_threshold = 20 if self._iteration > self._config['opacity_reset_interval'] else None
                        self._gaussians.densify_and_prune(
                            self._config['densify_grad_threshold'],
                            0.005,
                            self._cameras_extent,
                            size_threshold,
                            radii,
                        )
                        
                        # Log if approaching limit
                        new_count = self._gaussians.get_xyz.shape[0]
                        if new_count > max_gaussians * 0.9:
                            logger.warning(f"Approaching Gaussian limit: {new_count}/{max_gaussians}")
                
                elif current_count >= max_gaussians:
                    # Over limit - prune aggressively 
                    if self._iteration % 100 == 0:
                        logger.info(f"Pruning Gaussians (at limit {current_count}/{max_gaussians})")
                        # Prune low opacity Gaussians
                        prune_mask = (self._gaussians.get_opacity < 0.02).squeeze()
                        if prune_mask.sum() > 0:
                            # Ensure tmp_radii exists
                            if not hasattr(self._gaussians, 'tmp_radii') or self._gaussians.tmp_radii is None:
                                self._gaussians.tmp_radii = torch.zeros(current_count, device="cuda")
                            self._gaussians.prune_points(prune_mask)
                    
                if self._iteration % self._config['opacity_reset_interval'] == 0:
                    self._gaussians.reset_opacity()
                
                # Optimizer step
                self._gaussians.optimizer.step()
                self._gaussians.optimizer.zero_grad(set_to_none=True)
                self._gaussians.exposure_optimizer.step()
                self._gaussians.exposure_optimizer.zero_grad(set_to_none=True)
        
        metrics['num_gaussians'] = self._gaussians.get_xyz.shape[0]
        metrics['iteration'] = self._iteration
        
        return metrics
    
    def _sample_camera(self) -> OnlineCamera:
        """
        Sample a camera for training.
        
        Uses a biased sampling strategy that favors recent frames for online mapping.
        """
        n_cameras = len(self._cameras)
        
        if n_cameras == 1:
            return self._cameras[0]
        
        # Exponential decay weights favoring recent frames
        recency_weight = self._config.get('recency_weight', 0.7)
        
        if recency_weight > 0 and n_cameras > 1:
            # Higher weight for recent frames
            weights = np.array([recency_weight ** (n_cameras - 1 - i) for i in range(n_cameras)])
            weights /= weights.sum()
            idx = np.random.choice(n_cameras, p=weights)
        else:
            idx = np.random.randint(n_cameras)
        
        return self._cameras[idx]
    
    def render_view(
        self,
        pose_world_cam: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """Render an RGB image from the current scene at a given pose."""
        if not self._initialized:
            raise RuntimeError("Scene not initialized.")
        
        if self._gaussians.get_xyz.shape[0] == 0:
            # Return blank image if no Gaussians
            return np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        
        # Create temporary camera for rendering
        # Graphdeco stores R as transpose of world-to-camera rotation (due to GLM in CUDA)
        pose_cam_world = np.linalg.inv(pose_world_cam)
        R = np.transpose(pose_cam_world[:3, :3])  # TRANSPOSED
        T = pose_cam_world[:3, 3]
        
        width, height = image_size
        fx = self._intrinsics['fx'] * width / self._intrinsics['width']
        fy = self._intrinsics['fy'] * height / self._intrinsics['height']
        
        FoVx = 2 * math.atan(width / (2 * fx))
        FoVy = 2 * math.atan(height / (2 * fy))
        
        # Create MiniCam for rendering
        from utils.graphics_utils import getWorld2View2, getProjectionMatrix
        
        world_view = torch.tensor(getWorld2View2(R, T)).transpose(0, 1).cuda().float()
        projection = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy
        ).transpose(0, 1).cuda().float()
        full_proj = (world_view.unsqueeze(0).bmm(projection.unsqueeze(0))).squeeze(0)
        
        from scene.cameras import MiniCam
        mini_cam = MiniCam(
            width=width,
            height=height,
            fovy=FoVy,
            fovx=FoVx,
            znear=0.01,
            zfar=100.0,
            world_view_transform=world_view,
            full_proj_transform=full_proj,
        )
        
        # Render
        with torch.no_grad():
            render_pkg = self._render(
                mini_cam,
                self._gaussians,
                self._pipe_params,
                self._background,
                use_trained_exp=False,
                separate_sh=False,
            )
            
            image = render_pkg["render"]
            
            # Convert to numpy uint8
            image = (torch.clamp(image, 0, 1) * 255).byte()
            image = image.permute(1, 2, 0).cpu().numpy()
        
        return image
    
    def save_state(self, path: str) -> None:
        """Save the current Gaussian scene to disk."""
        if not self._initialized:
            raise RuntimeError("Scene not initialized.")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save PLY file
        ply_path = save_path / "point_cloud.ply"
        self._gaussians.save_ply(str(ply_path))
        logger.info(f"Saved {self._gaussians.get_xyz.shape[0]} Gaussians to {ply_path}")
        
        # Save config and intrinsics
        import json
        config_data = {
            'intrinsics': self._intrinsics,
            'config': self._config,
            'iteration': self._iteration,
            'num_cameras': len(self._cameras),
            'spatial_lr_scale': self._spatial_lr_scale,
            'cameras_extent': self._cameras_extent,
        }
        
        with open(save_path / "config.json", 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Save optimizer state
        checkpoint = {
            'gaussians': self._gaussians.capture(),
            'iteration': self._iteration,
        }
        torch.save(checkpoint, save_path / "checkpoint.pth")
        
        logger.info(f"Saved engine state to {save_path}")
    
    def load_state(self, path: str) -> None:
        """Load a previously saved Gaussian scene."""
        load_path = Path(path)
        
        if not load_path.exists():
            raise FileNotFoundError(f"State path not found: {load_path}")
        
        # Load config
        import json
        with open(load_path / "config.json", 'r') as f:
            config_data = json.load(f)
        
        # Initialize if needed
        if not self._initialized:
            self.initialize_scene(
                config_data['intrinsics'],
                config_data['config']
            )
        
        self._iteration = config_data['iteration']
        self._spatial_lr_scale = config_data['spatial_lr_scale']
        self._cameras_extent = config_data['cameras_extent']
        
        # Load PLY
        ply_path = load_path / "point_cloud.ply"
        if ply_path.exists():
            self._gaussians.load_ply(str(ply_path))
            self._gaussians.training_setup(self._opt_params)
        
        # Load checkpoint
        checkpoint_path = load_path / "checkpoint.pth"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self._gaussians.restore(checkpoint['gaussians'], self._opt_params)
            self._iteration = checkpoint['iteration']
        
        logger.info(f"Loaded engine state from {load_path}")
        logger.info(f"Resumed at iteration {self._iteration} with {self._gaussians.get_xyz.shape[0]} Gaussians")
    
    def reset(self) -> None:
        """Reset the engine to uninitialized state."""
        self._initialized = False
        self._intrinsics = None
        self._config = {}
        self._gaussians = None
        self._cameras = []
        self._background = None
        self._iteration = 0
        self._opt_params = None
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
        
        logger.info("Engine reset")
    
    def get_num_gaussians(self) -> int:
        """Get the current number of Gaussians."""
        if self._gaussians is None:
            return 0
        return self._gaussians.get_xyz.shape[0]
    
    def get_point_cloud(self) -> Optional[np.ndarray]:
        """Get the current Gaussian centers as a point cloud."""
        if self._gaussians is None:
            return None
        return self._gaussians.get_xyz.detach().cpu().numpy()
    
    def get_gaussian_colors(self) -> Optional[np.ndarray]:
        """Get RGB colors for each Gaussian from SH coefficients."""
        if self._gaussians is None:
            return None
        try:
            features = self._gaussians.get_features
            if features is None or len(features) == 0:
                return None
            # DC component (degree 0)
            C0 = 0.28209479177387814
            colors = features[:, 0, :3].detach() * C0 + 0.5
            return torch.clamp(colors, 0, 1).cpu().numpy()
        except:
            return None
    
    def get_gaussian_scales(self) -> Optional[np.ndarray]:
        """Get scales for each Gaussian."""
        if self._gaussians is None:
            return None
        try:
            scales = self._gaussians.get_scaling
            if scales is None:
                return None
            return scales.detach().cpu().numpy()
        except:
            return None
    
    def get_gaussian_opacities(self) -> Optional[np.ndarray]:
        """Get opacities for each Gaussian."""
        if self._gaussians is None:
            return None
        try:
            opacities = self._gaussians.get_opacity
            if opacities is None:
                return None
            return opacities.detach().cpu().numpy().flatten()
        except:
            return None

    
    @property
    def is_initialized(self) -> bool:
        """Check if the engine has been initialized."""
        return self._initialized
    
    @property
    def is_available(self) -> bool:
        """Check if Graphdeco modules are available."""
        return self._graphdeco_available
