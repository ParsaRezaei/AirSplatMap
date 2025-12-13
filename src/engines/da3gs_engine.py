"""
Depth Anything V3 Gaussian Splatting Engine
============================================

This engine uses Depth Anything V3's built-in Gaussian splatting capability.
DA3 can predict Gaussians directly from images without separate depth estimation.

Key features:
- End-to-end depth + Gaussian prediction in a single forward pass
- Built-in pose estimation and refinement
- Multi-view consistent Gaussian generation
- Supports metric depth output

Reference: https://github.com/ByteDance-Seed/Depth-Anything-3
"""

import numpy as np
import logging
import os
import sys
from typing import Optional, Dict, Any, Tuple

from .base import BaseGSEngine

logger = logging.getLogger(__name__)

# Suppress verbose DA3 logging
os.environ['DA3_LOG_LEVEL'] = 'ERROR'

# Check DA3 availability
_DA3_AVAILABLE = False
_DA3_GS_AVAILABLE = False
_torch = None

try:
    import torch as _torch
    
    # Add submodule path
    submodule_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        'submodules', 'Depth-Anything-3', 'src'
    )
    if os.path.exists(submodule_path) and submodule_path not in sys.path:
        sys.path.insert(0, submodule_path)
    
    from depth_anything_3.api import DepthAnything3
    _DA3_AVAILABLE = True
    
    # Check if GS-capable model is available
    # DA3-LARGE and above support Gaussian output
    _DA3_GS_AVAILABLE = True
    
except ImportError as e:
    logger.debug(f"DA3 not available: {e}")


class DA3GSEngine(BaseGSEngine):
    """
    Depth Anything V3 Gaussian Splatting Engine.
    
    This engine uses DA3's built-in capability to predict Gaussians directly
    from RGB images. Unlike other engines that require separate depth estimation
    and then convert to Gaussians, DA3 produces both in a single forward pass.
    
    Suitable for:
    - Quick reconstruction from few images
    - Novel view synthesis with limited input
    - End-to-end depth-to-Gaussian pipeline
    
    Limitations:
    - Requires DA3 submodule with GS support
    - Not designed for incremental/online mapping
    - Fixed model architecture (no fine-tuning during inference)
    
    Usage:
        engine = DA3GSEngine(model_size='large')
        engine.initialize_scene(intrinsics, config)
        
        # Add all frames first (batch processing)
        for i, (rgb, depth, pose) in enumerate(frames):
            engine.add_frame(i, rgb, depth, pose)
        
        # Then optimize (generates Gaussians)
        engine.optimize_step(1)
        
        # Render novel views
        img = engine.render_view(new_pose, (640, 480))
    """
    
    # This engine processes all frames in batch, not incrementally
    is_batch_engine = True
    """
    
    # HuggingFace model IDs for DA3 with GS support
    MODEL_IDS = {
        'base': 'depth-anything/DA3-BASE',
        'large': 'depth-anything/DA3-LARGE-1.1',
        'giant': 'depth-anything/DA3-GIANT-1.1',
    }
    
    def __init__(
        self,
        model_size: str = "large",
        device: str = "cuda",
        process_res: int = 504,
    ):
        """
        Initialize DA3 GS Engine.
        
        Args:
            model_size: 'base', 'large', or 'giant'
            device: 'cuda' or 'cpu'
            process_res: Processing resolution for DA3
        """
        self.model_size = model_size.lower()
        self.device = device
        self.process_res = process_res
        
        self._model = None
        self._initialized = False
        self._intrinsics = None
        self._config = None
        
        # Frame storage (DA3 processes in batch)
        self._frames = []  # List of (rgb, pose)
        self._depths = []  # Optional depths
        
        # Gaussian output
        self._gaussians = None
        self._prediction = None
    
    @staticmethod
    def is_available() -> bool:
        """Check if DA3 with GS support is available."""
        return _DA3_AVAILABLE and _DA3_GS_AVAILABLE
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    def _lazy_load_model(self):
        """Lazy load DA3 model on first use."""
        if self._model is not None:
            return
        
        if not _DA3_AVAILABLE:
            raise RuntimeError("DA3 not available. Install from submodules/Depth-Anything-3")
        
        from depth_anything_3.api import DepthAnything3
        
        model_id = self.MODEL_IDS.get(self.model_size, self.MODEL_IDS['large'])
        logger.info(f"Loading DA3 GS model: {model_id}")
        
        self._model = DepthAnything3.from_pretrained(model_id)
        self._model = self._model.to(device=_torch.device(self.device))
        
        logger.info(f"DA3 GS ({self.model_size}) initialized on {self.device}")
    
    def initialize_scene(
        self,
        intrinsics: Dict[str, float],
        config: Dict[str, Any]
    ) -> None:
        """
        Initialize the scene.
        
        Args:
            intrinsics: Camera intrinsics (fx, fy, cx, cy, width, height)
            config: Engine configuration
        """
        if self._initialized:
            self.reset()
        
        self._intrinsics = intrinsics.copy()
        self._config = config.copy()
        self._frames = []
        self._depths = []
        self._gaussians = None
        self._prediction = None
        self._initialized = True
        
        logger.info(f"DA3 GS scene initialized with intrinsics: {intrinsics}")
    
    def add_frame(
        self,
        frame_id: int,
        rgb: np.ndarray,
        depth: Optional[np.ndarray],
        pose_world_cam: np.ndarray
    ) -> None:
        """
        Add a frame to the scene.
        
        DA3 processes frames in batch, so this just stores the frame.
        Actual processing happens in optimize_step().
        
        Args:
            frame_id: Frame identifier (not used for ordering)
            rgb: RGB image (HxWx3)
            depth: Optional depth (HxW) - DA3 predicts its own depth
            pose_world_cam: Camera-to-world pose (4x4)
        """
        if not self._initialized:
            raise RuntimeError("Scene not initialized. Call initialize_scene first.")
        
        # Ensure uint8
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        
        self._frames.append((rgb.copy(), pose_world_cam.copy()))
        self._depths.append(depth.copy() if depth is not None else None)
        
        # Reset cached Gaussians when new frame added
        self._gaussians = None
        self._prediction = None
    
    def optimize_step(self, n_steps: int = 1) -> Dict[str, float]:
        """
        Run DA3 inference to generate Gaussians.
        
        Unlike other engines, DA3 doesn't do iterative optimization.
        It generates Gaussians in a single forward pass.
        
        Args:
            n_steps: Ignored (DA3 is single-pass)
            
        Returns:
            Dict with metrics
        """
        if not self._initialized:
            raise RuntimeError("Scene not initialized")
        
        if len(self._frames) == 0:
            raise RuntimeError("No frames added")
        
        # Lazy load model
        self._lazy_load_model()
        
        # Prepare inputs for DA3
        images = [frame[0] for frame in self._frames]
        poses = np.array([frame[1] for frame in self._frames])
        
        # Convert poses to extrinsics (world-to-camera)
        extrinsics = np.array([np.linalg.inv(p) for p in poses])
        
        # Build intrinsics matrix
        fx, fy = self._intrinsics['fx'], self._intrinsics['fy']
        cx, cy = self._intrinsics['cx'], self._intrinsics['cy']
        w, h = int(self._intrinsics['width']), int(self._intrinsics['height'])
        
        intrinsics_mat = np.array([
            [[fx, 0, cx],
             [0, fy, cy],
             [0, 0, 1]]
        ] * len(images), dtype=np.float32)
        
        # Run DA3 with Gaussian output enabled
        logger.info(f"Running DA3 inference on {len(images)} frames with GS output")
        
        self._prediction = self._model.inference(
            images,
            extrinsics=extrinsics.astype(np.float32),
            intrinsics=intrinsics_mat,
            infer_gs=True,  # Enable Gaussian splatting branch
            process_res=self.process_res,
        )
        
        # Extract Gaussians
        if hasattr(self._prediction, 'gaussians') and self._prediction.gaussians is not None:
            self._gaussians = self._prediction.gaussians
            num_gaussians = self._gaussians.means.shape[1] if self._gaussians.means is not None else 0
            logger.info(f"DA3 generated {num_gaussians} Gaussians")
        else:
            num_gaussians = 0
            logger.warning("DA3 did not generate Gaussians")
        
        return {
            'num_gaussians': num_gaussians,
            'num_frames': len(self._frames),
        }
    
    def render_view(
        self,
        pose_world_cam: np.ndarray,
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Render a view using DA3's Gaussian renderer.
        
        Args:
            pose_world_cam: Camera-to-world pose (4x4)
            image_size: (width, height)
            
        Returns:
            Rendered RGB image (HxWx3, uint8)
        """
        if not self._initialized:
            raise RuntimeError("Scene not initialized")
        
        width, height = image_size
        
        if self._gaussians is None:
            # No Gaussians yet, return black image
            return np.zeros((height, width, 3), dtype=np.uint8)
        
        # Use DA3's built-in renderer
        try:
            from depth_anything_3.model.utils.gs_renderer import run_renderer_in_chunk_w_trj_mode
            
            # Convert pose to extrinsic
            extrinsic = _torch.from_numpy(
                np.linalg.inv(pose_world_cam).astype(np.float32)
            ).unsqueeze(0).unsqueeze(0).to(self._gaussians.means.device)
            
            # Build intrinsics
            fx, fy = self._intrinsics['fx'], self._intrinsics['fy']
            cx, cy = self._intrinsics['cx'], self._intrinsics['cy']
            intrinsic = _torch.tensor([
                [[fx, 0, cx],
                 [0, fy, cy],
                 [0, 0, 1]]
            ], dtype=_torch.float32, device=self._gaussians.means.device)
            
            # Render
            color, depth = run_renderer_in_chunk_w_trj_mode(
                gaussians=self._gaussians,
                extrinsics=extrinsic,
                intrinsics=intrinsic,
                image_shape=(height, width),
                chunk_size=1,
                trj_mode="original",
                use_sh=True,
                color_mode="RGB",
                enable_tqdm=False,
            )
            
            # Convert to numpy
            img = color[0, 0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            return img
            
        except Exception as e:
            logger.error(f"DA3 render failed: {e}")
            # Fallback: return processed image from prediction
            if (self._prediction is not None and 
                hasattr(self._prediction, 'processed_images') and 
                len(self._prediction.processed_images) > 0):
                img = self._prediction.processed_images[0]
                if img.shape[:2] != (height, width):
                    import cv2
                    img = cv2.resize(img, (width, height))
                return img
            return np.zeros((height, width, 3), dtype=np.uint8)
    
    def save_state(self, path: str) -> None:
        """
        Save Gaussians to PLY file.
        
        Args:
            path: Output directory or file path
        """
        if not self._initialized or self._gaussians is None:
            raise RuntimeError("No Gaussians to save")
        
        from depth_anything_3.utils.export.gs import export_to_gs_ply
        
        os.makedirs(path, exist_ok=True)
        export_to_gs_ply(self._prediction, path)
        logger.info(f"Saved DA3 Gaussians to {path}")
    
    def load_state(self, path: str) -> None:
        """
        Load is not supported for DA3 engine.
        
        DA3 generates Gaussians from scratch each time.
        """
        raise NotImplementedError(
            "DA3 GS engine does not support loading pre-trained Gaussians. "
            "Use optimize_step() to generate new Gaussians from input frames."
        )
    
    def reset(self) -> None:
        """Reset the engine state."""
        self._frames = []
        self._depths = []
        self._gaussians = None
        self._prediction = None
        self._initialized = False
        
        # Clear GPU memory
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    
    def get_num_gaussians(self) -> int:
        """Get number of Gaussians."""
        if self._gaussians is None or self._gaussians.means is None:
            return 0
        return int(self._gaussians.means.shape[1])
    
    def get_point_cloud(self) -> Optional[np.ndarray]:
        """Get Gaussian centers as point cloud."""
        if self._gaussians is None or self._gaussians.means is None:
            return None
        # means shape: (1, N, 3)
        return self._gaussians.means[0].cpu().numpy()
    
    def cleanup(self):
        """Release GPU memory."""
        self.reset()
        if self._model is not None:
            del self._model
            self._model = None
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
