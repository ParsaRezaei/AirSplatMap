"""
Apple Depth Pro Estimator
=========================

Wrapper for Apple's Depth Pro - Sharp Monocular Metric Depth in Less Than a Second.

Depth Pro is a foundation model for zero-shot metric monocular depth estimation
that produces high-resolution depth maps with excellent boundary sharpness.

Key Features:
- Metric depth (outputs depth in meters)
- Absolute scale without camera intrinsics
- High-resolution output (2.25 megapixels)
- Fast inference (~0.3s per image)
- Focal length estimation from single image

Installation:
    # Clone and install depth_pro
    git clone https://github.com/apple/ml-depth-pro.git
    cd ml-depth-pro
    pip install -e .
    
    # Download pretrained models
    source get_pretrained_models.sh

Paper: "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second"
       Bochkovskii et al., ICLR 2025
       https://arxiv.org/abs/2410.02073
"""

import numpy as np
import logging
from typing import Optional, Tuple
from pathlib import Path

from .base import BaseDepthEstimator, DepthResult

logger = logging.getLogger(__name__)

# Check availability
_AVAILABLE = False
_torch = None
_depth_pro = None

try:
    import torch as _torch
    try:
        import depth_pro as _depth_pro
        _AVAILABLE = True
        logger.debug("Apple Depth Pro is available")
    except ImportError:
        logger.debug("depth_pro package not installed")
except ImportError:
    logger.debug("PyTorch not available")


class DepthProEstimator(BaseDepthEstimator):
    """
    Metric depth estimation using Apple's Depth Pro.
    
    Produces sharp, high-resolution depth maps in actual meters.
    Also estimates camera focal length from the image.
    
    Features:
    - Zero-shot metric depth (no calibration needed)
    - High boundary accuracy
    - Focal length estimation
    - Fast inference
    
    Usage:
        estimator = DepthProEstimator()
        result = estimator.estimate(rgb_image)
        depth = result.depth  # Depth in meters
        
        # Also get focal length
        depth, focal_px = estimator.estimate_with_focal(rgb_image)
    
    Note:
        Requires installing the depth_pro package:
        pip install git+https://github.com/apple/ml-depth-pro.git
        
        Or clone and install:
        git clone https://github.com/apple/ml-depth-pro.git
        cd ml-depth-pro
        pip install -e .
    """
    
    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: Optional[str] = None,
    ):
        """
        Initialize Depth Pro.
        
        Args:
            device: 'cuda' or 'cpu'
            checkpoint_path: Optional path to model checkpoint.
                           If None, uses default from depth_pro package.
        """
        super().__init__(device)
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._transform = None
        self._focal_length_px: Optional[float] = None
    
    @staticmethod
    def is_available() -> bool:
        """Check if Depth Pro is installed and available."""
        return _AVAILABLE
    
    def _lazy_init(self):
        """Lazy initialization of model."""
        if self._initialized:
            return
        
        if not _AVAILABLE:
            logger.error(
                "Apple Depth Pro not available. Install with:\n"
                "  pip install git+https://github.com/apple/ml-depth-pro.git\n"
                "Or clone and install:\n"
                "  git clone https://github.com/apple/ml-depth-pro.git\n"
                "  cd ml-depth-pro && pip install -e ."
            )
            self._initialized = True
            return
        
        try:
            import sys
            import io
            
            logger.info("Loading Apple Depth Pro...")
            
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
            from dataclasses import replace as dataclass_replace
            
            # Find checkpoint path
            checkpoint = self.checkpoint_path
            if checkpoint is None:
                # Look for checkpoint in common locations
                possible_paths = [
                    Path(__file__).parent.parent.parent / "submodules" / "ml-depth-pro" / "checkpoints" / "depth_pro.pt",
                    Path.cwd() / "submodules" / "ml-depth-pro" / "checkpoints" / "depth_pro.pt",
                    Path.cwd() / "checkpoints" / "depth_pro.pt",
                    Path.home() / ".cache" / "depth_pro" / "depth_pro.pt",
                ]
                for p in possible_paths:
                    if p.exists():
                        checkpoint = str(p)
                        logger.info(f"Found checkpoint at: {checkpoint}")
                        break
            
            # Create config with custom checkpoint path
            if checkpoint and Path(checkpoint).exists():
                config = dataclass_replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=checkpoint)
            else:
                config = DEFAULT_MONODEPTH_CONFIG_DICT
                logger.warning(f"Using default checkpoint path: {config.checkpoint_uri}")
            
            # Suppress verbose output during model creation
            import warnings
            
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            
            # Also suppress logging from depth_pro module
            depth_pro_logger = logging.getLogger('depth_pro')
            old_level = depth_pro_logger.level
            depth_pro_logger.setLevel(logging.ERROR)
            
            # Suppress torch model printing
            old_torch_repr = _torch.nn.Module.__repr__
            _torch.nn.Module.__repr__ = lambda self: f"{self.__class__.__name__}(...)"
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # Create model and transforms
                    model, transform = _depth_pro.create_model_and_transforms(
                        config=config,
                        device=_torch.device(self.device)
                    )
            finally:
                # Restore everything
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                depth_pro_logger.setLevel(old_level)
                _torch.nn.Module.__repr__ = old_torch_repr
            
            self._model = model
            self._transform = transform
            self._model.eval()
            
            self._initialized = True
            logger.info(f"Apple Depth Pro initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Apple Depth Pro: {e}")
            logger.error(
                "Make sure you've downloaded the pretrained models:\n"
                "  cd submodules/ml-depth-pro && source get_pretrained_models.sh"
            )
            self._model = None
            self._initialized = True
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        """
        Estimate metric depth from RGB image.
        
        Args:
            rgb: HxWx3 RGB image (uint8 or float32 [0-1])
            
        Returns:
            DepthResult with metric depth in meters
        """
        self._lazy_init()
        
        h, w = rgb.shape[:2]
        
        if self._model is None:
            return DepthResult(
                depth=np.ones((h, w), dtype=np.float32) * 2.0,
                is_metric=True,
            )
        
        # Ensure uint8 RGB
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        try:
            from PIL import Image
            
            # Convert to PIL Image
            image = Image.fromarray(rgb)
            
            # Apply transform
            image_tensor = self._transform(image)
            
            # Run inference
            with _torch.no_grad():
                prediction = self._model.infer(image_tensor, f_px=None)
            
            # Extract outputs
            depth = prediction["depth"]  # Depth in meters
            self._focal_length_px = prediction.get("focallength_px")
            
            # Convert to numpy
            if isinstance(depth, _torch.Tensor):
                depth = depth.squeeze().cpu().numpy()
            
            # Resize to original size if needed
            if depth.shape != (h, w):
                import cv2
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            
            depth = depth.astype(np.float32)
            
            return DepthResult(
                depth=depth,
                is_metric=True,  # Depth Pro outputs metric depth!
                min_depth=float(np.maximum(depth.min(), 0.01)),
                max_depth=float(depth.max()),
            )
            
        except Exception as e:
            logger.error(f"Depth Pro estimation failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return DepthResult(
                depth=np.ones((h, w), dtype=np.float32) * 2.0,
                is_metric=True,
            )
    
    def estimate_with_focal(
        self,
        rgb: np.ndarray,
        known_focal_px: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Estimate depth and focal length from RGB image.
        
        Args:
            rgb: HxWx3 RGB image
            known_focal_px: Optional known focal length in pixels.
                          If provided, uses this instead of estimating.
            
        Returns:
            (depth, focal_length_px): Depth map and focal length in pixels
        """
        self._lazy_init()
        
        h, w = rgb.shape[:2]
        
        if self._model is None:
            return np.ones((h, w), dtype=np.float32) * 2.0, w  # Default to image width
        
        # Ensure uint8 RGB
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            else:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
        try:
            from PIL import Image
            
            image = Image.fromarray(rgb)
            image_tensor = self._transform(image)
            
            with _torch.no_grad():
                prediction = self._model.infer(image_tensor, f_px=known_focal_px)
            
            depth = prediction["depth"]
            focal_px = prediction.get("focallength_px", w)
            
            if isinstance(depth, _torch.Tensor):
                depth = depth.squeeze().cpu().numpy()
            
            if isinstance(focal_px, _torch.Tensor):
                focal_px = focal_px.item()
            
            if depth.shape != (h, w):
                import cv2
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            
            return depth.astype(np.float32), float(focal_px)
            
        except Exception as e:
            logger.error(f"Depth Pro estimation failed: {e}")
            return np.ones((h, w), dtype=np.float32) * 2.0, float(w)
    
    def get_focal_length(self) -> Optional[float]:
        """
        Get the focal length from the last estimation.
        
        Returns:
            Focal length in pixels, or None if not available
        """
        return self._focal_length_px
    
    def get_name(self) -> str:
        return "depth_pro"
    
    def is_metric(self) -> bool:
        return True
    
    def cleanup(self):
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._transform is not None:
            del self._transform
            self._transform = None
        self._initialized = False
        
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
            import gc
            gc.collect()
        
        logger.debug("Depth Pro resources released")


class DepthProLiteEstimator(DepthProEstimator):
    """
    Lighter version of Depth Pro for faster inference and lower memory usage.
    
    Uses lower resolution processing internally while maintaining
    output quality through upscaling.
    
    Note: This uses the same model but processes at lower resolution.
    If Apple releases a dedicated lite model, this will be updated.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        max_dimension: int = 384,  # Reduced to 384 for better memory efficiency
    ):
        """
        Initialize Depth Pro Lite.
        
        Args:
            device: 'cuda' or 'cpu'
            max_dimension: Maximum image dimension for processing (default 384)
        """
        super().__init__(device)
        self.max_dimension = max_dimension
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        """Estimate depth with reduced resolution for speed and memory."""
        h, w = rgb.shape[:2]
        
        # Downscale if needed
        scale = 1.0
        if max(h, w) > self.max_dimension:
            scale = self.max_dimension / max(h, w)
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            import cv2
            rgb_small = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            rgb_small = rgb
        
        # Clear CUDA cache before running
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        
        # Run estimation on smaller image
        result = super().estimate(rgb_small)
        
        # Clear cache after
        if _torch is not None and _torch.cuda.is_available():
            _torch.cuda.empty_cache()
        
        # Upscale depth back to original size
        if scale != 1.0:
            import cv2
            result.depth = cv2.resize(
                result.depth, (w, h), interpolation=cv2.INTER_LINEAR
            )
        
        return result
    
    def get_name(self) -> str:
        return f"depth_pro_lite_{self.max_dimension}"
