"""
MiDaS Depth Estimator
=====================

Wrapper for MiDaS - robust monocular depth estimation.

MiDaS is a classic depth estimation model that's well-tested
and works reliably across many scenarios.

Models:
- MiDaS_small: Fastest, lower accuracy
- DPT_Large: Highest accuracy, slower
- DPT_Hybrid: Good balance

Installation:
    # Models download automatically via torch hub
    pip install timm
"""

import numpy as np
import logging

from .base import BaseDepthEstimator, DepthResult

logger = logging.getLogger(__name__)

# Check availability
_AVAILABLE = False
_torch = None

try:
    import torch as _torch
    _AVAILABLE = True
except ImportError:
    pass


class MiDaSEstimator(BaseDepthEstimator):
    """
    Depth estimation using MiDaS.
    
    Produces relative (inverse) depth - higher values are closer.
    Output is inverted internally to give conventional depth.
    
    Models:
    - 'small' (MiDaS_small): Fast, ~50ms
    - 'hybrid' (DPT_Hybrid): Balanced
    - 'large' (DPT_Large): Most accurate
    
    Usage:
        estimator = MiDaSEstimator(model_type='small')
        result = estimator.estimate(rgb_image)
    """
    
    MODEL_NAMES = {
        'small': 'MiDaS_small',
        'midas_small': 'MiDaS_small',
        'hybrid': 'DPT_Hybrid',
        'dpt_hybrid': 'DPT_Hybrid',
        'large': 'DPT_Large',
        'dpt_large': 'DPT_Large',
    }
    
    def __init__(
        self,
        model_type: str = "small",
        device: str = "cuda",
    ):
        super().__init__(device)
        self.model_type = model_type.lower()
        self._model = None
        self._transform = None
    
    @staticmethod
    def is_available() -> bool:
        return _AVAILABLE
    
    def _lazy_init(self):
        if self._initialized:
            return
        
        if not _AVAILABLE:
            logger.error("PyTorch not available")
            self._initialized = True
            return
        
        try:
            model_name = self.MODEL_NAMES.get(self.model_type, 'MiDaS_small')
            
            logger.info(f"Loading MiDaS ({model_name})...")
            self._model = _torch.hub.load('intel-isl/MiDaS', model_name)
            self._model.to(self.device)
            self._model.eval()
            
            # Get appropriate transform
            midas_transforms = _torch.hub.load('intel-isl/MiDaS', 'transforms')
            if 'small' in model_name.lower():
                self._transform = midas_transforms.small_transform
            else:
                self._transform = midas_transforms.dpt_transform
            
            self._initialized = True
            logger.info(f"MiDaS ({model_name}) initialized on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load MiDaS: {e}")
            self._model = None
            self._initialized = True
    
    def estimate(self, rgb: np.ndarray) -> DepthResult:
        self._lazy_init()
        
        h, w = rgb.shape[:2]
        
        if self._model is None:
            return DepthResult(
                depth=np.ones((h, w), dtype=np.float32) * 2.0,
                is_metric=False,
            )
        
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        
        try:
            with _torch.no_grad():
                input_batch = self._transform(rgb).to(self.device)
                prediction = self._model(input_batch)
                
                # Resize to original
                prediction = _torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(h, w),
                    mode='bicubic',
                    align_corners=False,
                ).squeeze()
                
                disparity = prediction.cpu().numpy()
            
            # MiDaS outputs inverse depth (disparity)
            # Convert to depth (invert and normalize)
            depth = 1.0 / (disparity + 1e-6)
            
            # Normalize to reasonable range (0.1 to 10m typical)
            depth = depth / depth.max() * 10.0
            
            return DepthResult(
                depth=depth.astype(np.float32),
                is_metric=False,  # Relative depth
                min_depth=float(depth.min()),
                max_depth=float(depth.max()),
            )
            
        except Exception as e:
            logger.error(f"MiDaS estimation failed: {e}")
            return DepthResult(
                depth=np.ones((h, w), dtype=np.float32) * 2.0,
                is_metric=False,
            )
    
    def get_name(self) -> str:
        return f"midas_{self.model_type}"
    
    def is_metric(self) -> bool:
        return False
    
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
