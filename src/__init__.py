# AirSplatMap - Online 3D Gaussian Splatting Pipeline
# =================================================
# 
# A modular, engine-agnostic pipeline for incremental 3D Gaussian Splatting
# from live or replayed streams of RGB / RGB-D frames with poses.

__version__ = "0.1.0"

# Enable MPS fallback for unsupported operators on macOS
# This MUST be set before importing torch to take effect
import os
import platform
if platform.system() == 'Darwin':
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
