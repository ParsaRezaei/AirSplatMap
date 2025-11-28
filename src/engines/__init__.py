# Engine implementations for 3D Gaussian Splatting backends
# =========================================================
# 
# This package provides pluggable engine backends that implement the
# BaseGSEngine interface. Each engine wraps a different 3DGS implementation.

from .base import BaseGSEngine
from .graphdeco_engine import GraphdecoEngine

__all__ = ["BaseGSEngine", "GraphdecoEngine"]
