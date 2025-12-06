"""
Visualization Package for AirSplatMap Benchmarks
=================================================

Organized into specialized modules:
- plot_utils: Core plotting functions and utilities
- pose_visualizations: Trajectory and pose analysis
- depth_visualizations: Depth map analysis and comparisons
- gs_visualizations: Gaussian splatting quality analysis  
- cross_metric_plots: Combined multi-metric visualizations
- hardware_visualizations: Resource usage plots
- generate_visualizations: High-level visualization generator
- html_report: Interactive HTML report generation
"""

# Core utilities
from .plot_utils import (
    get_color,
    setup_plot_style,
    save_figure,
)

# High-level generators
from .generate_visualizations import generate_comprehensive_visualizations
from .html_report import (
    generate_html_report,
    generate_simple_html_report,
)

__all__ = [
    # Core
    'get_color',
    'setup_plot_style', 
    'save_figure',
    # Generators
    'generate_comprehensive_visualizations',
    'generate_html_report',
    'generate_simple_html_report',
]