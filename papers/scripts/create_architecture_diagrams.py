#!/usr/bin/env python3
"""
Create architecture diagrams for AirSplatMap papers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

DPI = 300

def create_pipeline_architecture(output_path):
    """Create the main pipeline architecture diagram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    input_color = '#E8F4FD'
    pose_color = '#FFE4B5'
    depth_color = '#98FB98'
    gs_color = '#DDA0DD'
    output_color = '#FFB6C1'
    arrow_color = '#333333'
    
    # Input block
    input_box = FancyBboxPatch((0.3, 3), 1.8, 2, boxstyle="round,pad=0.05",
                                facecolor=input_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(input_box)
    ax.text(1.2, 4.3, 'RGB\nVideo', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(1.2, 3.5, 'Stream', ha='center', va='center', fontsize=8)
    
    # Pose Estimation Module
    pose_box = FancyBboxPatch((2.8, 5), 2.8, 2.5, boxstyle="round,pad=0.05",
                               facecolor=pose_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(pose_box)
    ax.text(4.2, 7, 'Pose Estimation', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(4.2, 6.3, '• ORB / SIFT', ha='center', va='center', fontsize=7)
    ax.text(4.2, 5.8, '• Optical Flow', ha='center', va='center', fontsize=7)
    ax.text(4.2, 5.3, '• SuperPoint/R2D2', ha='center', va='center', fontsize=7)
    
    # Depth Estimation Module
    depth_box = FancyBboxPatch((2.8, 0.5), 2.8, 2.5, boxstyle="round,pad=0.05",
                                facecolor=depth_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(depth_box)
    ax.text(4.2, 2.5, 'Depth Estimation', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(4.2, 1.8, '• MiDaS', ha='center', va='center', fontsize=7)
    ax.text(4.2, 1.3, '• Depth Anything', ha='center', va='center', fontsize=7)
    ax.text(4.2, 0.8, '• Depth Pro', ha='center', va='center', fontsize=7)
    
    # 3DGS Engine
    gs_box = FancyBboxPatch((6.5, 2.5), 3, 3, boxstyle="round,pad=0.05",
                             facecolor=gs_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(gs_box)
    ax.text(8, 5, '3D Gaussian', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(8, 4.4, 'Splatting Engine', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(8, 3.6, '• GraphDeco', ha='center', va='center', fontsize=7)
    ax.text(8, 3.1, '• gsplat / MonoGS', ha='center', va='center', fontsize=7)
    ax.text(8, 2.6, '• SplaTAM / G-SLAM', ha='center', va='center', fontsize=7)
    
    # Output
    output_box = FancyBboxPatch((10.5, 2.5), 2.5, 3, boxstyle="round,pad=0.05",
                                 facecolor=output_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(11.75, 5, 'Output', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(11.75, 4.2, '3D Gaussians', ha='center', va='center', fontsize=8)
    ax.text(11.75, 3.5, 'Novel Views', ha='center', va='center', fontsize=8)
    ax.text(11.75, 2.8, 'Point Cloud', ha='center', va='center', fontsize=8)
    
    # Arrows
    ax.annotate('', xy=(2.8, 6.25), xytext=(2.1, 4.5),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
    ax.annotate('', xy=(2.8, 1.75), xytext=(2.1, 3.5),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
    ax.annotate('', xy=(6.5, 4.5), xytext=(5.6, 6),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
    ax.annotate('', xy=(6.5, 3.5), xytext=(5.6, 2),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
    ax.annotate('', xy=(10.5, 4), xytext=(9.5, 4),
                arrowprops=dict(arrowstyle='->', color=arrow_color, lw=1.5))
    
    # Labels on arrows
    ax.text(2.3, 5.5, 'T∈SE(3)', fontsize=7, style='italic')
    ax.text(2.3, 2.5, 'D∈ℝ^(H×W)', fontsize=7, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {output_path}')


def create_gaussian_representation(output_path):
    """Create diagram showing Gaussian representation."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Single Gaussian illustration
    circle = Circle((2, 3), 1, facecolor='#FFB6C1', edgecolor='black', alpha=0.7, linewidth=1.5)
    ax.add_patch(circle)
    ax.text(2, 3, 'G', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Parameters box
    params_box = FancyBboxPatch((4, 0.5), 7, 5, boxstyle="round,pad=0.1",
                                 facecolor='#F0F0F0', edgecolor='black', linewidth=1)
    ax.add_patch(params_box)
    
    ax.text(7.5, 5, 'Gaussian Parameters', ha='center', va='center', fontsize=10, fontweight='bold')
    
    params = [
        ('μ ∈ ℝ³', 'Position (x, y, z)'),
        ('Σ ∈ ℝ³ˣ³', 'Covariance (shape/orientation)'),
        ('α ∈ [0,1]', 'Opacity'),
        ('c ∈ ℝ^(3×(l+1)²)', 'SH coefficients (color)'),
    ]
    
    for i, (symbol, desc) in enumerate(params):
        y = 4 - i * 0.9
        ax.text(4.5, y, symbol, fontsize=9, fontweight='bold', family='monospace')
        ax.text(6.5, y, ':', fontsize=9)
        ax.text(6.8, y, desc, fontsize=8)
    
    # Arrow
    ax.annotate('', xy=(4, 3), xytext=(3.2, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {output_path}')


def create_training_pipeline(output_path):
    """Create training pipeline diagram."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Colors
    data_color = '#E8F4FD'
    process_color = '#DDA0DD'
    loss_color = '#FFB6C1'
    
    # Input image
    img_box = FancyBboxPatch((0.3, 2.5), 1.5, 2, boxstyle="round,pad=0.05",
                              facecolor=data_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(img_box)
    ax.text(1.05, 3.5, 'Input\nImage', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Gaussians
    gauss_box = FancyBboxPatch((2.5, 2.5), 1.8, 2, boxstyle="round,pad=0.05",
                                facecolor=process_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(gauss_box)
    ax.text(3.4, 3.5, 'Gaussians\n{Gᵢ}', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Rasterizer
    rast_box = FancyBboxPatch((5, 2.5), 2, 2, boxstyle="round,pad=0.05",
                               facecolor=process_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rast_box)
    ax.text(6, 4, 'Differentiable', ha='center', va='center', fontsize=8)
    ax.text(6, 3.3, 'Rasterizer', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Rendered image
    render_box = FancyBboxPatch((7.8, 2.5), 1.5, 2, boxstyle="round,pad=0.05",
                                 facecolor=data_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(render_box)
    ax.text(8.55, 3.5, 'Rendered\nImage', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Loss computation
    loss_box = FancyBboxPatch((10, 2.5), 2.2, 2, boxstyle="round,pad=0.05",
                               facecolor=loss_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(loss_box)
    ax.text(11.1, 4, 'Loss', ha='center', va='center', fontsize=8, fontweight='bold')
    ax.text(11.1, 3.2, 'L₁ + λ·SSIM', ha='center', va='center', fontsize=8)
    
    # GT image
    gt_box = FancyBboxPatch((10, 5), 2.2, 1.3, boxstyle="round,pad=0.05",
                             facecolor=data_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(gt_box)
    ax.text(11.1, 5.65, 'GT Image', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Gradient flow (back)
    grad_box = FancyBboxPatch((5, 0.3), 2, 1.2, boxstyle="round,pad=0.05",
                               facecolor='#98FB98', edgecolor='black', linewidth=1.5)
    ax.add_patch(grad_box)
    ax.text(6, 0.9, '∇L → Update', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(2.5, 3.5), xytext=(1.8, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(5, 3.5), xytext=(4.3, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(7.8, 3.5), xytext=(7, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(10, 3.5), xytext=(9.3, 3.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(11.1, 4.5), xytext=(11.1, 5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Backprop arrow
    ax.annotate('', xy=(4.3, 2.5), xytext=(5, 1.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5, ls='--'))
    ax.annotate('', xy=(7, 1.5), xytext=(10, 2.5),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5, ls='--'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {output_path}')


def create_pose_estimation_architecture(output_path):
    """Create pose estimation architecture diagram."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    img_color = '#E8F4FD'
    classical_color = '#FFE4B5'
    learned_color = '#98FB98'
    output_color = '#DDA0DD'
    
    # Input images
    img1 = FancyBboxPatch((0.3, 5), 1.5, 2, boxstyle="round,pad=0.05",
                           facecolor=img_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(img1)
    ax.text(1.05, 6, 'Frame\nt-1', ha='center', va='center', fontsize=8, fontweight='bold')
    
    img2 = FancyBboxPatch((0.3, 1), 1.5, 2, boxstyle="round,pad=0.05",
                           facecolor=img_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(img2)
    ax.text(1.05, 2, 'Frame\nt', ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Classical path
    classical_box = FancyBboxPatch((3, 5), 3.5, 2.5, boxstyle="round,pad=0.05",
                                    facecolor=classical_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(classical_box)
    ax.text(4.75, 7, 'Classical Methods', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(4.75, 6.2, '• Feature Detection', ha='center', va='center', fontsize=7)
    ax.text(4.75, 5.6, '• Matching + RANSAC', ha='center', va='center', fontsize=7)
    ax.text(4.75, 5.1, '• Essential Matrix', ha='center', va='center', fontsize=7)
    
    # Learned path
    learned_box = FancyBboxPatch((3, 0.5), 3.5, 2.5, boxstyle="round,pad=0.05",
                                  facecolor=learned_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(learned_box)
    ax.text(4.75, 2.5, 'Learned Methods', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(4.75, 1.7, '• CNN Feature Extraction', ha='center', va='center', fontsize=7)
    ax.text(4.75, 1.1, '• Attention Matching', ha='center', va='center', fontsize=7)
    ax.text(4.75, 0.6, '• Pose Regression', ha='center', va='center', fontsize=7)
    
    # Pose output
    pose_box = FancyBboxPatch((8, 2.75), 2.5, 2.5, boxstyle="round,pad=0.05",
                               facecolor=output_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(pose_box)
    ax.text(9.25, 4.7, 'Relative Pose', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(9.25, 3.9, 'T = [R | t]', ha='center', va='center', fontsize=9, family='monospace')
    ax.text(9.25, 3.2, 'R ∈ SO(3)', ha='center', va='center', fontsize=8)
    
    # Trajectory output
    traj_box = FancyBboxPatch((11.5, 2.75), 2, 2.5, boxstyle="round,pad=0.05",
                               facecolor='#FFB6C1', edgecolor='black', linewidth=1.5)
    ax.add_patch(traj_box)
    ax.text(12.5, 4.7, 'Trajectory', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(12.5, 3.7, '{T₁...Tₙ}', ha='center', va='center', fontsize=9, family='monospace')
    
    # Arrows
    ax.annotate('', xy=(3, 6), xytext=(1.8, 6),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(3, 2), xytext=(1.8, 2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(3, 6), xytext=(1.8, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(3, 2), xytext=(1.8, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(8, 4.5), xytext=(6.5, 6),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(8, 3.5), xytext=(6.5, 2),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(11.5, 4), xytext=(10.5, 4),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {output_path}')


def create_depth_estimation_architecture(output_path):
    """Create depth estimation architecture diagram."""
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Colors
    img_color = '#E8F4FD'
    enc_color = '#98FB98'
    dec_color = '#DDA0DD'
    out_color = '#FFB6C1'
    
    # Input
    img_box = FancyBboxPatch((0.3, 2), 1.8, 2, boxstyle="round,pad=0.05",
                              facecolor=img_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(img_box)
    ax.text(1.2, 3, 'RGB\nImage', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Encoder
    enc_box = FancyBboxPatch((2.8, 1.5), 2.5, 3, boxstyle="round,pad=0.05",
                              facecolor=enc_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(enc_box)
    ax.text(4.05, 4, 'Encoder', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(4.05, 3.2, 'ViT / CNN', ha='center', va='center', fontsize=8)
    ax.text(4.05, 2.5, 'Backbone', ha='center', va='center', fontsize=8)
    
    # Decoder
    dec_box = FancyBboxPatch((6, 1.5), 2.5, 3, boxstyle="round,pad=0.05",
                              facecolor=dec_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(dec_box)
    ax.text(7.25, 4, 'Decoder', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(7.25, 3.2, 'Multi-scale', ha='center', va='center', fontsize=8)
    ax.text(7.25, 2.5, 'Upsampling', ha='center', va='center', fontsize=8)
    
    # Alignment
    align_box = FancyBboxPatch((9.2, 1.5), 2.2, 3, boxstyle="round,pad=0.05",
                                facecolor='#FFE4B5', edgecolor='black', linewidth=1.5)
    ax.add_patch(align_box)
    ax.text(10.3, 4, 'Scale-Shift', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(10.3, 3.2, 'Alignment', ha='center', va='center', fontsize=8)
    ax.text(10.3, 2.5, 'd′=s·d+b', ha='center', va='center', fontsize=8, family='monospace')
    
    # Output
    out_box = FancyBboxPatch((12.1, 2), 1.5, 2, boxstyle="round,pad=0.05",
                              facecolor=out_color, edgecolor='black', linewidth=1.5)
    ax.add_patch(out_box)
    ax.text(12.85, 3, 'Depth\nMap', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(2.8, 3), xytext=(2.1, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(6, 3), xytext=(5.3, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(9.2, 3), xytext=(8.5, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(12.1, 3), xytext=(11.4, 3),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {output_path}')


def create_evaluation_framework(output_path):
    """Create evaluation framework diagram."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    # Title
    ax.text(7, 8.5, 'Systematic Evaluation Framework', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    
    # Dataset box
    data_box = FancyBboxPatch((0.5, 5.5), 3, 2.3, boxstyle="round,pad=0.05",
                               facecolor='#E8F4FD', edgecolor='black', linewidth=1.5)
    ax.add_patch(data_box)
    ax.text(2, 7.3, 'Datasets', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(2, 6.6, '• TUM RGB-D (7)', ha='center', va='center', fontsize=7)
    ax.text(2, 6.1, '• Replica (8)', ha='center', va='center', fontsize=7)
    ax.text(2, 5.6, '• 7-Scenes (2)', ha='center', va='center', fontsize=7)
    
    # Methods box
    method_box = FancyBboxPatch((4.5, 5.5), 3.5, 2.3, boxstyle="round,pad=0.05",
                                 facecolor='#FFE4B5', edgecolor='black', linewidth=1.5)
    ax.add_patch(method_box)
    ax.text(6.25, 7.3, 'Methods', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(6.25, 6.6, '• 11 Pose Methods', ha='center', va='center', fontsize=7)
    ax.text(6.25, 6.1, '• 4 Depth Methods', ha='center', va='center', fontsize=7)
    ax.text(6.25, 5.6, '• 5 GS Engines', ha='center', va='center', fontsize=7)
    
    # Hardware box
    hw_box = FancyBboxPatch((9, 5.5), 3.5, 2.3, boxstyle="round,pad=0.05",
                             facecolor='#98FB98', edgecolor='black', linewidth=1.5)
    ax.add_patch(hw_box)
    ax.text(10.75, 7.3, 'Hardware', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(10.75, 6.6, '• Desktop (2080 Ti)', ha='center', va='center', fontsize=7)
    ax.text(10.75, 6.1, '• Jetson Orin', ha='center', va='center', fontsize=7)
    ax.text(10.75, 5.6, '• Power Monitoring', ha='center', va='center', fontsize=7)
    
    # Evaluation
    eval_box = FancyBboxPatch((4, 2.5), 5.5, 2.3, boxstyle="round,pad=0.05",
                               facecolor='#DDA0DD', edgecolor='black', linewidth=1.5)
    ax.add_patch(eval_box)
    ax.text(6.75, 4.3, 'Evaluation Pipeline', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(5, 3.5, 'Pose: ATE, RPE', ha='center', va='center', fontsize=7)
    ax.text(5, 3, 'Depth: AbsRel, δ₁', ha='center', va='center', fontsize=7)
    ax.text(8.5, 3.5, 'GS: PSNR, SSIM', ha='center', va='center', fontsize=7)
    ax.text(8.5, 3, 'Time: FPS, Latency', ha='center', va='center', fontsize=7)
    
    # Results
    result_box = FancyBboxPatch((4, 0.3), 5.5, 1.5, boxstyle="round,pad=0.05",
                                 facecolor='#FFB6C1', edgecolor='black', linewidth=1.5)
    ax.add_patch(result_box)
    ax.text(6.75, 1.3, 'Results: 537 Pose + 204 Depth + 271 GS + 2979 Pipeline', 
            ha='center', va='center', fontsize=8, fontweight='bold')
    
    # Arrows
    ax.annotate('', xy=(6.75, 5.5), xytext=(2, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.annotate('', xy=(6.75, 5.5), xytext=(6.25, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.annotate('', xy=(6.75, 5.5), xytext=(10.75, 5.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2))
    ax.annotate('', xy=(6.75, 2.5), xytext=(6.75, 4.8),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.annotate('', xy=(6.75, 1.8), xytext=(6.75, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {output_path}')


def create_densification_diagram(output_path):
    """Create Gaussian densification diagram."""
    fig, ax = plt.subplots(figsize=(7, 2.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    # Initial state
    ax.text(2, 4.5, 'Clone (large gradient)', ha='center', va='center', fontsize=9, fontweight='bold')
    circle1 = Circle((1, 2.5), 0.6, facecolor='#FFB6C1', edgecolor='black', alpha=0.7, linewidth=1.5)
    ax.add_patch(circle1)
    ax.annotate('', xy=(2.5, 2.5), xytext=(1.8, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    circle2 = Circle((3, 2.8), 0.5, facecolor='#FFB6C1', edgecolor='black', alpha=0.7, linewidth=1.5)
    ax.add_patch(circle2)
    circle3 = Circle((3.3, 2.2), 0.5, facecolor='#FFB6C1', edgecolor='black', alpha=0.7, linewidth=1.5)
    ax.add_patch(circle3)
    
    # Split
    ax.text(7, 4.5, 'Split (large variance)', ha='center', va='center', fontsize=9, fontweight='bold')
    from matplotlib.patches import Ellipse
    ellipse1 = Ellipse((5.5, 2.5), 1.5, 0.6, angle=30, facecolor='#98FB98', edgecolor='black', alpha=0.7, linewidth=1.5)
    ax.add_patch(ellipse1)
    ax.annotate('', xy=(7, 2.5), xytext=(6.5, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    circle4 = Circle((7.5, 2.8), 0.35, facecolor='#98FB98', edgecolor='black', alpha=0.7, linewidth=1.5)
    ax.add_patch(circle4)
    circle5 = Circle((8, 2.2), 0.35, facecolor='#98FB98', edgecolor='black', alpha=0.7, linewidth=1.5)
    ax.add_patch(circle5)
    
    # Prune
    ax.text(11.5, 4.5, 'Prune (low opacity)', ha='center', va='center', fontsize=9, fontweight='bold')
    circle6 = Circle((10, 2.5), 0.4, facecolor='#DDA0DD', edgecolor='black', alpha=0.3, linewidth=1.5)
    ax.add_patch(circle6)
    ax.annotate('', xy=(11.5, 2.5), xytext=(10.6, 2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    ax.text(12.5, 2.5, '∅', ha='center', va='center', fontsize=16, color='red')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  Saved: {output_path}')


if __name__ == '__main__':
    import os
    
    dl_dir = '../deep_learning/figures'
    cv_dir = '../computer_vision/figures'
    
    os.makedirs(dl_dir, exist_ok=True)
    os.makedirs(cv_dir, exist_ok=True)
    
    print('Creating architecture diagrams...')
    
    # For both papers
    create_pipeline_architecture(f'{dl_dir}/pipeline_architecture.png')
    create_pipeline_architecture(f'{cv_dir}/pipeline_architecture.png')
    
    # DL paper specific
    create_gaussian_representation(f'{dl_dir}/gaussian_representation.png')
    create_training_pipeline(f'{dl_dir}/training_pipeline.png')
    create_densification_diagram(f'{dl_dir}/densification.png')
    create_evaluation_framework(f'{dl_dir}/evaluation_framework.png')
    
    # CV paper specific
    create_pose_estimation_architecture(f'{cv_dir}/pose_architecture.png')
    create_depth_estimation_architecture(f'{cv_dir}/depth_architecture.png')
    create_evaluation_framework(f'{cv_dir}/evaluation_framework.png')
    
    print('Done!')
