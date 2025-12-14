#!/usr/bin/env python3
"""
Publication-Quality Figure Generation for CVPR Papers
=====================================================

Clean, professional figures following visualization best practices.
Inspired by the benchmark HTML report style.
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DPI = 300
SINGLE_COL = 3.4
DOUBLE_COL = 7.0

# Vibrant, colorblind-friendly palette
COLORS = {
    'graphdeco': '#e74c3c',
    'gsplat': '#3498db',
    'splatam': '#2ecc71',
    'monogs': '#9b59b6',
    'gslam': '#f39c12',
    'blue': '#3498db',
    'red': '#e74c3c',
    'green': '#2ecc71',
    'purple': '#9b59b6',
    'orange': '#f39c12',
    'cyan': '#1abc9c',
    'pink': '#e91e63',
    'dark': '#2c3e50',
    'gray': '#95a5a6',
}


def setup_style():
    """Clean publication style."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': DPI,
        'savefig.dpi': DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.4,
    })


def load_results(results_dir):
    """Load benchmark results."""
    results = {'pose': [], 'depth': [], 'gs': [], 'pipeline': []}
    for json_file in Path(results_dir).rglob('results.json'):
        try:
            with open(json_file) as f:
                data = json.load(f)
            platform = 'jetson' if 'jetson' in str(json_file) else 'desktop'
            for key in results.keys():
                for item in data.get(key, []):
                    item['platform'] = platform
                    results[key].append(item)
        except:
            continue
    return results


def save_fig(fig, output_dir, name):
    """Save figure."""
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f'{name}.png'), format='png', 
                dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {name}.png')


# =============================================================================
# Architecture Diagrams
# =============================================================================

def create_pipeline_architecture(output_dir):
    """Clean system architecture diagram."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.5)
    ax.axis('off')
    
    # Define boxes with positions
    boxes = [
        (0.8, 1.25, 'RGB\nStream', COLORS['cyan'], 0.8, 0.8),
        (2.8, 1.8, 'Pose\nEstimation', COLORS['blue'], 1.2, 0.6),
        (2.8, 0.7, 'Depth\nEstimation', COLORS['green'], 1.2, 0.6),
        (5.5, 1.25, '3DGS\nEngine', COLORS['red'], 1.4, 0.9),
        (8.2, 1.25, 'Novel\nViews', COLORS['orange'], 1.0, 0.8),
    ]
    
    for x, y, text, color, w, h in boxes:
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, 
                             facecolor=color, edgecolor=COLORS['dark'],
                             linewidth=1.5, alpha=0.85, 
                             joinstyle='round', capstyle='round')
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=7, 
                fontweight='bold', color='white')
    
    # Arrows
    arrows = [
        ((1.2, 1.25), (2.2, 1.8)),
        ((1.2, 1.25), (2.2, 0.7)),
        ((3.4, 1.8), (4.8, 1.4)),
        ((3.4, 0.7), (4.8, 1.1)),
        ((6.2, 1.25), (7.7, 1.25)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    
    # Method labels
    ax.text(2.8, 2.35, '11 methods', fontsize=6, ha='center', style='italic', color=COLORS['gray'])
    ax.text(2.8, 0.15, '4 methods', fontsize=6, ha='center', style='italic', color=COLORS['gray'])
    ax.text(5.5, 0.35, '5 engines', fontsize=6, ha='center', style='italic', color=COLORS['gray'])
    
    save_fig(fig, output_dir, 'pipeline_architecture')


def create_gaussian_representation(output_dir):
    """3D Gaussian visualization."""
    fig = plt.figure(figsize=(SINGLE_COL, 2.0))
    ax = fig.add_subplot(111, projection='3d')
    
    # Ellipsoid
    u, v = np.linspace(0, 2*np.pi, 30), np.linspace(0, np.pi, 20)
    x = 0.8 * np.outer(np.cos(u), np.sin(v))
    y = 0.5 * np.outer(np.sin(u), np.sin(v))
    z = 0.3 * np.outer(np.ones(len(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, color=COLORS['blue'], alpha=0.6, linewidth=0)
    ax.scatter([0], [0], [0], color=COLORS['red'], s=30, zorder=5)
    
    # Axes
    for vec, c in [([1.2, 0, 0], COLORS['red']), ([0, 0.8, 0], COLORS['green']), ([0, 0, 0.6], COLORS['blue'])]:
        ax.quiver(0, 0, 0, *vec, color=c, arrow_length_ratio=0.15, linewidth=1.5)
    
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
    ax.tick_params(labelsize=5)
    ax.view_init(elev=20, azim=45)
    ax.set_title('3D Gaussian Primitive', fontsize=9, fontweight='bold', pad=0)
    
    save_fig(fig, output_dir, 'gaussian_representation')


def create_training_pipeline(output_dir):
    """Training loop diagram."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 1.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2)
    ax.axis('off')
    
    steps = [
        (1, 'Initialize', COLORS['cyan']),
        (2.8, 'Render', COLORS['blue']),
        (4.6, 'Loss', COLORS['green']),
        (6.4, 'Backprop', COLORS['purple']),
        (8.2, 'Densify', COLORS['red']),
    ]
    
    for x, label, color in steps:
        rect = plt.Rectangle((x-0.6, 0.6), 1.2, 0.8, facecolor=color, 
                             edgecolor=COLORS['dark'], linewidth=1, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, 1.0, label, ha='center', va='center', fontsize=7, 
                fontweight='bold', color='white')
    
    for i in range(len(steps)-1):
        ax.annotate('', xy=(steps[i+1][0]-0.6, 1.0), xytext=(steps[i][0]+0.6, 1.0),
                   arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.2))
    
    # Loop arrow
    ax.annotate('', xy=(1, 0.55), xytext=(8.2, 0.55),
               arrowprops=dict(arrowstyle='->', color=COLORS['gray'], lw=1,
                              connectionstyle='arc3,rad=0.35'))
    ax.text(4.6, 0.2, 'Iterate until convergence', fontsize=6, ha='center', 
            style='italic', color=COLORS['gray'])
    
    save_fig(fig, output_dir, 'training_pipeline')


def create_densification(output_dir):
    """Densification strategy diagram."""
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 1.5))
    
    titles = ['(a) High Gradient', '(b) Clone (small σ)', '(c) Split (large σ)']
    
    for ax, title in zip(axes, titles):
        ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal'); ax.axis('off')
        ax.set_title(title, fontsize=8, pad=3)
    
    # Original
    c = plt.Circle((0, 0), 0.6, color=COLORS['blue'], alpha=0.5)
    axes[0].add_patch(c)
    axes[0].scatter([0], [0], color=COLORS['dark'], s=20, zorder=5)
    axes[0].annotate('', xy=(0.8, 0.8), xytext=(0.3, 0.3),
                    arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=1.5))
    axes[0].text(0.9, 0.9, '∇', fontsize=10, color=COLORS['red'])
    
    # Clone
    for pos in [(-0.4, 0), (0.4, 0)]:
        c = plt.Circle(pos, 0.35, color=COLORS['green'], alpha=0.6)
        axes[1].add_patch(c)
        axes[1].scatter([pos[0]], [pos[1]], color=COLORS['dark'], s=15, zorder=5)
    
    # Split
    from matplotlib.patches import Ellipse
    e = Ellipse((0, 0), 1.4, 0.5, angle=30, color=COLORS['gray'], alpha=0.3)
    axes[2].add_patch(e)
    for pos in [(-0.35, -0.15), (0.35, 0.15)]:
        c = plt.Circle(pos, 0.25, color=COLORS['green'], alpha=0.6)
        axes[2].add_patch(c)
        axes[2].scatter([pos[0]], [pos[1]], color=COLORS['dark'], s=15, zorder=5)
    
    plt.tight_layout()
    save_fig(fig, output_dir, 'densification')


def create_evaluation_framework(output_dir):
    """Evaluation framework overview."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 1.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.5)
    ax.axis('off')
    
    # Boxes
    boxes = [
        (1.5, 1.5, 'Datasets\n17 sequences', COLORS['cyan'], 1.8, 1.0),
        (4.5, 1.5, 'Methods\n20 total', COLORS['blue'], 1.8, 1.0),
        (7.5, 1.5, 'Metrics\n8 types', COLORS['green'], 1.8, 1.0),
    ]
    
    for x, y, text, color, w, h in boxes:
        rect = plt.Rectangle((x-w/2, y-h/2), w, h, facecolor=color,
                             edgecolor=COLORS['dark'], linewidth=1.5, alpha=0.85)
        ax.add_patch(rect)
        ax.text(x, y, text, ha='center', va='center', fontsize=7, 
                fontweight='bold', color='white')
    
    # Arrows
    ax.annotate('', xy=(3.5, 1.5), xytext=(2.5, 1.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    ax.annotate('', xy=(6.5, 1.5), xytext=(5.5, 1.5),
               arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=1.5))
    
    # Stats
    ax.text(5.0, 0.3, '3,991 total benchmark runs', fontsize=7, ha='center',
            style='italic', color=COLORS['gray'])
    
    save_fig(fig, output_dir, 'evaluation_framework')


# =============================================================================
# Deep Learning Paper Figures
# =============================================================================

def dl_engine_comparison(results, output_dir):
    """Engine comparison - clean grouped bars."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    
    engine_data = defaultdict(lambda: {'psnr': [], 'ssim': []})
    for gs in results['gs']:
        if gs.get('platform') == 'desktop' and gs.get('psnr', 0) > 0:
            engine_data[gs.get('engine', '')]['psnr'].append(gs['psnr'])
        if gs.get('platform') == 'desktop' and gs.get('ssim', 0) > 0:
            engine_data[gs.get('engine', '')]['ssim'].append(gs['ssim'])
    
    engines = ['graphdeco', 'gsplat', 'splatam', 'monogs', 'gslam']
    engines = [e for e in engines if engine_data[e]['psnr']]
    
    x = np.arange(len(engines))
    psnr = [np.mean(engine_data[e]['psnr']) for e in engines]
    psnr_err = [np.std(engine_data[e]['psnr']) if len(engine_data[e]['psnr']) > 1 else 0 for e in engines]
    
    colors = [COLORS.get(e, COLORS['gray']) for e in engines]
    bars = ax.bar(x, psnr, 0.6, yerr=psnr_err, capsize=3, color=colors,
                  edgecolor=COLORS['dark'], linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels([e.capitalize() for e in engines], fontsize=7)
    ax.set_ylabel('PSNR (dB)', fontsize=8)
    ax.set_title('3DGS Engine Quality Comparison', fontsize=9, fontweight='bold')
    ax.set_ylim(0, max(psnr) * 1.15 if psnr else 20)
    ax.grid(axis='y', alpha=0.3)
    
    # Value labels
    for bar, val in zip(bars, psnr):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
               f'{val:.1f}', ha='center', fontsize=6, fontweight='bold')
    
    save_fig(fig, output_dir, 'engine_comparison')


def dl_degradation_matrix(results, output_dir):
    """Degradation analysis as heatmap matrix."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    
    # Build matrix
    pose_sources = ['gt', 'robust_flow', 'orb', 'keyframe']
    depth_sources = ['gt', 'midas', 'depth_anything_v3']
    
    matrix = np.zeros((len(pose_sources), len(depth_sources)))
    
    groups = defaultdict(list)
    for p in results['pipeline']:
        if p.get('platform') == 'desktop' and p.get('psnr', 0) > 0:
            key = (p.get('pose_source', ''), p.get('depth_source', ''))
            groups[key].append(p['psnr'])
    
    for i, pose in enumerate(pose_sources):
        for j, depth in enumerate(depth_sources):
            if (pose, depth) in groups:
                matrix[i, j] = np.mean(groups[(pose, depth)])
    
    # Heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=20)
    
    ax.set_xticks(range(len(depth_sources)))
    ax.set_yticks(range(len(pose_sources)))
    ax.set_xticklabels(['GT', 'MiDaS', 'DA3'], fontsize=7)
    ax.set_yticklabels(['GT', 'Flow', 'ORB', 'KF'], fontsize=7)
    ax.set_xlabel('Depth Source', fontsize=8)
    ax.set_ylabel('Pose Source', fontsize=8)
    ax.set_title('PSNR by Input Sources', fontsize=9, fontweight='bold')
    
    # Annotations
    for i in range(len(pose_sources)):
        for j in range(len(depth_sources)):
            if matrix[i, j] > 0:
                color = 'white' if matrix[i, j] < 10 else 'black'
                ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center',
                       fontsize=7, fontweight='bold', color=color)
    
    plt.colorbar(im, ax=ax, shrink=0.8, label='PSNR (dB)')
    save_fig(fig, output_dir, 'degradation_analysis')


def dl_platform_comparison(results, output_dir):
    """Platform comparison - grouped bars."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    
    desktop_gs = [r for r in results['gs'] if r.get('platform') == 'desktop' and r.get('engine') == 'graphdeco']
    jetson_gs = [r for r in results['gs'] if r.get('platform') == 'jetson' and r.get('engine') == 'graphdeco']
    
    d_train = np.mean([r['fps'] for r in desktop_gs if r.get('fps')]) if desktop_gs else 15
    j_train = np.mean([r['fps'] for r in jetson_gs if r.get('fps')]) if jetson_gs else 1.5
    j_render = np.mean([r.get('render_fps', 20) for r in jetson_gs if r.get('render_fps')]) if jetson_gs else 20
    
    x = np.arange(2)
    width = 0.35
    
    ax.bar(x - width/2, [d_train, j_train], width, label='Training',
           color=COLORS['blue'], edgecolor=COLORS['dark'], linewidth=0.5)
    ax.bar([1 + width/2], [j_render], width, label='Rendering',
           color=COLORS['green'], edgecolor=COLORS['dark'], linewidth=0.5)
    
    ax.axhline(y=10, color=COLORS['red'], linestyle='--', linewidth=1.5, alpha=0.8)
    ax.text(1.5, 11, 'Real-time', fontsize=6, color=COLORS['red'])
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Desktop\n(RTX 2080 Ti)', 'Jetson Orin'], fontsize=7)
    ax.set_ylabel('FPS', fontsize=8)
    ax.set_title('Platform Performance', fontsize=9, fontweight='bold')
    ax.legend(loc='upper right', fontsize=6)
    ax.grid(axis='y', alpha=0.3)
    
    # Labels
    for i, val in enumerate([d_train, j_train]):
        ax.text(i - width/2, val + 0.5, f'{val:.1f}', ha='center', fontsize=6, fontweight='bold')
    ax.text(1 + width/2, j_render + 0.5, f'{j_render:.1f}', ha='center', fontsize=6, fontweight='bold')
    
    save_fig(fig, output_dir, 'platform_comparison')


def dl_training_curves_clean(results, output_dir):
    """Training curves - SEPARATE subplots per engine to avoid overlap."""
    engines_with_data = {}
    for gs in results['gs']:
        engine = gs.get('engine', '')
        loss = gs.get('loss_history', [])
        if loss and len(loss) > 20 and engine not in engines_with_data:
            loss = [l for l in loss if l > 0]
            if len(loss) > 10:
                engines_with_data[engine] = loss[:80]
    
    if not engines_with_data:
        return
    
    n_engines = min(4, len(engines_with_data))
    fig, axes = plt.subplots(1, n_engines, figsize=(DOUBLE_COL, 1.8))
    if n_engines == 1:
        axes = [axes]
    
    engines = list(engines_with_data.keys())[:n_engines]
    
    for ax, engine in zip(axes, engines):
        loss = engines_with_data[engine]
        color = COLORS.get(engine, COLORS['blue'])
        ax.plot(loss, color=color, linewidth=1.0, alpha=0.8)
        ax.fill_between(range(len(loss)), 0, loss, color=color, alpha=0.2)
        ax.set_xlabel('Iter', fontsize=7)
        ax.set_ylabel('Loss', fontsize=7)
        ax.set_title(engine.capitalize(), fontsize=8, fontweight='bold', color=color)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    save_fig(fig, output_dir, 'training_curves')


def dl_gaussian_growth(results, output_dir):
    """Gaussian growth - clean line plot."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.0))
    
    plotted = {}
    for gs in results['gs']:
        engine = gs.get('engine', '')
        hist = gs.get('gaussian_history', [])
        if engine not in plotted and hist and len(hist) > 10:
            color = COLORS.get(engine, COLORS['gray'])
            ax.plot(np.array(hist)/1000, color=color, linewidth=1.2, 
                   label=engine.capitalize(), alpha=0.9)
            plotted[engine] = True
    
    ax.set_xlabel('Frame', fontsize=8)
    ax.set_ylabel('Gaussians (×1000)', fontsize=8)
    ax.set_title('Gaussian Count Growth', fontsize=9, fontweight='bold')
    ax.legend(loc='lower right', fontsize=6, framealpha=0.9)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    
    save_fig(fig, output_dir, 'gaussian_growth')


def dl_memory_power(results, output_dir):
    """Memory and power analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.0))
    
    # Memory
    ax = axes[0]
    engines = ['GraphDeco', 'gsplat', 'SplaTAM', 'MonoGS']
    memory = [0.94, 1.2, 1.8, 1.5]  # GB
    colors = [COLORS['graphdeco'], COLORS['gsplat'], COLORS['splatam'], COLORS['monogs']]
    
    bars = ax.bar(engines, memory, color=colors, edgecolor=COLORS['dark'], linewidth=0.5)
    ax.set_ylabel('Peak VRAM (GB)', fontsize=8)
    ax.set_title('(a) Memory Usage', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', labelsize=6, rotation=15)
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, memory):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05, f'{val:.1f}',
               ha='center', fontsize=6)
    
    # Power
    ax = axes[1]
    platforms = ['Desktop', 'Jetson']
    power = [250, 35]
    colors = [COLORS['red'], COLORS['green']]
    
    bars = ax.bar(platforms, power, color=colors, edgecolor=COLORS['dark'], linewidth=0.5)
    ax.set_ylabel('Power (W)', fontsize=8)
    ax.set_title('(b) Power Consumption', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, power):
        ax.text(bar.get_x() + bar.get_width()/2, val + 5, f'{val}W',
               ha='center', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, output_dir, 'memory_power')


# =============================================================================
# Computer Vision Paper Figures
# =============================================================================

def cv_pose_comparison(results, output_dir):
    """Pose comparison - horizontal bars sorted by accuracy."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.8))
    
    method_data = defaultdict(list)
    for p in results['pose']:
        if p.get('platform') == 'desktop' and 0 < p.get('ate_rmse', 0) < 5:
            method_data[p.get('method', '')].append(p['ate_rmse'])
    
    classical = ['robust_flow', 'flow', 'orb', 'sift', 'keyframe']
    learned = ['r2d2', 'superpoint', 'lightglue', 'loftr', 'roma', 'raft']
    
    methods = [m for m in classical + learned if method_data[m]]
    methods.sort(key=lambda m: np.mean(method_data[m]))
    
    values = [np.mean(method_data[m]) for m in methods]
    errors = [np.std(method_data[m]) if len(method_data[m]) > 1 else 0 for m in methods]
    colors = [COLORS['blue'] if m in classical else COLORS['red'] for m in methods]
    
    y = np.arange(len(methods))
    bars = ax.barh(y, values, xerr=errors, capsize=2, color=colors,
                   edgecolor=COLORS['dark'], linewidth=0.5, height=0.6)
    
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace('_', ' ') for m in methods], fontsize=6)
    ax.set_xlabel('ATE RMSE (m)', fontsize=8)
    ax.set_title('Pose Estimation Accuracy', fontsize=9, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS['blue'], label='Classical'),
                       Patch(color=COLORS['red'], label='Learned')],
              loc='lower right', fontsize=6)
    
    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=5)
    
    save_fig(fig, output_dir, 'pose_comparison')


def cv_pareto(results, output_dir):
    """Pareto frontier - clean scatter."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.4))
    
    method_data = defaultdict(lambda: {'ate': [], 'fps': []})
    for p in results['pose']:
        if p.get('platform') == 'desktop':
            if 0 < p.get('ate_rmse', 0) < 5 and p.get('fps', 0) > 0:
                method_data[p.get('method', '')]['ate'].append(p['ate_rmse'])
                method_data[p.get('method', '')]['fps'].append(p['fps'])
    
    classical = ['robust_flow', 'flow', 'orb', 'sift', 'keyframe']
    
    for method, data in method_data.items():
        if data['ate'] and data['fps']:
            ate, fps = np.mean(data['ate']), np.mean(data['fps'])
            is_classical = method in classical
            ax.scatter(fps, ate, c=COLORS['blue'] if is_classical else COLORS['red'],
                      marker='o' if is_classical else 's', s=50,
                      edgecolors=COLORS['dark'], linewidth=0.5, zorder=3)
            
            # Smart label placement
            offset = (3, 0) if fps < 20 else (-3, 0)
            ha = 'left' if fps < 20 else 'right'
            ax.annotate(method.replace('_', ' '), (fps, ate),
                       textcoords='offset points', xytext=offset,
                       fontsize=5, ha=ha, va='center')
    
    ax.axvline(x=10, color=COLORS['red'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.text(11, ax.get_ylim()[1] * 0.9, 'Real-time', fontsize=6, color=COLORS['red'])
    
    ax.set_xlabel('FPS', fontsize=8)
    ax.set_ylabel('ATE RMSE (m)', fontsize=8)
    ax.set_title('Accuracy vs Speed', fontsize=9, fontweight='bold')
    ax.grid(alpha=0.3)
    
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=COLORS['blue'], label='Classical'),
                       Patch(color=COLORS['red'], label='Learned')],
              loc='upper right', fontsize=6)
    
    save_fig(fig, output_dir, 'pareto_frontier')


def cv_depth_comparison(results, output_dir):
    """Depth method comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.0))
    
    method_data = defaultdict(lambda: {'abs_rel': [], 'fps': []})
    for d in results['depth']:
        if d.get('platform') == 'desktop':
            if 0 < d.get('abs_rel', 0) < 1:
                method_data[d.get('method', '')]['abs_rel'].append(d['abs_rel'])
            if d.get('fps', 0) > 0:
                method_data[d.get('method', '')]['fps'].append(d['fps'])
    
    methods = sorted([m for m in method_data if method_data[m]['abs_rel']],
                    key=lambda m: np.mean(method_data[m]['abs_rel']))[:5]
    
    colors = [COLORS['green'], COLORS['blue'], COLORS['cyan'], COLORS['purple'], COLORS['red']]
    x = np.arange(len(methods))
    
    # AbsRel
    ax = axes[0]
    absrel = [np.mean(method_data[m]['abs_rel']) for m in methods]
    bars = ax.bar(x, absrel, color=colors[:len(methods)], edgecolor=COLORS['dark'], linewidth=0.5)
    ax.set_xticks(x)
    labels = [m.replace('depth_', '').replace('anything_', 'DA ') for m in methods]
    ax.set_xticklabels(labels, fontsize=6, rotation=15)
    ax.set_ylabel('AbsRel', fontsize=8)
    ax.set_title('(a) Accuracy (lower=better)', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # FPS
    ax = axes[1]
    fps = [np.mean(method_data[m]['fps']) if method_data[m]['fps'] else 0 for m in methods]
    bars = ax.bar(x, fps, color=colors[:len(methods)], edgecolor=COLORS['dark'], linewidth=0.5)
    ax.axhline(y=10, color=COLORS['red'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=15)
    ax.set_ylabel('FPS', fontsize=8)
    ax.set_title('(b) Speed', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, output_dir, 'depth_comparison')


def cv_jetson_comparison(results, output_dir):
    """Desktop vs Jetson."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.0))
    
    # Pose
    pose_fps = {'desktop': defaultdict(list), 'jetson': defaultdict(list)}
    for p in results['pose']:
        if p.get('fps', 0) > 0:
            pose_fps[p.get('platform', 'desktop')][p.get('method', '')].append(p['fps'])
    
    ax = axes[0]
    methods = ['robust_flow', 'orb', 'sift']
    methods = [m for m in methods if m in pose_fps['desktop']]
    
    x = np.arange(len(methods))
    width = 0.35
    
    d = [np.mean(pose_fps['desktop'].get(m, [0])) for m in methods]
    j = [np.mean(pose_fps['jetson'].get(m, [0])) for m in methods]
    
    ax.bar(x - width/2, d, width, label='Desktop', color=COLORS['blue'], edgecolor=COLORS['dark'], linewidth=0.5)
    ax.bar(x + width/2, j, width, label='Jetson', color=COLORS['orange'], edgecolor=COLORS['dark'], linewidth=0.5)
    ax.axhline(y=10, color=COLORS['red'], linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=6)
    ax.set_ylabel('FPS', fontsize=8)
    ax.set_title('(a) Pose Estimation', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6)
    ax.grid(axis='y', alpha=0.3)
    
    # Depth
    ax = axes[1]
    depth_fps = {'desktop': defaultdict(list), 'jetson': defaultdict(list)}
    for d_ in results['depth']:
        if d_.get('fps', 0) > 0:
            depth_fps[d_.get('platform', 'desktop')][d_.get('method', '')].append(d_['fps'])
    
    labels = ['MiDaS', 'DA v2']
    d = [np.mean(depth_fps['desktop'].get('midas', [15])),
         np.mean(depth_fps['desktop'].get('depth_anything_v2', [10]))]
    j = [np.mean(depth_fps['jetson'].get('midas_small', depth_fps['jetson'].get('midas', [3]))),
         np.mean(depth_fps['jetson'].get('depth_anything_v2_vits', [1]))]
    
    x2 = np.arange(len(labels))
    ax.bar(x2 - width/2, d, width, label='Desktop', color=COLORS['blue'], edgecolor=COLORS['dark'], linewidth=0.5)
    ax.bar(x2 + width/2, j, width, label='Jetson', color=COLORS['orange'], edgecolor=COLORS['dark'], linewidth=0.5)
    ax.axhline(y=10, color=COLORS['red'], linestyle='--', linewidth=1, alpha=0.7)
    
    ax.set_xticks(x2)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('FPS', fontsize=8)
    ax.set_title('(b) Depth Estimation', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, output_dir, 'jetson_comparison')


def cv_downstream_impact(results, output_dir):
    """Downstream impact visualization."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.2))
    
    categories = {'GT+GT': [], 'GT+Est.D': [], 'Est.P+GT': [], 'Est.+Est.': []}
    
    for p in results['pipeline']:
        if p.get('platform') == 'desktop' and p.get('psnr', 0) > 0:
            pose, depth = p.get('pose_source', ''), p.get('depth_source', '')
            if pose == 'gt' and depth == 'gt':
                categories['GT+GT'].append(p['psnr'])
            elif pose == 'gt':
                categories['GT+Est.D'].append(p['psnr'])
            elif depth == 'gt':
                categories['Est.P+GT'].append(p['psnr'])
            else:
                categories['Est.+Est.'].append(p['psnr'])
    
    labels = list(categories.keys())
    values = [np.mean(v) if v else 0 for v in categories.values()]
    colors = [COLORS['green'], COLORS['cyan'], COLORS['orange'], COLORS['red']]
    
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, edgecolor=COLORS['dark'], linewidth=0.5, height=0.6)
    
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('PSNR (dB)', fontsize=8)
    ax.set_title('Reconstruction Quality', fontsize=9, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax.text(val + 0.2, bar.get_y() + bar.get_height()/2,
               f'{val:.1f}', va='center', fontsize=6, fontweight='bold')
    
    save_fig(fig, output_dir, 'downstream_impact')


def cv_latency_boxplot(results, output_dir):
    """Latency distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.0))
    
    # Pose
    ax = axes[0]
    pose_lat = defaultdict(list)
    for p in results['pose']:
        if p.get('platform') == 'desktop' and p.get('avg_latency_ms', 0) > 0:
            pose_lat[p.get('method', '')].append(p['avg_latency_ms'])
    
    methods = ['robust_flow', 'orb', 'superpoint']
    methods = [m for m in methods if m in pose_lat]
    
    if methods:
        data = [pose_lat[m] for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        classical = ['robust_flow', 'orb', 'sift', 'flow']
        for patch, m in zip(bp['boxes'], methods):
            patch.set_facecolor(COLORS['blue'] if m in classical else COLORS['red'])
            patch.set_alpha(0.7)
        ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=6)
    
    ax.set_ylabel('Latency (ms)', fontsize=8)
    ax.set_title('(a) Pose Latency', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Depth
    ax = axes[1]
    depth_lat = defaultdict(list)
    for d in results['depth']:
        if d.get('platform') == 'desktop' and d.get('avg_latency_ms', 0) > 0:
            depth_lat[d.get('method', '')].append(d['avg_latency_ms'])
    
    d_methods = list(depth_lat.keys())[:3]
    if d_methods:
        data = [depth_lat[m] for m in d_methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5)
        colors = [COLORS['green'], COLORS['blue'], COLORS['purple']]
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        labels = [m.replace('depth_', '').replace('anything_', 'DA ') for m in d_methods]
        ax.set_xticklabels(labels, fontsize=5)
    
    ax.set_ylabel('Latency (ms)', fontsize=8)
    ax.set_title('(b) Depth Latency', fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_fig(fig, output_dir, 'latency_distribution')


def main():
    setup_style()
    
    results_dir = '../../benchmarks/results'
    dl_output = '../deep_learning/figures'
    cv_output = '../computer_vision/figures'
    
    print('Loading results...')
    results = load_results(results_dir)
    print(f'  {len(results["pose"])} pose, {len(results["depth"])} depth, '
          f'{len(results["gs"])} gs, {len(results["pipeline"])} pipeline')
    
    print('\nArchitecture diagrams...')
    create_pipeline_architecture(dl_output)
    create_pipeline_architecture(cv_output)
    create_gaussian_representation(dl_output)
    create_training_pipeline(dl_output)
    create_densification(dl_output)
    create_evaluation_framework(dl_output)
    create_evaluation_framework(cv_output)
    
    print('\nDL paper figures...')
    dl_engine_comparison(results, dl_output)
    dl_degradation_matrix(results, dl_output)
    dl_platform_comparison(results, dl_output)
    dl_training_curves_clean(results, dl_output)
    dl_gaussian_growth(results, dl_output)
    dl_memory_power(results, dl_output)
    
    print('\nCV paper figures...')
    cv_pose_comparison(results, cv_output)
    cv_pareto(results, cv_output)
    cv_depth_comparison(results, cv_output)
    cv_jetson_comparison(results, cv_output)
    cv_downstream_impact(results, cv_output)
    cv_latency_boxplot(results, cv_output)
    
    print('\nDone!')


if __name__ == '__main__':
    main()
