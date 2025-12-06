"""
Cross-Metric Visualizations (Paper-Level Quality)
==================================================

Combined visualizations showing relationships between:
- Pose errors and depth errors
- Pose/depth errors and 3DGS quality
- Comprehensive panel figures for publications
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def setup_style():
    """Setup publication-quality plot style."""
    if not HAS_MATPLOTLIB:
        return
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
    })


# =============================================================================
# Combined Accuracy Panel (Paper-Grade)
# =============================================================================

def plot_combined_accuracy_panel(
    pose_results: List[Dict],
    depth_results: List[Dict],
    gs_results: List[Dict],
    pipeline_results: List[Dict],
    output_path: Path,
    dataset_name: str = "Dataset",
):
    """
    Create comprehensive panel figure combining all benchmark results.
    This is the main visualization for academic papers.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # Row 0: Pose estimation metrics
    # ATE comparison
    ax_ate = fig.add_subplot(gs[0, 0])
    if pose_results:
        methods = [r['method'] for r in pose_results]
        ates = [r.get('ate_rmse', 0) for r in pose_results]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(methods)))
        sorted_idx = np.argsort(ates)
        bars = ax_ate.barh([methods[i] for i in sorted_idx], [ates[i] for i in sorted_idx],
                          color=[colors[i] for i in sorted_idx], edgecolor='black')
        ax_ate.set_xlabel('ATE RMSE (m)')
        ax_ate.set_title('Pose: Absolute Trajectory Error', fontweight='bold')
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(2)
    
    # RPE comparison
    ax_rpe = fig.add_subplot(gs[0, 1])
    if pose_results:
        rpes = [r.get('rpe_trans_rmse', 0) for r in pose_results]
        colors = plt.cm.Greens(np.linspace(0.4, 0.9, len(methods)))
        sorted_idx = np.argsort(rpes)
        bars = ax_rpe.barh([methods[i] for i in sorted_idx], [rpes[i] for i in sorted_idx],
                          color=[colors[i] for i in sorted_idx], edgecolor='black')
        ax_rpe.set_xlabel('RPE Trans (m)')
        ax_rpe.set_title('Pose: Relative Pose Error', fontweight='bold')
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(2)
    
    # Row 0: Depth estimation metrics
    # AbsRel comparison
    ax_absrel = fig.add_subplot(gs[0, 2])
    if depth_results:
        methods_d = [r['method'] for r in depth_results]
        absrels = [r.get('abs_rel', 0) for r in depth_results]
        colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(methods_d)))
        sorted_idx = np.argsort(absrels)
        bars = ax_absrel.barh([methods_d[i] for i in sorted_idx], [absrels[i] for i in sorted_idx],
                             color=[colors[i] for i in sorted_idx], edgecolor='black')
        ax_absrel.set_xlabel('Absolute Relative Error')
        ax_absrel.set_title('Depth: AbsRel Error', fontweight='bold')
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(2)
    
    # Delta accuracy
    ax_delta = fig.add_subplot(gs[0, 3])
    if depth_results:
        deltas = [r.get('delta1', 0) for r in depth_results]
        colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(methods_d)))
        sorted_idx = np.argsort(deltas)[::-1]
        bars = ax_delta.barh([methods_d[i] for i in sorted_idx], [deltas[i] for i in sorted_idx],
                            color=[colors[i] for i in sorted_idx], edgecolor='black')
        ax_delta.set_xlabel('δ < 1.25 Accuracy')
        ax_delta.set_title('Depth: Accuracy Threshold', fontweight='bold')
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(2)
    
    # Row 1: GS metrics
    # PSNR
    ax_psnr = fig.add_subplot(gs[1, 0])
    if gs_results:
        engines = [r['engine'] for r in gs_results]
        psnrs = [r.get('psnr', 0) for r in gs_results]
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(engines)))
        sorted_idx = np.argsort(psnrs)[::-1]
        bars = ax_psnr.barh([engines[i] for i in sorted_idx], [psnrs[i] for i in sorted_idx],
                           color=[colors[i] for i in sorted_idx], edgecolor='black')
        ax_psnr.set_xlabel('PSNR (dB)')
        ax_psnr.set_title('3DGS: Peak SNR', fontweight='bold')
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(2)
    
    # SSIM
    ax_ssim = fig.add_subplot(gs[1, 1])
    if gs_results:
        ssims = [r.get('ssim', 0) for r in gs_results]
        sorted_idx = np.argsort(ssims)[::-1]
        bars = ax_ssim.barh([engines[i] for i in sorted_idx], [ssims[i] for i in sorted_idx],
                           color=[colors[i] for i in sorted_idx], edgecolor='black')
        ax_ssim.set_xlabel('SSIM')
        ax_ssim.set_title('3DGS: Structural Similarity', fontweight='bold')
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(2)
    
    # Gaussians count
    ax_gauss = fig.add_subplot(gs[1, 2])
    if gs_results:
        gaussians = [r.get('final_gaussians', 0) for r in gs_results]
        bars = ax_gauss.barh(engines, gaussians, color=plt.cm.Set2(np.linspace(0, 1, len(engines))), edgecolor='black')
        ax_gauss.set_xlabel('Number of Gaussians')
        ax_gauss.set_title('3DGS: Model Complexity', fontweight='bold')
    
    # FPS
    ax_fps = fig.add_subplot(gs[1, 3])
    if gs_results:
        fps = [r.get('fps', 0) for r in gs_results]
        bars = ax_fps.barh(engines, fps, color=plt.cm.Set3(np.linspace(0, 1, len(engines))), edgecolor='black')
        ax_fps.set_xlabel('Frames per Second')
        ax_fps.set_title('3DGS: Processing Speed', fontweight='bold')
    
    # Row 2: Pipeline comparisons
    ax_pipeline = fig.add_subplot(gs[2, :2])
    if pipeline_results:
        configs = [r['name'] for r in pipeline_results]
        psnrs = [r.get('psnr', 0) for r in pipeline_results]
        
        def get_config_color(name):
            if 'GT_pose_GT_depth' in name:
                return '#27ae60'
            elif 'GT_pose' in name:
                return '#3498db'
            elif 'GT_depth' in name:
                return '#e74c3c'
            else:
                return '#9b59b6'
        
        colors = [get_config_color(c) for c in configs]
        sorted_idx = np.argsort(psnrs)[::-1]
        
        y_pos = np.arange(len(configs))
        bars = ax_pipeline.barh(y_pos, [psnrs[i] for i in sorted_idx],
                               color=[colors[i] for i in sorted_idx], edgecolor='black')
        ax_pipeline.set_yticks(y_pos)
        ax_pipeline.set_yticklabels([configs[i] for i in sorted_idx], fontsize=9)
        ax_pipeline.set_xlabel('PSNR (dB)')
        ax_pipeline.set_title('Pipeline Comparison: Impact of Pose/Depth Source', fontweight='bold')
        ax_pipeline.invert_yaxis()
        
        for bar, val in zip(bars, [psnrs[i] for i in sorted_idx]):
            ax_pipeline.text(val + 0.1, bar.get_y() + bar.get_height()/2, f'{val:.2f}', va='center', fontsize=9)
    
    # Pipeline legend
    ax_legend = fig.add_subplot(gs[2, 2])
    ax_legend.axis('off')
    legend_elements = [
        Patch(facecolor='#27ae60', edgecolor='black', label='GT Pose + GT Depth (Upper Bound)'),
        Patch(facecolor='#3498db', edgecolor='black', label='GT Pose + Estimated Depth'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Estimated Pose + GT Depth'),
        Patch(facecolor='#9b59b6', edgecolor='black', label='Estimated Pose + Estimated Depth'),
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=11, frameon=True)
    ax_legend.set_title('Pipeline Configurations', fontweight='bold')
    
    # Latency comparison
    ax_latency = fig.add_subplot(gs[2, 3])
    latency_data = []
    latency_labels = []
    latency_colors = []
    
    if pose_results:
        for r in pose_results[:3]:
            latency_data.append(r.get('avg_latency_ms', 0))
            latency_labels.append(f"Pose:{r['method'][:6]}")
            latency_colors.append('#3498db')
    if depth_results:
        for r in depth_results[:3]:
            latency_data.append(1000 / r.get('fps', 1))
            latency_labels.append(f"Depth:{r['method'][:6]}")
            latency_colors.append('#e74c3c')
    
    if latency_data:
        bars = ax_latency.barh(latency_labels, latency_data, color=latency_colors, edgecolor='black')
        ax_latency.set_xlabel('Latency (ms)')
        ax_latency.set_title('Processing Latency', fontweight='bold')
    
    # Row 3: Summary statistics table
    ax_table = fig.add_subplot(gs[3, :])
    ax_table.axis('off')
    
    # Create summary text
    summary_text = f"📊 BENCHMARK SUMMARY - {dataset_name}\n"
    summary_text += "=" * 80 + "\n\n"
    
    if pose_results:
        best_pose = min(pose_results, key=lambda x: x.get('ate_rmse', float('inf')))
        summary_text += f"🏆 Best Pose Estimator: {best_pose['method']} (ATE: {best_pose.get('ate_rmse', 0):.4f}m)\n"
    
    if depth_results:
        best_depth = min(depth_results, key=lambda x: x.get('abs_rel', float('inf')))
        summary_text += f"🏆 Best Depth Estimator: {best_depth['method']} (AbsRel: {best_depth.get('abs_rel', 0):.4f})\n"
    
    if gs_results:
        best_gs = max(gs_results, key=lambda x: x.get('psnr', 0))
        summary_text += f"🏆 Best GS Engine: {best_gs['engine']} (PSNR: {best_gs.get('psnr', 0):.2f}dB)\n"
    
    if pipeline_results:
        best_pipeline = max(pipeline_results, key=lambda x: x.get('psnr', 0))
        summary_text += f"🏆 Best Pipeline: {best_pipeline['name']} (PSNR: {best_pipeline.get('psnr', 0):.2f}dB)\n"
    
    ax_table.text(0.05, 0.5, summary_text, transform=ax_table.transAxes,
                  fontsize=12, family='monospace', verticalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle(f'AirSplatMap Comprehensive Benchmark Results\n{dataset_name}', 
                 fontsize=20, fontweight='bold')
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Violin Plots for All Metrics
# =============================================================================

def plot_all_metrics_violin(
    pose_results: List[Dict],
    depth_results: List[Dict],
    gs_results: List[Dict],
    output_path: Path,
):
    """
    Combined violin/bar plots for all metrics.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Pose metrics
    if pose_results:
        methods = [r['method'] for r in pose_results]
        
        # ATE
        ates = [r.get('ate_rmse', 0) for r in pose_results]
        axes[0, 0].bar(methods, ates, color=plt.cm.Blues(0.6), edgecolor='black')
        axes[0, 0].set_ylabel('ATE RMSE (m)')
        axes[0, 0].set_title('Pose: ATE', fontweight='bold')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # RPE Trans
        rpes = [r.get('rpe_trans_rmse', 0) for r in pose_results]
        axes[0, 1].bar(methods, rpes, color=plt.cm.Greens(0.6), edgecolor='black')
        axes[0, 1].set_ylabel('RPE Trans (m)')
        axes[0, 1].set_title('Pose: RPE Translation', fontweight='bold')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Depth metrics
    if depth_results:
        methods_d = [r['method'] for r in depth_results]
        
        # AbsRel
        absrels = [r.get('abs_rel', 0) for r in depth_results]
        axes[0, 2].bar(methods_d, absrels, color=plt.cm.Oranges(0.6), edgecolor='black')
        axes[0, 2].set_ylabel('AbsRel')
        axes[0, 2].set_title('Depth: Absolute Relative', fontweight='bold')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # RMSE
        rmses = [r.get('rmse', 0) for r in depth_results]
        axes[0, 3].bar(methods_d, rmses, color=plt.cm.Purples(0.6), edgecolor='black')
        axes[0, 3].set_ylabel('RMSE')
        axes[0, 3].set_title('Depth: RMSE', fontweight='bold')
        axes[0, 3].tick_params(axis='x', rotation=45)
    
    # GS metrics
    if gs_results:
        engines = [r['engine'] for r in gs_results]
        
        # PSNR
        psnrs = [r.get('psnr', 0) for r in gs_results]
        axes[1, 0].bar(engines, psnrs, color=plt.cm.Reds(0.6), edgecolor='black')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].set_title('3DGS: PSNR', fontweight='bold')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # SSIM
        ssims = [r.get('ssim', 0) for r in gs_results]
        axes[1, 1].bar(engines, ssims, color=plt.cm.YlOrRd(0.6), edgecolor='black')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].set_title('3DGS: SSIM', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Gaussians
        gaussians = [r.get('final_gaussians', 0) for r in gs_results]
        axes[1, 2].bar(engines, gaussians, color=plt.cm.Set2(0.6), edgecolor='black')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('3DGS: Gaussian Count', fontweight='bold')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        # FPS
        fps = [r.get('fps', 0) for r in gs_results]
        axes[1, 3].bar(engines, fps, color=plt.cm.Set3(0.6), edgecolor='black')
        axes[1, 3].set_ylabel('FPS')
        axes[1, 3].set_title('3DGS: Processing Speed', fontweight='bold')
        axes[1, 3].tick_params(axis='x', rotation=45)
    
    plt.suptitle('All Benchmark Metrics Overview', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Radar/Spider Chart
# =============================================================================

def plot_method_radar_chart(
    results: List[Dict],
    result_type: str,  # 'pose', 'depth', or 'gs'
    output_path: Path,
):
    """
    Radar chart comparing methods across multiple metrics.
    """
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    if result_type == 'pose':
        metrics = ['ate_rmse', 'rpe_trans_rmse', 'fps']
        metric_names = ['ATE (inv)', 'RPE (inv)', 'FPS']
        invert = [True, True, False]
    elif result_type == 'depth':
        metrics = ['abs_rel', 'rmse', 'delta1', 'fps']
        metric_names = ['AbsRel (inv)', 'RMSE (inv)', 'δ<1.25', 'FPS']
        invert = [True, True, False, False]
    else:  # gs
        metrics = ['psnr', 'ssim', 'fps']
        metric_names = ['PSNR', 'SSIM', 'FPS']
        invert = [False, False, False]
    
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(results)))
    
    for i, r in enumerate(results):
        name = r.get('method', r.get('engine', 'Unknown'))
        
        # Normalize values
        values = []
        for j, metric in enumerate(metrics):
            val = r.get(metric, 0)
            # Normalize to 0-1 range
            all_vals = [res.get(metric, 0) for res in results]
            if max(all_vals) > min(all_vals):
                norm_val = (val - min(all_vals)) / (max(all_vals) - min(all_vals))
            else:
                norm_val = 0.5
            if invert[j]:
                norm_val = 1 - norm_val
            values.append(norm_val)
        
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.title(f'{result_type.capitalize()} Method Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =============================================================================
# Main Generator
# =============================================================================

def generate_all_cross_metric_plots(
    pose_results: List[Dict],
    depth_results: List[Dict],
    gs_results: List[Dict],
    pipeline_results: List[Dict],
    output_dir: Path,
    dataset_name: str = "Dataset",
):
    """
    Generate all cross-metric visualizations.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available, skipping cross-metric plots")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating cross-metric plots in {output_dir}")
    
    # Combined accuracy panel (main figure)
    try:
        plot_combined_accuracy_panel(
            pose_results, depth_results, gs_results, pipeline_results,
            output_dir / 'combined_accuracy_panel.png',
            dataset_name=dataset_name,
        )
        logger.info("  ✓ combined_accuracy_panel.png")
    except Exception as e:
        logger.warning(f"  ✗ combined_accuracy_panel failed: {e}")
    
    # All metrics overview
    try:
        plot_all_metrics_violin(
            pose_results, depth_results, gs_results,
            output_dir / 'all_metrics_overview.png',
        )
        logger.info("  ✓ all_metrics_overview.png")
    except Exception as e:
        logger.warning(f"  ✗ all_metrics_overview failed: {e}")
    
    # Radar charts
    if pose_results:
        try:
            plot_method_radar_chart(pose_results, 'pose', output_dir / 'pose_radar.png')
            logger.info("  ✓ pose_radar.png")
        except Exception as e:
            logger.warning(f"  ✗ pose_radar failed: {e}")
    
    if depth_results:
        try:
            plot_method_radar_chart(depth_results, 'depth', output_dir / 'depth_radar.png')
            logger.info("  ✓ depth_radar.png")
        except Exception as e:
            logger.warning(f"  ✗ depth_radar failed: {e}")
    
    if gs_results:
        try:
            plot_method_radar_chart(gs_results, 'gs', output_dir / 'gs_radar.png')
            logger.info("  ✓ gs_radar.png")
        except Exception as e:
            logger.warning(f"  ✗ gs_radar failed: {e}")
