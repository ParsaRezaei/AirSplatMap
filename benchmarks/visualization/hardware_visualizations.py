"""
Hardware Usage Visualizations
=============================

Generate plots for hardware resource usage during benchmarks.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

logger = logging.getLogger(__name__)


def setup_style():
    """Setup plot style."""
    plt.style.use('default')
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'font.size': 10,
    })


def plot_hardware_summary(hardware_data: Dict[str, Any], output_path: Path) -> bool:
    """
    Create a summary plot of hardware usage.
    
    Args:
        hardware_data: Hardware stats from HardwareMonitor.get_summary()
        output_path: Path to save the plot
        
    Returns:
        True if plot was generated successfully
    """
    if not hardware_data or 'overall' not in hardware_data:
        logger.warning("No hardware data to plot")
        return False
    
    setup_style()
    
    overall = hardware_data['overall']
    phases = hardware_data.get('phases', {})
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Hardware Resource Usage During Benchmark', fontsize=14, fontweight='bold')
    
    # 1. GPU Utilization by Phase
    ax1 = axes[0, 0]
    if phases:
        phase_names = list(phases.keys())
        gpu_utils = [phases[p]['gpu']['utilization_mean'] for p in phase_names]
        gpu_utils_max = [phases[p]['gpu']['utilization_max'] for p in phase_names]
        
        x = np.arange(len(phase_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, gpu_utils, width, label='Mean', color='#3498db', alpha=0.8)
        bars2 = ax1.bar(x + width/2, gpu_utils_max, width, label='Max', color='#e74c3c', alpha=0.8)
        
        ax1.set_ylabel('GPU Utilization (%)')
        ax1.set_title('GPU Compute Utilization by Phase')
        ax1.set_xticks(x)
        ax1.set_xticklabels([p.upper() for p in phase_names])
        ax1.legend()
        ax1.set_ylim(0, 105)
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.0f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No phase data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('GPU Compute Utilization by Phase')
    
    # 2. GPU Memory by Phase
    ax2 = axes[0, 1]
    if phases:
        mem_used = [phases[p]['gpu']['memory_used_gb_mean'] for p in phase_names]
        mem_max = [phases[p]['gpu']['memory_used_gb_max'] for p in phase_names]
        
        x = np.arange(len(phase_names))
        
        bars1 = ax2.bar(x - width/2, mem_used, width, label='Mean', color='#2ecc71', alpha=0.8)
        bars2 = ax2.bar(x + width/2, mem_max, width, label='Max', color='#f39c12', alpha=0.8)
        
        ax2.set_ylabel('GPU Memory (GB)')
        ax2.set_title('GPU Memory Usage by Phase')
        ax2.set_xticks(x)
        ax2.set_xticklabels([p.upper() for p in phase_names])
        ax2.legend()
        
        # Add total GPU memory line
        total_mem = overall['gpu'].get('memory_used_gb_max', 0) * 1.1  # Approximate total
        if total_mem > 0:
            ax2.axhline(y=total_mem, color='red', linestyle='--', alpha=0.5, label='Approx. Total')
        
        for bar in bars1:
            height = bar.get_height()
            ax2.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No phase data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('GPU Memory Usage by Phase')
    
    # 3. CPU & RAM Summary
    ax3 = axes[1, 0]
    
    categories = ['CPU Usage (%)', 'RAM Usage (GB)', 'Process RAM (GB)']
    mean_values = [
        overall['cpu']['percent_mean'],
        overall['ram']['used_gb_mean'],
        overall['process']['ram_gb_mean']
    ]
    max_values = [
        overall['cpu']['percent_max'],
        overall['ram']['used_gb_max'],
        overall['process']['ram_gb_max']
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, mean_values, width, label='Mean', color='#9b59b6', alpha=0.8)
    bars2 = ax3.bar(x + width/2, max_values, width, label='Max', color='#e91e63', alpha=0.8)
    
    ax3.set_ylabel('Value')
    ax3.set_title('CPU & Memory Usage')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, rotation=15)
    ax3.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    # 4. GPU Power & Temperature
    ax4 = axes[1, 1]
    
    # Create dual y-axis
    ax4_temp = ax4.twinx()
    
    if phases:
        temps = [phases[p]['gpu']['temperature_c_mean'] for p in phase_names]
        temps_max = [phases[p]['gpu']['temperature_c_max'] for p in phase_names]
        powers = [phases[p]['gpu']['power_w_mean'] for p in phase_names]
        powers_max = [phases[p]['gpu']['power_w_max'] for p in phase_names]
        
        x = np.arange(len(phase_names))
        width = 0.2
        
        # Power bars
        bars_power = ax4.bar(x - width, powers, width, label='Power Mean (W)', color='#e74c3c', alpha=0.7)
        ax4.bar(x, powers_max, width, label='Power Max (W)', color='#c0392b', alpha=0.7)
        
        # Temperature line
        line_temp = ax4_temp.plot(x, temps, 'o-', color='#3498db', label='Temp Mean (°C)', linewidth=2)
        ax4_temp.plot(x, temps_max, 's--', color='#2980b9', label='Temp Max (°C)', linewidth=1.5)
        
        ax4.set_ylabel('Power (W)', color='#e74c3c')
        ax4_temp.set_ylabel('Temperature (°C)', color='#3498db')
        ax4.set_title('GPU Power & Temperature by Phase')
        ax4.set_xticks(x)
        ax4.set_xticklabels([p.upper() for p in phase_names])
        
        # Combined legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_temp.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    else:
        ax4.text(0.5, 0.5, 'No phase data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('GPU Power & Temperature by Phase')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    ✓ {output_path.name}")
    return True


def plot_hardware_timeline(snapshots: List[Dict], output_path: Path) -> bool:
    """
    Plot hardware metrics over time (if snapshot data available).
    
    Args:
        snapshots: List of HardwareSnapshot dictionaries
        output_path: Path to save the plot
    """
    # This would require storing individual snapshots
    # For now, we only have aggregated stats
    logger.debug("Timeline plot requires raw snapshot data")
    return False


def plot_energy_comparison(hardware_data: Dict[str, Any], output_path: Path) -> bool:
    """
    Create energy consumption comparison chart.
    """
    if not hardware_data or 'phases' not in hardware_data:
        return False
    
    setup_style()
    phases = hardware_data['phases']
    
    if not phases:
        return False
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    phase_names = list(phases.keys())
    energies = [phases[p].get('energy_wh', 0) * 1000 for p in phase_names]  # Convert to mWh
    durations = [phases[p].get('duration_seconds', 0) for p in phase_names]
    
    # Bar chart
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(phase_names)))
    bars = ax.bar(phase_names, energies, color=colors, alpha=0.8)
    
    ax.set_ylabel('Energy Consumption (mWh)')
    ax.set_xlabel('Benchmark Phase')
    ax.set_title('Energy Consumption by Benchmark Phase')
    
    # Add duration annotations
    for i, (bar, dur) in enumerate(zip(bars, durations)):
        height = bar.get_height()
        ax.annotate(f'{height:.1f} mWh\n({dur:.0f}s)',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 5), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    # Add total
    total_energy = sum(energies)
    ax.axhline(y=total_energy/len(phases), color='red', linestyle='--', alpha=0.5)
    ax.text(len(phases)-0.5, total_energy/len(phases), f'Avg: {total_energy/len(phases):.1f} mWh',
           color='red', fontsize=9, va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"    ✓ {output_path.name}")
    return True


def generate_all_hardware_plots(hardware_data: Dict[str, Any], output_dir: Path) -> int:
    """
    Generate all hardware visualization plots.
    
    Args:
        hardware_data: Hardware stats from HardwareMonitor.get_summary()
        output_dir: Directory to save plots
        
    Returns:
        Number of plots generated
    """
    if not hardware_data:
        logger.info("No hardware data available for visualization")
        return 0
    
    hw_dir = output_dir / "hardware"
    hw_dir.mkdir(exist_ok=True)
    
    plots_generated = 0
    
    # Main summary plot
    if plot_hardware_summary(hardware_data, hw_dir / "usage_summary.png"):
        plots_generated += 1
    
    # Energy comparison
    if plot_energy_comparison(hardware_data, hw_dir / "energy_comparison.png"):
        plots_generated += 1
    
    return plots_generated
