"""
Hardware Monitoring for Benchmarks
==================================

Tracks GPU, CPU, and memory usage during benchmark runs.
Provides per-method and aggregate statistics.
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict

logger = logging.getLogger(__name__)

# Try to import monitoring libraries
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    logger.debug("psutil not available - CPU/RAM monitoring disabled")

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception:
    _NVML_AVAILABLE = False
    logger.debug("pynvml not available - GPU monitoring disabled")

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class HardwareSnapshot:
    """Single point-in-time hardware measurement."""
    timestamp: float
    
    # CPU - system-wide
    cpu_percent: float = 0.0  # Overall CPU usage %
    cpu_percent_per_core: List[float] = field(default_factory=list)
    
    # Memory - system-wide
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    ram_percent: float = 0.0
    
    # Process-specific CPU/RAM
    process_cpu_percent: float = 0.0
    process_ram_gb: float = 0.0
    process_ram_percent: float = 0.0
    
    # GPU - system-wide
    gpu_utilization: float = 0.0  # GPU compute %
    gpu_memory_used_gb: float = 0.0
    gpu_memory_total_gb: float = 0.0
    gpu_memory_percent: float = 0.0
    gpu_temperature_c: float = 0.0
    gpu_power_w: float = 0.0
    gpu_power_limit_w: float = 0.0
    
    # Process-specific GPU memory
    process_gpu_memory_gb: float = 0.0


@dataclass
class HardwareStats:
    """Aggregated hardware statistics for a benchmark run."""
    # Timing
    duration_seconds: float = 0.0
    num_samples: int = 0
    
    # CPU stats - system-wide
    cpu_percent_mean: float = 0.0
    cpu_percent_max: float = 0.0
    cpu_percent_min: float = 0.0
    
    # RAM stats - system-wide
    ram_used_gb_mean: float = 0.0
    ram_used_gb_max: float = 0.0
    ram_percent_mean: float = 0.0
    
    # Process stats - CPU/RAM
    process_cpu_percent_mean: float = 0.0
    process_cpu_percent_max: float = 0.0
    process_ram_gb_mean: float = 0.0
    process_ram_gb_max: float = 0.0
    
    # GPU stats - system-wide
    gpu_utilization_mean: float = 0.0
    gpu_utilization_max: float = 0.0
    gpu_memory_used_gb_mean: float = 0.0
    gpu_memory_used_gb_max: float = 0.0
    gpu_memory_percent_mean: float = 0.0
    gpu_memory_percent_max: float = 0.0
    gpu_temperature_c_mean: float = 0.0
    gpu_temperature_c_max: float = 0.0
    gpu_power_w_mean: float = 0.0
    gpu_power_w_max: float = 0.0
    
    # Process-specific GPU memory
    process_gpu_memory_gb_mean: float = 0.0
    process_gpu_memory_gb_max: float = 0.0
    
    # Energy estimate
    total_energy_wh: float = 0.0  # Watt-hours consumed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'duration_seconds': round(self.duration_seconds, 2),
            'num_samples': self.num_samples,
            'cpu': {
                'system_percent_mean': round(self.cpu_percent_mean, 1),
                'system_percent_max': round(self.cpu_percent_max, 1),
            },
            'ram': {
                'system_used_gb_mean': round(self.ram_used_gb_mean, 2),
                'system_used_gb_max': round(self.ram_used_gb_max, 2),
                'system_percent_mean': round(self.ram_percent_mean, 1),
            },
            'process': {
                'cpu_percent_mean': round(self.process_cpu_percent_mean, 1),
                'cpu_percent_max': round(self.process_cpu_percent_max, 1),
                'ram_gb_mean': round(self.process_ram_gb_mean, 2),
                'ram_gb_max': round(self.process_ram_gb_max, 2),
                'gpu_memory_gb_mean': round(self.process_gpu_memory_gb_mean, 2),
                'gpu_memory_gb_max': round(self.process_gpu_memory_gb_max, 2),
            },
            'gpu': {
                'utilization_mean': round(self.gpu_utilization_mean, 1),
                'utilization_max': round(self.gpu_utilization_max, 1),
                'memory_used_gb_mean': round(self.gpu_memory_used_gb_mean, 2),
                'memory_used_gb_max': round(self.gpu_memory_used_gb_max, 2),
                'memory_percent_mean': round(self.gpu_memory_percent_mean, 1),
                'memory_percent_max': round(self.gpu_memory_percent_max, 1),
                'temperature_c_mean': round(self.gpu_temperature_c_mean, 1),
                'temperature_c_max': round(self.gpu_temperature_c_max, 1),
                'power_w_mean': round(self.gpu_power_w_mean, 1),
                'power_w_max': round(self.gpu_power_w_max, 1),
            },
            'energy_wh': round(self.total_energy_wh, 4),
        }


class HardwareMonitor:
    """
    Monitor hardware usage during benchmark execution.
    
    Supports both system-wide and process-specific monitoring.
    
    Usage:
        # Monitor current process
        monitor = HardwareMonitor(sample_interval=0.5)
        
        # Monitor a specific PID
        monitor = HardwareMonitor(sample_interval=0.5, pid=12345)
        
        # Start monitoring
        monitor.start()
        
        # Run benchmark...
        monitor.mark("pose_orb")  # Mark start of a specific method
        run_pose_benchmark("orb")
        
        monitor.mark("depth_midas")
        run_depth_benchmark("midas")
        
        # Stop and get results
        monitor.stop()
        
        # Get stats for specific method
        orb_stats = monitor.get_stats("pose_orb")
        
        # Get overall stats
        total_stats = monitor.get_stats()
    """
    
    def __init__(self, sample_interval: float = 0.5, gpu_index: int = 0, pid: Optional[int] = None):
        """
        Initialize hardware monitor.
        
        Args:
            sample_interval: Seconds between samples (default 0.5s)
            gpu_index: GPU index to monitor (default 0)
            pid: Process ID to monitor. If None, monitors current process.
                 Can also be set later with set_pid()
        """
        self.sample_interval = sample_interval
        self.gpu_index = gpu_index
        self._target_pid = pid
        
        self._snapshots: List[HardwareSnapshot] = []
        self._marks: Dict[str, int] = {}  # method_name -> snapshot_index
        self._current_mark: Optional[str] = None
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._process: Optional[Any] = None
        self._gpu_handle: Optional[Any] = None
        
        # Initialize process handle
        self._init_process_handle(pid)
        
        # Initialize GPU handle
        if _NVML_AVAILABLE:
            try:
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception as e:
                logger.warning(f"Could not get GPU handle: {e}")
                self._gpu_handle = None
    
    def _init_process_handle(self, pid: Optional[int] = None):
        """Initialize process handle for monitoring."""
        if _PSUTIL_AVAILABLE:
            try:
                import os
                target_pid = pid if pid is not None else os.getpid()
                self._target_pid = target_pid
                self._process = psutil.Process(target_pid)
                logger.debug(f"Monitoring process PID: {target_pid}")
            except psutil.NoSuchProcess:
                logger.warning(f"Process {pid} not found")
                self._process = None
            except Exception as e:
                logger.warning(f"Could not initialize process handle: {e}")
                self._process = None
    
    def set_pid(self, pid: int):
        """
        Set the process ID to monitor.
        
        Args:
            pid: Process ID to monitor
        """
        self._init_process_handle(pid)
    
    def _get_process_gpu_memory(self) -> float:
        """Get GPU memory used by the monitored process (in GB)."""
        if not _NVML_AVAILABLE or not self._gpu_handle or not self._target_pid:
            return 0.0
        
        try:
            # Get compute processes on GPU
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self._gpu_handle)
            for proc in processes:
                if proc.pid == self._target_pid:
                    return proc.usedGpuMemory / (1024**3)
            
            # Also check graphics processes
            try:
                graphics_procs = pynvml.nvmlDeviceGetGraphicsRunningProcesses(self._gpu_handle)
                for proc in graphics_procs:
                    if proc.pid == self._target_pid:
                        return proc.usedGpuMemory / (1024**3)
            except:
                pass
                
        except Exception as e:
            logger.debug(f"Could not get process GPU memory: {e}")
        
        return 0.0
    
    def _get_child_processes(self) -> List[Any]:
        """Get all child processes of the monitored process."""
        if not _PSUTIL_AVAILABLE or not self._process:
            return []
        
        try:
            return self._process.children(recursive=True)
        except:
            return []
    
    def _take_snapshot(self) -> HardwareSnapshot:
        """Take a single hardware measurement."""
        snapshot = HardwareSnapshot(timestamp=time.time())
        
        # CPU and RAM via psutil - system-wide
        if _PSUTIL_AVAILABLE:
            try:
                snapshot.cpu_percent = psutil.cpu_percent(interval=None)
                snapshot.cpu_percent_per_core = psutil.cpu_percent(interval=None, percpu=True)
                
                mem = psutil.virtual_memory()
                snapshot.ram_used_gb = mem.used / (1024**3)
                snapshot.ram_total_gb = mem.total / (1024**3)
                snapshot.ram_percent = mem.percent
                
                # Process-specific (including children)
                if self._process:
                    try:
                        # Main process
                        proc_cpu = self._process.cpu_percent()
                        proc_mem = self._process.memory_info().rss
                        
                        # Add child processes
                        for child in self._get_child_processes():
                            try:
                                proc_cpu += child.cpu_percent()
                                proc_mem += child.memory_info().rss
                            except (psutil.NoSuchProcess, psutil.AccessDenied):
                                pass
                        
                        snapshot.process_cpu_percent = proc_cpu
                        snapshot.process_ram_gb = proc_mem / (1024**3)
                        snapshot.process_ram_percent = (proc_mem / mem.total) * 100
                    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                        logger.debug(f"Process monitoring error: {e}")
            except Exception as e:
                logger.debug(f"psutil error: {e}")
        
        # GPU via pynvml - system-wide stats
        if _NVML_AVAILABLE and self._gpu_handle:
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
                snapshot.gpu_utilization = util.gpu
                
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
                snapshot.gpu_memory_used_gb = mem.used / (1024**3)
                snapshot.gpu_memory_total_gb = mem.total / (1024**3)
                snapshot.gpu_memory_percent = (mem.used / mem.total) * 100
                
                # Process-specific GPU memory
                snapshot.process_gpu_memory_gb = self._get_process_gpu_memory()
                
                try:
                    snapshot.gpu_temperature_c = pynvml.nvmlDeviceGetTemperature(
                        self._gpu_handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except:
                    pass
                
                try:
                    power = pynvml.nvmlDeviceGetPowerUsage(self._gpu_handle)
                    snapshot.gpu_power_w = power / 1000.0  # mW to W
                    
                    power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(self._gpu_handle)
                    snapshot.gpu_power_limit_w = power_limit / 1000.0
                except:
                    pass
            except Exception as e:
                logger.debug(f"NVML error: {e}")
        
        # Fallback GPU memory via PyTorch
        elif _TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                snapshot.gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                snapshot.gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                snapshot.gpu_memory_percent = (snapshot.gpu_memory_used_gb / snapshot.gpu_memory_total_gb) * 100
            except:
                pass
        
        return snapshot
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        # Initial CPU measurement to prime psutil
        if _PSUTIL_AVAILABLE:
            psutil.cpu_percent(interval=None)
            if self._process:
                self._process.cpu_percent()
        
        while self._running:
            snapshot = self._take_snapshot()
            self._snapshots.append(snapshot)
            time.sleep(self.sample_interval)
    
    def start(self):
        """Start monitoring in background thread."""
        if self._running:
            return
        
        self._running = True
        self._snapshots = []
        self._marks = {'__start__': 0}
        self._current_mark = '__start__'
        
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        
        logger.debug("Hardware monitoring started")
    
    def stop(self):
        """Stop monitoring and finalize."""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        
        # Mark end
        self._marks['__end__'] = len(self._snapshots)
        
        logger.debug(f"Hardware monitoring stopped. {len(self._snapshots)} samples collected.")
    
    def mark(self, name: str):
        """
        Mark the start of a new benchmark phase.
        
        Args:
            name: Identifier for this phase (e.g., "pose_orb", "depth_midas")
        """
        self._marks[name] = len(self._snapshots)
        self._current_mark = name
    
    def _compute_stats(self, snapshots: List[HardwareSnapshot]) -> HardwareStats:
        """Compute aggregate statistics from snapshots."""
        if not snapshots:
            return HardwareStats()
        
        stats = HardwareStats()
        stats.num_samples = len(snapshots)
        stats.duration_seconds = snapshots[-1].timestamp - snapshots[0].timestamp
        
        # Helper to safely compute stats
        def safe_mean(values):
            return sum(values) / len(values) if values else 0
        
        def safe_max(values):
            return max(values) if values else 0
        
        def safe_min(values):
            return min(values) if values else 0
        
        # System-wide CPU
        cpu_percents = [s.cpu_percent for s in snapshots]
        stats.cpu_percent_mean = safe_mean(cpu_percents)
        stats.cpu_percent_max = safe_max(cpu_percents)
        stats.cpu_percent_min = safe_min(cpu_percents)
        
        # System-wide RAM
        ram_used = [s.ram_used_gb for s in snapshots]
        stats.ram_used_gb_mean = safe_mean(ram_used)
        stats.ram_used_gb_max = safe_max(ram_used)
        
        ram_percent = [s.ram_percent for s in snapshots]
        stats.ram_percent_mean = safe_mean(ram_percent)
        
        # Process-specific CPU/RAM
        proc_cpu = [s.process_cpu_percent for s in snapshots]
        stats.process_cpu_percent_mean = safe_mean(proc_cpu)
        stats.process_cpu_percent_max = safe_max(proc_cpu)
        
        proc_ram = [s.process_ram_gb for s in snapshots]
        stats.process_ram_gb_mean = safe_mean(proc_ram)
        stats.process_ram_gb_max = safe_max(proc_ram)
        
        # System-wide GPU
        gpu_util = [s.gpu_utilization for s in snapshots]
        stats.gpu_utilization_mean = safe_mean(gpu_util)
        stats.gpu_utilization_max = safe_max(gpu_util)
        
        gpu_mem = [s.gpu_memory_used_gb for s in snapshots]
        stats.gpu_memory_used_gb_mean = safe_mean(gpu_mem)
        stats.gpu_memory_used_gb_max = safe_max(gpu_mem)
        
        gpu_mem_pct = [s.gpu_memory_percent for s in snapshots]
        stats.gpu_memory_percent_mean = safe_mean(gpu_mem_pct)
        stats.gpu_memory_percent_max = safe_max(gpu_mem_pct)
        
        gpu_temp = [s.gpu_temperature_c for s in snapshots]
        stats.gpu_temperature_c_mean = safe_mean(gpu_temp)
        stats.gpu_temperature_c_max = safe_max(gpu_temp)
        
        gpu_power = [s.gpu_power_w for s in snapshots]
        stats.gpu_power_w_mean = safe_mean(gpu_power)
        stats.gpu_power_w_max = safe_max(gpu_power)
        
        # Process-specific GPU memory
        proc_gpu_mem = [s.process_gpu_memory_gb for s in snapshots]
        stats.process_gpu_memory_gb_mean = safe_mean(proc_gpu_mem)
        stats.process_gpu_memory_gb_max = safe_max(proc_gpu_mem)
        
        # Estimate energy consumption (Watt-hours)
        # Rough estimate: average power * duration
        if stats.duration_seconds > 0 and stats.gpu_power_w_mean > 0:
            stats.total_energy_wh = stats.gpu_power_w_mean * (stats.duration_seconds / 3600)
        
        return stats
    
    def get_stats(self, name: Optional[str] = None) -> HardwareStats:
        """
        Get hardware statistics for a specific phase or overall.
        
        Args:
            name: Phase name (from mark()), or None for overall stats
            
        Returns:
            HardwareStats with aggregated metrics
        """
        if not self._snapshots:
            return HardwareStats()
        
        if name is None:
            # Overall stats
            return self._compute_stats(self._snapshots)
        
        if name not in self._marks:
            logger.warning(f"Unknown mark: {name}")
            return HardwareStats()
        
        # Find range for this mark
        start_idx = self._marks[name]
        
        # Find next mark
        sorted_marks = sorted(self._marks.items(), key=lambda x: x[1])
        end_idx = len(self._snapshots)
        
        for i, (mark_name, mark_idx) in enumerate(sorted_marks):
            if mark_name == name and i + 1 < len(sorted_marks):
                end_idx = sorted_marks[i + 1][1]
                break
        
        return self._compute_stats(self._snapshots[start_idx:end_idx])
    
    def get_all_stats(self) -> Dict[str, HardwareStats]:
        """Get stats for all marked phases plus overall."""
        result = {}
        
        # Stats for each phase
        for name in self._marks:
            if not name.startswith('__'):
                result[name] = self.get_stats(name)
        
        # Overall stats
        result['__overall__'] = self.get_stats()
        
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary suitable for JSON output."""
        overall = self.get_stats()
        
        phases = {}
        for name in self._marks:
            if not name.startswith('__'):
                phases[name] = self.get_stats(name).to_dict()
        
        return {
            'overall': overall.to_dict(),
            'phases': phases,
            'monitoring': {
                'sample_interval_s': self.sample_interval,
                'total_samples': len(self._snapshots),
                'psutil_available': _PSUTIL_AVAILABLE,
                'nvml_available': _NVML_AVAILABLE,
            }
        }


def get_system_info() -> Dict[str, Any]:
    """Get static system information."""
    info = {
        'cpu': {},
        'memory': {},
        'gpu': {},
    }
    
    if _PSUTIL_AVAILABLE:
        info['cpu']['cores_physical'] = psutil.cpu_count(logical=False)
        info['cpu']['cores_logical'] = psutil.cpu_count(logical=True)
        
        try:
            freq = psutil.cpu_freq()
            if freq:
                info['cpu']['freq_mhz'] = freq.current
                info['cpu']['freq_max_mhz'] = freq.max
        except:
            pass
        
        mem = psutil.virtual_memory()
        info['memory']['total_gb'] = round(mem.total / (1024**3), 2)
    
    if _NVML_AVAILABLE:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info['gpu']['name'] = pynvml.nvmlDeviceGetName(handle)
            if isinstance(info['gpu']['name'], bytes):
                info['gpu']['name'] = info['gpu']['name'].decode('utf-8')
            
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            info['gpu']['memory_total_gb'] = round(mem.total / (1024**3), 2)
            
            try:
                power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle)
                info['gpu']['power_limit_w'] = power_limit / 1000
            except:
                pass
        except Exception as e:
            logger.debug(f"Could not get GPU info: {e}")
    
    elif _TORCH_AVAILABLE and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info['gpu']['name'] = props.name
        info['gpu']['memory_total_gb'] = round(props.total_memory / (1024**3), 2)
    
    return info


def print_hardware_stats(stats: HardwareStats, title: str = "Hardware Usage"):
    """Print hardware stats in a nice format."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Duration: {stats.duration_seconds:.1f}s ({stats.num_samples} samples)")
    print()
    print("  SYSTEM-WIDE:")
    print(f"    CPU:        {stats.cpu_percent_mean:.1f}% avg, {stats.cpu_percent_max:.1f}% max")
    print(f"    RAM:        {stats.ram_used_gb_mean:.1f} GB avg, {stats.ram_used_gb_max:.1f} GB max ({stats.ram_percent_mean:.0f}%)")
    print()
    print("  PROCESS-SPECIFIC:")
    print(f"    CPU:        {stats.process_cpu_percent_mean:.1f}% avg, {stats.process_cpu_percent_max:.1f}% max")
    print(f"    RAM:        {stats.process_ram_gb_mean:.2f} GB avg, {stats.process_ram_gb_max:.2f} GB max")
    print(f"    GPU Memory: {stats.process_gpu_memory_gb_mean:.2f} GB avg, {stats.process_gpu_memory_gb_max:.2f} GB max")
    print()
    print("  GPU:")
    print(f"    Utilization: {stats.gpu_utilization_mean:.1f}% avg, {stats.gpu_utilization_max:.1f}% max")
    print(f"    Memory:      {stats.gpu_memory_used_gb_mean:.1f} GB avg, {stats.gpu_memory_used_gb_max:.1f} GB max ({stats.gpu_memory_percent_mean:.0f}%)")
    print(f"    Temperature: {stats.gpu_temperature_c_mean:.0f}°C avg, {stats.gpu_temperature_c_max:.0f}°C max")
    print(f"    Power:       {stats.gpu_power_w_mean:.0f}W avg, {stats.gpu_power_w_max:.0f}W max")
    print()
    print(f"  Energy Est:    {stats.total_energy_wh:.4f} Wh")
    print(f"{'=' * 70}")


# Convenience singleton for simple usage
_global_monitor: Optional[HardwareMonitor] = None


def start_monitoring(sample_interval: float = 0.5):
    """Start global hardware monitoring."""
    global _global_monitor
    _global_monitor = HardwareMonitor(sample_interval=sample_interval)
    _global_monitor.start()


def stop_monitoring() -> Optional[HardwareStats]:
    """Stop global monitoring and return overall stats."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.stop()
        return _global_monitor.get_stats()
    return None


def mark_phase(name: str):
    """Mark a phase in global monitoring."""
    if _global_monitor:
        _global_monitor.mark(name)


def get_monitor() -> Optional[HardwareMonitor]:
    """Get the global monitor instance."""
    return _global_monitor


if __name__ == "__main__":
    # Demo usage
    print("System Info:")
    print(get_system_info())
    print()
    
    monitor = HardwareMonitor(sample_interval=0.2)
    monitor.start()
    
    print("Running test workload...")
    
    monitor.mark("cpu_test")
    # CPU workload
    import math
    for _ in range(1000000):
        math.sqrt(12345.6789)
    
    monitor.mark("gpu_test")
    # GPU workload (if available)
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        x = torch.randn(5000, 5000, device='cuda')
        for _ in range(10):
            y = torch.mm(x, x)
        torch.cuda.synchronize()
    
    time.sleep(0.5)
    monitor.stop()
    
    # Print results
    print_hardware_stats(monitor.get_stats("cpu_test"), "CPU Test")
    print_hardware_stats(monitor.get_stats("gpu_test"), "GPU Test")
    print_hardware_stats(monitor.get_stats(), "Overall")
