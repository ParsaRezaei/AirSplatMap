#!/usr/bin/env python3
"""
AirSplatMap Quick System Test
=============================

Tests all major components to verify installation is working correctly.
Run this after setup to diagnose any issues.

Usage:
    python quick_test.py
    python quick_test.py --verbose  # Show more details
"""

import sys
import argparse
from pathlib import Path


def print_header(title):
    print(f'\n{"="*60}')
    print(f'  {title}')
    print(f'{"="*60}')


def print_section(title, num, total):
    print(f'\n[{num}/{total}] {title}...')


def test_core_imports(verbose=False):
    """Test core library imports."""
    results = {}
    
    # PyTorch
    try:
        import torch
        results['PyTorch'] = f'{torch.__version__}'
        results['CUDA'] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            results['GPU'] = torch.cuda.get_device_name(0)
            results['CUDA Version'] = torch.version.cuda
    except ImportError as e:
        results['PyTorch'] = f'FAILED: {e}'
    
    # NumPy
    try:
        import numpy as np
        results['NumPy'] = np.__version__
    except ImportError as e:
        results['NumPy'] = f'FAILED: {e}'
    
    # OpenCV
    try:
        import cv2
        results['OpenCV'] = cv2.__version__
    except ImportError as e:
        results['OpenCV'] = f'FAILED: {e}'
    
    # Open3D
    try:
        import open3d as o3d
        results['Open3D'] = o3d.__version__
    except ImportError as e:
        results['Open3D'] = f'N/A: {e}'
    
    for name, value in results.items():
        print(f'  {name}: {value}')
    
    return 'FAILED' not in str(results.get('PyTorch', '')) and results.get('CUDA') == 'True'


def test_hardware_monitoring(verbose=False):
    """Test hardware monitoring (pynvml/nvidia-ml-py)."""
    results = {}
    
    # Check pynvml (from nvidia-ml-py)
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        results['nvidia-ml-py'] = f'OK ({device_count} GPU(s))'
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            results['GPU Name'] = name
        pynvml.nvmlShutdown()
    except ImportError:
        results['nvidia-ml-py'] = 'NOT INSTALLED (pip install nvidia-ml-py)'
    except Exception as e:
        results['nvidia-ml-py'] = f'FAILED: {e}'
    
    # Check psutil
    try:
        import psutil
        results['psutil'] = f'OK (CPU: {psutil.cpu_percent()}%)'
    except ImportError:
        results['psutil'] = 'NOT INSTALLED'
    
    for name, value in results.items():
        print(f'  {name}: {value}')
    
    return 'OK' in str(results.get('nvidia-ml-py', ''))


def test_depth_estimators(verbose=False):
    """Test depth estimation availability."""
    try:
        from src.depth import list_depth_estimators
        estimators = list_depth_estimators()
        available = []
        unavailable = []
        
        for name, info in estimators.items():
            if info.get('available'):
                available.append(name)
                print(f'  ✓ {name}')
            else:
                unavailable.append(name)
                if verbose:
                    print(f'  ✗ {name}: {info.get("error", "Not available")}')
        
        if not verbose and unavailable:
            print(f'  ({len(unavailable)} unavailable: {", ".join(unavailable[:3])}{"..." if len(unavailable) > 3 else ""})')
        
        return len(available) > 0
    except Exception as e:
        print(f'  FAILED: {e}')
        return False


def test_pose_estimators(verbose=False):
    """Test pose estimation availability."""
    try:
        from src.pose import list_pose_estimators
        estimators = list_pose_estimators()
        available = []
        unavailable = []
        
        for name, info in estimators.items():
            if info.get('available'):
                available.append(name)
                print(f'  ✓ {name}')
            else:
                unavailable.append(name)
                if verbose:
                    print(f'  ✗ {name}')
        
        if not verbose and unavailable:
            print(f'  ({len(unavailable)} unavailable)')
        
        return len(available) > 0
    except Exception as e:
        print(f'  FAILED: {e}')
        return False


def test_gs_engines(verbose=False):
    """Test Gaussian Splatting engines."""
    try:
        from src.engines import list_engines
        engines = list_engines()
        available = []
        unavailable = []
        
        for name, info in engines.items():
            if info.get('available'):
                available.append(name)
                print(f'  ✓ {name}: {info.get("description", "")[:50]}')
            else:
                unavailable.append(name)
                if verbose:
                    print(f'  ✗ {name}: {info.get("error", "Not available")}')
        
        if not verbose and unavailable:
            print(f'  ({len(unavailable)} unavailable: {", ".join(unavailable)})')
        
        return len(available) > 0
    except Exception as e:
        print(f'  FAILED: {e}')
        return False


def test_gsplat_cuda(verbose=False):
    """Test gsplat CUDA kernels."""
    try:
        import torch
        from gsplat import rasterization
        
        # Create minimal test data
        means = torch.rand(10, 3, device='cuda')
        quats = torch.zeros(10, 4, device='cuda')
        quats[:, 0] = 1  # Identity quaternion
        scales = torch.ones(10, 3, device='cuda') * 0.1
        opacities = torch.ones(10, device='cuda')
        colors = torch.rand(10, 3, device='cuda')
        viewmat = torch.eye(4, device='cuda')
        viewmat[2, 3] = -3  # Camera at z=-3
        K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], 
                        dtype=torch.float32, device='cuda')
        
        # Run rasterization
        result = rasterization(
            means=means, quats=quats, scales=scales, 
            opacities=opacities, colors=colors,
            viewmats=viewmat[None], Ks=K[None], 
            width=640, height=480
        )
        print(f'  gsplat rasterization: OK')
        return True
    except ImportError as e:
        print(f'  gsplat: NOT INSTALLED ({e})')
        return False
    except Exception as e:
        print(f'  gsplat: FAILED - {e}')
        return False


def test_depth_pro_checkpoint(verbose=False):
    """Check Depth Pro checkpoint availability."""
    checkpoint_paths = [
        Path('submodules/ml-depth-pro/checkpoints/depth_pro.pt'),
        Path('checkpoints/depth_pro.pt'),
    ]
    
    for path in checkpoint_paths:
        if path.exists():
            size_gb = path.stat().st_size / 1e9
            print(f'  Checkpoint: OK ({path}, {size_gb:.2f} GB)')
            return True
    
    print(f'  Checkpoint: MISSING')
    print(f'    Download from: https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt')
    print(f'    Place in: submodules/ml-depth-pro/checkpoints/')
    return False


def test_datasets(verbose=False):
    """Check dataset availability."""
    datasets = [
        ('TUM fr1_desk', 'datasets/tum/rgbd_dataset_freiburg1_desk'),
        ('TUM fr1_room', 'datasets/tum/rgbd_dataset_freiburg1_room'),
        ('TUM fr3_office', 'datasets/tum/rgbd_dataset_freiburg3_long_office_household'),
        ('7-Scenes chess', 'datasets/7scenes/chess'),
        ('Replica room0', 'datasets/Replica/room0'),
    ]
    
    found = 0
    for name, path in datasets:
        exists = Path(path).exists()
        if exists:
            found += 1
            print(f'  ✓ {name}')
        elif verbose:
            print(f'  ✗ {name}: {path}')
    
    if found == 0:
        print(f'  No datasets found. Download with:')
        print(f'    python scripts/tools/download_datasets.ps1 datasets tum')
    elif found < len(datasets) and not verbose:
        print(f'  ({len(datasets) - found} datasets not found, use --verbose to see)')
    
    return found > 0


def test_quick_inference(verbose=False):
    """Quick inference test with depth estimation."""
    try:
        import numpy as np
        from src.depth import get_depth_estimator
        
        # Create test image
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Try fastest estimator
        est = get_depth_estimator('midas', model_type='MiDaS_small')
        result = est.estimate(test_img)
        print(f'  MiDaS inference: OK (output shape: {result.depth.shape})')
        est.cleanup()
        return True
    except Exception as e:
        print(f'  Inference test: FAILED - {e}')
        return False


def main():
    parser = argparse.ArgumentParser(description='AirSplatMap system test')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Show detailed output')
    args = parser.parse_args()
    
    print_header('AirSplatMap Quick System Test')
    
    tests = [
        ('Core imports', test_core_imports),
        ('Hardware monitoring', test_hardware_monitoring),
        ('Depth estimators', test_depth_estimators),
        ('Pose estimators', test_pose_estimators),
        ('GS engines', test_gs_engines),
        ('gsplat CUDA kernels', test_gsplat_cuda),
        ('Depth Pro checkpoint', test_depth_pro_checkpoint),
        ('Dataset availability', test_datasets),
        ('Quick inference', test_quick_inference),
    ]
    
    results = {}
    total = len(tests)
    
    for i, (name, test_func) in enumerate(tests, 1):
        print_section(name, i, total)
        try:
            results[name] = test_func(verbose=args.verbose)
        except Exception as e:
            print(f'  ERROR: {e}')
            results[name] = False
    
    # Summary
    print_header('Summary')
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for name, passed_test in results.items():
        status = '✓' if passed_test else '✗'
        print(f'  {status} {name}')
    
    print(f'\n  {passed}/{total} tests passed')
    
    if failed > 0:
        print(f'\n  ⚠️  Some tests failed. Check output above for details.')
        print(f'  Run with --verbose for more information.')
    else:
        print(f'\n  ✓ All systems operational!')
    
    print('=' * 60)
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
