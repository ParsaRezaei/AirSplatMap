#!/usr/bin/env python3
"""
AirSplatMap Comprehensive System Test
=====================================
Tests all major components to identify what works and what has issues.
"""

import sys
sys.path.insert(0, '.')
import os
import traceback

def main():
    print('=' * 60)
    print('AirSplatMap Comprehensive System Test')
    print('=' * 60)

    # Track results
    results = {}

    # ============================================
    # 1. Core Dependencies
    # ============================================
    print('\n[1/8] CORE DEPENDENCIES')
    print('-' * 40)

    deps = [
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('opencv', 'cv2'),
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('PIL', 'PIL'),
        ('matplotlib', 'matplotlib'),
        ('open3d', 'open3d'),
        ('kornia', 'kornia'),
        ('transformers', 'transformers'),
        ('evo', 'evo'),
    ]

    for name, module in deps:
        try:
            m = __import__(module)
            ver = getattr(m, '__version__', 'OK')
            print(f'  ✓ {name}: {ver}')
            results[f'dep_{name}'] = 'OK'
        except ImportError as e:
            print(f'  ✗ {name}: MISSING - {e}')
            results[f'dep_{name}'] = 'MISSING'

    # ============================================
    # 2. PyTorch Device Test
    # ============================================
    print('\n[2/8] PYTORCH DEVICES')
    print('-' * 40)

    import torch
    print(f'  PyTorch version: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    print(f'  MPS available: {mps_available}')

    # Test CPU
    try:
        x = torch.randn(100, 100)
        y = x @ x.T
        print(f'  ✓ CPU tensor ops: OK')
        results['torch_cpu'] = 'OK'
    except Exception as e:
        print(f'  ✗ CPU tensor ops: {e}')
        results['torch_cpu'] = 'FAILED'

    # Test MPS (skip on Intel Macs - known to be unstable)
    import platform
    is_intel_mac = platform.system() == 'Darwin' and platform.machine() == 'x86_64'
    
    if mps_available and not is_intel_mac:
        try:
            x = torch.randn(10, 10, device='mps')
            y = x @ x.T
            torch.mps.synchronize()
            print(f'  ✓ MPS basic ops: OK')
            results['torch_mps'] = 'OK'
        except Exception as e:
            print(f'  ✗ MPS basic ops: {e}')
            results['torch_mps'] = 'FAILED'
    elif is_intel_mac:
        print(f'  - MPS: Skipped (unstable on Intel Mac with AMD GPU)')
        results['torch_mps'] = 'SKIPPED'
    else:
        print(f'  - MPS: Not available')
        results['torch_mps'] = 'N/A'

    # ============================================
    # 3. Depth Estimators
    # ============================================
    print('\n[3/8] DEPTH ESTIMATORS')
    print('-' * 40)

    # Test imports only (no model loading to avoid crashes)
    depth_estimators = [
        ('DepthAnythingV2', 'src.depth.depth_anything', 'DepthAnythingV2Estimator'),
        ('DepthAnythingV3', 'src.depth.depth_anything', 'DepthAnythingV3Estimator'),
        ('MiDaS', 'src.depth.midas', 'MiDaSEstimator'),
        ('DepthPro', 'src.depth.depth_pro', 'DepthProEstimator'),
    ]

    for name, module, cls in depth_estimators:
        try:
            mod = __import__(module, fromlist=[cls])
            estimator_cls = getattr(mod, cls)
            available = estimator_cls.is_available() if hasattr(estimator_cls, 'is_available') else True
            if available:
                print(f'  ✓ {name}: Available')
                results[f'depth_{name}'] = 'OK'
            else:
                print(f'  - {name}: Dependencies missing')
                results[f'depth_{name}'] = 'DEPS'
        except Exception as e:
            print(f'  ✗ {name}: {e}')
            results[f'depth_{name}'] = 'FAILED'

    # ============================================
    # 4. Pose Estimators
    # ============================================
    print('\n[4/8] POSE ESTIMATORS')
    print('-' * 40)

    try:
        from src.pose import get_pose_estimator, AVAILABLE_ESTIMATORS
        print(f'  Available estimators: {list(AVAILABLE_ESTIMATORS.keys())}')
        
        for name in ['orb', 'sift', 'robust_flow']:
            try:
                est = get_pose_estimator(name)
                print(f'  ✓ {name}: OK')
                results[f'pose_{name}'] = 'OK'
            except Exception as e:
                print(f'  ✗ {name}: {e}')
                results[f'pose_{name}'] = 'FAILED'
    except Exception as e:
        print(f'  ✗ Pose module: {e}')
        results['pose_module'] = 'FAILED'

    # ============================================
    # 5. 3DGS Engines
    # ============================================
    print('\n[5/8] 3DGS ENGINES')
    print('-' * 40)

    try:
        from src.engines import list_engines
        engines = list_engines()
        for name, info in engines.items():
            status = '✓' if info.get('available', False) else '✗'
            reason = info.get('reason', '')
            avail_str = "Available" if info.get("available") else reason
            print(f'  {status} {name}: {avail_str}')
            results[f'engine_{name}'] = 'OK' if info.get('available') else 'N/A'
    except Exception as e:
        print(f'  ✗ Engines module: {e}')
        results['engines'] = 'FAILED'

    # ============================================
    # 6. Evaluation Metrics
    # ============================================
    print('\n[6/8] EVALUATION METRICS')
    print('-' * 40)

    try:
        from src.evaluation import metrics
        import numpy as np
        
        # Test PSNR
        img1 = np.random.rand(100, 100, 3).astype(np.float32)
        img2 = img1 + np.random.rand(100, 100, 3).astype(np.float32) * 0.1
        
        psnr = metrics.compute_psnr(img1, img2)
        print(f'  ✓ PSNR: {psnr:.2f} dB')
        results['metric_psnr'] = 'OK'
        
        ssim = metrics.compute_ssim(img1, img2)
        print(f'  ✓ SSIM: {ssim:.4f}')
        results['metric_ssim'] = 'OK'
        
    except Exception as e:
        print(f'  ✗ Metrics: {e}')
        results['metrics'] = 'FAILED'

    # ============================================
    # 7. Hardware Monitor
    # ============================================
    print('\n[7/8] HARDWARE MONITORING')
    print('-' * 40)

    try:
        from benchmarks.hardware_monitor import get_system_info, HardwareMonitor
        
        info = get_system_info()
        print(f'  ✓ CPU cores: {info.get("cpu", {}).get("cores_logical", "?")}')
        print(f'  ✓ Memory: {info.get("memory", {}).get("total_gb", "?")} GB')
        print(f'  ✓ Platform: {info.get("platform", "?")}')
        results['hw_monitor'] = 'OK'
    except Exception as e:
        print(f'  ✗ Hardware monitor: {e}')
        results['hw_monitor'] = 'FAILED'

    # ============================================
    # 8. Web Dashboard
    # ============================================
    print('\n[8/8] WEB DASHBOARD')
    print('-' * 40)

    dashboard_deps = ['flask', 'dash', 'websockets']
    for dep in dashboard_deps:
        try:
            __import__(dep)
            print(f'  ✓ {dep}: OK')
            results[f'dashboard_{dep}'] = 'OK'
        except ImportError:
            print(f'  ✗ {dep}: MISSING')
            results[f'dashboard_{dep}'] = 'MISSING'

    # ============================================
    # 9. Deep Learning Inference Test (CPU only)
    # ============================================
    print('\n[BONUS] DEEP LEARNING INFERENCE (CPU)')
    print('-' * 40)
    
    import numpy as np
    test_image = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
    
    # Test depth estimation on CPU
    try:
        from src.depth.depth_anything import DepthAnythingV2Estimator
        print('  Loading DepthAnythingV2 (small) on CPU...')
        estimator = DepthAnythingV2Estimator(model_size='vits', device='cpu')
        print(f'    Device: {estimator.device}')
        
        import time
        start = time.time()
        result = estimator.estimate(test_image)
        elapsed = time.time() - start
        
        print(f'  ✓ DepthAnythingV2: {elapsed:.2f}s, shape={result.depth.shape}')
        results['inference_depth'] = 'OK'
    except Exception as e:
        print(f'  ✗ DepthAnythingV2: {e}')
        results['inference_depth'] = 'FAILED'
    
    # Test pose estimation on CPU
    try:
        from src.pose import get_pose_estimator
        print('  Testing ORB pose estimator...')
        est = get_pose_estimator('orb')
        est.set_intrinsics_from_dict({'fx': 525, 'fy': 525, 'cx': 160, 'cy': 120})
        
        # Need two frames for pose estimation
        result1 = est.estimate(test_image)
        test_image2 = np.roll(test_image, 5, axis=1)  # Shift image slightly
        result2 = est.estimate(test_image2)
        
        print(f'  ✓ ORB pose: status={result2.tracking_status}, inliers={result2.num_inliers}')
        results['inference_pose'] = 'OK'
    except Exception as e:
        print(f'  ✗ ORB pose: {e}')
        results['inference_pose'] = 'FAILED'

    # ============================================
    # Summary
    # ============================================
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)

    ok_count = sum(1 for v in results.values() if v == 'OK')
    failed_count = sum(1 for v in results.values() if v in ['FAILED', 'MISSING'])
    na_count = sum(1 for v in results.values() if v in ['N/A', 'DEPS'])

    print(f'  ✓ Working: {ok_count}')
    print(f'  ✗ Failed/Missing: {failed_count}')
    print(f'  - Not applicable/unavailable: {na_count}')

    if failed_count > 0:
        print('\nFailed components:')
        for k, v in results.items():
            if v in ['FAILED', 'MISSING']:
                print(f'  - {k}: {v}')

    # Platform-specific notes
    print('\n' + '=' * 60)
    print('PLATFORM NOTES')
    print('=' * 60)
    
    import platform
    if platform.system() == 'Darwin':
        print('  macOS detected:')
        print('  - CUDA is NOT available (NVIDIA GPUs not supported)')
        print('  - MPS (Metal) has limited support for complex models')
        print('  - 3DGS engines require CUDA and will NOT work')
        print('  - Depth/Pose estimation works on CPU')
        print('  - For GPU acceleration, use a Linux machine with NVIDIA GPU')
    
    return results


if __name__ == '__main__':
    main()
