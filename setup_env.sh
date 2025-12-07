#!/bin/bash
# AirSplatMap Environment Setup Script
# =====================================
# This script creates the conda environment and installs PyTorch with CUDA support.
#
# Usage:
#   ./setup_env.sh
#
# Supports:
#   - Jetson (JetPack 6.x) with NVIDIA PyTorch wheels (includes SM 8.7 support)
#   - x86_64 Linux with CUDA 12.4+

set -e

echo "=============================================="
echo "AirSplatMap Environment Setup"
echo "=============================================="

# Detect platform
ARCH=$(uname -m)
IS_JETSON=false

if [[ "$ARCH" == "aarch64" ]]; then
    # Check if this is a Jetson device
    if [ -f /etc/nv_tegra_release ] || [ -d /usr/lib/aarch64-linux-gnu/tegra ]; then
        IS_JETSON=true
        echo "Detected: NVIDIA Jetson (aarch64)"
        
        # Try to detect JetPack version
        if command -v dpkg-query &> /dev/null; then
            JP_VERSION=$(dpkg-query --showformat='${Version}' --show nvidia-jetpack 2>/dev/null || echo "unknown")
            echo "JetPack version: $JP_VERSION"
        fi
    else
        echo "Detected: aarch64 (non-Jetson)"
    fi
else
    echo "Detected: $ARCH"
fi

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^airsplatmap "; then
    echo "Removing existing airsplatmap environment..."
    conda env remove -n airsplatmap -y
fi

# Create the environment
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate the environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate airsplatmap

# Uninstall any CPU-only PyTorch that may have been installed as a dependency
# (e.g., LightGlue pulls torch from PyPI which is CPU-only for aarch64)
echo "Replacing CPU-only PyTorch with CUDA-enabled version..."
pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."

if [[ "$IS_JETSON" == "true" ]]; then
    # ============================================
    # NVIDIA Jetson: Use NVIDIA's Jetson PyTorch wheels
    # ============================================
    # These wheels include SM 8.7 (Jetson Orin) support
    # Reference: https://developer.nvidia.com/embedded/jetson-linux
    # Compatibility: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform-release-notes/
    
    echo "Installing NVIDIA PyTorch for Jetson..."
    
    # NVIDIA Jetson PyTorch wheel index for JetPack 6.x
    # PyTorch 2.5.0 for JetPack 6.0/6.1 includes SM 8.7 support
    pip install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
    
    # Install torchvision from source or compatible wheel
    # Note: We install without dependencies to avoid overwriting NVIDIA torch
    echo "Installing torchvision..."
    pip install torchvision --no-deps 2>/dev/null || \
        pip install 'torchvision>=0.19,<0.20' --no-deps 2>/dev/null || \
        echo "WARNING: torchvision installation may need manual setup"
    
    # Install torchaudio if needed
    echo "Installing torchaudio..."
    pip install torchaudio --no-deps 2>/dev/null || \
        echo "WARNING: torchaudio installation skipped (optional)"
        
else
    # ============================================
    # x86_64: Use standard PyTorch with CUDA 12.4
    # ============================================
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

# ============================================
# Setup conda activation scripts for libstdc++ fix
# ============================================
# On Jetson/ARM, there's a conflict between system libstdc++ and conda's version.
# OpenCV and other libraries need the newer GLIBCXX symbols from conda's libstdc++.
# We create activation/deactivation scripts to automatically set LD_PRELOAD.
echo ""
echo "=============================================="
echo "Setting up environment activation scripts..."
echo "=============================================="

CONDA_ENV_PATH="$CONDA_PREFIX"
mkdir -p "$CONDA_ENV_PATH/etc/conda/activate.d"
mkdir -p "$CONDA_ENV_PATH/etc/conda/deactivate.d"

# Create activation script
cat > "$CONDA_ENV_PATH/etc/conda/activate.d/airsplatmap_env.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# AirSplatMap environment activation script
# Fixes libstdc++ ABI compatibility on Jetson/ARM platforms

# Only apply fix on aarch64 (Jetson) or if libstdc++ issues are detected
if [[ "$(uname -m)" == "aarch64" ]] || [[ -n "$AIRSPLATMAP_FORCE_LIBSTDCXX" ]]; then
    # Save old LD_PRELOAD
    export _OLD_LD_PRELOAD="$LD_PRELOAD"
    # Preload conda's libstdc++ to fix GLIBCXX version issues
    if [ -f "$CONDA_PREFIX/lib/libstdc++.so.6" ]; then
        export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
    fi
fi

# Set CUDA architecture for Jetson Orin (SM 8.7) if not already set
if [[ "$(uname -m)" == "aarch64" ]] && [[ -z "$TORCH_CUDA_ARCH_LIST" ]]; then
    export TORCH_CUDA_ARCH_LIST="8.7"
fi
ACTIVATE_EOF

# Create deactivation script
cat > "$CONDA_ENV_PATH/etc/conda/deactivate.d/airsplatmap_env.sh" << 'DEACTIVATE_EOF'
#!/bin/bash
# AirSplatMap environment deactivation script

# Restore original LD_PRELOAD
if [[ -n "$_OLD_LD_PRELOAD" ]]; then
    export LD_PRELOAD="$_OLD_LD_PRELOAD"
else
    unset LD_PRELOAD
fi
unset _OLD_LD_PRELOAD

# Unset CUDA arch if we set it
if [[ "$(uname -m)" == "aarch64" ]]; then
    unset TORCH_CUDA_ARCH_LIST
fi
DEACTIVATE_EOF

chmod +x "$CONDA_ENV_PATH/etc/conda/activate.d/airsplatmap_env.sh"
chmod +x "$CONDA_ENV_PATH/etc/conda/deactivate.d/airsplatmap_env.sh"

echo "Created activation scripts for libstdc++ compatibility fix"

# Re-activate to apply the new scripts
conda deactivate
conda activate airsplatmap

# Verify PyTorch installation
echo ""
echo "=============================================="
echo "Verifying PyTorch installation..."
echo "=============================================="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Device capability: {torch.cuda.get_device_capability()}')
    if hasattr(torch.cuda, 'get_arch_list'):
        archs = torch.cuda.get_arch_list()
        print(f'Supported CUDA architectures: {archs}')
        # Check if Jetson Orin (SM 8.7) is supported
        if any('87' in arch for arch in archs):
            print('✓ Jetson Orin (SM 8.7) support: YES')
        else:
            print('✗ Jetson Orin (SM 8.7) support: NO - some engines may not work')
else:
    print('WARNING: CUDA not available! Check your NVIDIA drivers.')
"

# ============================================
# Install 3DGS Engine Dependencies
# ============================================
echo ""
echo "=============================================="
echo "Installing 3DGS Engine Dependencies..."
echo "=============================================="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 1. Install gsplat (Nerfstudio's optimized 3DGS - fastest engine, ~17 FPS)
echo ""
echo "[1/4] Installing gsplat (recommended for real-time performance)..."

if [[ "$IS_JETSON" == "true" ]]; then
    # For Jetson, we need to build gsplat from source with SM 8.7 support
    # The pip wheel doesn't include Jetson CUDA kernels
    echo "  Building gsplat from source for Jetson (SM 8.7)..."
    
    GSPLAT_BUILD_DIR="/tmp/gsplat_build_$$"
    if git clone --recursive --depth 1 --branch v1.0.0 https://github.com/nerfstudio-project/gsplat.git "$GSPLAT_BUILD_DIR" 2>/dev/null; then
        cd "$GSPLAT_BUILD_DIR"
        TORCH_CUDA_ARCH_LIST="8.7" pip install . --no-build-isolation 2>&1 || \
            echo "WARNING: gsplat build failed. Try manually: cd $GSPLAT_BUILD_DIR && TORCH_CUDA_ARCH_LIST=8.7 pip install . --no-build-isolation"
        cd - > /dev/null
        rm -rf "$GSPLAT_BUILD_DIR"
    else
        echo "WARNING: Failed to clone gsplat repository"
    fi
else
    # For x86_64, the pip wheel should work
    pip install gsplat 2>/dev/null || echo "WARNING: gsplat installation failed. May require matching CUDA toolkit version."
fi

# 2. Install GraphDeco CUDA extensions (original 3DGS)
echo ""
echo "[2/4] Installing GraphDeco CUDA extensions (original 3DGS)..."
if [ -d "$SCRIPT_DIR/submodules/gaussian-splatting/submodules/simple-knn" ]; then
    echo "  Installing simple-knn..."
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.7}" pip install --no-build-isolation "$SCRIPT_DIR/submodules/gaussian-splatting/submodules/simple-knn" 2>&1 || \
        echo "WARNING: simple-knn installation failed. GraphDeco engine may not be available."
    
    echo "  Installing diff-gaussian-rasterization..."
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.7}" pip install --no-build-isolation "$SCRIPT_DIR/submodules/gaussian-splatting/submodules/diff-gaussian-rasterization" 2>&1 || \
        echo "WARNING: diff-gaussian-rasterization installation failed. GraphDeco engine may not be available."
    
    if [ -d "$SCRIPT_DIR/submodules/gaussian-splatting/submodules/fused-ssim" ]; then
        echo "  Installing fused-ssim..."
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.7}" pip install --no-build-isolation "$SCRIPT_DIR/submodules/gaussian-splatting/submodules/fused-ssim" 2>&1 || \
            echo "WARNING: fused-ssim installation failed (optional)."
    fi
else
    echo "WARNING: gaussian-splatting submodule not found. Run: git submodule update --init --recursive"
fi

# 3. Install MonoGS dependencies (if submodule exists)
echo ""
echo "[3/4] Installing MonoGS dependencies..."
if [ -d "$SCRIPT_DIR/submodules/MonoGS" ]; then
    # MonoGS uses the same diff-gaussian-rasterization, already installed above
    # Install any additional MonoGS-specific requirements
    if [ -f "$SCRIPT_DIR/submodules/MonoGS/requirements.txt" ]; then
        pip install -r "$SCRIPT_DIR/submodules/MonoGS/requirements.txt" 2>/dev/null || \
            echo "WARNING: Some MonoGS requirements failed to install."
    fi
    # Add MonoGS to Python path by installing in editable mode if setup.py exists
    if [ -f "$SCRIPT_DIR/submodules/MonoGS/setup.py" ]; then
        pip install -e "$SCRIPT_DIR/submodules/MonoGS" 2>/dev/null || \
            echo "WARNING: MonoGS installation failed."
    fi
else
    echo "WARNING: MonoGS submodule not found. Run: git submodule update --init --recursive"
fi

# 4. Verify SplaTAM and Gaussian-SLAM (already handled via environment.yml typically)
echo ""
echo "[4/4] Verifying SLAM engines (SplaTAM, Gaussian-SLAM)..."
# These are typically installed via the environment.yml or as submodules
# Just verify they're accessible
python -c "
try:
    from src.engines import list_engines
    engines = list_engines()
    available = [k for k, v in engines.items() if v['available']]
    print(f'Available engines: {available}')
    if not available:
        print('WARNING: No engines available. Check CUDA extensions.')
except Exception as e:
    print(f'WARNING: Could not verify engines: {e}')
"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate airsplatmap"
echo ""
echo "Available engines can be checked with:"
echo "  python -c \"from src.engines import list_engines; print(list_engines())\""
echo ""
