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
OS=$(uname -s)
IS_JETSON=false
IS_MACOS=false

if [[ "$OS" == "Darwin" ]]; then
    IS_MACOS=true
    echo "Detected: macOS ($ARCH)"
    if [[ "$ARCH" == "x86_64" ]]; then
        echo "  Intel Mac - MPS will use AMD GPU if available"
    else
        echo "  Apple Silicon - MPS will use integrated GPU"
    fi
    echo "NOTE: CUDA is not available on macOS. PyTorch will use MPS (Metal) for GPU acceleration."
    echo "NOTE: pyrealsense2 is not available via pip on macOS."
    echo "      If you need RealSense support, install via: brew install librealsense"
elif [[ "$ARCH" == "aarch64" ]]; then
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
    echo "Detected: $OS $ARCH"
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
python -m pip uninstall torch torchvision torchaudio -y 2>/dev/null || true

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
    python -m pip install --no-cache-dir https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
    
    # Install torchvision from source or compatible wheel
    # Note: We install without dependencies to avoid overwriting NVIDIA torch
    echo "Installing torchvision..."
    python -m pip install torchvision --no-deps 2>/dev/null || \
        python -m pip install 'torchvision>=0.19,<0.20' --no-deps 2>/dev/null || \
        echo "WARNING: torchvision installation may need manual setup"
    
    # Install torchaudio if needed
    echo "Installing torchaudio..."
    python -m pip install torchaudio --no-deps 2>/dev/null || \
        echo "WARNING: torchaudio installation skipped (optional)"
    
    # Install jetson-stats for GPU monitoring
    echo "Installing jetson-stats for Jetson GPU monitoring..."
    python -m pip install jetson-stats 2>/dev/null || \
        echo "WARNING: jetson-stats installation failed (optional, for GPU monitoring)"
    
    # Add user to jtop group if not already (requires re-login to take effect)
    if ! groups | grep -q jtop; then
        echo "Adding user to jtop group (may require re-login for full access)..."
        sudo usermod -aG jtop "$USER" 2>/dev/null || \
            echo "NOTE: Could not add user to jtop group. Run: sudo usermod -aG jtop $USER"
    fi
        
elif [[ "$IS_MACOS" == "true" ]]; then
    # ============================================
    # macOS: PyTorch NIGHTLY with MPS support (Apple Silicon or AMD GPU)
    # ============================================
    # Reference: https://developer.apple.com/metal/pytorch/
    # MPS works on both Apple Silicon AND Intel Macs with AMD GPUs
    # 
    # IMPORTANT: Use nightly/preview builds for best MPS support!
    # The nightly version has many more MPS operations implemented and
    # better performance than the stable release.
    #
    # NOTE: macOS wheels are on PyPI, not the pytorch index
    
    echo "Installing PyTorch for macOS with MPS support..."
    echo "  Reference: https://developer.apple.com/metal/pytorch/"
    # Use conda for macOS - pip only has old versions for x86_64
    # conda-forge has latest PyTorch with MPS support for both Intel and Apple Silicon
    conda install pytorch torchvision torchaudio -c pytorch-nightly -y
    
    # Install OpenCV via pip (conda version has protobuf/abseil conflicts on macOS)
    echo "Installing OpenCV via pip (avoids conda library conflicts)..."
    # First remove any conda-installed OpenCV
    conda uninstall opencv py-opencv libopencv -y 2>/dev/null || true
    python -m pip install opencv-python opencv-contrib-python
    
    # Verify MPS availability
    echo ""
    echo "Checking MPS (Metal Performance Shaders) availability..."
    python -c "
import torch
if torch.backends.mps.is_available():
    print('✓ MPS is available - your GPU will be used for acceleration')
    print('  Device: Apple Metal GPU (AMD Radeon or Apple Silicon)')
else:
    print('⚠ MPS not available - will use CPU')
    print('  (MPS requires macOS 12.3+ and compatible GPU)')
"
else
    # ============================================
    # x86_64 Linux: Use standard PyTorch with CUDA 12.1
    # Using specific versions that are known to work
    # ============================================
    python -m pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
fi

# ============================================
# Install OpenCV (platform-specific)
# ============================================
echo ""
echo "Installing OpenCV..."
if [[ "$IS_MACOS" != "true" ]]; then
    # Linux/Windows: Install from conda (already in environment.yml, but ensure it's there)
    conda install -c conda-forge opencv -y 2>/dev/null || \
        python -m pip install opencv-python opencv-contrib-python
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
# Enables MPS fallback for unsupported operators on macOS

# macOS MPS support: Enable CPU fallback for unsupported MPS operators
if [[ "$(uname -s)" == "Darwin" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

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

# macOS: Unset MPS fallback
if [[ "$(uname -s)" == "Darwin" ]]; then
    unset PYTORCH_ENABLE_MPS_FALLBACK
fi

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
import platform
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
elif platform.system() == 'Darwin':
    # macOS - check for MPS (Apple Silicon or AMD GPU)
    import os
    mps_fallback = os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0')
    print(f'MPS fallback enabled: {mps_fallback == \"1\"}')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print('✓ MPS (Metal Performance Shaders) available: YES')
        # Detect CPU architecture
        import platform as pf
        arch = pf.machine()
        if arch == 'arm64':
            print('  GPU: Apple Silicon (M1/M2/M3)')
        else:
            print('  GPU: AMD Radeon (Intel Mac with discrete GPU)')
        print('  Using Apple Metal for GPU acceleration')
    else:
        print('MPS not available. Using CPU only.')
        print('  (MPS requires macOS 12.3+ and Apple Silicon or AMD GPU)')
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

# Skip CUDA extensions on macOS (no CUDA support)
if [[ "$IS_MACOS" == "true" ]]; then
    echo "Skipping CUDA-dependent 3DGS engines on macOS."
    echo "Note: 3DGS engines require CUDA and are not available on macOS."
    echo "      You can still use depth estimation and other CPU/MPS features."
else

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
        TORCH_CUDA_ARCH_LIST="8.7" python -m pip install . --no-build-isolation 2>&1 || \
            echo "WARNING: gsplat build failed. Try manually: cd $GSPLAT_BUILD_DIR && TORCH_CUDA_ARCH_LIST=8.7 python -m pip install . --no-build-isolation"
        cd - > /dev/null
        rm -rf "$GSPLAT_BUILD_DIR"
    else
        echo "WARNING: Failed to clone gsplat repository"
    fi
else
    # For x86_64, the pip wheel should work
    python -m pip install gsplat 2>/dev/null || echo "WARNING: gsplat installation failed. May require matching CUDA toolkit version."
fi

# 2. Install GraphDeco CUDA extensions (original 3DGS)
echo ""
echo "[2/4] Installing GraphDeco CUDA extensions (original 3DGS)..."
if [ -d "$SCRIPT_DIR/submodules/gaussian-splatting/submodules/simple-knn" ]; then
    echo "  Installing simple-knn..."
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.7}" python -m pip install --no-build-isolation "$SCRIPT_DIR/submodules/gaussian-splatting/submodules/simple-knn" 2>&1 || \
        echo "WARNING: simple-knn installation failed. GraphDeco engine may not be available."
    
    echo "  Installing diff-gaussian-rasterization..."
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.7}" python -m pip install --no-build-isolation "$SCRIPT_DIR/submodules/gaussian-splatting/submodules/diff-gaussian-rasterization" 2>&1 || \
        echo "WARNING: diff-gaussian-rasterization installation failed. GraphDeco engine may not be available."
    
    if [ -d "$SCRIPT_DIR/submodules/gaussian-splatting/submodules/fused-ssim" ]; then
        echo "  Installing fused-ssim..."
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.7}" python -m pip install --no-build-isolation "$SCRIPT_DIR/submodules/gaussian-splatting/submodules/fused-ssim" 2>&1 || \
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
        python -m pip install -r "$SCRIPT_DIR/submodules/MonoGS/requirements.txt" 2>/dev/null || \
            echo "WARNING: Some MonoGS requirements failed to install."
    fi
    # Add MonoGS to Python path by installing in editable mode if setup.py exists
    if [ -f "$SCRIPT_DIR/submodules/MonoGS/setup.py" ]; then
        python -m pip install -e "$SCRIPT_DIR/submodules/MonoGS" 2>/dev/null || \
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

fi  # End of non-macOS CUDA engine installation

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

# ============================================
# Install Depth Anything V3 (DA3)
# ============================================
echo ""
echo "=============================================="
echo "Installing Depth Anything V3 (DA3)..."
echo "=============================================="

DA3_PATH="$SCRIPT_DIR/submodules/Depth-Anything-3"

if [ -f "$DA3_PATH/pyproject.toml" ]; then
    echo "Found DA3 at: $DA3_PATH"
    
    # Install DA3 dependencies
    echo "Installing DA3 dependencies..."
    pip install einops huggingface_hub omegaconf evo e3nn moviepy plyfile safetensors trimesh open3d pillow_heif 2>/dev/null || \
        echo "WARNING: Some DA3 dependencies failed to install"
    
    # Install xformers (required for DA3 attention)
    echo "Installing xformers..."
    pip install xformers 2>/dev/null || \
        echo "WARNING: xformers installation failed. DA3 may fall back to slower attention."
    
    # Install DA3 in editable mode
    echo "Installing DA3 package..."
    pip install -e "$DA3_PATH" --no-deps 2>/dev/null || \
        echo "WARNING: DA3 installation failed"
    
    # Verify DA3 installation
    echo ""
    echo "Verifying DA3 installation..."
    python -c "from depth_anything_3.api import DepthAnything3; print('  DA3 import: OK')" 2>&1 || \
        echo "  DA3 import: FAILED"
else
    echo "WARNING: DA3 submodule not found at $DA3_PATH"
    echo "Run: git submodule update --init --recursive"
fi

echo ""
echo "=============================================="
echo "All installations complete!"
echo "=============================================="
echo ""
echo "Available components:"
echo "  - PyTorch with CUDA"
echo "  - Gaussian Splatting (diff-gaussian-rasterization)"
echo "  - gsplat (JIT compilation)"
echo "  - Depth Anything V3 (DA3)"
echo ""
