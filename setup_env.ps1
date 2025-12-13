# AirSplatMap Environment Setup Script for Windows
# =================================================
# This script creates the conda environment and installs PyTorch with CUDA support.
# It also creates activation hooks that automatically set up MSVC and CUDA.
#
# Usage (run in PowerShell):
#   .\setup_env.ps1
#
# For Windows with NVIDIA GPU and CUDA 12.x
# Requires: Miniconda or Anaconda installed

$ErrorActionPreference = "Stop"

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "AirSplatMap Environment Setup (Windows)" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Check if conda is available
try {
    $condaVersion = conda --version
    Write-Host "Found: $condaVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: conda not found. Please install Miniconda or Anaconda first." -ForegroundColor Red
    Write-Host "Download from: https://docs.conda.io/en/latest/miniconda.html" -ForegroundColor Yellow
    exit 1
}

# Check if environment exists and remove it
$envList = conda env list
if ($envList -match "airsplatmap") {
    Write-Host "Removing existing airsplatmap environment..." -ForegroundColor Yellow
    conda env remove -n airsplatmap -y
}

# Create the environment
Write-Host "Creating conda environment from environment.yml..." -ForegroundColor Cyan
conda env create -f environment.yml
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create conda environment" -ForegroundColor Red
    exit 1
}

# Get conda env path
$condaBase = (conda info --base).Trim()
$condaEnvPath = Join-Path $condaBase "envs\airsplatmap"

Write-Host "Environment path: $condaEnvPath" -ForegroundColor Gray

# ============================================
# Create Conda Activation Hooks for CUDA/MSVC
# ============================================
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Creating CUDA/MSVC activation hooks..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Create activate.d and deactivate.d directories
$activateDir = Join-Path $condaEnvPath "etc\conda\activate.d"
$deactivateDir = Join-Path $condaEnvPath "etc\conda\deactivate.d"

New-Item -ItemType Directory -Force -Path $activateDir | Out-Null
New-Item -ItemType Directory -Force -Path $deactivateDir | Out-Null

# Create activation script (batch file for Windows cmd)
$activateScript = @'
@echo off
REM AirSplatMap CUDA/MSVC Environment Setup
REM This script runs automatically when you activate the airsplatmap environment

REM Save original PATH for deactivation
set "AIRSPLATMAP_OLD_PATH=%PATH%"

REM Required for PyTorch CUDA extension building with MSVC
set "DISTUTILS_USE_SDK=1"

REM ============================================
REM Setup Visual Studio Build Tools (for gsplat JIT compilation)
REM IMPORTANT: Use VS 2019 for CUDA 12.1 compatibility (VS 2022 v17.10+ is too new)
REM ============================================
set "VSCMD_START_DIR=%CD%"

REM Try VS 2019 first (required for CUDA 12.1), then 2022
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)

REM ============================================
REM Setup CUDA Toolkit
REM ============================================
REM Try multiple CUDA versions (prefer 12.1 to match PyTorch)
if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\nvcc.exe" (
    set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1"
    set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin;%PATH%"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\nvcc.exe" (
    set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2"
    set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin;%PATH%"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe" (
    set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
    set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%"
) else if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" (
    set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
    set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;%PATH%"
)
'@

$deactivateScript = @'
@echo off
REM Restore original PATH on deactivation
if defined AIRSPLATMAP_OLD_PATH (
    set "PATH=%AIRSPLATMAP_OLD_PATH%"
    set "AIRSPLATMAP_OLD_PATH="
)
set "CUDA_HOME="
set "CUDA_PATH="
set "DISTUTILS_USE_SDK="
'@

# Create PowerShell activation script
$activateScriptPS = @'
# AirSplatMap CUDA/MSVC Environment Setup for PowerShell
# This script runs automatically when you activate the airsplatmap environment

# Save original PATH for deactivation
$env:AIRSPLATMAP_OLD_PATH = $env:PATH

# Required for PyTorch CUDA extension building with MSVC
$env:DISTUTILS_USE_SDK = "1"

# Setup CUDA Toolkit - try multiple versions
$cudaPaths = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
)
foreach ($cudaPath in $cudaPaths) {
    if (Test-Path "$cudaPath\bin\nvcc.exe") {
        $env:CUDA_HOME = $cudaPath
        $env:CUDA_PATH = $cudaPath
        $env:PATH = "$cudaPath\bin;" + $env:PATH
        break
    }
}
'@

$deactivateScriptPS = @'
# Restore original PATH on deactivation
if ($env:AIRSPLATMAP_OLD_PATH) {
    $env:PATH = $env:AIRSPLATMAP_OLD_PATH
    Remove-Item Env:\AIRSPLATMAP_OLD_PATH -ErrorAction SilentlyContinue
}
Remove-Item Env:\CUDA_HOME -ErrorAction SilentlyContinue
Remove-Item Env:\CUDA_PATH -ErrorAction SilentlyContinue
Remove-Item Env:\DISTUTILS_USE_SDK -ErrorAction SilentlyContinue
'@

# Write activation scripts - both .bat and .ps1
$activateScriptPath = Join-Path $activateDir "cuda_msvc_setup.bat"
$deactivateScriptPath = Join-Path $deactivateDir "cuda_msvc_cleanup.bat"
$activateScriptPathPS = Join-Path $activateDir "cuda_msvc_setup.ps1"
$deactivateScriptPathPS = Join-Path $deactivateDir "cuda_msvc_cleanup.ps1"

$activateScript | Out-File -FilePath $activateScriptPath -Encoding ASCII
$deactivateScript | Out-File -FilePath $deactivateScriptPath -Encoding ASCII
$activateScriptPS | Out-File -FilePath $activateScriptPathPS -Encoding ASCII
$deactivateScriptPS | Out-File -FilePath $deactivateScriptPathPS -Encoding ASCII

Write-Host "Created activation hook (cmd): $activateScriptPath" -ForegroundColor Green
Write-Host "Created activation hook (ps1): $activateScriptPathPS" -ForegroundColor Green
Write-Host "Created deactivation hook (cmd): $deactivateScriptPath" -ForegroundColor Green
Write-Host "Created deactivation hook (ps1): $deactivateScriptPathPS" -ForegroundColor Green

# Activate the environment
Write-Host ""
Write-Host "Activating environment..." -ForegroundColor Cyan
conda activate airsplatmap

# Uninstall any CPU-only PyTorch that may have been installed as a dependency
# Note: We temporarily disable error action preference because pip warnings go to stderr
Write-Host "Replacing CPU-only PyTorch with CUDA-enabled version..." -ForegroundColor Yellow
$ErrorActionPreference = "Continue"
pip uninstall torch torchvision torchaudio -y 2>&1 | Out-Null
$ErrorActionPreference = "Stop"

# Install PyTorch with CUDA from the cu121 index
# Using specific versions that are known to work with Windows + Python 3.10 + CUDA 12.1
Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Cyan
$ErrorActionPreference = "Continue"
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
$pipExitCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($pipExitCode -ne 0) {
    Write-Host "ERROR: Failed to install PyTorch" -ForegroundColor Red
    exit 1
}

# Verify PyTorch installation
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Verifying PyTorch installation..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

$ErrorActionPreference = "Continue"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else print('WARNING: CUDA not available!'); print(f'Device: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else None"
$ErrorActionPreference = "Stop"

# ============================================
# Build Gaussian Splatting CUDA Extensions
# ============================================
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Building Gaussian Splatting CUDA Extensions..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# Find CUDA toolkit
$cudaPaths = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
)

$cudaPath = $null
foreach ($path in $cudaPaths) {
    if (Test-Path "$path\bin\nvcc.exe") {
        $cudaPath = $path
        break
    }
}

if ($cudaPath) {
    Write-Host "Found CUDA toolkit at: $cudaPath" -ForegroundColor Green
    $env:Path = "$cudaPath\bin;" + $env:Path
    $env:CUDA_HOME = $cudaPath
    
    # Also setup MSVC for this session (needed for building extensions)
    # IMPORTANT: Use VS 2019 for CUDA 12.1 compatibility (VS 2022 v17.10+ is too new)
    $vsPathsToTry = @(
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat",
        "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
    )
    
    $vsPath = $null
    foreach ($path in $vsPathsToTry) {
        if (Test-Path $path) {
            $vsPath = $path
            break
        }
    }
    
    if ($vsPath) {
        Write-Host "Found Visual Studio at: $vsPath" -ForegroundColor Green
        # Run vcvars64.bat and capture environment changes
        $tempBat = [System.IO.Path]::GetTempFileName() -replace '\.tmp$', '.bat'
        $tempEnv = [System.IO.Path]::GetTempFileName()
        @"
@echo off
call "$vsPath" >nul 2>&1
set > "$tempEnv"
"@ | Out-File -FilePath $tempBat -Encoding ASCII
        cmd /c $tempBat
        Get-Content $tempEnv | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
            }
        }
        Remove-Item $tempBat, $tempEnv -ErrorAction SilentlyContinue
    } else {
        Write-Host "WARNING: Visual Studio Build Tools not found. gsplat JIT compilation may not work." -ForegroundColor Yellow
    }
    
    $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $gsSubmodules = "$scriptDir\submodules\gaussian-splatting\submodules"
    
    # Set DISTUTILS_USE_SDK to avoid VC environment activation issues
    $env:DISTUTILS_USE_SDK = "1"
    
    $ErrorActionPreference = "Continue"
    
    # Build simple_knn
    if (Test-Path "$gsSubmodules\simple-knn\setup.py") {
        Write-Host "Building simple_knn..." -ForegroundColor Cyan
        pip install "$gsSubmodules\simple-knn" --no-build-isolation 2>&1 | Out-Host
    }
    
    # Build diff-gaussian-rasterization
    if (Test-Path "$gsSubmodules\diff-gaussian-rasterization\setup.py") {
        Write-Host "Building diff-gaussian-rasterization..." -ForegroundColor Cyan
        pip install "$gsSubmodules\diff-gaussian-rasterization" --no-build-isolation 2>&1 | Out-Host
    }
    
    # Build fused-ssim (optional)
    if (Test-Path "$gsSubmodules\fused-ssim\setup.py") {
        Write-Host "Building fused-ssim..." -ForegroundColor Cyan
        pip install "$gsSubmodules\fused-ssim" --no-build-isolation 2>&1 | Out-Host
    }
    
    $ErrorActionPreference = "Stop"
    
    # Verify CUDA extensions
    Write-Host ""
    Write-Host "Verifying CUDA extensions..." -ForegroundColor Cyan
    $ErrorActionPreference = "Continue"
    python -c "import simple_knn; print('  simple_knn: OK')" 2>&1
    python -c "import diff_gaussian_rasterization; print('  diff_gaussian_rasterization: OK')" 2>&1
    python -c "import fused_ssim; print('  fused_ssim: OK')" 2>&1
    $ErrorActionPreference = "Stop"
} else {
    Write-Host "WARNING: CUDA toolkit not found. Gaussian Splatting extensions will not be available." -ForegroundColor Yellow
    Write-Host "Install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
}

# ============================================
# Pre-build gsplat CUDA kernels (JIT compilation)
# ============================================
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Pre-building gsplat CUDA kernels..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

# gsplat uses JIT compilation which requires both nvcc and cl (MSVC) in PATH
# Since MSVC environment is already active from building gaussian-splatting extensions,
# we can trigger the gsplat JIT compilation now

$ErrorActionPreference = "Continue"

# Test if gsplat can compile by importing and running a simple operation
$gsplatTestScript = @'
import torch
import gsplat

# Trigger JIT compilation by calling project_gaussians
try:
    means = torch.randn(10, 3, device='cuda')
    scales = torch.ones(10, 3, device='cuda') * 0.1
    quats = torch.randn(10, 4, device='cuda')
    quats = quats / quats.norm(dim=-1, keepdim=True)
    viewmat = torch.eye(4, device='cuda')
    
    xys, depths, radii, conics, comp, num_tiles, cov3d = gsplat.project_gaussians(
        means, scales, 1.0, quats, viewmat, 500.0, 500.0, 320.0, 240.0, 480, 640, 16, 0.01
    )
    print('gsplat JIT compilation: SUCCESS')
except Exception as e:
    print(f'gsplat JIT compilation: FAILED - {e}')
'@

$gsplatTestScript | Out-File -FilePath "$env:TEMP\test_gsplat.py" -Encoding UTF8
python "$env:TEMP\test_gsplat.py" 2>&1 | Out-Host
Remove-Item "$env:TEMP\test_gsplat.py" -ErrorAction SilentlyContinue

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT: The environment now automatically sets up CUDA and MSVC" -ForegroundColor Yellow
Write-Host "when activated. This enables gsplat JIT compilation." -ForegroundColor Yellow
Write-Host ""
Write-Host "To use the environment:" -ForegroundColor Cyan
Write-Host "  1. Open a NEW terminal (cmd.exe recommended for activation hooks)" -ForegroundColor White
Write-Host "  2. Run: conda activate airsplatmap" -ForegroundColor White
Write-Host "  3. Run your commands normally" -ForegroundColor White
Write-Host ""

# ============================================
# Install Depth Anything V3 (DA3)
# ============================================
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Installing Depth Anything V3 (DA3)..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$da3Path = "$scriptDir\submodules\Depth-Anything-3"

if (Test-Path "$da3Path\pyproject.toml") {
    Write-Host "Found DA3 at: $da3Path" -ForegroundColor Green
    
    # Install xformers first (required for DA3 attention) - must match torch version
    # Use xformers 0.0.27 which is compatible with torch 2.3.1
    Write-Host "Installing xformers (compatible with torch 2.3.1)..." -ForegroundColor Cyan
    $ErrorActionPreference = "Continue"
    pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121 2>&1 | Out-Host
    
    # Install addict (missing from DA3 pyproject.toml but required)
    pip install addict 2>&1 | Out-Host
    
    # Install DA3 in editable mode WITH all its dependencies
    Write-Host "Installing DA3 package with all dependencies..." -ForegroundColor Cyan
    pip install -e "$da3Path" 2>&1 | Out-Host
    $ErrorActionPreference = "Stop"
    
    # Verify DA3 installation
    Write-Host ""
    Write-Host "Verifying DA3 installation..." -ForegroundColor Cyan
    $ErrorActionPreference = "Continue"
    python -c "from depth_anything_3.api import DepthAnything3; print('  DA3 import: OK')" 2>&1
    $ErrorActionPreference = "Stop"
} else {
    Write-Host "WARNING: DA3 submodule not found at $da3Path" -ForegroundColor Yellow
    Write-Host "Run: git submodule update --init --recursive" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "All installations complete!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "Available components:" -ForegroundColor Cyan
Write-Host "  - PyTorch with CUDA" -ForegroundColor White
Write-Host "  - Gaussian Splatting (diff-gaussian-rasterization)" -ForegroundColor White
Write-Host "  - gsplat (JIT compilation)" -ForegroundColor White  
Write-Host "  - Depth Anything V3 (DA3)" -ForegroundColor White
Write-Host ""

# ============================================
# Install Apple Depth Pro
# ============================================
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Installing Apple Depth Pro..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

$depthProPath = "$scriptDir\submodules\ml-depth-pro"

if (Test-Path "$depthProPath\setup.py") {
    Write-Host "Found Depth Pro at: $depthProPath" -ForegroundColor Green
    
    $ErrorActionPreference = "Continue"
    # Install Depth Pro in editable mode
    Write-Host "Installing Depth Pro package..." -ForegroundColor Cyan
    pip install -e "$depthProPath" 2>&1 | Out-Host
    $ErrorActionPreference = "Stop"
    
    # Verify Depth Pro installation
    Write-Host ""
    Write-Host "Verifying Depth Pro installation..." -ForegroundColor Cyan
    $ErrorActionPreference = "Continue"
    python -c "import depth_pro; print('  Depth Pro import: OK')" 2>&1
    $ErrorActionPreference = "Stop"
} else {
    Write-Host "WARNING: Depth Pro submodule not found at $depthProPath" -ForegroundColor Yellow
    Write-Host "Run: git submodule update --init --recursive" -ForegroundColor Yellow
}

# ============================================
# Final Step: Reinstall correct PyTorch with CUDA
# (Some packages like gsplat/xformers may have installed CPU-only torch)
# ============================================
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Final: Ensuring correct PyTorch with CUDA..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

$ErrorActionPreference = "Continue"
pip uninstall torch torchvision torchaudio -y 2>&1 | Out-Null
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121 2>&1 | Out-Host

# Reinstall xformers after torch to ensure compatibility
Write-Host "Reinstalling xformers for torch 2.3.1..." -ForegroundColor Cyan
pip install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121 2>&1 | Out-Host
$ErrorActionPreference = "Stop"

# Final verification
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Final Verification..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

$ErrorActionPreference = "Continue"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  CUDA version: {torch.version.cuda}') if torch.cuda.is_available() else print('  WARNING: CUDA not available!')"
python -c "import simple_knn; print('  simple_knn: OK')" 2>&1
python -c "import diff_gaussian_rasterization; print('  diff_gaussian_rasterization: OK')" 2>&1
python -c "import gsplat; print('  gsplat: OK')" 2>&1
python -c "import xformers; print('  xformers: OK')" 2>&1
python -c "from depth_anything_3.api import DepthAnything3; print('  DA3: OK')" 2>&1
python -c "import depth_pro; print('  Depth Pro: OK')" 2>&1
python -c "from src.engines import list_engines; print(f'  Engines: {list(list_engines().keys())}')" 2>&1
$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "All components installed:" -ForegroundColor Cyan
Write-Host "  - PyTorch 2.3.1 with CUDA 12.1" -ForegroundColor White
Write-Host "  - Gaussian Splatting (diff-gaussian-rasterization, simple_knn)" -ForegroundColor White
Write-Host "  - gsplat (JIT compilation)" -ForegroundColor White  
Write-Host "  - xformers 0.0.27" -ForegroundColor White
Write-Host "  - Depth Anything V3 (DA3)" -ForegroundColor White
Write-Host "  - Apple Depth Pro" -ForegroundColor White
Write-Host ""
Write-Host "To use the environment:" -ForegroundColor Yellow
Write-Host "  conda activate airsplatmap" -ForegroundColor White
Write-Host ""

