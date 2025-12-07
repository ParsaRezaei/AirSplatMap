# AirSplatMap Environment Setup Script for Windows
# =================================================
# This script creates the conda environment and installs PyTorch with CUDA support.
#
# Usage (run in PowerShell):
#   .\setup_env.ps1
#
# For Windows with NVIDIA GPU and CUDA 12.4+
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

# Activate the environment
Write-Host "Activating environment..." -ForegroundColor Cyan
conda activate airsplatmap

# Uninstall any CPU-only PyTorch that may have been installed as a dependency
Write-Host "Replacing CPU-only PyTorch with CUDA-enabled version..." -ForegroundColor Yellow
pip uninstall torch torchvision torchaudio -y 2>$null

# Install PyTorch with CUDA from the cu124 index
# For Windows x86_64, cu124 provides CUDA-enabled wheels
Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Cyan
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install PyTorch" -ForegroundColor Red
    exit 1
}

# Verify PyTorch installation
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Verifying PyTorch installation..." -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan

python -c @"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Device: {torch.cuda.get_device_name(0)}')
else:
    print('WARNING: CUDA not available! Check your NVIDIA drivers.')
"@

Write-Host ""
Write-Host "==============================================" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment, run:" -ForegroundColor Yellow
Write-Host "  conda activate airsplatmap" -ForegroundColor White
Write-Host ""
