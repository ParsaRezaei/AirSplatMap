# AirSplatMap Papers

This directory contains two CVPR-format academic papers for the AirSplatMap project.

## Directory Structure

```
papers/
├── deep_learning/          # Paper 1: Deep Learning (3DGS focus)
│   ├── main.tex           # Main document
│   ├── cvpr.sty           # CVPR style file
│   ├── preamble.tex       # LaTeX preamble
│   ├── sections/          # Paper sections
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── related_work.tex
│   │   ├── approach.tex
│   │   ├── experiments.tex
│   │   └── conclusion.tex
│   └── figures/           # Generated figures (PDF)
│
├── computer_vision/        # Paper 2: Computer Vision (Depth/Pose focus)
│   ├── main.tex
│   ├── cvpr.sty
│   ├── preamble.tex
│   ├── sections/
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── related_work.tex
│   │   ├── approach.tex
│   │   ├── experiments.tex
│   │   └── conclusion.tex
│   └── figures/
│
├── shared/                 # Shared resources
│   ├── references.bib     # Bibliography (40+ references)
│   └── cvpr/              # Official CVPR 2025 template
│
└── scripts/
    └── generate_figures.py # Figure generation script
```

## Paper 1: Deep Learning (3DGS Focus)

**Title**: AirSplatMap: A Modular Real-Time Pipeline for Learning-Based 3D Gaussian Splatting

**Focus**:
- 3D Gaussian representation and rendering
- L1 + λSSIM loss function and optimization
- Unified engine architecture (GraphDeco, gsplat, MonoGS, SplaTAM, Gaussian-SLAM)
- Degradation analysis: pose errors cause 2.5× more quality loss than depth errors
- Edge deployment on Jetson Orin (19.6 FPS render, 1.43 FPS train)

## Paper 2: Computer Vision (Depth/Pose Focus)

**Title**: AirSplatMap: A Comprehensive Evaluation of Depth and Pose Estimation for Real-Time 3D Reconstruction

**Focus**:
- 11 visual odometry methods (ORB, SIFT, optical flow, LoFTR, SuperPoint, LightGlue, R2D2, RoMa, RAFT)
- 4 monocular depth estimators (MiDaS, Depth Anything V2/V3, Depth Pro)
- Accuracy vs speed Pareto analysis
- Cross-platform evaluation (Desktop vs Jetson)
- Downstream impact on 3D reconstruction quality

## Building the Papers

### Prerequisites

```bash
# Install LaTeX (Ubuntu/Debian)
sudo apt-get install texlive-full

# Or minimal install
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

### Compile

```bash
# Deep Learning paper
cd deep_learning
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Computer Vision paper
cd ../computer_vision
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using latexmk (recommended)

```bash
cd deep_learning
latexmk -pdf main.tex

cd ../computer_vision
latexmk -pdf main.tex
```

## Generating Figures

The figure generation script creates CVPR-style plots:

```bash
cd scripts

# Install dependencies
pip install matplotlib numpy

# Generate all figures
python generate_figures.py --paper both

# Generate figures for specific paper
python generate_figures.py --paper dl --output ../deep_learning/figures/
python generate_figures.py --paper cv --output ../computer_vision/figures/
```

## Key Results

### Deep Learning Paper

| Engine | PSNR (dB) | SSIM | FPS |
|--------|-----------|------|-----|
| GraphDeco | **15.85** | **0.560** | 20.7 |
| gsplat | 4.07 | 0.018 | 52.7 |
| MonoGS | 6.68 | 0.048 | 8.6 |

### Computer Vision Paper

| Pose Method | ATE (m) | FPS | Category |
|-------------|---------|-----|----------|
| R2D2 | **0.069** | 7.8 | Learned |
| robust_flow | 0.075 | **39.5** | Classical |
| ORB | 0.102 | 31.4 | Classical |

### Platform Comparison (Jetson Orin)

| Component | Desktop FPS | Jetson FPS | Real-time? |
|-----------|-------------|------------|------------|
| Pose (robust_flow) | 39.5 | 13.4 | ✅ |
| Pose (SIFT) | 14.5 | 4.2 | ❌ |
| Depth (MiDaS) | 20.6 | 4.8 | ❌ |
| GS Training | 20.7 | 1.4 | ❌ |
| GS Rendering | - | 19.6 | ✅ |

## Online Resources

- **Interactive Benchmark Viewer**: https://ParsaRezaei.github.io/AirSplatMap/
- **Source Code**: https://github.com/ParsaRezaei/AirSplatMap

## Citation

```bibtex
@inproceedings{airsplatmap2025dl,
  title={AirSplatMap: A Modular Real-Time Pipeline for Learning-Based 3D Gaussian Splatting},
  author={Author One and Author Two},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@inproceedings{airsplatmap2025cv,
  title={AirSplatMap: A Comprehensive Evaluation of Depth and Pose Estimation for Real-Time 3D Reconstruction},
  author={Author One},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

## CVPR Formatting Guidelines

Both papers follow the official CVPR 2025 format:
- Minimum 9 pages (excluding references)
- Two-column layout
- 10pt font
- All figures are original (generated from benchmark data)
- References use IEEE format

## Notes

1. **Architecture diagrams** in `figures/architecture.pdf` are placeholders. Create manually using:
   - draw.io / diagrams.net (recommended)
   - TikZ in LaTeX
   - Adobe Illustrator / Inkscape

2. **Contribution statements** are split between two authors for the DL paper. Adjust names and contributions as needed.

3. **For drone deployment context**: The papers emphasize that drones provide their own pose via GPS/IMU, eliminating the largest source of error (pose estimation).
