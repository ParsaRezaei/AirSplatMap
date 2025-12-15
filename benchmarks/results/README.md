# Benchmark Results Viewer

This directory contains benchmark results and a web-based viewer for GitHub Pages.

> 📊 **View Results Online**: [ParsaRezaei.github.io/AirSplatMap](https://ParsaRezaei.github.io/AirSplatMap/)

---

## Viewing Results

### Online (GitHub Pages)

The easiest way to view results is the online viewer:
```
https://ParsaRezaei.github.io/AirSplatMap/
```

### Locally

You can serve the files locally with any static file server:

```bash
# Using Python
cd benchmarks/results
python -m http.server 8000
# Then open http://localhost:8000

# Using Node.js
npx serve .
```

---

## Directory Structure

```
results/
├── index.html              # Main viewer page
├── manifest.json           # Auto-generated benchmark manifest
├── generate_manifest.py    # Script to regenerate manifest
├── jetson/                 # Device: Jetson
│   ├── benchmark_YYYYMMDD_HHMMSS/
│   │   ├── report.html     # Benchmark report
│   │   ├── results.json    # Raw results data
│   │   └── plots/          # Generated plots
│   └── latest -> benchmark_...  # Symlink to latest
├── pastup/                 # Device: pastup
│   └── ...
└── <other-devices>/
```

## Adding New Devices

Simply create a new directory with the device name and add benchmark results following the same structure. Run `generate_manifest.py` to update the manifest.

## Regenerating the Manifest

The manifest must be regenerated when benchmarks are added or removed:

```bash
python generate_manifest.py
```

Or with custom paths:
```bash
python generate_manifest.py --results-dir /path/to/results --output manifest.json
```

## Setting Up GitHub Pages

This project uses GitHub Actions to deploy to GitHub Pages (static HTML, no Jekyll).

### One-time Setup

1. Go to your repository **Settings → Pages**
2. Under "Build and deployment", select **Source: GitHub Actions**
3. That's it! The workflow at `.github/workflows/deploy-pages.yml` handles deployment

### How It Works

- When you push changes to `benchmarks/results/` on `main`, the workflow:
  1. Regenerates `manifest.json` from the current benchmark folders
  2. Uploads the entire `benchmarks/results/` directory
  3. Deploys to GitHub Pages

- You can also trigger a deployment manually from the Actions tab

### Manual Deployment

Go to **Actions → Deploy Benchmark Results to GitHub Pages → Run workflow**

## URL Parameters

The viewer supports URL parameters for deep linking:

- `?device=jetson` - Pre-select a device
- `?device=jetson&benchmark=benchmark_20251208_111826` - Open specific benchmark

Example: `https://ParsaRezaei.github.io/AirSplatMap/?device=jetson&benchmark=latest`

---

## See Also

- [Benchmarks Guide](../../docs/benchmarks.md) - How to run benchmarks
- [benchmarks/README.md](../README.md) - Benchmark suite documentation
- [Getting Started](../../docs/getting_started.md) - Installation and setup
