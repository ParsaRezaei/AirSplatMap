# Datasets

Place your datasets here for the dashboard to detect them.

## TUM RGB-D Dataset

Download TUM RGB-D sequences and extract them here. Each sequence folder should contain:
- `rgb/` - RGB images
- `depth/` - Depth images  
- `rgb.txt` - RGB image timestamps and paths
- `depth.txt` - Depth image timestamps and paths
- `groundtruth.txt` - Ground truth poses (optional)

### Download Script (Linux/WSL)

```bash
# Download common TUM scenes
./scripts/tools/download_tum.sh ./datasets/tum
```

### Manual Download

1. Visit: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download
2. Download desired sequences (e.g., `rgbd_dataset_freiburg1_desk.tgz`)
3. Extract to `datasets/` or `datasets/tum/`

### Expected Structure

```
datasets/
├── tum/
│   ├── rgbd_dataset_freiburg1_desk/
│   │   ├── rgb/
│   │   ├── depth/
│   │   ├── rgb.txt
│   │   └── depth.txt
│   └── rgbd_dataset_freiburg1_room/
│       └── ...
└── README.md
```

Or directly in datasets folder:
```
datasets/
├── rgbd_dataset_freiburg1_desk/
│   ├── rgb/
│   ├── depth/
│   ├── rgb.txt
│   └── depth.txt
└── ...
```

## Live Sources

You can also add live sources (RTSP, webcam, video files) directly from the dashboard UI without downloading datasets.
