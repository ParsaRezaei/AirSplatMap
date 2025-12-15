# Datasets

Place your datasets here for the dashboard and benchmarks to detect them.

> 📖 **Related Documentation**:
> - [Getting Started](../docs/getting_started.md) - Installation guide
> - [Benchmarks Guide](../docs/benchmarks.md) - Running evaluations
> - [Interactive Results](https://ParsaRezaei.github.io/AirSplatMap/) - View benchmark results

---

## Supported Datasets

All datasets must have:
- **RGB images** - Color frames
- **Depth images** - Depth maps (for RGB-D datasets)
- **Ground truth poses** - Camera trajectories (for benchmarking)

---

## 1. TUM RGB-D Dataset

The standard benchmark dataset for RGB-D SLAM. Real-world indoor scenes captured with Kinect/Xtion sensors.

**Website:** [cvg.cit.tum.de/data/datasets/rgbd-dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)

### Quick Download

```bash
# Download all TUM scenes (~15GB)
./scripts/tools/download_datasets.sh ./datasets tum all

# Download standard benchmark set (~6GB)
./scripts/tools/download_datasets.sh ./datasets tum

# Download minimal set for testing (~1.5GB)
./scripts/tools/download_datasets.sh ./datasets tum minimal

# Download specific scenes
./scripts/tools/download_datasets.sh ./datasets tum fr1_xyz,fr1_desk,fr2_desk
```

### Available Sequences

| Category | Sequences | Description |
|----------|-----------|-------------|
| **Testing** | fr1_xyz, fr1_rpy, fr2_xyz, fr2_rpy | Simple movements for debugging |
| **Handheld SLAM** | fr1_desk, fr1_desk2, fr1_room, fr1_360, fr1_floor | Office/room scenes |
| **Robot SLAM** | fr2_desk, fr2_pioneer_360, fr2_pioneer_slam | Larger scenes, robot mounted |
| **Structure/Texture** | fr3_str_tex_far, fr3_str_tex_near, fr3_nstr_tex_near | Challenging scenes |
| **3D Objects** | fr1_plant, fr1_teddy, fr3_teddy, fr3_cabinet | Object reconstruction |
| **Dynamic** | fr2_desk_person, fr3_sitting_xyz, fr3_walking_xyz | Moving people |
| **Long Trajectory** | fr3_long_office_household | Large loop closure |

### Data Format

```
rgbd_dataset_freiburg1_desk/
├── rgb/                    # RGB images (640x480, PNG)
│   ├── 1305031102.175304.png
│   └── ...
├── depth/                  # Depth images (640x480, 16-bit PNG)
│   ├── 1305031102.160407.png
│   └── ...
├── rgb.txt                 # Timestamps + RGB paths
├── depth.txt               # Timestamps + depth paths
├── groundtruth.txt         # GT poses (timestamp tx ty tz qx qy qz qw)
└── accelerometer.txt       # IMU data (optional)
```

---

## 2. ICL-NUIM Dataset

Synthetic RGB-D dataset with **perfect** ground truth. Ideal for baseline accuracy testing.

**Website:** https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html

### Quick Download

```bash
# Download standard ICL-NUIM sequences
./scripts/tools/download_datasets.sh ./datasets icl

# Download all sequences
./scripts/tools/download_datasets.sh ./datasets icl all
```

### Available Sequences

| Scene | Trajectories | Description |
|-------|-------------|-------------|
| Living Room | lr_kt0, lr_kt1, lr_kt2, lr_kt3 | Synthetic living room |
| Office | of_kt0, of_kt1, of_kt2, of_kt3 | Synthetic office |

---

## 3. Replica Dataset

High-quality synthetic indoor reconstructions. Photorealistic rendering.

**Website:** https://github.com/facebookresearch/Replica-Dataset

### Quick Download

```bash
# Download Replica (NICE-SLAM version, ~5GB)
./scripts/tools/download_datasets.sh ./datasets replica
```

### Available Scenes

- office0, office1, office2, office3, office4
- room0, room1, room2

---

## 4. Microsoft 7-Scenes Dataset

RGB-D dataset with KinectFusion ground truth poses. Indoor scenes.

**Website:** https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/

### Quick Download

```bash
# Download default scenes (chess, fire, office, redkitchen)
./scripts/tools/download_datasets.sh ./datasets 7scenes

# Download all 7 scenes (~17GB)
./scripts/tools/download_datasets.sh ./datasets 7scenes all
```

### Available Scenes

| Scene | Size | Description |
|-------|------|-------------|
| chess | 1GB | Chess board and pieces |
| fire | 1GB | Fire extinguisher area |
| heads | 0.5GB | Heads/busts display |
| office | 2GB | Office desk scene |
| pumpkin | 2GB | Pumpkin on table |
| redkitchen | 4GB | Kitchen scene (best for GS) |
| stairs | 2GB | Staircase |

### Data Format

```
chess/
├── seq-01/
│   ├── frame-000000.color.png   # RGB (640x480)
│   ├── frame-000000.depth.png   # Depth (16-bit, millimeters)
│   └── frame-000000.pose.txt    # 4x4 camera-to-world matrix
├── seq-02/
└── ...
```

---

## 5. ScanNet Dataset

Large-scale real-world RGB-D scans with semantic annotations.

**Website:** http://www.scan-net.org/

### Access

ScanNet requires accepting a license agreement:
1. Visit http://www.scan-net.org/
2. Fill out the Terms of Use
3. Receive download script via email

---

## Download All Datasets

```bash
# Download all supported datasets (TUM + Replica + 7-Scenes + ICL-NUIM)
./scripts/tools/download_datasets.sh ./datasets all
```

---

## Expected Directory Structure

```
datasets/
├── tum/
│   ├── rgbd_dataset_freiburg1_xyz/
│   ├── rgbd_dataset_freiburg1_desk/
│   ├── rgbd_dataset_freiburg1_desk2/
│   ├── rgbd_dataset_freiburg1_room/
│   ├── rgbd_dataset_freiburg2_desk/
│   └── rgbd_dataset_freiburg3_long_office_household/
├── icl_nuim/
│   ├── living_room_traj0_frei_png/
│   ├── living_room_traj1_frei_png/
│   ├── office_room_traj0_frei_png/
│   └── office_room_traj1_frei_png/
├── replica/
│   ├── office0/
│   ├── office1/
│   ├── room0/
│   └── room1/
└── 7scenes/
    ├── chess/
    ├── fire/
    ├── office/
    └── redkitchen/
```

---

## Recommended Sequences for Gaussian Splatting

For best Gaussian Splatting results, use sequences with:
- Good texture (avoid plain walls)
- Static scenes (no moving people/objects)
- Complete coverage (full room/object views)

### Best TUM Sequences for GS

| Sequence | Type | Why Good for GS |
|----------|------|-----------------|
| fr1_desk | Office | Good texture, complete desk view |
| fr1_desk2 | Office | Alternative angle, good coverage |
| fr1_room | Room | Full room with loop closure |
| fr1_plant | Object | 360° object reconstruction |
| fr1_teddy | Object | Complete object coverage |
| fr2_desk | Office | High quality, slow motion |
| fr3_long_office | Office | Large scene, good texture |
| fr3_str_tex_far | Synthetic | Strong structure + texture |

### Best ICL-NUIM for GS

All ICL-NUIM sequences work well as they have perfect GT and consistent textures.

### Best Replica for GS

- office0, office1 - Good office scenes
- room0, room1 - Apartment rooms with furniture

---

## Live Sources

You can also add live sources (RTSP, webcam, video files) directly from the dashboard UI without downloading datasets.

---

## See Also

- [Getting Started](../docs/getting_started.md) - Installation and first run
- [Benchmarks Guide](../docs/benchmarks.md) - Running evaluations
- [Dashboard Guide](../docs/dashboard.md) - Web dashboard usage
- [Interactive Benchmark Results](https://ParsaRezaei.github.io/AirSplatMap/)
