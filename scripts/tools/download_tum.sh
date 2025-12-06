#!/bin/bash
# Download TUM RGB-D Dataset Scenes
# https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download

TUM_DIR="${1:-./datasets/tum}"
mkdir -p "$TUM_DIR"
cd "$TUM_DIR"

echo "Downloading TUM RGB-D scenes to: $TUM_DIR"
echo ""

# Freiburg1 sequences (handheld SLAM, office scenes)
SCENES=(
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz"
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz"
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz"
    "https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"
)

for url in "${SCENES[@]}"; do
    filename=$(basename "$url")
    dirname="${filename%.tgz}"
    
    if [ -d "$dirname" ]; then
        echo "✓ Already exists: $dirname"
        continue
    fi
    
    echo "Downloading: $filename"
    wget -q --show-progress "$url" -O "$filename"
    
    echo "Extracting: $filename"
    tar -xzf "$filename"
    rm "$filename"
    
    echo "✓ Done: $dirname"
    echo ""
done

echo ""
echo "Available TUM scenes:"
ls -d */ 2>/dev/null | sed 's/\///'
