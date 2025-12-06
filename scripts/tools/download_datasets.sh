#!/bin/bash
# Download Benchmark Datasets for Gaussian Splatting
# Supports: TUM RGB-D, Replica, 7-Scenes, ICL-NUIM
# All datasets have RGB color images + depth + ground truth poses

set -e

DATASETS_DIR="${1:-./datasets}"
mkdir -p "$DATASETS_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}→ $1${NC}"
}

# ============================================
# TUM RGB-D Dataset
# https://cvg.cit.tum.de/data/datasets/rgbd-dataset
# ============================================
download_tum() {
    print_header "TUM RGB-D Dataset"
    
    TUM_DIR="$DATASETS_DIR/tum"
    mkdir -p "$TUM_DIR"
    cd "$TUM_DIR"
    
    # Available TUM sequences
    declare -A TUM_SCENES=(
        # Freiburg1 - Handheld SLAM, office/desk scenes
        ["fr1_xyz"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
        ["fr1_desk"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz"
        ["fr1_desk2"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz"
        ["fr1_room"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_room.tgz"
        ["fr1_360"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_360.tgz"
        ["fr1_floor"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_floor.tgz"
        ["fr1_plant"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_plant.tgz"
        ["fr1_teddy"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_teddy.tgz"
        
        # Freiburg2 - Robot SLAM, larger office
        ["fr2_xyz"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_xyz.tgz"
        ["fr2_desk"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz"
        ["fr2_desk_person"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk_with_person.tgz"
        ["fr2_large_no_loop"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_large_no_loop.tgz"
        ["fr2_pioneer_360"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_360.tgz"
        ["fr2_pioneer_slam"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_pioneer_slam.tgz"
        
        # Freiburg3 - Structure vs texture, large office
        ["fr3_office"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz"
        ["fr3_nstr_tex_near"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_nostructure_texture_near_withloop.tgz"
        ["fr3_str_notex_far"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_structure_notexture_far.tgz"
        ["fr3_str_tex_far"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_structure_texture_far.tgz"
        ["fr3_sitting_xyz"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_xyz.tgz"
        ["fr3_sitting_halfsphere"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_sitting_halfsphere.tgz"
        ["fr3_walking_xyz"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz"
        ["fr3_cabinet"]="https://cvg.cit.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_cabinet.tgz"
    )
    
    # Default selection (commonly used for benchmarks)
    DEFAULT_TUM=("fr1_xyz" "fr1_desk" "fr1_desk2" "fr1_room" "fr2_xyz" "fr2_desk" "fr3_office")
    
    # Parse arguments
    if [ "$2" == "all" ]; then
        SELECTED=("${!TUM_SCENES[@]}")
    elif [ "$2" == "minimal" ]; then
        SELECTED=("fr1_xyz" "fr1_desk2" "fr3_office")
    elif [ -n "$2" ]; then
        IFS=',' read -ra SELECTED <<< "$2"
    else
        SELECTED=("${DEFAULT_TUM[@]}")
    fi
    
    echo "Downloading TUM RGB-D scenes to: $TUM_DIR"
    echo "Selected scenes: ${SELECTED[*]}"
    echo ""
    
    for scene in "${SELECTED[@]}"; do
        url="${TUM_SCENES[$scene]}"
        if [ -z "$url" ]; then
            print_warning "Unknown scene: $scene (skipping)"
            continue
        fi
        
        filename=$(basename "$url")
        dirname="${filename%.tgz}"
        
        if [ -d "$dirname" ]; then
            print_success "Already exists: $dirname"
            continue
        fi
        
        print_info "Downloading: $scene"
        wget -q --show-progress "$url" -O "$filename"
        
        print_info "Extracting: $filename"
        tar -xzf "$filename"
        rm "$filename"
        
        print_success "Done: $dirname"
        echo ""
    done
    
    cd - > /dev/null
}

# ============================================
# Replica Dataset (NICE-SLAM version)
# High-quality synthetic indoor scenes
# ============================================
download_replica() {
    print_header "Replica Dataset"
    
    REPLICA_DIR="$DATASETS_DIR/replica"
    
    if [ -d "$REPLICA_DIR" ] && [ "$(ls -A $REPLICA_DIR 2>/dev/null)" ]; then
        print_success "Replica dataset already exists at: $REPLICA_DIR"
        return
    fi
    
    mkdir -p "$DATASETS_DIR"
    cd "$DATASETS_DIR"
    
    print_info "Downloading Replica dataset (NICE-SLAM version)..."
    print_info "This may take a while (~5GB)..."
    
    wget -q --show-progress https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip -O Replica.zip
    
    print_info "Extracting..."
    unzip -q Replica.zip
    mv Replica replica
    rm Replica.zip
    
    print_success "Replica dataset downloaded to: $REPLICA_DIR"
    echo ""
    echo "Available scenes:"
    ls -d replica/*/ 2>/dev/null | xargs -n1 basename
    
    cd - > /dev/null
}

# ============================================
# Microsoft 7-Scenes Dataset
# RGB-D with KinectFusion ground truth poses
# https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
# ============================================
download_7scenes() {
    print_header "Microsoft 7-Scenes Dataset"
    
    SCENES_DIR="$DATASETS_DIR/7scenes"
    mkdir -p "$SCENES_DIR"
    cd "$SCENES_DIR"
    
    # Available 7-Scenes (all have RGB + Depth + GT poses)
    declare -A SEVEN_SCENES=(
        ["chess"]="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip"
        ["fire"]="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/fire.zip"
        ["heads"]="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/heads.zip"
        ["office"]="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/office.zip"
        ["pumpkin"]="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/pumpkin.zip"
        ["redkitchen"]="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/redkitchen.zip"
        ["stairs"]="http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/stairs.zip"
    )
    
    # Default selection (good for GS: textured indoor scenes)
    DEFAULT_7SCENES=("chess" "fire" "office" "redkitchen")
    
    # Parse arguments
    if [ "$2" == "all" ]; then
        SELECTED=("${!SEVEN_SCENES[@]}")
    elif [ "$2" == "minimal" ]; then
        SELECTED=("chess")
    elif [ -n "$2" ]; then
        IFS=',' read -ra SELECTED <<< "$2"
    else
        SELECTED=("${DEFAULT_7SCENES[@]}")
    fi
    
    echo "Downloading 7-Scenes to: $SCENES_DIR"
    echo "Selected scenes: ${SELECTED[*]}"
    echo ""
    echo "Note: Each scene is 1-4 GB. Total ~17GB for all scenes."
    echo ""
    
    for scene in "${SELECTED[@]}"; do
        url="${SEVEN_SCENES[$scene]}"
        if [ -z "$url" ]; then
            print_warning "Unknown scene: $scene (skipping)"
            continue
        fi
        
        # Check if scene already fully extracted (has seq-XX directories with frame files)
        if [ -d "$scene" ]; then
            seq_count=$(find "$scene" -maxdepth 1 -type d -name "seq-*" 2>/dev/null | wc -l)
            if [ "$seq_count" -gt 0 ]; then
                # Check if at least one seq dir has frame files (not just zips)
                frame_count=$(find "$scene/seq-01" -name "frame-*.color.png" 2>/dev/null | head -1)
                if [ -n "$frame_count" ]; then
                    print_success "Already exists: $scene"
                    continue
                fi
            fi
        fi
        
        # Download if zip doesn't exist
        if [ ! -f "${scene}.zip" ]; then
            print_info "Downloading: $scene"
            wget -q --show-progress "$url" -O "${scene}.zip"
        fi
        
        print_info "Extracting: ${scene}.zip"
        unzip -q -o "${scene}.zip"
        rm -f "${scene}.zip"
        
        # 7-Scenes has nested seq-XX.zip files that need extraction
        if [ -d "$scene" ]; then
            cd "$scene"
            for seq_zip in seq-*.zip; do
                if [ -f "$seq_zip" ]; then
                    seq_name="${seq_zip%.zip}"
                    print_info "  Extracting sequence: $seq_name"
                    unzip -q -o "$seq_zip"
                    rm -f "$seq_zip"
                fi
            done
            cd ..
        fi
        
        print_success "Done: $scene"
        echo ""
    done
    
    cd - > /dev/null
    
    echo ""
    echo "7-Scenes format:"
    echo "  Each frame has: color.png, depth.png, pose.txt"
    echo "  Depth: 16-bit PNG in millimeters"
    echo "  Pose: 4x4 camera-to-world matrix"
}

# ============================================
# ICL-NUIM Dataset
# Synthetic RGB-D dataset with perfect ground truth
# ============================================
download_icl_nuim() {
    print_header "ICL-NUIM Dataset"
    
    ICL_DIR="$DATASETS_DIR/icl_nuim"
    mkdir -p "$ICL_DIR"
    cd "$ICL_DIR"
    
    # ICL-NUIM sequences (living room and office)
    declare -A ICL_SCENES=(
        ["lr_kt0"]="https://www.doc.ic.ac.uk/~ahanda/living_room_traj0_frei_png.tar.gz"
        ["lr_kt1"]="https://www.doc.ic.ac.uk/~ahanda/living_room_traj1_frei_png.tar.gz"
        ["lr_kt2"]="https://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz"
        ["lr_kt3"]="https://www.doc.ic.ac.uk/~ahanda/living_room_traj3_frei_png.tar.gz"
        ["of_kt0"]="https://www.doc.ic.ac.uk/~ahanda/office_room_traj0_frei_png.tar.gz"
        ["of_kt1"]="https://www.doc.ic.ac.uk/~ahanda/office_room_traj1_frei_png.tar.gz"
        ["of_kt2"]="https://www.doc.ic.ac.uk/~ahanda/office_room_traj2_frei_png.tar.gz"
        ["of_kt3"]="https://www.doc.ic.ac.uk/~ahanda/office_room_traj3_frei_png.tar.gz"
    )
    
    DEFAULT_ICL=("lr_kt0" "lr_kt1" "of_kt0" "of_kt1")
    
    if [ "$2" == "all" ]; then
        SELECTED=("${!ICL_SCENES[@]}")
    elif [ "$2" == "minimal" ]; then
        SELECTED=("lr_kt0")
    elif [ -n "$2" ]; then
        IFS=',' read -ra SELECTED <<< "$2"
    else
        SELECTED=("${DEFAULT_ICL[@]}")
    fi
    
    echo "Downloading ICL-NUIM scenes to: $ICL_DIR"
    echo "Selected scenes: ${SELECTED[*]}"
    echo ""
    
    for scene in "${SELECTED[@]}"; do
        url="${ICL_SCENES[$scene]}"
        if [ -z "$url" ]; then
            print_warning "Unknown scene: $scene (skipping)"
            continue
        fi
        
        filename=$(basename "$url")
        
        # Check if already exists
        if [ -d "living_room_traj${scene:3:1}_frei_png" ] || [ -d "office_room_traj${scene:3:1}_frei_png" ]; then
            print_success "Already exists: $scene"
            continue
        fi
        
        print_info "Downloading: $scene"
        wget -q --show-progress "$url" -O "$filename"
        
        print_info "Extracting: $filename"
        tar -xzf "$filename"
        rm "$filename"
        
        print_success "Done: $scene"
        echo ""
    done
    
    cd - > /dev/null
}

# ============================================
# ScanNet Dataset (requires agreement)
# Real-world RGB-D scans
# ============================================
download_scannet_info() {
    print_header "ScanNet Dataset"
    
    echo "ScanNet requires accepting a license agreement."
    echo ""
    echo "To download ScanNet:"
    echo "1. Go to: http://www.scan-net.org/"
    echo "2. Fill out the Terms of Use agreement"
    echo "3. You will receive a download script via email"
    echo ""
    echo "After receiving access, place scenes in: $DATASETS_DIR/scannet/"
    echo ""
    echo "Recommended scenes for benchmarking:"
    echo "  - scene0000_00, scene0059_00, scene0106_00"
    echo "  - scene0169_00, scene0181_00, scene0207_00"
}

# ============================================
# Print Usage
# ============================================
print_usage() {
    echo "AirSplatMap Dataset Downloader"
    echo ""
    echo "Usage: $0 [DATASETS_DIR] <command> [options]"
    echo ""
    echo "Commands:"
    echo "  tum [scenes]      Download TUM RGB-D dataset (real indoor scenes)"
    echo "  7scenes [scenes]  Download Microsoft 7-Scenes (RGB-D with GT poses)"
    echo "  replica           Download Replica dataset (synthetic, NICE-SLAM version)"
    echo "  icl [scenes]      Download ICL-NUIM dataset (synthetic with perfect GT)"
    echo "  scannet           Show ScanNet download instructions"
    echo "  all               Download all datasets (default selection)"
    echo "  list              List available scenes"
    echo ""
    echo "Scene options:"
    echo "  all               Download all available scenes"
    echo "  minimal           Download minimal set for testing"
    echo "  scene1,scene2     Comma-separated list of scenes"
    echo ""
    echo "Examples:"
    echo "  $0 ./datasets tum                    # Default TUM scenes"
    echo "  $0 ./datasets tum all                # All TUM scenes"
    echo "  $0 ./datasets tum fr1_xyz,fr2_desk   # Specific scenes"
    echo "  $0 ./datasets replica                # Replica dataset"
    echo "  $0 ./datasets 7scenes                # 7-Scenes (chess, fire, office, redkitchen)"
    echo "  $0 ./datasets 7scenes all            # All 7-Scenes (~17GB)"
    echo "  $0 ./datasets all                    # All datasets"
    echo ""
}

list_scenes() {
    print_header "Available Scenes"
    
    echo "TUM RGB-D Dataset:"
    echo "  Freiburg1: fr1_xyz, fr1_desk, fr1_desk2, fr1_room, fr1_360, fr1_floor, fr1_plant, fr1_teddy"
    echo "  Freiburg2: fr2_xyz, fr2_desk, fr2_desk_person, fr2_large_no_loop, fr2_pioneer_360, fr2_pioneer_slam"
    echo "  Freiburg3: fr3_office, fr3_nstr_tex_near, fr3_str_notex_far, fr3_str_tex_far, fr3_sitting_xyz, fr3_walking_xyz, fr3_cabinet"
    echo ""
    echo "Replica Dataset (synthetic):"
    echo "  office0, office1, office2, office3, office4"
    echo "  room0, room1, room2"
    echo ""
    echo "7-Scenes Dataset (RGB-D with KinectFusion GT):"
    echo "  chess, fire, heads, office, pumpkin, redkitchen, stairs"
    echo "  Best for GS: chess, office, redkitchen (good texture)"
    echo ""
    echo "ICL-NUIM Dataset (synthetic):"
    echo "  Living Room: lr_kt0, lr_kt1, lr_kt2, lr_kt3"
    echo "  Office: of_kt0, of_kt1, of_kt2, of_kt3"
    echo ""
}

# ============================================
# Main
# ============================================

# Handle first argument as directory or command
if [[ "$1" == -* ]] || [ -z "$1" ]; then
    print_usage
    exit 0
fi

# Check if first arg is a directory path
if [[ "$1" == ./* ]] || [[ "$1" == /* ]] || [[ "$1" == *datasets* ]]; then
    DATASETS_DIR="$1"
    shift
fi

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
    tum)
        download_tum "$DATASETS_DIR" "$1"
        ;;
    replica)
        download_replica "$DATASETS_DIR"
        ;;
    7scenes|7-scenes|seven-scenes)
        download_7scenes "$DATASETS_DIR" "$1"
        ;;
    icl|icl_nuim|icl-nuim)
        download_icl_nuim "$DATASETS_DIR" "$1"
        ;;
    scannet)
        download_scannet_info
        ;;
    all)
        download_tum "$DATASETS_DIR" ""
        download_replica "$DATASETS_DIR"
        download_7scenes "$DATASETS_DIR" ""
        download_icl_nuim "$DATASETS_DIR" ""
        ;;
    list)
        list_scenes
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac

print_header "Done!"
echo "Datasets directory: $DATASETS_DIR"
echo ""
echo "Current datasets:"
ls -d "$DATASETS_DIR"/*/ 2>/dev/null | xargs -n1 basename || echo "  (none)"
