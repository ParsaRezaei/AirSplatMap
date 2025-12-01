#!/bin/bash
# Pose Estimation Benchmark
# 
# Usage:
#   ./scripts/benchmark_pose.sh                      # Quick test (100 frames, CPU methods)
#   ./scripts/benchmark_pose.sh --full               # Full benchmark (all frames, all methods)
#   ./scripts/benchmark_pose.sh --methods orb flow   # Specific methods
#   ./scripts/benchmark_pose.sh --include-gpu        # Include GPU methods (LoFTR)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find conda
if [ -f ~/miniconda/etc/profile.d/conda.sh ]; then
    source ~/miniconda/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi

conda activate airsplatmap 2>/dev/null || true

# Parse args
FULL=0
for arg in "$@"; do
    if [ "$arg" == "--full" ]; then
        FULL=1
        shift
    fi
done

echo "🎯 Pose Estimation Benchmark"
echo "============================"
echo ""

if [ "$FULL" == "1" ]; then
    echo "Running FULL benchmark (this may take a while)..."
    python "$SCRIPT_DIR/benchmark_pose.py" "$@"
else
    echo "Running QUICK benchmark (100 frames)..."
    echo "Use --full for complete evaluation"
    echo ""
    python "$SCRIPT_DIR/benchmark_pose.py" --max-frames 100 "$@"
fi
