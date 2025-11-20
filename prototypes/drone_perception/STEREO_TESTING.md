# Stereo Pipeline Testing Guide

This guide explains how to test the stereo perception pipeline with synthetic depth maps generated from MiDaS.

## Overview

The stereo test suite allows you to:
- Test stereo pipeline without physical hardware (RealSense/OAK-D)
- Generate synthetic depth maps from existing RGB videos using MiDaS
- Compare monocular vs stereo pipeline performance
- Validate depth accuracy and tracking stability

## Test Data Structure

```
test_data/videos/traffic/
├── 247589_tiny.mp4              # RGB video (monocular)
├── 247589_tiny_depth.mp4        # Generated depth map video
├── 247589_tiny_output.mp4       # Monocular pipeline output
├── 247589_tiny_stereo_output.mp4 # Stereo pipeline output
├── 247589_tiny.log              # Monocular logs
├── 247589_tiny_stereo.log       # Stereo logs
└── 247589_tiny_depth.yaml       # Depth metadata
```

## Quick Start

### 1. Generate Depth Maps

First time only - generate depth maps from RGB videos:

```bash
# Install PyTorch (if not already installed)
pip install torch torchvision

# Generate depth maps for all toll booth videos
python scripts/generate_depth_maps.py --batch toll_booth
```

This will:
- Download MiDaS model (first run only)
- Process each video frame-by-frame
- Generate synthetic depth map videos
- Save metadata YAML files

**Expected time:** ~5-10 minutes for all videos (GPU recommended)

### 2. Run Stereo Test Suite

```bash
# Run stereo pipeline on all test videos
./run_stereo_test_suite.sh
```

This will automatically:
- Check for depth maps (generate if missing)
- Run stereo pipeline on all videos
- Save outputs and logs
- Generate comparison metrics

### 3. Compare Results

```bash
# Compare monocular vs stereo for all videos
python scripts/compare_mono_vs_stereo.py --batch

# Or compare a specific video
python scripts/compare_mono_vs_stereo.py --video-name 247589_tiny
```

## Manual Testing

### Generate Depth for a Single Video

```bash
python scripts/generate_depth_maps.py \
    --input test_data/videos/traffic/247589_tiny.mp4 \
    --output test_data/videos/traffic/247589_tiny_depth.mp4 \
    --model dpt_hybrid \
    --max-depth 15.0 \
    --preview
```

Options:
- `--model`: `dpt_large` (best quality), `dpt_hybrid` (balanced), `midas_small` (fastest)
- `--max-depth`: Maximum depth in meters for normalization
- `--preview`: Show live preview during processing

### Run Stereo Pipeline on Recorded Data

```bash
python examples/stereo_pipeline.py \
    --backend recorded \
    --rgb-video test_data/videos/traffic/247589_tiny.mp4 \
    --depth-video test_data/videos/traffic/247589_tiny_depth.mp4 \
    --model s \
    --classes 0 1 2 3 4 5 6 7 8 9 \
    --save-video output.mp4 \
    --depth-method median
```

Depth extraction methods:
- `center`: Use center point of bounding box
- `median`: Median depth within bbox (most robust)
- `bottom`: Bottom center point (good for ground plane)

## Expected Results

### Stereo Advantages

Compared to monocular pipeline, stereo should show:

✅ **More accurate depth**
- Real metric depth vs. heuristic estimation
- No scale ambiguity

✅ **More stable tracking**
- Lower variance in 3D positions
- Smoother trajectories

✅ **Better velocity estimates**
- Metric accuracy enables better velocity/acceleration
- More consistent speed measurements

### Performance Trade-offs

❗ **Slightly slower**
- Depth processing adds ~10-20% overhead
- Still real-time capable (>20 FPS)

❗ **Hardware requirements**
- Requires stereo camera or depth generation step
- Depth generation requires PyTorch + GPU (recommended)

## Depth Map Quality

### MiDaS Limitations

⚠️ **Relative depth only**
- MiDaS outputs relative (not absolute) depth
- Scale is normalized during generation
- Good for testing pipeline, not ground truth validation

⚠️ **Best for**
- Outdoor scenes with clear depth cues
- Well-lit environments
- Scenes with texture

⚠️ **Challenging for**
- Textureless surfaces (walls, sky)
- Transparent objects (glass, water)
- Low-light conditions

### For Real Hardware Testing

To test with actual stereo cameras:

```bash
# RealSense D435
python examples/stereo_pipeline.py --backend realsense

# OAK-D
python examples/stereo_pipeline.py --backend oakd

# Validation with known distances
python scripts/validate_stereo_accuracy.py --backend realsense --interactive
```

## Troubleshooting

### PyTorch Installation

If you get errors about missing PyTorch:

```bash
# CPU only (faster install)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# With CUDA (for GPU acceleration)
pip install torch torchvision
```

### Depth Generation Too Slow

Use smaller/faster model:

```bash
python scripts/generate_depth_maps.py \
    --batch toll_booth \
    --model midas_small  # Faster but less accurate
```

### Out of Memory

Reduce video resolution or process in chunks:

```bash
# Resize video first
ffmpeg -i input.mp4 -vf scale=640:480 input_small.mp4

# Then generate depth
python scripts/generate_depth_maps.py --input input_small.mp4
```

## Pre-generated Depth Maps

If you don't want to install PyTorch, you can download pre-generated depth maps:

```bash
# TODO: Add download link when available
wget https://example.com/depth_maps.tar.gz
tar -xzf depth_maps.tar.gz -C test_data/videos/traffic/
```

## Next Steps

- **Phase 3**: Add HDF5 recording to save stereo data streams
- **Phase 4**: Integrate with real LiDAR for ground truth comparison
- **Validation**: Compare with real RealSense depth measurements

## References

- MiDaS: https://github.com/isl-org/MiDaS
- Intel RealSense: https://github.com/IntelRealSense/librealsense
- OAK-D: https://docs.luxonis.com/
