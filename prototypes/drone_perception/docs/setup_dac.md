# Depth Any Camera (DAC) Setup Guide

This guide explains how to integrate Depth Any Camera for wide-angle/fisheye depth estimation in the drone perception pipeline.

## Overview

**Depth Any Camera (DAC)** is a CVPR 2025 model that provides:
- ✅ Zero-shot metric depth estimation on any camera type
- ✅ Fisheye and wide-angle camera support (180°-200°+ FOV)
- ✅ No camera-specific training required
- ✅ Dense depth maps from monocular images

Perfect for Skydio-style wide-angle stereo cameras!

## Installation

### Step 1: Clone DAC Repository

```bash
cd prototypes/drone_perception
mkdir -p third_party
cd third_party

# Clone Depth Any Camera
git clone https://github.com/yuliangguo/depth_any_camera.git
cd depth_any_camera
```

### Step 2: Install Dependencies

**Using Conda (Recommended):**

```bash
# Create environment
conda create -n dac python=3.9 -y
conda activate dac

# Install PyTorch (adjust CUDA version as needed)
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 \
    --extra-index-url https://download.pytorch.org/whl/cu116

# Install DAC requirements
pip install -r requirements.txt

# Build custom ops
cd dac/models/ops/
pip install -e .
cd ../../..
```

**Using pip (Alternative):**

```bash
# Install in your existing venv
pip install torch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1
pip install timm>=0.9.12 einops>=0.7.0 transformers>=4.30.0 huggingface-hub>=0.20.0

# Clone and install DAC
cd third_party/depth_any_camera
pip install -r requirements.txt
cd dac/models/ops/
pip install -e .
```

### Step 3: Download Pre-trained Models

DAC provides several pre-trained models:

```bash
cd prototypes/drone_perception
mkdir -p checkpoints

# Download from Hugging Face
# Option 1: Using huggingface-cli
huggingface-cli download yuliangguo/depth-any-camera \
    dac_swinl_indoor.pt dac_swinl_indoor.json \
    --local-dir checkpoints/

# Option 2: Using Python
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='yuliangguo/depth-any-camera',
                filename='dac_swinl_indoor.pt',
                local_dir='checkpoints/')
hf_hub_download(repo_id='yuliangguo/depth-any-camera',
                filename='dac_swinl_indoor.json',
                local_dir='checkpoints/')
"

# Option 3: Manual download
# Visit: https://huggingface.co/yuliangguo/depth-any-camera
# Download both .pt and .json files to checkpoints/
```

**Available Models:**

| Model | Dataset | Use Case | Size |
|-------|---------|----------|------|
| `dac_swinl_indoor.pt` | 670K indoor | Indoor scenes | ~600MB |
| `dac_swinl_outdoor.pt` | 130K outdoor | Outdoor/drone | ~600MB |
| `dac_resnet101_indoor.pt` | 670K indoor | Faster indoor | ~200MB |
| `dac_resnet101_outdoor.pt` | 130K outdoor | Faster outdoor | ~200MB |

**Recommendation:** Start with `dac_swinl_outdoor` for drone applications.

## Usage

### Quick Test

```bash
# Test without depth (image only)
python examples/wide_angle_pipeline.py \
    --camera 0 \
    --fov 200 \
    --no-dac \
    --model s

# Test with DAC depth estimation
python examples/wide_angle_pipeline.py \
    --camera 0 \
    --fov 200 \
    --dac-model checkpoints/dac_swinl_outdoor.pt \
    --dac-config checkpoints/dac_swinl_outdoor.json \
    --model s \
    --classes 0 2 3 7
```

### Single Wide-Angle Camera

```bash
python examples/wide_angle_pipeline.py \
    --mode single \
    --camera 0 \
    --fov 200 \
    --dac-model checkpoints/dac_swinl_outdoor.pt \
    --dac-config checkpoints/dac_swinl_outdoor.json \
    --model s \
    --save-video output/wide_angle_test.mp4
```

### Dual Wide-Angle Cameras (Skydio-style)

```bash
python examples/wide_angle_pipeline.py \
    --mode dual \
    --camera 0 \
    --camera-right 1 \
    --fov 200 \
    --baseline 0.12 \
    --dac-model checkpoints/dac_swinl_outdoor.pt \
    --dac-config checkpoints/dac_swinl_outdoor.json \
    --model s \
    --classes 0 2 3 7
```

## Troubleshooting

### Import Errors

**Error:** `ModuleNotFoundError: No module named 'dac'`

**Solution:**
```bash
# Add DAC to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/third_party/depth_any_camera"

# Or install in development mode
cd third_party/depth_any_camera
pip install -e .
```

### CUDA Out of Memory

**Solution:** Use smaller model or reduce resolution

```bash
# Use ResNet101 instead of SwinL (smaller)
python examples/wide_angle_pipeline.py \
    --dac-model checkpoints/dac_resnet101_outdoor.pt \
    --dac-config checkpoints/dac_resnet101_outdoor.json \
    --resolution 640x480
```

### Slow Inference

**Solutions:**
1. Use GPU: Ensure CUDA is available (`torch.cuda.is_available()`)
2. Use smaller model: ResNet101 is ~3x faster than SwinL
3. Lower resolution: `--resolution 640x480`
4. Skip frames: Modify pipeline to run depth every N frames

### Model Not Found

**Error:** `FileNotFoundError: checkpoints/dac_swinl_outdoor.pt`

**Solution:** Download models as described in Step 3 above

## Architecture Overview

### How DAC Works

1. **Input:** RGB image + camera intrinsics (fx, fy, cx, cy)
2. **Encoder:** Vision transformer extracts multi-scale features
3. **Decoder:** Predicts metric depth map
4. **Output:** Dense depth map in meters

### Key Features

- **Camera-agnostic:** Trained only on perspective images, generalizes to fisheye
- **Metric depth:** Real-world scale (meters) without post-processing
- **Dense prediction:** Every pixel gets a depth value
- **Fast inference:** ~30 FPS on GPU (ResNet101), ~10 FPS (SwinL)

## Integration Details

### Sensor Class: `WideAngleCamera`

Located in `sensors/wide_angle.py`:

```python
from sensors import WideAngleCamera

# Initialize
camera = WideAngleCamera(
    camera_id=0,
    resolution=(1280, 720),
    fov=200.0,  # Field of view in degrees
    dac_model_path="checkpoints/dac_swinl_outdoor.pt",
    dac_config_path="checkpoints/dac_swinl_outdoor.json",
    use_dac=True
)

# Get frame with depth
frame = camera.get_frame()
# frame.image: RGB image (H, W, 3)
# frame.depth: Depth map (H, W) in meters
```

### Dual Camera: `DualWideAngleCamera`

```python
from sensors import DualWideAngleCamera

camera = DualWideAngleCamera(
    left_camera_id=0,
    right_camera_id=1,
    fov=200.0,
    baseline=0.12,  # meters
    use_dac=True
)

frame = camera.get_frame()
# Automatically fuses left and right depth
```

## Performance Tips

### For Drone Applications

1. **Use outdoor model:** `dac_swinl_outdoor.pt` or `dac_resnet101_outdoor.pt`
2. **Higher FOV:** Set `--fov 200` or higher for wide-angle lenses
3. **Calibrate camera:** Provide accurate intrinsics for better depth accuracy
4. **GPU recommended:** Enables real-time processing (30 FPS)

### For Indoor Testing

1. **Use indoor model:** `dac_swinl_indoor.pt`
2. **Lower FOV:** If using normal webcam, set `--fov 90`
3. **Good lighting:** Helps YOLO detection quality

## References

- **Paper:** "Depth Any Camera: Zero-Shot Metric Depth Estimation from Any Camera" (CVPR 2025)
- **Repository:** https://github.com/yuliangguo/depth_any_camera
- **Models:** https://huggingface.co/yuliangguo/depth-any-camera
- **Authors:** Yuliang Guo, Xinyi Ye, Liu Ren

## Next Steps

1. ✅ Install DAC and download models
2. ✅ Test with your webcam first (`--no-dac` then with DAC)
3. ✅ Calibrate your fisheye camera for accurate intrinsics
4. ✅ Test with actual wide-angle/fisheye cameras
5. ✅ Integrate into your drone perception pipeline

## Support

If you encounter issues:

1. Check DAC repository issues: https://github.com/yuliangguo/depth_any_camera/issues
2. Verify PyTorch CUDA setup: `python -c "import torch; print(torch.cuda.is_available())"`
3. Test DAC standalone before integration
4. Open issue in this repository with error details
