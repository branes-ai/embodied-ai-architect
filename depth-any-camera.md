# DAC: Depth Any Camera

  Files Created:

  1. sensors/wide_angle.py - Wide-angle camera sensors:
    - WideAngleCamera: Single fisheye camera with DAC depth
    - DualWideAngleCamera: Skydio-style dual wide-angle setup
    - Supports 180°-200°+ FOV cameras
  2. examples/wide_angle_pipeline.py - Complete pipeline:
    - Single or dual wide-angle camera modes
    - DAC-based metric depth estimation
    - YOLO detection + ByteTrack tracking
    - 3D scene graph and visualization
  3. docs/setup_dac.md - Comprehensive setup guide:
    - Installation instructions
    - Model downloads
    - Usage examples
    - Troubleshooting guide
  4. Updated requirements.txt with DAC dependencies

  ---

##🚀 Quick Start

Without DAC (Test First):

```python
  # Test camera and detection without depth
  python examples/wide_angle_pipeline.py \
      --camera 0 \
      --fov 200 \
      --no-dac \
      --model s
```

With DAC (Full Pipeline):

```bash
  # Step 1: Install DAC
  cd third_party
  git clone https://github.com/yuliangguo/depth_any_camera.git
  cd depth_any_camera
  pip install -r requirements.txt
  cd dac/models/ops && pip install -e .

  # Step 2: Download Models
  cd ../../../..  # Back to drone_perception root
  mkdir -p checkpoints
  # Download dac_swinl_outdoor.pt and .json from:
  # https://huggingface.co/yuliangguo/depth-any-camera

  Step 3: Run Pipeline
  # Single wide-angle camera
  python examples/wide_angle_pipeline.py \
      --mode single \
      --camera 0 \
      --fov 200 \
      --dac-model checkpoints/dac_swinl_outdoor.pt \
      --dac-config checkpoints/dac_swinl_outdoor.json \
      --model s

  # Dual wide-angle (Skydio-style)
  python examples/wide_angle_pipeline.py \
      --mode dual \
      --camera 0 \
      --camera-right 1 \
      --fov 200 \
      --baseline 0.12 \
      --dac-model checkpoints/dac_swinl_outdoor.pt \
      --dac-config checkpoints/dac_swinl_outdoor.json
```
  ---

## Key Advantages

Depth Any Camera Benefits:
  - ✅ Zero-shot: Works on any fisheye camera without training
  - ✅ Wide FOV: Handles 180°-200°+ field of view
  - ✅ Metric depth: Real-world scale (meters)
  - ✅ Dense: Every pixel gets depth
  - ✅ CVPR 2025: State-of-the-art performance

DAC vs Stereo Matching:
  - Single camera works (don't need stereo pair)
  - No calibration required
  - Better for wide FOV (stereo matching struggles with fisheye distortion)

