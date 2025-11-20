## Quick Start Guide

Get the drone perception pipeline running in 5 minutes!

### 1. Install Dependencies

```bash
cd prototypes/drone_perception
pip install -r requirements.txt
```

### 2. Test with Webcam

Run the full pipeline with your webcam:

```bash
python examples/full_pipeline.py --video 0 --model n --device cpu
```

You should see:
- **2D Window**: Video with detection bboxes and track IDs
- **3D Window**: Interactive 3D plot showing object positions, velocities, and trajectories

### 3. Test with Sample Video

If you don't have a webcam, download a sample video:

```bash
# Download a sample traffic video
wget https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4 -O test.mp4

# Run pipeline
python examples/full_pipeline.py --video test.mp4 --model n
```

Or use YouTube:
```bash
# Install youtube-dl
pip install yt-dlp

# Download a video (e.g., city traffic)
yt-dlp -f 'bestvideo[height<=720]' -o test_video.mp4 'YOUTUBE_URL_HERE'

# Run pipeline
python examples/full_pipeline.py --video test_video.mp4 --conf 0.3
```

### 4. Understanding the Output

**2D View (OpenCV Window)**:
- Colored bounding boxes around detected objects
- Class labels with confidence scores
- Green track IDs (persistent across frames)
- Stats overlay: Frame #, FPS, detection/track counts

**3D View (Matplotlib Window)**:
- Colored spheres = object positions
- Lines = trajectories (position history)
- Arrows = velocity vectors
- Text labels = class name, ID, speed

**Console Output**:
```
Frame 0030 | FPS: 28.3 | Detections: 5 | Tracks: 4 | 3D Objects: 4
```

### 5. Advanced Usage

**Track only specific classes** (0=person, 2=car, 5=bus, 7=truck):
```bash
python examples/full_pipeline.py --video 0 --classes 0 2  # Only people and cars
```

**Use GPU for faster inference**:
```bash
python examples/full_pipeline.py --video test.mp4 --device cuda --model s
```

**Save output video**:
```bash
python examples/full_pipeline.py --video test.mp4 --save-video output.mp4
```

**Simple detection only** (no tracking/3D):
```bash
python examples/simple_detection.py --video 0 --model n
```

### 6. Controls

While running:
- `q` - Quit
- `p` - Pause/Resume
- `s` - Save 3D view screenshot

### 7. Troubleshooting

**Problem**: "ultralytics not found"
```bash
pip install ultralytics
```

**Problem**: "No module named 'filterpy'"
```bash
pip install filterpy
```

**Problem**: "CUDA out of memory"
```bash
# Use smaller model or CPU
python examples/full_pipeline.py --model n --device cpu
```

**Problem**: "Can't open webcam"
```bash
# Try different device IDs
python examples/full_pipeline.py --video 1  # or 2, 3, etc.

# Or use a video file instead
```

**Problem**: 3D visualization is slow
```bash
# Disable 3D view for better performance
python examples/full_pipeline.py --video 0 --no-viz-3d
```

### 8. Next Steps

Once the basic pipeline works:

1. **Try stereo camera**: Implement `sensors/stereo.py` for RealSense
2. **Add recording**: Implement HDF5 recording for replay
3. **Optimize performance**: Profile and optimize bottlenecks
4. **Integrate with multi-rate framework**: Port to components

### Expected Performance

| Hardware | Model | FPS | Notes |
|----------|-------|-----|-------|
| Laptop CPU (i7) | yolov8n | 20-30 | Good for development |
| Laptop GPU (RTX 3060) | yolov8n | 60-100 | Real-time capable |
| Laptop GPU (RTX 3060) | yolov8s | 40-60 | Better accuracy |
| Jetson Orin Nano | yolov8n | 30-40 | Edge deployment ready |

### Sensor Progression Roadmap

**✅ Phase 1: Monocular (DONE)**
- Video/webcam input
- Heuristic depth estimation
- Basic 3D scene graph

**📋 Phase 2: Stereo (Next)**
- RealSense D435 support
- True metric depth
- Improved velocity accuracy

**📋 Phase 3: LiDAR (Future)**
- Point cloud fusion
- Industrial-grade accuracy
- Works in low light

## Common YOLO Class IDs

For `--classes` filter:

```
0: person
1: bicycle
2: car
3: motorcycle
5: bus
7: truck
14: bird
15: cat
16: dog
```

Full list: https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml
