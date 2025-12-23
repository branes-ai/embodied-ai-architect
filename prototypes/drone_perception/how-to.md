# Drone Perception Pipeline

## Quick Start

### Installation

```bash
cd prototypes/drone_perception
pip install -r requirements.txt
```

### Run Full Pipeline (✅ Working!)

```bash
# Basic usage with webcam
python examples/full_pipeline.py --video 0

# Or with a video file
python examples/full_pipeline.py --video your_video.mp4

# Advanced: GPU + specific classes + save output
python examples/full_pipeline.py \
    --video test.mp4 \
    --device cuda \
    --model s \
    --classes 0 2 7 \
    --save-video output.mp4
```

$ python examples/full_pipeline.py --video test_data/videos/traffic/247589_tiny.mp4 --model s --classes 0 1 2 3 4 5 6 7 8 9 --save-video output.mp4

======================================================================
FULL DRONE PERCEPTION PIPELINE
======================================================================
Video: test_data/videos/traffic/247589_tiny.mp4
Model: yolov8s on cpu
Detection threshold: 0.3
Tracking classes: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Pipeline: Camera → YOLO → ByteTrack → Scene Graph → 3D Viz

Controls:
  'q' - Quit
  'p' - Pause/Resume
  's' - Save 3D view screenshot
=======                ===============================================================

[1/5] Initializing camera...
[MonocularCamera] Opened: test_data/videos/traffic/247589_tiny                .mp4
  Resolution: 360x640
  FPS: 25.0 → 25.0
  Total frames: 258
[2/5] Loading detection model...
[YOLODetector] Loading yolov8s.pt...
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt to 'yolov8s.pt': 8% ╸─────────── 1.7/21.5
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt to 'yolov8s.pt': 34% ━━━━──────── 7.3/21.5
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt to 'yolov8s.pt': 82% ━━━━━━━━━╸── 17.8/21.5
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt to 'yolov8s.pt': 100% ━━━━━━━━━━━━ 21.5MB
Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt to 'yolov8s.pt': 100% ━━━━━━━━━━━━ 21.5MB 62.4MB/s 0.3s
[YOLODetector] Warming up on cpu...
[YOLODetector] Ready! Detecting 80 classes
  Filtered to: ['person', 'bicycle', 'car', 'airplane', 'motorcycle', 'bus', 'train', 'truck', 'boat', 'traffic light']
[3/5] Initializing tracker...
[4/5] Creating scene graph...
[5/5] Setting up visualization...

[READY] Starting pipeline...

F:\Users\tomtz\dev\branes\clones\embodied-ai-architect\prototypes\drone_perception\visualization\live_view.py:158: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.pause(0.001)
Frame 0000 | FPS: 6.7 | Detections: 3 | Tracks: 3 | 3D Objects: 3
Frame 0030 | FPS: 7.9 | Detections: 2 | Tracks: 2 | 3D Objects: 6
Frame 0060 | FPS: 7.9 | Detections: 3 | Tracks: 3 | 3D Objects: 4
Frame 0090 | FPS: 7.9 | Detections: 2 | Tracks: 2 | 3D Objects: 4
Frame 0120 | FPS: 7.9 | Detections: 1 | Tracks: 1 | 3D Objects: 3
Frame 0150 | FPS: 8.0 | Detections: 0 | Tracks: 0 | 3D Objects: 3
Frame 0180 | FPS: 7.8 | Detections: 1 | Tracks: 1 | 3D Objects: 8
Frame 0210 | FPS: 7.6 | Detections: 0 | Tracks: 0 | 3D Objects: 2                                                   
Frame 0240 | FPS: 7.6 | Detections: 2 | Tracks: 2 | 3D Objects: 4

[INFO] End of video

[CLEANUP] Shutting down...
[MonocularCamera] Released

[STATS] Average FPS: 7.5
[INFO] Done!


======================================================================
FULL TOLL BOOTH EXAMPLE
======================================================================
#!/usr/bash
python examples/full_pipeline.py --video test_data/videos/traffic/247589_tiny.mp4 --model s --classes 0 1 2 3 4 5 6 7 8 9 --save-video test_data/videos/traffic/247589_tiny_output.mp4 > test_data/videos/traffic/247589_tiny.log
python examples/full_pipeline.py --video test_data/videos/traffic/247589_small.mp4 --model s --classes 0 1 2 3 4 5 6 7 8 9 --save-video test_data/videos/traffic/247589_small_output.mp4 > test_data/videos/traffic/247589_small.log
python examples/full_pipeline.py --video test_data/videos/traffic/247589_medium.mp4 --model s --classes 0 1 2 3 4 5 6 7 8 9 --save-video test_data/videos/traffic/247589_medium_output.mp4 > test_data/videos/traffic/247589_medium.log

