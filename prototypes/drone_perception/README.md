# Drone Perception Pipeline

A progressive perception pipeline for drone-based object detection, tracking, and situational awareness.

## Quick Start

### Installation

```bash
cd prototypes/drone_perception
pip install -r requirements.txt
```

### Run Full Pipeline (✅ Working!)

```bash
# Basic usage with webcam (monocular)
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

# Stereo mode with RealSense D435 (Phase 2)
python examples/full_pipeline.py --stereo --stereo-backend realsense

# Stereo mode with OAK-D
python examples/full_pipeline.py --stereo --stereo-backend oakd

# Dedicated stereo pipeline example
python examples/stereo_pipeline.py --backend realsense --model s
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed instructions!**

## Progressive Sensor Support

This pipeline is designed to work with three levels of sensor complexity:

### Level 1: Monocular Camera
- **Input**: Video file or webcam
- **Depth**: Estimated via heuristics or MiDaS
- **Use Case**: Development, testing, recorded data
- **Status**: 🚧 In Progress

### Level 2: Stereo Camera
- **Input**: RealSense D435, OAK-D
- **Depth**: Stereo depth map
- **Use Case**: Metric tracking, velocity estimation
- **Status**: ✅ Complete

### Level 3: LiDAR + Camera
- **Input**: Livox/Velodyne + Camera
- **Depth**: 3D point cloud
- **Use Case**: Industrial deployment, high accuracy
- **Status**: 📋 Planned

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design.

```
Camera → Detection → Tracking → Scene Graph → Visualization
         (YOLOv8)   (ByteTrack)  (Kalman)      (3D Plot)
```

## Features

- ✅ **Object detection** with YOLOv8 (nano to xlarge models)
- ✅ **Multi-object tracking** with ByteTrack (ID persistence, re-identification)
- ✅ **3D scene graph** with position/velocity/acceleration estimation
- ✅ **Kalman filtering** for smooth state estimation (9D state per object)
- ✅ **Real-time 3D visualization** with matplotlib (position, velocity, trajectories)
- ✅ **Sensor abstraction** ready for monocular → stereo → LiDAR progression
- 📋 HDF5 recording for replay (coming soon)

## Project Structure

```
drone_perception/
├── sensors/           # Camera abstractions
│   ├── base.py       # BaseSensor interface
│   ├── monocular.py  # Video/webcam
│   ├── stereo.py     # RealSense/OAK-D
│   └── lidar.py      # LiDAR fusion
├── detection/         # Object detection
│   └── yolo.py       # YOLOv8 wrapper
├── tracking/          # Object tracking
│   └── bytetrack.py  # ByteTrack implementation
├── scene_graph/       # World state
│   ├── objects.py    # TrackedObject dataclass
│   ├── manager.py    # Scene graph manager
│   └── kalman.py     # State estimation
├── visualization/     # Rendering
│   ├── live_view.py  # Real-time 3D plot
│   └── replay.py     # Playback from HDF5
└── examples/          # Usage examples
    └── monocular_tracking.py
```

## Development Status

### ✅ Phase 1: Monocular Pipeline (COMPLETE)
- [x] Project structure and architecture
- [x] Sensor abstraction layer (monocular camera)
- [x] YOLOv8 detection integration
- [x] ByteTrack multi-object tracking
- [x] 3D scene graph with Kalman filtering
- [x] Real-time 3D visualization
- [x] Full end-to-end example

### ✅ Phase 2: Stereo Support (COMPLETE)
- [x] RealSense D435 integration
- [x] OAK-D support
- [x] Depth map fusion
- [x] Metric accuracy validation
- [x] Stereo pipeline example
- [x] Updated full_pipeline.py with --stereo flag

### 📋 Phase 3: Recording & Replay
- [ ] HDF5 data recording
- [ ] Replay viewer with timeline
- [ ] Export to common formats

### 📋 Phase 4: Production Ready
- [ ] LiDAR sensor support
- [ ] Multi-rate framework integration
- [ ] Performance optimization (30+ FPS on edge)
- [ ] Unit tests and CI/CD

## References

- Research: `../../docs/research/drone-pipeline.md`
- Multi-Rate Framework: `../multi_rate_framework/`
- ByteTrack: https://github.com/ifzhang/ByteTrack
- YOLOv8: https://github.com/ultralytics/ultralytics
