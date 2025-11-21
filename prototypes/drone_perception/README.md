# Drone Perception Pipeline

A progressive perception pipeline for drone-based object detection, tracking, 3D reasoning, and situational awareness.

📋 **[CHANGELOG](CHANGELOG.md)** | 📝 **[Latest Session Log](../../docs/sessions/2025-11-21-drone-perception-phase3-tracking-improvements.md)**

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

# 3D Reasoning Pipeline (Phase 3) - Trajectory prediction, collision detection, behavior analysis
python examples/reasoning_pipeline.py --camera 0 --model s --prediction-horizon 3.0
```

**See [QUICKSTART.md](QUICKSTART.md) for detailed instructions!**

**NEW: [Phase 3 Reasoning Documentation](docs/phase3_reasoning.md)** - Trajectory prediction, collision detection, spatial analysis, and behavior classification

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
Camera → Detection → Tracking → Scene Graph → Reasoning → Visualization
         (YOLOv8)   (ByteTrack)  (Kalman)      (Phase 3)   (3D Plot)
                                                   ↓
                                    ┌──────────────┴──────────────┐
                                    │  - Trajectory Prediction    │
                                    │  - Collision Detection      │
                                    │  - Spatial Analysis         │
                                    │  - Behavior Classification  │
                                    └─────────────────────────────┘
```

## Features

### Perception (Phase 1 & 2)
- ✅ **Object detection** with YOLOv8 (nano to xlarge models)
- ✅ **Multi-object tracking** with ByteTrack (ID persistence, re-identification)
- ✅ **3D scene graph** with position/velocity/acceleration estimation
- ✅ **Kalman filtering** for smooth state estimation (9D state per object)
- ✅ **Real-time 3D visualization** with matplotlib (position, velocity, trajectories)
- ✅ **Sensor abstraction** ready for monocular → stereo → LiDAR progression

### 3D Reasoning & Planning (Phase 3) - NEW!
- ✅ **Trajectory prediction** - Constant velocity, acceleration, and physics-based models
- ✅ **Collision detection** - Time-to-collision with 5-level risk assessment
- ✅ **Spatial analysis** - Relative positioning, proximity detection, clustering
- ✅ **Behavior classification** - Stationary, moving, turning, accelerating, approaching
- ✅ **Real-time visualization** - Predicted trajectories with color-coded risk levels

### Coming Soon
- 📋 HDF5 recording for replay
- 📋 LiDAR sensor support

## Project Structure

```
drone_perception/
├── sensors/                    # Camera abstractions
│   ├── base.py                # BaseSensor interface
│   ├── monocular.py           # Video/webcam
│   ├── stereo.py              # RealSense/OAK-D
│   ├── wide_angle.py          # Fisheye/wide-angle cameras
│   └── lidar.py               # LiDAR fusion
├── detection/                  # Object detection
│   └── yolo.py                # YOLOv8 wrapper
├── tracking/                   # Object tracking
│   ├── bytetrack.py           # ByteTrack implementation
│   └── kalman_filter.py       # Kalman box filter
├── scene_graph/                # World state management
│   └── manager.py             # 3D scene graph with Kalman filtering
├── reasoning/                  # 3D reasoning & planning (Phase 3)
│   ├── trajectory_predictor.py   # Future path prediction
│   ├── collision_detector.py     # Risk assessment & avoidance
│   ├── spatial_analyzer.py       # Relative positioning & proximity
│   └── behavior_classifier.py    # Motion pattern classification
├── visualization/              # Rendering
│   ├── live_view.py           # Real-time 3D plot
│   └── replay.py              # Playback from HDF5
├── examples/                   # Usage examples
│   ├── full_pipeline.py       # Complete monocular/stereo pipeline
│   ├── stereo_pipeline.py     # Dedicated stereo example
│   └── reasoning_pipeline.py  # 3D reasoning demo (Phase 3)
├── docs/                       # Documentation
│   └── phase3_reasoning.md    # Phase 3 guide
└── CHANGELOG.md               # Version history
```

**Note:** Development session logs are maintained in `../../docs/sessions/`

## Development Status

### ✅ Phase 1: Monocular Pipeline (COMPLETE)
- [x] Project structure and architecture
- [x] Sensor abstraction layer (monocular camera)
- [x] YOLOv8 detection integration
- [x] ByteTrack multi-object tracking
- [x] 3D scene graph with Kalman filtering
- [x] Real-time 3D visualization
- [x] Full end-to-end example

### ✅ Phase 2: Multi-Sensor Support (COMPLETE)
- [x] RealSense D435 integration
- [x] OAK-D support
- [x] Wide-angle/fisheye camera support
- [x] Depth Any Camera (DAC) integration
- [x] Depth map fusion
- [x] Metric accuracy validation
- [x] Stereo pipeline example
- [x] Updated full_pipeline.py with --stereo flag

### ✅ Phase 3: 3D Reasoning & Planning (COMPLETE - Nov 2025)
- [x] Trajectory prediction (constant velocity, acceleration, physics-based)
- [x] Collision detection with risk assessment
- [x] Spatial analysis (relative positioning, proximity)
- [x] Behavior classification (stationary, moving, turning, etc.)
- [x] Real-time reasoning pipeline example
- [x] Comprehensive documentation (docs/phase3_reasoning.md)
- [x] Performance optimizations (frame skipping, reduced resolution)
- [x] Enhanced tracking (improved re-identification, object pruning)

### 📋 Phase 4: Recording & Replay
- [ ] HDF5 data recording
- [ ] Replay viewer with timeline
- [ ] Export to common formats

### 📋 Phase 5: Production Ready
- [ ] LiDAR sensor support (Livox/Velodyne)
- [ ] Multi-rate framework integration
- [ ] Performance optimization (30+ FPS on edge devices)
- [ ] Unit tests and CI/CD
- [ ] Docker deployment

## References

- Research: `../../docs/research/drone-pipeline.md`
- Multi-Rate Framework: `../multi_rate_framework/`
- ByteTrack: https://github.com/ifzhang/ByteTrack
- YOLOv8: https://github.com/ultralytics/ultralytics
