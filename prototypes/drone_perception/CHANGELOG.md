# Changelog

All notable changes to the Drone Perception pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 3: 3D Reasoning & Planning modules
  - Trajectory prediction (constant velocity, constant acceleration, physics-based)
  - Collision detection with risk levels (NONE, LOW, MEDIUM, HIGH, CRITICAL)
  - Spatial analysis (relative positioning, proximity detection)
  - Behavior classification (stationary, moving, turning, accelerating, approaching)
- `examples/reasoning_pipeline.py` - Complete demonstration of all reasoning capabilities
- `docs/phase3_reasoning.md` - Comprehensive documentation for Phase 3
- Enhanced debug output ('d' key) showing detailed object states and tracking info
- Pruning notifications when stale objects are removed from scene graph
- Info overlay showing Tracks/Scene/Active object counts

### Changed
- **Tracking Improvements** (2025-11-21)
  - Lowered ByteTracker `match_thresh` from 0.8 → 0.5 → 0.3 for better re-identification
  - Increased ByteTracker `max_time_lost` from 15 → 30 frames (~1s) for better re-ID
  - Adjusted scene graph TTL from 5.0s → 1.0s → 1.5s to match tracker timeout
  - Added manual object pruning logic in reasoning pipeline
- **Performance Optimizations**
  - Increased trajectory prediction timestep from 0.1s → 0.2s (50% fewer points)
  - Added frame skipping for predictions (every 3rd frame instead of every frame)
  - Reduced trajectory visualization resolution from 50 → 15 points
- **Visualization Fixes**
  - Fixed BGR/RGB color space conversion for camera display
  - Fixed trajectory color gradient to use BGR format for OpenCV
  - Added filtering to only show trajectories for currently visible objects
- **Depth Estimation**
  - Implemented bbox-based depth estimation for monocular cameras
  - Depth calculated from bbox height: `depth = (focal_length * object_height) / bbox_height`
  - Assumed object heights: person=1.7m, car=1.5m, truck=2.5m, bus=3.0m
  - Depth clamped to 1-20m range for stability

### Fixed
- **Object Tracking & Pruning**
  - Fixed stale objects not being removed when they leave the frame
  - Added manual pruning call since `update_node()` doesn't auto-prune
  - Fixed objects accumulating in scene graph with zero velocities
- **Trajectory Prediction**
  - Fixed trajectories not clearing when 'x' key pressed (now clears predictions, collision_risks, behaviors)
  - Fixed trajectory dots accumulating on screen
  - Fixed stationary objects (zero velocity) not getting valid predictions
- **Import & API Issues**
  - Fixed missing `RiskLevel` export in `reasoning/__init__.py`
  - Fixed `Track.track_id` → `Track.id` attribute access
  - Fixed `MonocularCamera` constructor parameter (`camera_id` → `source`)
  - Fixed `YOLODetector` constructor parameter (`classes_filter` → `classes`)

### Deprecated
- None

### Removed
- None

### Security
- None

## [0.2.0] - Phase 2: Stereo and LiDAR Integration

### Added
- Stereo camera support with depth map generation
- LiDAR + Camera fusion
- Wide-angle/fisheye camera support
- Depth Any Camera (DAC) integration for zero-shot depth estimation
- `sensors/stereo.py` - Stereo camera sensor
- `sensors/lidar.py` - LiDAR + Camera sensor
- `sensors/wide_angle.py` - Wide-angle camera with DAC depth
- `docs/setup_dac.md` - Setup guide for Depth Any Camera

### Changed
- Enhanced scene graph to support multiple depth sources
- Added depth estimation modes: 'heuristic' (bbox-based) and 'depth_map'

### Fixed
- MiDaS model name compatibility (lowercase → capitalized: `dpt_hybrid` → `DPT_Hybrid`)
- FFV1 codec incompatibility with MP4 (changed to mp4v codec)
- ByteTracker API signature (removed `frame_id` parameter)
- Virtual environment path issues on Windows (backslash handling)

## [0.1.0] - Phase 1: Monocular Perception Pipeline

### Added
- Initial monocular camera perception pipeline
- YOLO object detection (YOLOv8)
- ByteTrack multi-object tracking
- 3D scene graph with Kalman filtering
- Basic visualization and control interface
- Project structure and documentation

### Dependencies
- Added `tqdm>=4.65.0` for progress bars
- Added torch/torchvision/timm for depth estimation
- Added filterpy for Kalman filtering
- Added ultralytics for YOLO detection

---

## Version History Summary

- **v0.1.0** - Monocular perception pipeline (detection, tracking, scene graph)
- **v0.2.0** - Multi-sensor support (stereo, LiDAR, wide-angle)
- **v0.3.0** - 3D reasoning and planning (current development)
