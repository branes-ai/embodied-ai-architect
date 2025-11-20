# Drone Perception Pipeline Architecture

## Overview

A lean, progressive perception pipeline for drone-based object detection, tracking, and situational awareness. Designed to start simple (monocular camera) and scale to industrial sensors (stereo/LiDAR).

## Design Principles

### 1. **Sensor Abstraction**
Support progressive sensor complexity without architectural rewrites:
- **Level 1**: Monocular camera (2D detection + depth estimation via heuristics)
- **Level 2**: Stereo camera (2D detection + depth map fusion)
- **Level 3**: LiDAR + Camera (3D point cloud + image fusion)

### 2. **The Visual Channel Approach**
Following the drone research (docs/research/drone-pipeline.md):
- **Goal**: 5-8W power budget (vs 30-60W for SLAM)
- **Strategy**: Stateless per-frame processing, local awareness only
- **Trade-off**: No global map, but fast and energy-efficient

### 3. **Modular Pipeline**
```
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌───────────┐   ┌──────────┐
│ Sensor  │──▶│ Detector │──▶│ Tracker │──▶│Scene Graph│──▶│  Replay  │
│ (Image) │   │ (YOLOv8) │   │(ByteTrack)│  │  (State)  │   │  (Viz)   │
└─────────┘   └──────────┘   └─────────┘   └───────────┘   └──────────┘
                                                    │
                                                    ▼
                                            ┌──────────────┐
                                            │Kalman Filter │
                                            │(Vel/Accel)   │
                                            └──────────────┘
```

## Architecture Components

### 1. Sensor Layer (`sensors/`)

**Abstract Interface**: `BaseSensor`
```python
class BaseSensor(ABC):
    @abstractmethod
    def get_frame(self) -> Frame:
        """Returns Frame with image and optional depth"""
        pass
```

**Implementations**:
- `MonocularCamera`: Video file or webcam, estimated depth
- `StereoCamera`: RealSense/OAK-D integration, depth map
- `LiDARCamera`: Point cloud + image fusion

**Frame Data Structure**:
```python
@dataclass
class Frame:
    timestamp: float
    image: np.ndarray  # RGB image
    depth: Optional[np.ndarray]  # Depth map or None
    camera_params: CameraParams  # Intrinsics for 2D→3D projection
```

### 2. Detection Layer (`detection/`)

**YOLOv8 Integration**:
- Model: YOLOv8-Nano for edge deployment
- Output: Bounding boxes [x, y, w, h], class, confidence
- Batching: Single frame (real-time) or batch mode (replay)

**Detector Interface**:
```python
class ObjectDetector:
    def detect(self, image: np.ndarray) -> List[Detection]
```

### 3. Tracking Layer (`tracking/`)

**ByteTrack Implementation**:
- ID Association: IOU + Kalman motion model
- Re-identification: Track continuity across occlusions
- No separate ReID network needed (simpler than DeepSORT)

**Tracker Interface**:
```python
class ObjectTracker:
    def update(self, detections: List[Detection], frame_id: int) -> List[Track]
```

**Track Data**:
```python
@dataclass
class Track:
    id: int  # Unique track ID
    bbox: BBox  # Current bounding box
    class_name: str
    confidence: float
    age: int  # Number of frames tracked
```

### 4. Scene Graph Layer (`scene_graph/`)

**World-Frame State Representation**:
```python
@dataclass
class TrackedObject:
    # Identity
    track_id: int
    class_name: str

    # 3D State (in drone body frame or world frame)
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s
    acceleration: np.ndarray  # [ax, ay, az] in m/s²

    # Metadata
    timestamp: float
    confidence: float
    last_seen: float
```

**Scene Graph Manager**:
- In-memory store: `Dict[track_id, TrackedObject]`
- TTL (Time-to-Live): Remove objects not seen for N seconds
- State propagation: Kalman Filter per object

**Kalman Filter Design**:
- **State Vector**: [x, y, z, vx, vy, vz, ax, ay, az] (9D)
- **Measurement**: [x, y, z] from 2D→3D projection
- **Process Model**: Constant acceleration
- **Update Rate**: Per detection (asynchronous per object)

### 5. Visualization Layer (`visualization/`)

**Live Situational Awareness**:
- 3D matplotlib scatter plot
- Object positions (spheres)
- Velocity vectors (arrows)
- Trajectories (trails)

**Replay System**:
- **Storage**: HDF5 or Parquet
  - `/frames/{frame_id}/detections`
  - `/frames/{frame_id}/tracks`
  - `/scene_graph/{timestamp}/objects`
- **Playback**: Slider to scrub through time
- **Overlay**: Detected bboxes on original video

## Sensor Progression Strategy

### Phase 1: Monocular (Weeks 1-2)
**Input**: Video file or webcam
**Depth Estimation**:
- Heuristic: Assume bbox bottom = ground plane
- Calculate Z from bbox height (known object size assumptions)
- OR: Use MiDaS/Depth-Anything for monocular depth estimation

**Limitations**: Scale ambiguity (person at 10m looks like child at 5m)

### Phase 2: Stereo (Weeks 3-4)
**Input**: RealSense D435/OAK-D
**Depth Estimation**:
- Direct depth lookup from depth map at bbox centroid
- Median depth within bbox for robustness

**Advantages**: True metric depth, no scale ambiguity

### Phase 3: LiDAR + Camera (Weeks 5-6)
**Input**: LiDAR (e.g., Livox/Velodyne) + Camera
**Fusion Strategy**:
- 3D bbox from point cloud clustering
- Class label from 2D detection
- Association via projection + IOU

**Advantages**: Best accuracy, works in low light

## Data Flow Example

```python
# 1. Capture frame
frame = sensor.get_frame()

# 2. Detect objects
detections = detector.detect(frame.image)

# 3. Track objects (assign IDs)
tracks = tracker.update(detections, frame_id)

# 4. Promote 2D tracks to 3D world state
for track in tracks:
    # Get depth at bbox location
    depth = get_depth_at_bbox(track.bbox, frame.depth)

    # Project to 3D
    position_3d = project_2d_to_3d(track.bbox, depth, frame.camera_params)

    # Update scene graph
    scene_graph.update(track.id, position_3d, frame.timestamp)

# 5. State estimation (Kalman filter updates velocities)
scene_graph.propagate_states()

# 6. Visualize
visualizer.render(scene_graph.get_objects())

# 7. Record for replay
recorder.save_frame(frame, detections, tracks, scene_graph)
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Detection FPS | 30 | YOLOv8-Nano on CPU/GPU |
| Tracking Latency | < 10ms | ByteTrack is very fast |
| Memory | < 500MB | Scene graph with ~100 objects |
| Power (edge) | 5-8W | Without SLAM (visual channel only) |

## Integration with Multi-Rate Framework

Once validated standalone, integrate as components:

```python
@control_loop(rate_hz=30)
class PerceptionComponent(Component):
    image = Input("camera/image")
    detections = Output("perception/detections")

    def step(self):
        frame = self.image.value
        self.detections.value = self.detector.detect(frame)

@control_loop(rate_hz=10)
class SceneGraphComponent(Component):
    detections = Input("perception/detections")
    scene_state = Output("scene/state")

    def step(self):
        self.scene_graph.update(self.detections.value)
        self.scene_state.value = self.scene_graph.get_objects()

@control_loop(rate_hz=1)
class VisualizationComponent(Component):
    scene_state = Input("scene/state")

    def step(self):
        self.visualizer.render(self.scene_state.value)
```

## Dependencies

```
# Detection
ultralytics>=8.0.0  # YOLOv8

# Tracking
opencv-python>=4.8.0
scipy>=1.11.0
filterpy>=1.4.5  # Kalman filter

# Visualization
matplotlib>=3.7.0
plotly>=5.17.0  # Optional: interactive plots

# Data storage
h5py>=3.9.0  # HDF5 for replay
pyarrow>=13.0.0  # Parquet (alternative)

# Sensors (optional, progressive)
pyrealsense2>=2.54.0  # RealSense cameras
depthai>=2.23.0  # OAK-D cameras
```

## Success Criteria

- ✅ **Monocular**: Detect + track objects in video, visualize 3D trajectories
- ✅ **Stereo**: Accurate metric depth, velocity estimation < 10% error
- ✅ **Replay**: Record 5-minute flight, scrub through timeline
- ✅ **Integration**: Plug into multi-rate framework without rewrites

## References

- Drone Pipeline Research: `docs/research/drone-pipeline.md`
- Multi-Rate Framework: `prototypes/multi_rate_framework/`
- ByteTrack Paper: https://arxiv.org/abs/2110.06864
- YOLOv8 Docs: https://docs.ultralytics.com/
