# Phase 3: 3D Reasoning & Planning

Advanced reasoning capabilities built on top of 3D perception to understand object behaviors, predict future states, and enable intelligent decision-making.

## Overview

Phase 3 takes the 3D object tracking from Phases 1-2 and adds high-level reasoning:

| Module | Purpose | Key Features |
|--------|---------|--------------|
| **Trajectory Prediction** | Predict where objects will move | Constant velocity, acceleration, physics-based |
| **Collision Detection** | Assess collision risks | Time-to-collision, risk levels, avoidance vectors |
| **Spatial Analysis** | Understand spatial relationships | Relative positions, proximity, clustering |
| **Behavior Classification** | Classify object behaviors | Stationary, moving, turning, approaching |

---

## Modules

### 1. Trajectory Prediction (`reasoning/trajectory_predictor.py`)

Predicts future object positions based on current state and motion history.

**Methods:**
- **Constant Velocity:** `p(t) = p0 + v*t`
- **Constant Acceleration:** `p(t) = p0 + v0*t + 0.5*a*t^2`
- **Physics-based:** Includes gravity, ground plane constraints

**Usage:**
```python
from reasoning import TrajectoryPredictor

predictor = TrajectoryPredictor(
    prediction_horizon=3.0,  # seconds into future
    dt=0.1,  # time step
    method="constant_velocity"
)

# Predict single object
prediction = predictor.predict(scene_graph_node)

# Predict all objects
predictions = predictor.predict_all(all_nodes)

# Access predictions
print(f"Current position: {prediction.current_position}")
print(f"Predicted endpoint: {prediction.predicted_endpoint}")
print(f"Future positions: {prediction.future_positions}")  # (N, 3) array
print(f"Confidence: {prediction.confidence}")
```

**Output:**
- `PredictedTrajectory` object with:
  - Future positions at timesteps
  - Confidence scores
  - Uncertainty estimates
  - Path length

---

### 2. Collision Detection (`reasoning/collision_detector.py`)

Detects potential collisions between predicted trajectories.

**Features:**
- Trajectory-trajectory collision checking
- Point-trajectory collision (e.g., drone vs objects)
- Time-to-collision estimation
- Risk level assessment (NONE, LOW, MEDIUM, HIGH, CRITICAL)
- Avoidance vector calculation

**Usage:**
```python
from reasoning import CollisionDetector, RiskLevel

detector = CollisionDetector(
    safety_margin=2.0,  # meters
    time_horizon=3.0,  # seconds
    critical_distance=1.0,  # meters
    warning_distance=3.0  # meters
)

# Check all collisions
collision_risks = detector.check_all_collisions(
    trajectories=predictions,
    drone_position=np.array([0, 0, 1.5])
)

# Analyze risks
for risk in collision_risks:
    if risk.risk_level >= RiskLevel.HIGH:
        print(f"ALERT: {risk.object_a_class} & {risk.object_b_class}")
        print(f"  Time to collision: {risk.time_to_collision:.1f}s")
        print(f"  Closest distance: {risk.closest_distance:.1f}m")

# Get avoidance direction
avoidance_vector = detector.get_avoidance_vector(drone_position, collision_risks)
if avoidance_vector is not None:
    print(f"Suggested avoidance direction: {avoidance_vector}")
```

**Output:**
- `CollisionRisk` objects with:
  - Risk level (enum)
  - Time to collision
  - Closest approach distance
  - Collision point
  - Confidence

---

### 3. Spatial Analysis (`reasoning/spatial_analyzer.py`)

Analyzes spatial relationships between objects.

**Relationships Detected:**
- **Directional:** in_front, behind, left, right, above, below
- **Proximity:** near, far
- **Special:** same_level

**Features:**
- Pairwise relationship analysis
- K-nearest neighbors
- Spatial clustering
- Natural language scene descriptions

**Usage:**
```python
from reasoning import SpatialAnalyzer

analyzer = SpatialAnalyzer(
    near_threshold=3.0,  # meters
    far_threshold=10.0,  # meters
    height_threshold=0.5  # meters
)

# Analyze relationship between two objects
relation = analyzer.analyze_pair(node_a, node_b)
print(relation.describe())
# Output: "car is in front of truck (5.2m)"

# Find nearest objects
nearest = analyzer.find_nearest_objects(reference_node, all_nodes, k=3)
for node, distance in nearest:
    print(f"{node.class_name} at {distance:.1f}m")

# Find spatial clusters
clusters = analyzer.find_clusters(all_nodes, max_cluster_distance=5.0)
print(f"Found {len(clusters)} groups of objects")

# Generate scene description
description = analyzer.describe_scene(all_nodes, drone_position)
print(description)
```

**Output:**
- `SpatialRelation` objects with:
  - Relative position (enum)
  - Distance
  - Angle
  - Natural language description

---

### 4. Behavior Classification (`reasoning/behavior_classifier.py`)

Classifies object behaviors from movement patterns.

**Behaviors Detected:**
- **State:** stationary, hovering
- **Motion:** moving_straight, turning
- **Dynamics:** accelerating, decelerating, stopping, starting
- **Direction:** approaching, receding

**Features:**
- Speed and acceleration analysis
- Angular velocity estimation
- Heading calculation
- Threat assessment

**Usage:**
```python
from reasoning import BehaviorClassifier

classifier = BehaviorClassifier(
    stationary_threshold=0.1,  # m/s
    straight_threshold=5.0,  # degrees/s
    acceleration_threshold=1.0  # m/s^2
)

# Classify single object
behavior = classifier.classify(node, reference_position=drone_position)

print(f"Behavior: {behavior.behavior_type.value}")
print(f"Description: {behavior.description}")
print(f"Speed: {behavior.speed:.1f} m/s")
print(f"Acceleration: {behavior.acceleration:.1f} m/s^2")
print(f"Is approaching: {behavior.is_approaching}")
print(f"Is threat: {behavior.is_potential_threat()}")

# Classify all objects
behaviors = classifier.classify_all(all_nodes, drone_position)
```

**Output:**
- `ObjectBehavior` objects with:
  - Behavior type (enum)
  - Movement characteristics
  - Threat assessment
  - Natural language description

---

## Complete Pipeline Example

The `examples/reasoning_pipeline.py` demonstrates all reasoning capabilities together:

```bash
# Basic usage (with webcam)
python examples/reasoning_pipeline.py \
    --camera 0 \
    --model s \
    --prediction-horizon 3.0 \
    --drone-x 0 --drone-y 0 --drone-z 1.5

# With specific classes and video save
python examples/reasoning_pipeline.py \
    --camera 0 \
    --model s \
    --classes 0 2 3 7 \
    --prediction-method physics \
    --save-video output/reasoning_demo.mp4
```

**Controls:**
- `q` - Quit
- `p` - Pause/Resume
- `t` - Toggle trajectory visualization
- `c` - Toggle collision warnings
- `b` - Toggle behavior info
- `s` - Print scene description

---

## Integration with Other Phases

Phase 3 reasoning works with ANY perception backend:

### With Stereo (Phase 2a)
```bash
# Modify stereo_pipeline.py to include reasoning
from reasoning import TrajectoryPredictor, CollisionDetector

predictor = TrajectoryPredictor(prediction_horizon=3.0)
detector = CollisionDetector()

# In main loop after scene graph update:
predictions = predictor.predict_all(scene_graph.get_active_nodes())
risks = detector.check_all_collisions(predictions, drone_position)
```

### With LiDAR (Phase 2b)
```bash
# Same as above - reasoning is sensor-agnostic
# Works with any 3D position and velocity data
```

### With Wide-Angle (Phase 4)
```bash
# Add to wide_angle_pipeline.py
from reasoning import BehaviorClassifier

behavior_classifier = BehaviorClassifier()
behaviors = behavior_classifier.classify_all(nodes, drone_position)
```

---

## Use Cases

### 1. Drone Navigation
```python
# Get collision risks
risks = detector.check_all_collisions(predictions, drone_position)

# Get avoidance vector
avoidance = detector.get_avoidance_vector(drone_position, risks)

# Apply to drone control
if avoidance is not None:
    drone.move(avoidance * avoidance_gain)
```

### 2. Traffic Monitoring
```python
# Classify vehicle behaviors
behaviors = classifier.classify_all(nodes)

# Find congestion (clusters of slow-moving vehicles)
slow_vehicles = [b for b in behaviors if b.speed < 2.0 and b.behavior_type != BehaviorType.STATIONARY]
```

### 3. Safety Monitoring
```python
# Detect fast-approaching objects
for behavior in behaviors:
    if behavior.is_approaching and behavior.speed > 5.0:
        print(f"ALERT: Fast {behavior.class_name} approaching!")
```

### 4. Scene Understanding
```python
# Generate natural language description
description = spatial_analyzer.describe_scene(nodes, drone_position)
print(description)
# Output:
# "Scene contains 5 objects:
#   - 2 car(s)
#   - 2 person(s)
#   - 1 truck(s)
# Objects form 2 spatial groups.
# Closest object: person at 3.2m"
```

---

## Performance Considerations

### Computational Cost

| Module | Complexity | Cost per Frame (CPU) |
|--------|------------|---------------------|
| Trajectory Prediction | O(N) | ~1ms for 10 objects |
| Collision Detection | O(N²) | ~5ms for 10 objects |
| Spatial Analysis | O(N²) | ~3ms for 10 objects |
| Behavior Classification | O(N) | ~1ms for 10 objects |

**Total:** ~10-15ms for 10 objects (can run at 60+ FPS)

### Optimization Tips

1. **Limit prediction horizon** - Shorter horizons are faster
2. **Filter objects** - Only reason about nearby/relevant objects
3. **Cache results** - Reuse predictions for multiple frames
4. **Parallel processing** - Run modules in separate threads

---

## Future Enhancements

### 1. Learning-Based Prediction
- Train neural network on trajectory data
- Learn object-specific motion patterns
- Incorporate scene context

### 2. Multi-Agent Interaction
- Model object interactions
- Predict coordinated movements
- Understand group behaviors

### 3. Semantic Reasoning
- Understand traffic rules
- Recognize activity patterns
- Contextual behavior analysis

### 4. Path Planning
- Plan drone paths avoiding obstacles
- Optimize for objectives (speed, safety, efficiency)
- Dynamic replanning based on predictions

---

## Testing

```bash
# Test trajectory prediction
python -m reasoning.trajectory_predictor

# Test collision detection
python -m reasoning.collision_detector

# Test full pipeline
python examples/reasoning_pipeline.py --camera 0
```

---

## Summary

Phase 3 transforms raw 3D perception into actionable intelligence:

| Input | Processing | Output |
|-------|-----------|--------|
| 3D positions | Trajectory Prediction | Future positions |
| Predicted paths | Collision Detection | Risk assessment |
| Object locations | Spatial Analysis | Relationships |
| Motion history | Behavior Classification | Intent understanding |

**Result:** Intelligent scene understanding for autonomous navigation and decision-making.
