# Drone Autonomy

## 3D Reasoning & Planning

Modules Created:

  1. reasoning/trajectory_predictor.py - Trajectory Prediction
    - Constant velocity model
    - Constant acceleration model
    - Physics-based prediction (gravity, ground plane)
    - 3 seconds into future by default
    - Confidence scores
  2. reasoning/collision_detector.py - Collision Detection
    - Trajectory-trajectory collision checking
    - Point-trajectory collision (drone vs objects)
    - Risk levels: NONE, LOW, MEDIUM, HIGH, CRITICAL
    - Time-to-collision estimation
    - Avoidance vector calculation
  3. reasoning/spatial_analyzer.py - Spatial Reasoning
    - Relative positions (in_front, behind, left, right, above, below)
    - Proximity analysis (near, far)
    - K-nearest neighbors
    - Spatial clustering
    - Natural language scene descriptions
  4. reasoning/behavior_classifier.py - Behavior Classification
    - Stationary vs moving
    - Accelerating/decelerating
    - Turning vs straight
    - Approaching vs receding
    - Threat assessment
  5. examples/reasoning_pipeline.py - Complete Demo
    - All reasoning capabilities integrated
    - Real-time visualization
    - Collision warnings
    - Behavior annotations
  6. docs/phase3_reasoning.md - Comprehensive documentation

  ---
## Quick Start

```bash
  # Basic reasoning demo (webcam + monocular)
  python examples/reasoning_pipeline.py \
      --camera 0 \
      --model s \
      --prediction-horizon 3.0 \
      --drone-x 0 --drone-y 0 --drone-z 1.5

  # Save output
  python examples/reasoning_pipeline.py \
      --camera 0 \
      --model s \
      --classes 0 2 3 7 \
      --prediction-method physics \
      --save-video output/reasoning_demo.mp4

  Controls:
  - t - Toggle trajectory visualization
  - c - Toggle collision warnings
  - b - Toggle behavior info
  - s - Print scene description
  - q - Quit
```

  ---
## What You Can Do Now

  1. Collision Avoidance

  risks = collision_detector.check_all_collisions(predictions, drone_position)
  avoidance_vector = detector.get_avoidance_vector(drone_position, risks)
  # Use avoidance_vector to steer drone away from danger

  2. Scene Understanding

  description = spatial_analyzer.describe_scene(nodes, drone_position)
  # "Scene contains 3 cars and 2 persons. Closest object: car at 5.2m"

  3. Behavior Monitoring

  for behavior in behaviors:
      if behavior.is_potential_threat():
          print(f"ALERT: {behavior.class_name} approaching at {behavior.speed}m/s!")

  4. Trajectory Prediction

  prediction = predictor.predict(node)
  print(f"Object will be at {prediction.predicted_endpoint} in 3s")

  ---
 ## Architecture Summary

  Phase 1 (Monocular)  ───┐
  Phase 2a (Stereo)    ───┼──> 3D Positions + Velocities
  Phase 2b (LiDAR)     ───┤                 ↓
  Phase 4 (Wide-Angle) ───┘     ┌──────────────────────┐
                                │  Scene Graph Manager │
                                └──────────────────────┘
                                           ↓
                          ┌────────────────┴────────────────┐
                          │     PHASE 3: REASONING          │
                          ├─────────────────────────────────┤
                          │ • Trajectory Prediction         │
                          │ • Collision Detection           │
                          │ • Spatial Analysis              │
                          │ • Behavior Classification       │
                          └────────────────┬────────────────┘
                                           ↓
                          ┌─────────────────────────────────┐
                          │  Intelligent Decision Making    │
                          │  • Navigation                   │
                          │  • Avoidance                    │
                          │  • Monitoring                   │
                          └─────────────────────────────────┘

  ---

## Real-World Applications

  1. Autonomous Drone Navigation - Predict and avoid collisions
  2. Traffic Monitoring - Understand vehicle behaviors and congestion
  3. Safety Systems - Alert on approaching threats
  4. Surveillance - Track and classify activities
  5. Research - Study object interactions and patterns

  ---
  See docs/phase3_reasoning.md for detailed documentation!

