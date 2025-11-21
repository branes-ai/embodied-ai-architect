# Documentation Index

Welcome to the Drone Perception Pipeline documentation!

## 📚 Main Documentation

### Getting Started
- **[README](../README.md)** - Project overview and quick start
- **[QUICKSTART.md](../QUICKSTART.md)** - Detailed installation and usage guide
- **[ARCHITECTURE.md](../ARCHITECTURE.md)** - System architecture and design

### Phase Documentation
- **[Phase 3: 3D Reasoning & Planning](phase3_reasoning.md)** - Trajectory prediction, collision detection, spatial analysis, behavior classification
- **[Setup DAC (Depth Any Camera)](setup_dac.md)** - Wide-angle/fisheye camera depth estimation

### Change History
- **[CHANGELOG.md](../CHANGELOG.md)** - Complete version history and changes
- **[Session Logs](../../../../docs/sessions/2025-11-21-drone-perception-phase3-tracking-improvements.md)** - Detailed development session notes

---

## 🎯 By Use Case

### I want to...

**Run basic object tracking:**
→ Start with [QUICKSTART.md](../QUICKSTART.md) and use `examples/full_pipeline.py`

**Add trajectory prediction and collision detection:**
→ See [Phase 3 Reasoning](phase3_reasoning.md) and use `examples/reasoning_pipeline.py`

**Use a stereo camera (RealSense, OAK-D):**
→ Check [QUICKSTART.md](../QUICKSTART.md) stereo section and use `examples/stereo_pipeline.py`

**Use a wide-angle or fisheye camera:**
→ Follow [Setup DAC](setup_dac.md) and use `examples/wide_angle_pipeline.py`

**Understand what changed recently:**
→ Check [CHANGELOG.md](../CHANGELOG.md) for high-level changes
→ Check [../../../docs/sessions/2025-11-21-drone-perception-phase3-tracking-improvements.md](../../../../docs/sessions/2025-11-21-drone-perception-phase3-tracking-improvements.md) for implementation details

**Debug tracking issues:**
→ See [Session Log - Tracking Improvements](../../../../docs/sessions/2025-11-21-drone-perception-phase3-tracking-improvements.md#2-poor-re-identification-multiple-ids-for-same-person)

---

## 🔧 By Component

### Sensors
- **Monocular**: [README](../README.md#level-1-monocular-camera) | Code: `sensors/monocular.py`
- **Stereo**: [QUICKSTART](../QUICKSTART.md) | Code: `sensors/stereo.py`
- **Wide-Angle**: [Setup DAC](setup_dac.md) | Code: `sensors/wide_angle.py`
- **LiDAR**: Coming in Phase 5 | Code: `sensors/lidar.py`

### Detection
- **YOLO**: [README](../README.md#features) | Code: `detection/yolo.py`

### Tracking
- **ByteTrack**: [README](../README.md#features) | Code: `tracking/bytetrack.py`
- **Scene Graph**: [ARCHITECTURE](../ARCHITECTURE.md) | Code: `scene_graph/manager.py`

### Reasoning (Phase 3)
- **Trajectory Prediction**: [Phase 3 Docs](phase3_reasoning.md#trajectory-prediction) | Code: `reasoning/trajectory_predictor.py`
- **Collision Detection**: [Phase 3 Docs](phase3_reasoning.md#collision-detection) | Code: `reasoning/collision_detector.py`
- **Spatial Analysis**: [Phase 3 Docs](phase3_reasoning.md#spatial-analysis) | Code: `reasoning/spatial_analyzer.py`
- **Behavior Classification**: [Phase 3 Docs](phase3_reasoning.md#behavior-classification) | Code: `reasoning/behavior_classifier.py`

---

## 📊 Examples

All example scripts are in `examples/`:

| Example | Description | Documentation |
|---------|-------------|---------------|
| `full_pipeline.py` | Complete monocular/stereo pipeline | [QUICKSTART](../QUICKSTART.md) |
| `stereo_pipeline.py` | Dedicated stereo camera demo | [QUICKSTART](../QUICKSTART.md) |
| `reasoning_pipeline.py` | 3D reasoning & planning | [Phase 3](phase3_reasoning.md) |
| `wide_angle_pipeline.py` | Fisheye camera with DAC | [Setup DAC](setup_dac.md) |

---

## 🔍 Troubleshooting

### Common Issues

**Objects not being removed when they leave frame:**
→ [Session Log - Object Pruning Fix](../../../../docs/sessions/2025-11-21-drone-perception-phase3-tracking-improvements.md#1-object-pruning-not-working)

**Multiple IDs for the same person:**
→ [Session Log - Re-ID Improvements](../../../../docs/sessions/2025-11-21-drone-perception-phase3-tracking-improvements.md#2-poor-re-identification-multiple-ids-for-same-person)

**Zero velocities for all objects:**
→ [CHANGELOG - Depth Estimation](../CHANGELOG.md#changed)

**Slow trajectory predictions:**
→ [CHANGELOG - Performance Optimizations](../CHANGELOG.md#changed)

**Wrong colors in visualization:**
→ [CHANGELOG - Visualization Fixes](../CHANGELOG.md#changed)

**MiDaS model not found:**
→ [CHANGELOG - Fixed](../CHANGELOG.md#fixed)

---

## 📝 Contributing

When making changes:
1. Update [CHANGELOG.md](../CHANGELOG.md) with your changes
2. Create a session log (`SESSION_LOG_YYYY-MM-DD.md`) for significant work
3. Update relevant phase documentation if adding features
4. Add examples to demonstrate new capabilities

---

## 🗺️ Roadmap

- ✅ **Phase 1**: Monocular perception pipeline
- ✅ **Phase 2**: Multi-sensor support (stereo, wide-angle)
- ✅ **Phase 3**: 3D reasoning & planning
- 📋 **Phase 4**: Recording & replay (HDF5)
- 📋 **Phase 5**: Production ready (LiDAR, optimization, testing)

See [README - Development Status](../README.md#development-status) for detailed progress.

---

Last updated: November 21, 2025
