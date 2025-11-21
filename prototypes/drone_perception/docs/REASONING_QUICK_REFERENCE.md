# Reasoning Pipeline Quick Reference

Quick reference for `examples/reasoning_pipeline.py`

## Launch Command

```bash
python examples/reasoning_pipeline.py [OPTIONS]
```

## Common Options

```bash
--camera 0                        # Use webcam (default: 0)
--model s                         # YOLO model size: n, s, m, l, x (default: s)
--conf-threshold 0.3              # Detection confidence (default: 0.3)
--prediction-horizon 3.0          # Seconds into future (default: 3.0)
--prediction-method physics       # Method: constant_velocity, constant_acceleration, physics
--drone-x 0.0 --drone-y 0.0       # Simulated drone position
--drone-z 1.5                     # Simulated drone height (default: 1.5m)
--save-video output.mp4           # Save annotated video
--classes 0 2 7                   # Filter specific classes (person, car, truck)
```

## Keyboard Controls

| Key | Function |
|-----|----------|
| `q` | Quit the application |
| `p` | Pause/Resume pipeline |
| `t` | Toggle trajectory visualization ON/OFF |
| `c` | Toggle collision warnings ON/OFF |
| `b` | Toggle behavior info ON/OFF |
| `x` | Clear all trajectory histories and predictions |
| `s` | Print detailed scene description to console |
| `d` | Debug: show object states, velocities, tracking info |

## Info Overlay (Top of Screen)

```
Frame: 123 | Tracks: 1 | Scene: 1 | Active: 1 | Risks: 0
```

- **Frame**: Current frame number
- **Tracks**: Objects detected by ByteTracker this frame
- **Scene**: Total objects in scene graph (including stale)
- **Active**: Objects both tracked and currently visible
- **Risks**: Number of HIGH or CRITICAL collision risks

## Collision Risk Levels

| Level | Color | Meaning |
|-------|-------|---------|
| CRITICAL | Red | Time to collision < 1s or distance < 1m |
| HIGH | Orange | Time to collision 1-2s or distance 1-2m |
| MEDIUM | Yellow | Time to collision 2-3s or distance 2-3m |
| LOW | Green | Time to collision > 3s or distance > 3m |
| NONE | - | No collision risk |

## Trajectory Visualization

- **Green → Yellow → Red**: Time progression (green = near future, red = far future)
- **Blue circle**: Predicted endpoint (where object will be)
- **Line of dots**: Predicted path over next N seconds

## Behavior Types

| Type | Description |
|------|-------------|
| Stationary | Not moving (velocity < 0.1 m/s) |
| Moving Straight | Constant velocity, straight line |
| Turning | Changing direction |
| Accelerating | Speed increasing |
| Decelerating | Speed decreasing |
| Approaching | Moving toward drone |

## Debug Output ('d' key)

```
Current tracks from ByteTracker: 1
Objects in scene graph: 1
Active nodes (filtered): 1

ID 1 (person):
  Position: [0.5 -0.3 3.2]
  Velocity: [0.2 0.1 -0.05] (mag: 0.229 m/s)
  History length: 15
  Last seen: 0.03s ago
  In current tracks: True
```

## Configuration Summary

### ByteTracker Settings
```python
high_thresh=0.5      # High confidence detections
low_thresh=0.1       # Low confidence detections
match_thresh=0.3     # IOU threshold (lower = more permissive re-ID)
max_time_lost=30     # Frames to keep lost tracks (~1 second)
```

### Scene Graph Settings
```python
ttl_seconds=1.5      # Remove objects after 1.5 seconds off-screen
```

### Prediction Settings
```python
prediction_horizon=3.0   # Predict 3 seconds into future
dt=0.2                   # 0.2 second timesteps (15 points total)
prediction_skip_frames=3 # Update predictions every 3 frames
visualization_resolution=15  # 15 points for smooth curve
```

## Troubleshooting

### Objects not being removed
- Check console for `[Pruning]` messages
- Press 'd' to see "Last seen" times
- Objects removed after 1.5s (TTL timeout)

### Multiple IDs for same person
- Lower `match_thresh` in code (currently 0.3)
- Increase `max_time_lost` for longer re-ID window
- Ensure good lighting and consistent visibility

### Zero velocities
- Objects need 3+ frames to compute velocity
- Check depth estimation (bbox-based for monocular)
- Use 'd' key to verify position changes

### Slow performance
- Reduce `prediction_horizon` (less future time)
- Increase `prediction_skip_frames` (update less often)
- Use smaller YOLO model (`--model n`)
- Reduce `visualization_resolution` in code

## Examples

### Basic usage
```bash
python examples/reasoning_pipeline.py --camera 0 --model s
```

### High-performance mode (faster)
```bash
python examples/reasoning_pipeline.py --camera 0 --model n --prediction-horizon 2.0
```

### Save output with specific classes
```bash
python examples/reasoning_pipeline.py \
    --camera 0 \
    --model s \
    --classes 0 2 3 5 7 \
    --save-video reasoning_output.mp4
```

### Physics-based prediction
```bash
python examples/reasoning_pipeline.py \
    --camera 0 \
    --prediction-method physics \
    --prediction-horizon 5.0
```

## Performance Tips

1. **For speed**: Use `--model n` (nano) instead of s/m/l/x
2. **For accuracy**: Use `--model l` or `--model x`
3. **Balance**: Default `--model s` is good middle ground
4. **GPU**: Automatically uses GPU if CUDA available
5. **Filtering**: Use `--classes` to reduce detections (faster tracking)

## Coordinate System

- **X**: Right (positive) / Left (negative)
- **Y**: Down (positive) / Up (negative)
- **Z**: Forward/Depth (positive = away from camera)

Camera frame origin is at camera center.

---

For more details, see [Phase 3 Documentation](phase3_reasoning.md)

Last updated: November 21, 2025
