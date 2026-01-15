---
title: Constraints Reference
description: Reference for constraint tiers and classifications.
---

## Latency Tiers

| Tier | Range | Use Cases |
|------|-------|-----------|
| **Ultra-Low** | <1ms | High-frequency control loops (1kHz+) |
| **Real-Time** | 1-10ms | Motor control, servo loops (100Hz) |
| **Interactive** | 10-100ms | Object detection, tracking (10-30Hz) |
| **Near-Real-Time** | 100-1000ms | Scene understanding, planning |
| **Batch** | >1000ms | Offline analysis, training |

## Power Classes

| Class | Range | Platforms |
|-------|-------|-----------|
| **Ultra-Low** | <2W | Coral Edge TPU, MCUs |
| **Low** | 2-10W | Hailo-8, Jetson Nano |
| **Medium** | 10-30W | Jetson Orin Nano/NX |
| **High** | 30-100W | Jetson Orin AGX |
| **Datacenter** | >100W | T4, A100, H100 |

## Memory Classes

| Class | Range | Notes |
|-------|-------|-------|
| **Minimal** | <512MB | Microcontrollers, Edge TPU |
| **Constrained** | 512MB-2GB | Jetson Nano, RPi |
| **Standard** | 2-8GB | Jetson Orin Nano |
| **Extended** | 8-32GB | Jetson Orin NX/AGX |
| **Unlimited** | >32GB | Cloud GPUs |

## Use Case Templates

### Drone Obstacle Avoidance

```yaml
constraints:
  latency_ms: 33      # 30Hz perception
  power_watts: 15     # Battery budget
  memory_mb: 4096     # Jetson Orin Nano
  accuracy_map: 0.35  # Minimum detection quality
```

### Autonomous Vehicle Perception

```yaml
constraints:
  latency_ms: 50      # 20Hz perception
  power_watts: 100    # Drive unit budget
  memory_mb: 16384    # Multi-model pipeline
  accuracy_map: 0.45  # Safety-critical
```

### Edge Security Camera

```yaml
constraints:
  latency_ms: 100     # 10Hz detection
  power_watts: 5      # PoE budget
  memory_mb: 2048     # Low-cost hardware
  accuracy_map: 0.30  # Sufficient for alerts
```

## Confidence Levels

| Level | Source | Accuracy |
|-------|--------|----------|
| **HIGH** | Calibrated benchmarks | ±5% |
| **MEDIUM** | Roofline modeling | ±15% |
| **LOW** | Estimates, extrapolation | ±30% |
