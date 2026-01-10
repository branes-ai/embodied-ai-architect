# Branes Embodied AI Platform: Deployment Guide

## The Opportunity: Embodied AI for Humanity

We stand at a remarkable moment in history. For the first time, we can build machines that perceive, reason, and act in the physical world—autonomous systems that extend human capability into places we cannot go, perform tasks we cannot do safely, and operate at scales we cannot achieve alone.

**Embodied AI** is the intelligence that powers:
- Drones that inspect infrastructure, deliver medicine, and assist in search and rescue
- Robots that care for the elderly, assist surgeons, and explore hazardous environments
- Autonomous vehicles that provide mobility to those who cannot drive
- Agricultural systems that feed a growing population sustainably
- Space systems that extend humanity's reach beyond Earth

The Branes platform exists to help you build these systems—quickly, correctly, and with confidence that they will perform when it matters.

---

## What Makes Embodied AI Unique

Embodied AI systems operate in the real world under real constraints. Unlike cloud applications that can retry failed requests, an autonomous drone must perceive and react within milliseconds. Unlike data center workloads that draw from the grid, a surgical robot runs on a power budget measured in watts.

These systems integrate multiple computational domains:

| Domain | Examples | Why It Matters |
|--------|----------|----------------|
| **Perception** | Object detection, depth estimation, SLAM | See and understand the environment |
| **State Estimation** | Kalman filters, sensor fusion, localization | Know where you are and how you're moving |
| **Planning** | Path planning, trajectory optimization, MPC | Decide what to do next |
| **Control** | PID, impedance control, force feedback | Execute movements precisely |
| **Reasoning** | Behavior trees, decision networks, LLMs | Handle complex situations intelligently |

The Branes platform helps you design systems that integrate all of these—and validates that your design will meet real-world requirements before you build.

---

## The Branes Platform

### Design with Confidence

The Branes Embodied AI Platform is a **hardware/software codesign environment** that helps you move from concept to deployment with confidence:

```
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Branes Embodied AI Platform                                   │
├────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                            │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐               │
│  │  Model Zoo   │───→│   Analyze    │───→│    HW/SW     │───→│    Deploy   │               │
│  │              │    │              │    │  Optimizer   │    │             │               │
│  │ 100+ models  │    │   Compute    │    │              │    │  Optimized  │               │
│  │ Detection    │    │    Memory    │    │ 50+ targets  │    │             │               │
│  │ Segmentation │    │   Latency    │    │ Multi-core   │    │  for your   │               │
│  │ Tracking     │    │    Energy    │    │ Many-core    │    │             │               │
│  │ Planning     │    │  Accuracy    │    │ Data-flow    │    │ application │               │
│  │ Control      │    │              │    │ Domain-flow  │    │             │               │
│  └──────────────┘    └──────────────┘    └──────────────┘    └─────────────┘               │
│         │                   │                   │                   │                      │
│         ▼                   ▼                   ▼                   ▼                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐   │
│  │                         Interactive AI Architect                                    │   │
│  │                                                                                     │   │
│  │   "Design a drone perception pipeline to id a landmark at 300 mph, under 15W"       │   │
│  │   → Analyzes requirements, recommends architecture, validates feasibility           │   │
│  │   → Identifies optimization opportunities, deploys to target hardware               │   │
│  │                                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                            │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

### What Branes Provides

1. **Model Zoo**: 100+ pre-validated models for perception, tracking, and control
2. **Architecture Analysis**: Understand compute, memory, and bandwidth requirements
3. **Hardware Profiling**: Match your application to the best hardware target
4. **Deployment Tools**: Export optimized models for your target platform
5. **Interactive Architect**: AI-powered design assistant for complex decisions
6. **Validation**: Verify that your system meets requirements before deployment

---

## The Knowledge Processing Unit (KPU)

### Purpose-Built for Embodied AI

As embodied AI applications grow more sophisticated, specialized hardware enables new capabilities. The **Knowledge Processing Unit (KPU)** is designed specifically for the computational patterns of perception, guidance, navigation, and control:

| KPU Capability | What It Enables |
|----------------|-----------------|
| **Streaming data acquisition** | Fuse cameras, LiDAR, IMU, and force sensors at 1kHz |
| **Real-time linear algebra** | Run Kalman filters and MPC at control-loop rates |
| **Eigenvalue acceleration** | Enable real-time stability analysis and covariance estimation |
| **Constraint optimization** | Solve trajectory planning and collision avoidance problems |
| **Multi-rate scheduling** | Coordinate 1kHz control with 30Hz perception seamlessly |
| **Deterministic execution** | Guarantee worst-case timing for safety-critical applications |
| **Energy-proportional compute** | Extend mission duration with efficient power management |

The KPU complements existing platforms by handling the workloads that benefit most from specialized acceleration—freeing CPUs, GPUs and NPUs to do what they do best.

### When to Consider KPU

The KPU is ideal for applications requiring:
- Sub-millisecond control loops
- Hard real-time guarantees
- Extended battery life
- Integrated sensor-to-actuator pipelines

The Branes platform helps you determine whether your application would benefit from KPU acceleration through detailed architectural analysis.

---

## Development Workflow

### From Concept to Deployment

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Development Workflow                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. DEFINE REQUIREMENTS                                                     │
│     ├─ Latency budget (e.g., 33ms for 30 FPS perception)                    │
│     ├─ Power budget (e.g., 15W for 1-hour flight)                           │
│     ├─ Accuracy targets (e.g., mAP > 0.5 for detection)                     │
│     └─ Throughput needs (e.g., 6 cameras at 30 FPS)                         │
│                                                                             │
│  2. DESIGN ARCHITECTURE                                                     │
│     ├─ Select operators (detection, tracking, control)                      │
│     ├─ Define data flow between operators                                   │
│     ├─ Specify execution rates for each stage                               │
│     └─ Estimate resource requirements                                       │
│                                                                             │
│  3. ANALYZE & OPTIMIZE                                                      │
│     ├─ Profile compute and memory usage                                     │
│     ├─ Identify optimization opportunities                                  │
│     ├─ Apply quantization, fusion, and tuning                               │
│     └─ Match operators to compute resources                                 │
│                                                                             │
│  4. GENERATE HARDWARE/SOFTWARE CONFIGURATION                                │
│     ├─ Compare candidate platforms                                          │
│     ├─ Evaluate trade-offs (performance, power, cost)                       │
│     └─ Validate that requirements can be met                                │
│                                                                             │
│  5. DEPLOY & VALIDATE                                                       │
│     ├─ Export optimized models to target format                             │
│     ├─ Benchmark on target hardware                                         │
│     ├─ Measure latency distribution and power                               │
│     └─ Confirm requirements are satisfied                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Getting Started

### Installation

```bash
# Core platform
pip install embodied-ai-architect

# with HW/SW optimization support
pip install embodied-ai-architect[branes]     # Branes.ai hw/sw co-design platform

# With COTS deployment support
pip install embodied-ai-architect[jetson]     # NVIDIA Jetson
pip install embodied-ai-architect[ryzen]      # AMD Ryzen AI
pip install embodied-ai-architect[coral]      # Google Coral
pip install embodied-ai-architect[openvino]   # Intel OpenVINO

# With interactive AI architect
pip install embodied-ai-architect[chat]
export ANTHROPIC_API_KEY=your-key-here
```

### Supported Platforms

#### HW/SW Co-design platforms

The Branes HW/SW co-design platform is best for use-cases that require custom embodied AI acceleration
to meet the demand of the application. Examples are autonomous drone missions, scouting quadrupeds,
autonomous vehicles and AMRs.

| Platform | Best For | Precision Support |
|----------|----------|-------------------|
| **Microchip PolarFire** | Custom Sensor and Edge AI | Configurable |
| **Achronix Speedster** | Experimentation and Emulation | Configurable |
| **Stillwater KPU** | Custom embodied AI acceleration | Configurable |

#### Commodity Platforms

The commodity platform support is best for experimentation and competitive analysis.

| Platform | Best For | Precision Support |
|----------|----------|-------------------|
| **Ryzen AI NUC** | Desktop and industrial applications | FP32, FP16, INT8 |
| **Jetson Orin** | High-performance edge AI | FP32, FP16, INT8 |
| **Jetson Nano** | Cost-effective edge deployment | FP16, INT8 |
| **Coral Edge TPU** | Ultra-low-power inference | INT8 |
| **Intel OpenVINO** | Intel CPU and VPU deployment | FP32, FP16, INT8 |


---

## Workflow Examples

### Step 1: Explore the Model Zoo

Find models suited to your application:

```bash
# Search for detection models
branes zoo search --task detection

# Filter by size for edge deployment
branes zoo search --task detection --max-params 10M

# Download a model
branes zoo download yolov8n --format onnx
```

**Available model categories:**
- **Detection**: YOLOv5, YOLOv8, YOLOv11, DETR, EfficientDet
- **Classification**: ResNet, MobileNet, EfficientNet, ViT, ConvNeXt
- **Segmentation**: SegFormer, Mask R-CNN
- **Depth Estimation**: DPT, MiDaS
- **Pose Estimation**: MediaPipe, YOLOv8-pose

### Step 2: Analyze Your Model

Understand compute and memory requirements:

```bash
branes analyze yolov8n.onnx
```

**Example output:**
```
Model Analysis: YOLOv8n
=======================
Parameters: 3.2M
FLOPs: 8.7G
Memory footprint: 12.4 MB (FP32)

Layer breakdown:
  Conv2d: 72% compute
  Upsample: 12% compute
  Concat: 8% memory bandwidth

Recommendations:
  ✓ Suitable for edge deployment
  ✓ INT8 quantization recommended for best performance
```

### Step 3: Find the Best Hardware

Match your application to available platforms:

```bash
branes workflow run yolov8n.onnx \
  --latency-target 33 \
  --power-budget 15 \
  --use-case drone
```

**Example output:**
```
Hardware Recommendations for YOLOv8n
=====================================

Your requirements:
  Latency: ≤33ms | Power: ≤15W | Use case: drone

Recommended platforms:
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Hardware         ┃ Latency   ┃ Power  ┃ Meets Reqs   ┃ Score  ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Jetson Orin Nano │ 18ms      │ 10W    │ ✓ Yes        │ 0.92   │
│ Coral Edge TPU   │ 12ms      │ 2W     │ ✓ Yes        │ 0.88   │
│ Ryzen AI NUC     │ 8ms       │ 25W    │ ✓ Latency    │ 0.76   │
└──────────────────┴───────────┴────────┴──────────────┴────────┘

Top recommendation: Jetson Orin Nano
  • Excellent latency headroom (18ms vs 33ms target)
  • Well within power budget (10W vs 15W)
  • Strong ecosystem and tooling support
```

### Step 4: Deploy with Optimization

Export your model optimized for the target platform:

```bash
# Deploy to Jetson with INT8 quantization
branes deploy run yolov8n.pt \
  --target jetson \
  --precision int8 \
  --calibration-data ./calibration_images \
  --input-shape 1,3,640,640 \
  --test-data ./validation_images \
  --validate
```

**Example output:**
```
Deployment: YOLOv8n → Jetson (INT8)
===================================

Step 1: Preparing ONNX model...
  ✓ Exported to ONNX format

Step 2: Building TensorRT engine...
  ✓ INT8 calibration complete (100 samples)
  ✓ Engine optimized and serialized

Step 3: Validating accuracy...
  ✓ Accuracy within tolerance (0.2% difference)

Deployment complete!
  Engine: deployments/model_int8.engine
  Size: 4.8 MB
  Latency: 8.7ms (2.1x faster than FP32)
  Speedup: 2.1x with <1% accuracy impact
```

### Step 5: Validate Performance

Confirm your system meets requirements:

```bash
branes benchmark run deployments/model_int8.engine \
  --iterations 1000 \
  --report latency-distribution
```

**Example output:**
```
Performance Validation (1000 iterations)
========================================

Latency distribution:
  Mean:  8.7ms
  P50:   8.6ms
  P90:   9.1ms
  P99:   10.2ms
  P99.9: 12.8ms

✓ P99.9 latency (12.8ms) well under target (33ms)
✓ Consistent performance with low variance
✓ Ready for deployment
```

---

## Interactive AI Architect

For complex design decisions, use the interactive AI architect:

```bash
branes chat
```

**Example session:**

```
You: Design a perception pipeline for an agricultural robot that needs to:
     - Detect crops and weeds at 15 FPS
     - Estimate plant health from RGB images
     - Run on solar power (10W budget)
     - Operate in outdoor lighting conditions

Agent: I'll design an architecture optimized for your requirements.

Proposed Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ Agricultural Perception Pipeline                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Camera (15 FPS) ──→ Preprocess ──→ YOLOv8n ──→ Health Model    │
│       │                 2ms          8ms          5ms           │
│       │                                                         │
│       └──────────────→ Exposure Control (adaptive)              │
│                                                                 │
│  Total latency: 15ms (66 FPS capable)                           │
│  Power estimate: 7W (30% headroom)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Model recommendations:
  • YOLOv8n: Efficient detection, 3.2M parameters
  • MobileNetV3: Plant health classification, 2.5M parameters

Hardware recommendation: Coral Edge TPU
  • 2W inference power
  • Leaves 8W for camera, compute, and actuation
  • Excellent for solar-powered operation

Shall I deploy these models to Coral format?
```

---

## Target Platform Guides

### NVIDIA Jetson (Orin, Xavier, Nano)

**Ideal for**: High-performance edge AI with GPU acceleration

```bash
# Install dependencies
pip install embodied-ai-architect[jetson]

# Deploy with optimization
branes deploy run model.pt --target jetson --precision int8 \
  --calibration-data ./images --input-shape 1,3,640,640

# Transfer to device
scp deployments/model_int8.engine jetson@device:/models/
```

**Performance guide:**
| Model | FP32 | FP16 | INT8 |
|-------|------|------|------|
| YOLOv8n | 45ms | 22ms | 9ms |
| YOLOv8s | 85ms | 42ms | 18ms |
| ResNet50 | 28ms | 14ms | 6ms |
| MobileNetV3 | 12ms | 6ms | 3ms |

### AMD Ryzen AI NUC

**Ideal for**: Desktop, industrial, and robotics applications

```bash
# Install dependencies
pip install embodied-ai-architect[ryzen]

# Deploy for NPU acceleration
branes deploy run model.pt --target ryzen --precision int8 \
  --input-shape 1,3,640,640
```

**Compute allocation strategy:**
```
Perception (NPU)  →  Tracking (CPU)  →  Planning (CPU)  →  Control (CPU)
   YOLOv8             ByteTrack          RRT*/MPC           PID
   12ms               8ms                35ms               1ms
```

### Google Coral Edge TPU

**Ideal for**: Ultra-low-power, battery-operated systems

```bash
# Install dependencies
pip install embodied-ai-architect[coral]

# Deploy (INT8 only)
branes deploy run model.pt --target coral \
  --calibration-data ./images --input-shape 1,224,224,3
```

**Power efficiency:**
| Model | Latency | Power | Efficiency |
|-------|---------|-------|------------|
| MobileNetV2 | 3ms | 0.5W | 666 inf/J |
| EfficientDet-Lite | 12ms | 1.2W | 69 inf/J |
| YOLOv5n | 15ms | 1.5W | 44 inf/J |

### Stillwater KPU

**Ideal for**: Custom embodied AI applications with specialized requirements

The KPU is designed for applications that need:
- Deterministic sub-millisecond latency
- Integrated linear algebra acceleration
- Hard real-time guarantees
- Maximum energy efficiency

```bash
# Analyze whether KPU would benefit your application
branes chat

You: Would my 1kHz robot control loop benefit from KPU acceleration?

Agent: Let me analyze your control pipeline...

KPU Benefit Analysis:
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Operation           ┃ Current    ┃ With KPU   ┃ Improvement     ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ EKF update          │ 1.2ms      │ 0.15ms     │ 8x faster       │
│ MPC solve           │ 8ms        │ 0.4ms      │ 20x faster      │
│ Control loop total  │ 9.5ms      │ 0.6ms      │ Enables 1kHz    │
└─────────────────────┴────────────┴────────────┴─────────────────┘

Recommendation: Your application would significantly benefit from KPU.
Contact Stillwater for a design consultation.
```

---

## End-to-End Examples

### Example 1: Drone Perception Pipeline

**Goal**: Build a perception system for autonomous drone navigation

**Requirements**:
- 30 FPS real-time perception
- Object detection and tracking
- Obstacle distance estimation
- 15W power budget for 1-hour flight

```bash
# Start interactive design session
branes chat

You: Design a drone perception pipeline for autonomous navigation.
     Requirements: 30 FPS, obstacle detection, depth estimation, 15W power.

Agent: I'll design an optimized pipeline for your drone.

Recommended Architecture:
┌────────────────────────────────────────────────────────────────┐
│ Drone Perception Pipeline                                      │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌───────────────┐  │
│  │ Camera  │──→│ YOLOv8n  │──→│ByteTrack│──→│ Obstacle Map  │  │
│  │ 720p    │   │ INT8     │   │         │   │               │  │
│  │ 30 FPS  │   │ 9ms      │   │ 6ms     │   │ 3ms           │  │
│  └─────────┘   └──────────┘   └─────────┘   └───────────────┘  │
│       │                                                        │
│       │        ┌──────────────┐                                │
│       └───────→│ MiDaS-small  │ (depth estimation)             │
│                │ INT8 | 8ms   │                                │
│                └──────────────┘                                │
│                                                                │
│  Pipeline metrics:                                             │
│    Total latency: 26ms (38 FPS capable)                        │
│    Power: 11W (4W headroom)                                    │
│    ✓ Exceeds all requirements                                  │
│                                                                │
└────────────────────────────────────────────────────────────────┘

Target hardware: Jetson Orin Nano

Shall I deploy these models?

You: Yes, deploy to Jetson with INT8

Agent: Deploying models...

Deployment complete!
  • yolov8n_int8.engine: 4.8 MB, 9ms latency
  • midas_small_int8.engine: 12 MB, 8ms latency

  Combined pipeline: 26ms at 10.8W
  ✓ Ready for flight testing
```

### Example 2: Industrial Robot Control

**Goal**: Build a pick-and-place system with visual servoing

**Requirements**:
- 100Hz control loop for precision manipulation
- 30Hz visual feedback for object localization
- Force-feedback safety system
- 35W power budget

```bash
branes chat

You: Design a pick-and-place robot with visual servoing.
     100Hz control, 30Hz vision, force feedback, 35W budget.

Agent: This requires a multi-rate architecture. Let me design it.

Multi-Rate Robot Architecture:
┌───────────────────────────────────────────────────────────────┐
│ 100Hz Control Loop (10ms budget)                              │
│ ┌───────────────┐   ┌─────────────┐   ┌────────────────────┐  │
│ │ Force Sensors │──→│ Impedance   │──→│ Joint Controllers  │  │
│ │ 6-axis        │   │ Control     │   │ 6-DOF arm          │  │
│ │ <1ms read     │   │ 3ms         │   │ 2ms                │  │
│ └───────────────┘   └─────────────┘   └────────────────────┘  │
└───────────────────────────────────────────────────────────────┘
                              ▲
                              │ pose setpoints @ 30Hz
┌────────────────────────────────────────────────────────────────┐
│ 30Hz Vision Loop (33ms budget)                                 │
│ ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌─────────────┐  │
│ │ Stereo   │──→│ Detection │──→│ 6DOF     │──→│ Grasp       │  │
│ │ Cameras  │   │ YOLOv8s   │   │ Pose Est │   │ Planning    │  │
│ │ 3ms      │   │ 12ms NPU  │   │ 8ms      │   │ 6ms         │  │
│ └──────────┘   └───────────┘   └──────────┘   └─────────────┘  │
└────────────────────────────────────────────────────────────────┘

Target: Ryzen 9 AI NUC
  • NPU: Object detection (12ms, 3W)
  • GPU: Depth processing (5ms, 8W)
  • CPU: Control, planning (dedicated cores)
  • Total: 28W (7W headroom)

✓ Architecture validated for your requirements
```

### Example 3: Exploring KPU Benefits

**Goal**: Understand when specialized hardware enables new capabilities

```bash
branes chat

You: I'm building a high-speed drone for racing. I need 200Hz state
     estimation and 100Hz trajectory replanning. Is this possible?

Agent: Let me analyze the computational requirements.

High-Speed Drone Analysis:
┏━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Requirement             ┃ Time Budget   ┃ What It Enables          ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 200Hz State Estimation  │ 5ms           │ Stable flight at 50+ m/s │
│ 100Hz Trajectory Replan │ 10ms          │ Agile obstacle avoidance │
│ 500Hz Motor Control     │ 2ms           │ Precise attitude control │
└─────────────────────────┴───────────────┴──────────────────────────┘

Platform Comparison:
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Platform        ┃ State Est (EKF) ┃ Trajectory (MPC)┃ Achievable? ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ Jetson Orin     │ 1.8ms           │ 18ms            │ Partial     │
│ Ryzen AI NUC    │ 1.2ms           │ 12ms            │ Partial     │
│ Stillwater KPU  │ 0.15ms          │ 0.8ms           │ ✓ Full      │
└─────────────────┴─────────────────┴─────────────────┴─────────────┘

Analysis:
  • Existing platforms can achieve 200Hz state estimation
  • 100Hz MPC replanning requires specialized acceleration
  • KPU enables the full capability set for racing performance

The KPU's dedicated linear algebra and constraint solver hardware
enables real-time MPC that would otherwise require a much larger,
power-hungry system.

Would you like me to generate a KPU requirements specification?
```

---

## Best Practices

### Choosing the Right Model

| Application | Recommended Models | Why |
|-------------|-------------------|-----|
| Drone perception | YOLOv8n, MobileNet | Small, fast, power-efficient |
| Industrial inspection | YOLOv8s, EfficientDet | Balance of accuracy and speed |
| Autonomous vehicles | YOLOv8m, DETR | Higher accuracy for safety |
| Mobile robots | MobileNetV3, NanoDet | Optimized for edge CPUs |

### Optimization Strategies

1. **Quantization**: INT8 typically provides 2-3x speedup with <1% accuracy loss
2. **Model selection**: Smaller models often meet requirements with better efficiency
3. **Operator fusion**: Combine operations to reduce memory bandwidth
4. **Hardware matching**: Use NPU for inference, CPU for control, GPU for preprocessing

### Validation Checklist

Before deployment, verify:
- [ ] Mean latency meets requirements
- [ ] P99 latency provides safety margin
- [ ] Power consumption fits within budget
- [ ] Accuracy meets application needs
- [ ] System operates correctly under load

---

## Getting Help

### Resources

- **Model Zoo**: `branes zoo search --help`
- **Interactive Architect**: `branes chat`
- **Documentation**: `branes --help`

### Support

- **GitHub Issues**: Report bugs and request features
- **KPU Inquiries**: Contact Stillwater for custom acceleration needs

---

## Summary

The Branes Embodied AI Platform helps you build autonomous systems that work—systems that perceive, reason, and act in the real world within the constraints that matter: latency for responsiveness, power for mission duration, and accuracy for safety.

Whether you're deploying to existing hardware like Jetson or Coral, or exploring custom acceleration with the KPU, Branes provides the tools to design with confidence and deploy with certainty.

**Start building:**
```bash
pip install embodied-ai-architect
branes chat
```

The future of embodied AI is being built today. We're here to help you build it.
