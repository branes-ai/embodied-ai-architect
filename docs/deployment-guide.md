# Branes Embodied AI Platform: Deployment Guide

## Why This Platform Exists

### The Hidden Crisis in Embodied AI

Most organizations building autonomous systems—drones, robots, autonomous vehicles—are unknowingly creating **systems that cannot work**.

They select hardware based on benchmark scores, deploy neural networks that run "fast enough" in demos, and discover too late that their systems fail in the field. The failures are not bugs. They are **architectural impossibilities** baked in from day one.

**The root cause**: existing accelerators were never designed for embodied AI.

| Platform | Original Purpose | What It's Good At | What It Cannot Do |
|----------|------------------|-------------------|-------------------|
| **NVIDIA Jetson (GPU)** | 3D graphics, games | Parallel floating-point, CNN inference | Deterministic latency, real-time control, energy efficiency at edge |
| **Google Coral (Edge TPU)** | Cloud vision CNNs | INT8 matrix multiplication | Kalman filters, eigenvalue solvers, multi-rate control, sensor fusion |
| **Intel Movidius (VPU)** | Computer vision | Image processing pipelines | Linear algebra, Bayesian inference, constraint optimization |

These platforms excel at their designed purpose. But embodied AI is not their designed purpose.

### What Embodied AI Actually Requires

A perception, guidance, navigation & control (PGN&C) system demands capabilities that no existing accelerator provides:

| Capability | Why It's Essential | GPU/TPU Reality |
|------------|-------------------|-----------------|
| **High-bandwidth data acquisition** | Cameras, LiDAR, IMU, force sensors at 100Hz-1kHz | Designed for batch processing, not streaming |
| **Real-time signal processing** | Filter noise, fuse sensors, detect anomalies | No latency guarantees |
| **Linear algebra solvers** | Constraint optimization, inverse kinematics, MPC | Poor at small-matrix operations |
| **Eigenvalue decomposition** | Stability analysis, PCA, covariance estimation | Not accelerated |
| **Bayesian inference** | Kalman filters, particle filters, SLAM | Must run on CPU |
| **Statistical inference** | Uncertainty quantification, decision-making | Not a design priority |
| **Model Predictive Control** | Plan optimal trajectories under constraints | Iterative solvers are slow on GPUs |
| **Multi-rate control** | 1kHz motor control, 30Hz perception, 10Hz planning | Single-clock architectures struggle |

### Latency and Energy Are Functional Requirements

This is the insight most engineers miss:

> **In embodied AI, latency and energy are not performance metrics—they are functional requirements.**

If a drone's obstacle avoidance takes 100ms instead of 20ms, the drone crashes. It doesn't crash *slower*—it crashes. The latency requirement is binary: meet it or fail.

If a robot's PGN&C pipeline consumes 50W instead of 15W, the battery dies in 20 minutes instead of 60. The mission fails. The robot becomes a liability, not an asset.

**This is why benchmarks lie.** A system that achieves 50 FPS in a lab demo may be worthless if:
- Latency variance causes 1-in-1000 frames to take 200ms (crash)
- Power consumption prevents mission completion
- The GPU cannot run Kalman filters at the required rate alongside perception

### The Organizational Risk

When an autonomous system fails in deployment, the consequences extend beyond engineering:

- **Safety incidents** → regulatory scrutiny, liability
- **Mission failures** → lost revenue, damaged reputation
- **Delayed programs** → burned runway, missed market windows
- **Technical debt** → years spent working around architectural limitations

Most organizations discover these problems after committing to a hardware platform. By then, the only options are:
1. Accept degraded performance
2. Redesign the system (expensive, slow)
3. Cancel the program

The Branes platform exists to prevent this outcome.

---

## The Branes Solution

### Platform Overview

The Branes Embodied AI Platform is a **hardware/software codesign environment** that helps you:

1. **Identify** where latency and energy bottlenecks will occur—before you build
2. **Analyze** whether your target hardware can meet functional requirements
3. **Optimize** your architecture to eliminate bottlenecks
4. **Deploy** to the right hardware with validated performance
5. **Design** custom accelerators when no existing hardware suffices

```
┌────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        Branes Embodied AI Platform                                             │
├────────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                                │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐      ┌───────────┐  │
│  │  Model Zoo   │───→│   Analyze    │───→│  Hardware    │───→│  Deploy   │  ───→│    CPU    │  │
│  │              │    │              │    │  Profiler    │    │           │      └───────────┘  │
│  │ 100+ models  │    │ Compute/     │    │              │    │ TensorRT  │      ┌───────────┐  │
│  │ YOLO, ViT,   │    │ Memory/      │    │ 50+ targets  │    │ ONNX-RT   │  ───→│    GPU    │  │
│  │ EfficientNet │    │ Bandwidth    │    │ Bottleneck   │    │ OpenVINO  │      └───────────┘  │
│  └──────────────┘    └──────────────┘    │ Detection    │    │ Branes-RT │      ┌───────────┐  │
│                                          └──────────────┘    └───────────┘  ───→│    NPU    │  │
│                                                 │                               └───────────┘  │
│                                                 ▼                               ┌───────────┐  │
│                              ┌─────────────────────────────────┐            ───→│    KPU    │  │
│                              │     Benchmark & Validate        │                └───────────┘  │
│                              │                                 │                               │
│                              │  • Latency distribution         │                               │
│                              │  • Power consumption            │                               │
│                              │  • Functional requirement check │                               │
│                              └─────────────────────────────────┘                               │
│                                                                                                │
└────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### The Knowledge Processing Unit (KPU)

When existing hardware cannot meet your requirements, the Branes platform enables design of a **custom embodied AI accelerator**: the Knowledge Processing Unit (KPU).

The KPU is purpose-built for the computational patterns of embodied AI:

| KPU Capability | Implementation | Embodied AI Application |
|----------------|----------------|------------------------|
| **Streaming data acquisition** | Hardware DMA, sensor interfaces | Camera, LiDAR, IMU fusion at 1kHz |
| **Real-time linear algebra** | Systolic array, configurable precision | Kalman filters, MPC, inverse kinematics |
| **Eigenvalue acceleration** | QR iteration hardware | Stability analysis, PCA |
| **Constraint solver** | Interior point method accelerator | Trajectory optimization, collision avoidance |
| **Multi-rate scheduler** | Hardware task dispatcher | 1kHz control, 30Hz perception, 10Hz planning |
| **Deterministic latency** | Worst-case execution time guarantees | Hard real-time requirements |
| **Energy-proportional compute** | Clock/voltage scaling per operator | Mission-duration power budgets |

**The KPU is not a general-purpose accelerator.** It is a specialized engine for the specific computational needs of PGN&C systems—the workloads that GPUs and TPUs handle poorly or not at all.

### How Branes Helps You

The platform provides tools for every stage of embodied AI development:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Development Workflow                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. REQUIREMENTS CAPTURE                                                    │
│     ├─ Latency budget (e.g., 33ms for 30 FPS)                               │
│     ├─ Power budget (e.g., 15W for 1-hour flight)                           │
│     ├─ Accuracy requirements (e.g., mAP > 0.5)                              │
│     └─ Throughput requirements (e.g., 6 cameras × 30 FPS)                   │
│                                                                             │
│  2. ARCHITECTURE DESIGN                                                     │
│     ├─ Select operators (detection, tracking, control)                      │
│     ├─ Define data flow and rates                                           │
│     ├─ Allocate operators to compute units                                  │
│     └─ Estimate latency and power                                           │
│                                                                             │
│  3. BOTTLENECK ANALYSIS                                                     │
│     ├─ Identify critical path                                               │
│     ├─ Find compute/memory/bandwidth bottlenecks                            │
│     ├─ Quantify gap to requirements                                         │
│     └─ Generate optimization recommendations                                │
│                                                                             │
│  4. OPTIMIZATION                                                            │
│     ├─ Quantization (FP32 → INT8)                                           │
│     ├─ Operator fusion                                                      │
│     ├─ Hardware reallocation (GPU → NPU)                                    │
│     └─ Algorithm substitution (EKF → complementary filter)                  │
│                                                                             │
│  5. VALIDATION                                                              │
│     ├─ Benchmark on target hardware                                         │
│     ├─ Measure latency distribution (P50, P99, P99.9)                       │
│     ├─ Measure power under load                                             │
│     └─ Verify functional requirements met                                   │
│                                                                             │
│  6. DEPLOYMENT                                                              │
│     ├─ Export optimized models                                              │
│     ├─ Generate deployment artifacts                                        │
│     └─ Produce validation report                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Software Requirements

```bash
# Core platform
pip install embodied-ai-architect

# For Jetson deployment
pip install embodied-ai-architect[jetson]

# For interactive design sessions
pip install embodied-ai-architect[chat]
export ANTHROPIC_API_KEY=your-key-here

# For Kubernetes-based benchmarking
pip install embodied-ai-architect[kubernetes]
```

### Supported Hardware Targets

| Target | Precision | Use Case | Installation |
|--------|-----------|----------|--------------|
| **Jetson Orin** | FP32, FP16, INT8 | High-performance edge | `[jetson]` - requires TensorRT |
| **Jetson Nano** | FP16, INT8 | Cost-constrained edge | `[jetson]` - requires TensorRT |
| **Coral Edge TPU** | INT8 only | Ultra-low power | `[coral]` - requires Edge TPU runtime |
| **Ryzen AI NUC** | FP32, FP16, INT8 | Desktop/industrial | `[ryzen]` - requires ONNX Runtime + Ryzen AI EP |
| **Intel OpenVINO** | FP32, FP16, INT8 | Intel CPUs/VPUs | `[openvino]` - requires OpenVINO |
| **Stillwater KPU** | Configurable | Custom embodied AI | Contact Stillwater for SDK |

### Data Requirements

For INT8 quantization, you need:
- **Calibration dataset**: 100-500 representative images
- **Validation dataset**: Test set for accuracy verification

---

## Workflow: From Model to Deployment

### Step 1: Select a Model

Use the Model Zoo to find models suited to your task:

```bash
# Search for detection models under 10M parameters
branes zoo search --task detection --max-params 10M

# Download YOLOv8n for edge deployment
branes zoo download yolov8n --format onnx
```

**Available model families:**
- Detection: YOLOv5/v8/v11, DETR, EfficientDet
- Classification: ResNet, MobileNet, EfficientNet, ViT
- Segmentation: SegFormer, Mask R-CNN
- Depth: DPT, MiDaS
- Pose: MediaPipe, YOLOv8-pose

### Step 2: Analyze Model Requirements

```bash
# Analyze compute and memory requirements
branes analyze yolov8n.onnx

# Output:
#   Parameters: 3.2M
#   FLOPs: 8.7G
#   Memory: 12.4 MB (FP32)
#   Layer breakdown:
#     Conv2d: 72% compute
#     Upsample: 12% compute
#     Concat: 8% memory bandwidth
```

### Step 3: Profile Hardware Fit

```bash
# Get hardware recommendations with constraints
branes workflow run yolov8n.onnx \
  --latency-target 33 \
  --power-budget 15 \
  --use-case drone
```

**Example output:**
```
Hardware Recommendations for YOLOv8n
=====================================

Constraints:
  Latency target: ≤33ms
  Power budget: ≤15W
  Use case: drone

Top Recommendations:
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Hardware         ┃ Est. Time ┃ Power    ┃ Meets Reqs?  ┃ Score  ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Jetson Orin Nano │ 18ms      │ 10W      │ ✓ Yes        │ 0.92   │
│ Coral Edge TPU   │ 12ms      │ 2W       │ ✓ Yes        │ 0.88   │
│ Jetson Nano      │ 45ms      │ 8W       │ ✗ Latency    │ 0.61   │
│ Ryzen AI NUC     │ 8ms       │ 25W      │ ✗ Power      │ 0.54   │
└──────────────────┴───────────┴──────────┴──────────────┴────────┘

⚠ Warning: Jetson Nano cannot meet 33ms latency requirement
  Recommendation: Use Jetson Orin Nano or apply INT8 quantization
```

### Step 4: Identify Bottlenecks

```bash
# Interactive session for bottleneck analysis
branes chat
```

```
You: Analyze the drone_perception_v1 architecture for Jetson Orin Nano

Agent: [analyze_architecture]

Architecture Analysis: drone_perception_v1
==========================================

Pipeline: Camera → Preprocess → Detect → Track → Scene Graph
Target: Jetson Orin Nano (10W)

Operator Breakdown:
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Operator      ┃ Latency  ┃ % Total ┃ Bottleneck                       ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Preprocess    │ 2.1ms    │ 6%      │                                  │
│ YOLOv8n       │ 18.4ms   │ 55%     │ ← COMPUTE BOUND (GPU saturated)  │
│ ByteTrack     │ 8.2ms    │ 25%     │ ← MEMORY BOUND (association)     │
│ Scene Graph   │ 4.8ms    │ 14%     │                                  │
├───────────────┼──────────┼─────────┼──────────────────────────────────┤
│ TOTAL         │ 33.5ms   │ 100%    │ ✗ Exceeds 33ms target by 0.5ms   │
└───────────────┴──────────┴─────────┴──────────────────────────────────┘

Critical Path: Preprocess → Detect → Track → Scene Graph (sequential)

Recommendations:
1. Quantize YOLOv8n to INT8: 18.4ms → ~9ms (50% reduction)
2. Optimize ByteTrack association: use spatial hashing for O(n) instead of O(n²)
3. Pipeline scene graph update: overlap with next frame preprocessing
```

### Step 5: Deploy with Optimization

```bash
# Deploy to Jetson with INT8 quantization
branes deploy run yolov8n.pt \
  --target jetson \
  --precision int8 \
  --calibration-data ./calibration_images \
  --input-shape 1,3,640,640 \
  --test-data ./validation_images \
  --validate

# Output:
# Step 1: Preparing ONNX model...
#   Exporting to ONNX...
#   ONNX exported: deployments/model.onnx
#
# Step 2: Deploying to jetson with int8 precision...
#   Running INT8 calibration with 100 samples...
#   Calibration complete, building engine...
#   Engine saved: deployments/model_int8.engine
#   Engine size: 4.82 MB
#
# Step 3: Running validation...
#   Validation passed: True
#   Speedup: 2.1x
#   Max output diff: 0.0023
#
# Deployment completed
#   Engine: deployments/model_int8.engine
#   Precision: int8
#   Latency: 8.7ms (was 18.4ms)
```

### Step 6: Validate Functional Requirements

```bash
# Benchmark the deployed model
branes benchmark run deployments/model_int8.engine \
  --iterations 1000 \
  --report latency-distribution

# Output:
# Latency Distribution (1000 iterations)
# ======================================
#   P50:   8.7ms
#   P90:   9.1ms
#   P99:   10.2ms
#   P99.9: 12.8ms
#   Max:   14.1ms
#
# ✓ P99.9 (12.8ms) < target (33ms): PASS
# ✓ Deterministic: 99.9% of frames under 13ms
```

---

## Target-Specific Guides

### Jetson (Orin, Xavier, Nano)

**Best for**: High-performance edge inference with power constraints

**Workflow**:
```bash
# 1. Install TensorRT dependencies
pip install embodied-ai-architect[jetson]

# 2. Deploy with INT8 for maximum performance
branes deploy run model.pt --target jetson --precision int8 \
  --calibration-data ./images --input-shape 1,3,640,640

# 3. Copy engine to Jetson device
scp deployments/model_int8.engine jetson:/models/
```

**Performance expectations**:
| Model | FP32 | FP16 | INT8 |
|-------|------|------|------|
| YOLOv8n | 45ms | 22ms | 9ms |
| YOLOv8s | 85ms | 42ms | 18ms |
| ResNet50 | 28ms | 14ms | 6ms |

### Ryzen AI NUC

**Best for**: Desktop/industrial applications with NPU acceleration

**Workflow**:
```bash
# 1. Install ONNX Runtime with Ryzen AI EP
pip install embodied-ai-architect[ryzen]

# 2. Export to ONNX with NPU-compatible ops
branes deploy run model.pt --target ryzen --precision fp16 \
  --input-shape 1,3,640,640

# 3. Profile NPU vs GPU allocation
branes benchmark arch drone_perception_v1 --target ryzen-ai-nuc-7
```

**Operator allocation strategy**:
```
Camera → Preprocess → Detect → Track → Scene Graph → Control
  30Hz      GPU         NPU     CPU       CPU         CPU
             ↓           ↓                              ↓
          Radeon     XDNA NPU                      Real-time
          780M       10 TOPS                       thread
```

### Coral Edge TPU

**Best for**: Ultra-low-power deployment (<2W)

**Constraints**:
- INT8 only
- Limited operator support
- Single-model execution

**Workflow**:
```bash
# 1. Check model compatibility
branes analyze model.pt --target coral

# 2. Convert with Edge TPU compiler
branes deploy run model.pt --target coral \
  --calibration-data ./images --input-shape 1,224,224,3
```

### Stillwater KPU (Custom Accelerator)

**Best for**: When existing hardware cannot meet requirements

**When to consider KPU**:
- Latency requirements below 1ms for control loops
- Power budget below 5W with full PGN&C pipeline
- Need for deterministic worst-case execution time
- Linear algebra workloads (Kalman, MPC, optimization)

**Workflow**:
```bash
# 1. Profile your architecture to identify gaps
branes chat

You: Analyze drone_control_v1 for 1kHz control loop

Agent: [analyze_architecture]

Gap Analysis:
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Requirement           ┃ Target       ┃ Jetson Orin  ┃ Gap      ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Control loop latency  │ <1ms         │ 3.2ms        │ 2.2ms    │
│ EKF update            │ <500µs       │ 1.8ms        │ 1.3ms    │
│ MPC solve (10 steps)  │ <5ms         │ 18ms         │ 13ms     │
│ Total power           │ <8W          │ 15W          │ 7W       │
└───────────────────────┴──────────────┴──────────────┴──────────┘

⚠ No existing hardware can meet these requirements.
  Recommendation: Custom KPU with:
    - Dedicated EKF accelerator
    - MPC constraint solver
    - 1kHz hard real-time scheduler

# 2. Generate KPU requirements specification
branes design kpu-spec drone_control_v1 --output kpu_requirements.json

# 3. Contact Stillwater for KPU design engagement
```

---

## End-to-End Examples

### Example 1: Drone Perception Pipeline on Jetson Orin

**Objective**: Deploy a complete perception pipeline for autonomous drone navigation

**Requirements**:
- 30 FPS perception (33ms latency budget)
- <15W power consumption
- >0.5 mAP detection accuracy

**Step-by-step**:

```bash
# 1. Start interactive session
branes chat

# 2. Design the architecture
You: Design a drone perception pipeline for Jetson Orin Nano with:
     - 30 FPS from 720p camera
     - Object detection and tracking
     - Obstacle distance estimation
     - Power under 15W

Agent: I'll design a pipeline and validate it meets your requirements.

[Creates architecture with operators: Preprocess, YOLOv8n, ByteTrack, DepthEstimator]
[Analyzes latency: 28.3ms total - within budget]
[Analyzes power: 11.2W - within budget]

Recommended Architecture:
┌─────────┐   ┌──────────┐   ┌─────────┐   ┌───────────┐
│ Camera  │──→│ Preproc  │──→│ YOLOv8n │──→│ ByteTrack │
│ 720p    │   │ 2ms      │   │ 9ms     │   │ 8ms       │
└─────────┘   └──────────┘   └─────────┘   └───────────┘
                                                  │
                             ┌─────────────┐      │
                             │ MiDaS-small │◀─────┘
                             │ 9ms         │
                             └─────────────┘

Total: 28ms | Power: 11.2W | ✓ Meets all requirements

# 3. Deploy the detection model
You: Deploy the YOLOv8n model with INT8 quantization

Agent: [deploy_model] Deploying to Jetson with INT8...

Deployment complete:
  Engine: yolov8n_int8.engine
  Latency: 8.7ms (down from 18.4ms)
  Size: 4.8 MB
  Validation: PASSED (accuracy within 1%)

# 4. Benchmark the full pipeline
You: Benchmark the full pipeline with power monitoring

Agent: [benchmark_architecture]

Full Pipeline Benchmark (1000 frames):
  Mean latency: 27.8ms (36 FPS)
  P99 latency: 31.2ms
  Power: 10.8W

  ✓ All functional requirements met
```

### Example 2: Robot Control on Ryzen AI NUC

**Objective**: Deploy a pick-and-place robot controller with perception and planning

**Requirements**:
- 100Hz control loop (10ms)
- 30Hz perception (33ms)
- <35W total power

```bash
# 1. Analyze the multi-rate architecture
branes chat

You: Design a pick-and-place robot pipeline for Ryzen 9 AI NUC with:
     - 100Hz force-feedback control
     - 30Hz visual perception
     - 10Hz trajectory planning
     - Collision avoidance

Agent: This requires multi-rate scheduling. Let me design the architecture.

Multi-Rate Architecture:
┌─────────────────────────────────────────────────────────────────┐
│ 100Hz Control Loop (10ms budget)                                │
│ ┌─────────────┐   ┌──────────────┐   ┌───────────────────────┐  │
│ │ Force/Torque│──→│ Impedance    │──→│ Joint Controller      │  │
│ │ Sensors     │   │ Control      │   │ (PID)                 │  │
│ │ <1ms        │   │ 2ms          │   │ 1ms                   │  │
│ └─────────────┘   └──────────────┘   └───────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ setpoints @ 30Hz
┌────────────────────────────────────────────────────────────────┐
│ 30Hz Perception Loop (33ms budget)                             │
│ ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌────────────────┐  │
│ │ Stereo  │──→│ Detect   │──→│ 6DOF    │──→│ Collision      │  │
│ │ Camera  │   │ (NPU)    │   │ Pose    │   │ Check          │  │
│ │ 3ms     │   │ 12ms     │   │ 8ms     │   │ 5ms            │  │
│ └─────────┘   └──────────┘   └─────────┘   └────────────────┘  │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼ obstacles @ 10Hz
┌────────────────────────────────────────────────────────────────┐
│ 10Hz Planning Loop (100ms budget)                              │
│ ┌─────────────────────┐   ┌─────────────────────────────────┐  │
│ │ Motion Planner      │──→│ Trajectory Optimizer (MPC)      │  │
│ │ (RRT*)              │   │ 45ms on CPU                     │  │
│ │ 35ms                │   │                                 │  │
│ └─────────────────────┘   └─────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘

Allocation:
  NPU (XDNA): Detection (YOLOv8s)
  GPU (780M): Depth estimation, preprocessing
  CPU: Control, planning, tracking

Power estimate: 28W (within 35W budget)

# 2. Identify NPU deployment path
You: Deploy the detection model to Ryzen AI NPU

Agent: [deploy_model] Converting for ONNX Runtime with Ryzen AI EP...

Ryzen AI Deployment:
  Format: ONNX with INT8 quantization
  Execution Provider: VitisAI EP
  Latency: 11.8ms on NPU
  Power: 3.2W (NPU only)
```

### Example 3: When Hardware Falls Short—KPU Design

**Objective**: Sub-millisecond control for high-speed manipulation

**Requirements**:
- 1kHz control loop (<1ms latency)
- EKF sensor fusion at 500Hz
- <5W power budget

```bash
# 1. Analyze feasibility on existing hardware
branes chat

You: Can I run 1kHz model-predictive control on Jetson Orin?

Agent: [analyze_architecture] Let me check...

Gap Analysis for 1kHz MPC:
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Operation           ┃ Required  ┃ Jetson Orin  ┃ Feasible? ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Sensor read         │ <100µs    │ 50µs         │ ✓         │
│ EKF update (12-state)│ <200µs   │ 1.2ms        │ ✗         │
│ MPC solve (10-step) │ <500µs    │ 8ms          │ ✗         │
│ Actuator write      │ <50µs     │ 30µs         │ ✓         │
├─────────────────────┼───────────┼──────────────┼───────────┤
│ Total               │ <1ms      │ 9.3ms        │ ✗ 9.3x gap│
└─────────────────────┴───────────┴──────────────┴───────────┘

⚠ CRITICAL GAP: No existing hardware can meet 1kHz MPC requirements.

Root cause: GPU is optimized for throughput (batch processing), not
latency (single-sample real-time). Linear algebra operations like
EKF and MPC run on CPU, which cannot achieve sub-millisecond latency.

Recommendation: Custom accelerator (KPU) with:
  - Dedicated matrix multiply unit for 12×12 state matrices
  - Hardened QP solver for MPC constraints
  - Deterministic 200µs EKF, 400µs MPC
  - Total: 650µs control loop

You: Generate a KPU requirements specification

Agent: [generate_kpu_spec]

KPU Requirements Specification
==============================

Target Application: High-speed manipulation control
Control Rate: 1kHz (1ms period)

Compute Requirements:
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Operation           ┃ Specification                            ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Matrix multiply     │ 12×12 × 12×1 in <10µs                    │
│ Cholesky decomp     │ 12×12 in <30µs                           │
│ QP solve            │ 10 variables, 20 constraints in <300µs   │
│ Sensor interface    │ 6-axis IMU + encoders @ 1kHz             │
│ Actuator interface  │ 6 motors @ 1kHz PWM                      │
└─────────────────────┴──────────────────────────────────────────┘

Power Budget: 5W total
  - Compute: 3W
  - I/O: 1W
  - Memory: 1W

Real-Time Requirements:
  - WCET guarantee: 800µs (200µs margin)
  - Jitter: <10µs
  - No dynamic memory allocation

Output: kpu_requirements.json

Next step: Contact Stillwater for KPU architecture design.
```

---

## Troubleshooting

### "Target not available"

```bash
branes deploy list
# Shows: jetson - No

# Solution: Install target-specific dependencies
pip install embodied-ai-architect[jetson]
```

### "INT8 calibration failed"

Common causes:
1. **Insufficient calibration data**: Need 100+ representative images
2. **Data format mismatch**: Images must match training preprocessing
3. **Out of memory**: Reduce batch size with `--calibration-batch-size 1`

### "Latency exceeds target"

Use bottleneck analysis:
```bash
branes chat
You: Why is my pipeline slower than expected?

Agent: [identify_bottleneck]
  Bottleneck: ByteTrack association taking 15ms
  Cause: O(n²) Hungarian algorithm with 200 detections
  Solution: Reduce max detections or use spatial hashing
```

### "Power exceeds budget"

```bash
branes benchmark arch my_pipeline --power-profile

# Identify high-power operators and consider:
# 1. INT8 quantization (2-3x power reduction)
# 2. Smaller model variants
# 3. Reduce frame rate
# 4. Offload to lower-power compute unit
```

---

## Summary: Why Branes Matters

The Branes Embodied AI Platform exists because **building autonomous systems is harder than it appears**.

Most teams discover—too late—that their hardware cannot meet latency requirements, their power budget is blown, or their control loops are unstable. These are not software bugs. They are architectural failures that require hardware changes to fix.

Branes helps you:
1. **Understand requirements** before committing to hardware
2. **Identify bottlenecks** before they become blockers
3. **Validate solutions** with real measurements
4. **Design custom accelerators** when no existing hardware suffices

The Knowledge Processing Unit represents the endgame: purpose-built silicon for embodied AI, designed from first principles for the computational patterns of perception, guidance, navigation, and control.

**Latency and energy are functional requirements.** Branes ensures you can meet them.

---

## Next Steps

- **Explore the Model Zoo**: `branes zoo search --task detection`
- **Analyze your model**: `branes analyze your_model.pt`
- **Start an interactive session**: `branes chat`
- **Contact Stillwater**: For KPU design engagements, reach out at [stillwater-sc.com](https://stillwater-sc.com)
