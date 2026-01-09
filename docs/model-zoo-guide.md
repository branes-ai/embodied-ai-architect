# Model Zoo and Pipeline Design Guide

This guide explains how to use the Branes Model Zoo, Requirements Wizard, and Testbench to design, validate, and deploy perception pipelines for embodied AI systems.

## Purpose

Building perception systems for embodied AI (drones, robots, autonomous vehicles) requires solving several interconnected challenges:

1. **Model Selection**: Choosing the right model from hundreds of options across multiple providers
2. **Constraint Satisfaction**: Meeting accuracy, latency, and power requirements for edge deployment
3. **Acquisition**: Downloading and converting models to deployment-ready formats
4. **Validation**: Ensuring models meet accuracy requirements before deployment
5. **Monitoring**: Detecting performance drift in production

The Branes platform addresses these challenges with an integrated workflow:

```
Requirements → Model Discovery → Acquisition → Validation → Pipeline → Deployment
     ↓              ↓                ↓             ↓           ↓           ↓
  Wizard         Zoo Search       Download     Testbench    Generate    Monitor
```

## The Embodied AI Workflow

### Traditional Approach (Manual)

```
1. Research models online
2. Download from various sources
3. Convert to ONNX/TensorRT manually
4. Write custom integration code
5. Benchmark on target hardware
6. Hope it works in production
```

### Branes Approach (Automated)

```bash
# 1. Define requirements
branes design new -o requirements.yaml

# 2. Synthesize pipeline (finds models, downloads, validates)
branes design synthesize requirements.yaml --validate

# 3. Run pipeline
branes pipeline run pipeline.yaml

# 4. Monitor for drift
branes testbench drift yolov8n
```

---

## Model Zoo

The Model Zoo provides unified access to 187+ models across 6 providers, all optimized for edge deployment.

### Providers

| Provider | Models | Focus | Example Models |
|----------|--------|-------|----------------|
| **Ultralytics** | 15 | Real-time detection | YOLOv5n, YOLOv8s, YOLO11n |
| **TorchVision** | 34 | Classic architectures | ResNet, EfficientNet, MobileNet |
| **HuggingFace** | 27 | Transformers | ViT, DETR, SegFormer, DPT |
| **Timm** | 53 | Edge-optimized | FastViT, EdgeNeXt, GhostNet |
| **ONNX Zoo** | 30 | Deployment-ready | MiDaS, Ultraface, YOLO |
| **MediaPipe** | 28 | Real-time ML | Hand tracking, pose, face mesh |

### Searching for Models

```bash
# Find all detection models
branes zoo search --task detection

# Find models under 10M parameters
branes zoo search --task detection --max-params 10

# Find models with >60% accuracy
branes zoo search --task detection --min-accuracy 0.6

# Search specific provider
branes zoo list --provider mediapipe

# Get detailed model info
branes zoo info yolov8n
```

### Downloading Models

```bash
# Download to cache (~/.cache/branes/models/)
branes zoo download yolov8n

# Download specific format
branes zoo download yolov8n --format onnx

# View cached models
branes zoo list --cached

# Clear cache
branes zoo clear
```

### Programmatic Access

```python
from embodied_ai_architect.model_zoo import acquire, discover

# Find models matching constraints
candidates = discover(
    task="detection",
    max_params=10_000_000,  # 10M params
    min_accuracy=0.5,       # 50% mAP
)

for model in candidates[:5]:
    print(f"{model.name}: {model.format_params()}, {model.format_accuracy()}")

# Acquire a model (downloads if needed)
model_path = acquire("yolov8n", format="onnx")
print(f"Model ready at: {model_path}")
```

---

## Requirements Wizard

The Requirements Wizard guides you through defining pipeline requirements and automatically synthesizes a matching pipeline.

### Interactive Mode

```bash
branes design new -o requirements.yaml
```

This launches an interactive wizard that asks about:
- Perception tasks (detection, classification, segmentation, pose, depth)
- Target classes (person, car, drone, etc.)
- Accuracy constraints (minimum mAP, top-1, etc.)
- Latency constraints (maximum inference time)
- Hardware targets (CPU, GPU, NPU, edge)
- Power and memory budgets
- Deployment configuration (runtime, quantization)

### Requirements YAML Format

```yaml
name: drone-perception-pipeline
description: Multi-task perception for autonomous drone navigation

perception:
  tasks:
    - object_detection
    - depth_estimation
  target_classes:
    - person
    - vehicle
    - obstacle
  min_accuracy: 0.5          # 50% mAP minimum
  max_latency_ms: 50         # 50ms max inference time
  min_fps: 20                # 20 FPS minimum

hardware:
  execution_target: gpu       # cpu, gpu, npu, edge
  max_power_watts: 15         # Power budget
  max_memory_mb: 512          # Memory limit
  max_params_millions: 20     # Model size limit

deployment:
  runtime: onnxruntime        # onnxruntime, tensorrt, openvino
  quantization: fp16          # fp32, fp16, int8
  batch_size: 1
```

### From Use Case Templates

```bash
# Load from embodied-schemas catalog
branes design from-usecase drone_obstacle_avoidance -o requirements.yaml
branes design from-usecase industrial_inspection -o requirements.yaml
```

### Viewing Requirements

```bash
branes design show requirements.yaml
```

Output:
```
╭───────────────────── drone-perception-pipeline ────────────────────╮
│ Pipeline: drone-perception-pipeline                                │
│   Multi-task perception for autonomous drone navigation            │
│   Tasks: object_detection, depth_estimation                        │
│   Classes: person, vehicle, obstacle                               │
│   Constraints: accuracy≥50%, latency≤50ms                          │
│   Target: gpu                                                      │
╰────────────────────────────────────────────────────────────────────╯
```

---

## Pipeline Synthesis

The synthesize command finds matching models, downloads them, and generates a ready-to-run pipeline.

### Basic Synthesis

```bash
branes design synthesize requirements.yaml
```

Output:
```
Synthesizing pipeline: drone-perception-pipeline

Step 1: Finding models...
  ✓ object_detection: YOLOv5 Nano
  ✓ depth_estimation: MiDaS v2 Small

                            Selected Models
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┓
┃ Task             ┃ Model          ┃ Provider    ┃ Params ┃ Accuracy ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━┩
│ object_detection │ YOLOv5 Nano    │ ultralytics │   1.9M │    46.0% │
│ depth_estimation │ MiDaS v2 Small │ onnx-zoo    │  21.0M │      N/A │
└──────────────────┴────────────────┴─────────────┴────────┴──────────┘

Step 2: Acquiring models...
  Acquiring yolov5n... ✓
  Acquiring midas-small... ✓

Step 3: Generating pipeline...

✓ Pipeline saved to pipeline.yaml
```

### Synthesis with Validation

```bash
branes design synthesize requirements.yaml --validate
```

Adds a validation step that benchmarks each model:
```
Step 3: Validating models...
  Benchmarking yolov5n... ✓ 12.3ms (81.3 FPS)
  Benchmarking midas-small... ✓ 45.2ms (22.1 FPS)
```

### Dry Run Mode

Preview model selection without downloading:

```bash
branes design synthesize requirements.yaml --dry-run
```

### Generated Pipeline

The output `pipeline.yaml` is ready to run:

```yaml
name: drone-perception-pipeline
description: Multi-task perception for autonomous drone navigation
operators:
  - id: object_detection_operator
    type: YOLOv8ONNX
    config:
      model_id: yolov5n
      model_path: /home/user/.cache/branes/models/ultralytics/yolov5n.onnx
  - id: depth_estimation_operator
    type: DepthEstimator
    config:
      model_id: midas-small
      model_path: /home/user/.cache/branes/models/onnx-zoo/midas-small.onnx
execution:
  target: gpu
  runtime: onnxruntime
  batch_size: 1
  quantization: fp16
```

---

## Testbench

The Testbench validates model accuracy and monitors for performance drift over time.

### Inference Benchmarking

Measure latency and throughput:

```bash
branes testbench benchmark model.onnx
```

Output:
```
Benchmarking yolov8n.onnx
  Input shape: (1, 3, 640, 640)
  Iterations: 100
  Providers: CUDAExecutionProvider, CPUExecutionProvider

      Benchmark Results
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ Metric          ┃    Value ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│ Average Latency │ 17.53 ms │
│ Std Dev         │  0.38 ms │
│ P50 Latency     │ 17.37 ms │
│ P95 Latency     │ 18.33 ms │
│ P99 Latency     │ 18.50 ms │
│ Throughput      │ 57.1 FPS │
└─────────────────┴──────────┘
```

### Accuracy Validation

Validate against a ground truth dataset:

```bash
branes testbench validate model.onnx --dataset validation.json --task detection
```

Dataset format:
```json
{
  "samples": [
    {
      "input": "path/to/image.jpg",
      "ground_truth": {
        "boxes": [[10, 20, 100, 200], [150, 50, 300, 250]],
        "labels": [0, 1]
      }
    }
  ]
}
```

### Drift Monitoring

Track performance over time and detect degradation:

```bash
# Record validation results
branes testbench validate model.onnx --dataset week1.json --record

# Check for drift (compares recent to baseline)
branes testbench drift yolov8n
```

Output:
```
╭─────────────────────── Drift Report ───────────────────────╮
│ Drift Report: ⚠ WARNING ⚠                                 │
│ Model: yolov8n                                             │
│ Metric: mAP@50                                             │
│ Baseline: 52.3%                                            │
│ Current: 48.1%                                             │
│ Delta: -8.0%                                               │
│                                                            │
│ Recommendations:                                           │
│   • Model performance is declining                         │
│   • Monitor closely for further degradation                │
│   • Consider collecting more validation data               │
╰────────────────────────────────────────────────────────────╯
```

### Validation History

View past validation results:

```bash
branes testbench history yolov8n
```

```
              Validation History: yolov8n
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Timestamp           ┃ Dataset  ┃ Status ┃ mAP@50  ┃ Latency  ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ 2026-01-08T10:00:00 │ week1    │ ✓      │  52.3%  │  17.5ms  │
│ 2026-01-08T11:00:00 │ week2    │ ✓      │  51.8%  │  17.6ms  │
│ 2026-01-08T12:00:00 │ week3    │ ✓      │  48.1%  │  17.4ms  │
└─────────────────────┴──────────┴────────┴─────────┴──────────┘
```

---

## Complete Workflow Examples

### Example 1: Drone Obstacle Avoidance

```bash
# 1. Create requirements
cat > drone-requirements.yaml << 'EOF'
name: drone-avoidance
description: Real-time obstacle detection for autonomous drone
perception:
  tasks:
    - object_detection
  target_classes:
    - person
    - vehicle
    - tree
    - building
  min_accuracy: 0.45
  max_latency_ms: 33  # 30 FPS requirement
hardware:
  execution_target: gpu
  max_power_watts: 10
  max_params_millions: 5
deployment:
  runtime: tensorrt
  quantization: fp16
EOF

# 2. Synthesize with validation
branes design synthesize drone-requirements.yaml --validate -o drone-pipeline.yaml

# 3. View the generated pipeline
cat drone-pipeline.yaml

# 4. Benchmark on target hardware
branes testbench benchmark ~/.cache/branes/models/ultralytics/yolov8n.onnx
```

### Example 2: Industrial Inspection

```bash
# 1. Interactive requirements definition
branes design new -o inspection-requirements.yaml
# Select: classification, segmentation
# Set: min_accuracy 0.9, max_latency 100ms
# Target: cpu (no GPU on inspection device)

# 2. Preview model selection
branes design synthesize inspection-requirements.yaml --dry-run

# 3. Full synthesis
branes design synthesize inspection-requirements.yaml --validate

# 4. Validate against test dataset
branes testbench validate pipeline-models/classifier.onnx \
  --dataset test-images.json \
  --task classification \
  --threshold 0.9
```

### Example 3: Hand Tracking for HRI

```bash
# 1. Search for hand tracking models
branes zoo search --task hand_tracking

# 2. Download MediaPipe hand tracker
branes zoo download hand_landmarker

# 3. Benchmark latency
branes testbench benchmark ~/.cache/branes/models/mediapipe/hand_landmarker.task

# 4. Create simple requirements
cat > hand-tracking.yaml << 'EOF'
name: hand-tracking-hri
perception:
  tasks:
    - hand_tracking
  max_latency_ms: 30
hardware:
  execution_target: cpu
EOF

# 5. Synthesize
branes design synthesize hand-tracking.yaml
```

### Example 4: Multi-Model Pipeline with Monitoring

```bash
# 1. Create multi-task requirements
cat > perception-stack.yaml << 'EOF'
name: full-perception-stack
description: Complete perception for mobile robot
perception:
  tasks:
    - object_detection
    - depth_estimation
    - pose_estimation
  min_accuracy: 0.5
  max_latency_ms: 100
hardware:
  execution_target: gpu
  max_params_millions: 50
EOF

# 2. Synthesize pipeline
branes design synthesize perception-stack.yaml --validate

# 3. Run weekly validation
branes testbench validate detection-model.onnx --dataset week1.json --record
branes testbench validate detection-model.onnx --dataset week2.json --record
branes testbench validate detection-model.onnx --dataset week3.json --record

# 4. Check for drift
branes testbench drift detection-model

# 5. View history
branes testbench history detection-model --limit 10
```

---

## Integration with Operators

The Model Zoo integrates directly with Branes operators. When no model path is specified, operators automatically acquire models:

```python
from embodied_ai_architect.operators.perception import YOLOv8ONNX

# Operator automatically downloads yolov8n if not cached
detector = YOLOv8ONNX(variant="n")
detector.setup(config={}, execution_target="gpu")

# Or specify a model explicitly
detector.setup(
    config={"model_path": "/path/to/custom/model.onnx"},
    execution_target="gpu"
)
```

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Commands                            │
│  branes design new/show/synthesize                              │
│  branes zoo search/download/list/info                           │
│  branes testbench validate/benchmark/drift                      │
└─────────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Requirements    │  │ Model Discovery │  │ Testbench       │
│ Wizard + YAML   │  │ Service         │  │ Validation      │
│                 │  │                 │  │ Drift Monitor   │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Ultralytics     │  │ TorchVision     │  │ HuggingFace     │
│ YOLO v5/v8/v11  │  │ ResNet, MobNet  │  │ ViT, DETR, DPT  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Timm            │  │ ONNX Zoo        │  │ MediaPipe       │
│ FastViT, Edge   │  │ MiDaS, YOLO     │  │ Hands, Face     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Model Cache         │
                   │ ~/.cache/branes/    │
                   └─────────────────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ Pipeline Generation │
                   │ + Operator Setup    │
                   └─────────────────────┘
```

---

## Summary

The Branes Model Zoo and Pipeline Design system transforms the complex process of building perception systems into a streamlined workflow:

| Stage | Traditional | Branes |
|-------|-------------|--------|
| Model Research | Hours of searching | `branes zoo search` |
| Model Download | Manual, per-provider | `branes zoo download` |
| Format Conversion | Custom scripts | Automatic ONNX export |
| Pipeline Design | Manual integration | `branes design synthesize` |
| Validation | Ad-hoc testing | `branes testbench validate` |
| Monitoring | None | `branes testbench drift` |

This enables rapid iteration from requirements to deployment while ensuring models meet accuracy and latency constraints for edge deployment.
