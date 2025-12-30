# Graph Analysis Tools Guide

This guide covers the verdict-first graph analysis tools available in `embodied-ai-architect`.

## Overview

The graph analysis tools integrate with `branes-ai/graphs` to provide detailed DNN performance estimation with a verdict-first output format designed for LLM consumption.

## Installation

```bash
# Install graphs with Pydantic schema support
pip install graphs[schemas]

# Or install embodied-schemas separately
pip install embodied-schemas
```

## Available Tools

### check_latency

Check if a model meets a latency target on specific hardware.

```python
from embodied_ai_architect.llm.graphs_tools import check_latency

result = check_latency(
    model_name="resnet18",
    hardware_name="H100-SXM5-80GB",
    latency_target_ms=10.0,
    batch_size=1,
    precision="FP32"
)
```

**Output:**
```json
{
  "verdict": "PASS",
  "confidence": "high",
  "summary": "Latency of 0.4 is well under 10.0 target (96% headroom)",
  "constraint": {
    "metric": "latency",
    "threshold": 10.0,
    "actual": 0.43,
    "margin_pct": 95.7
  },
  "suggestions": ["Increase batch size to amortize static energy"]
}
```

### check_power

Check if a model meets a power budget on specific hardware.

```python
from embodied_ai_architect.llm.graphs_tools import check_power

result = check_power(
    model_name="mobilenet_v2",
    hardware_name="Jetson-Orin-Nano",
    power_budget_w=15.0,
    batch_size=1
)
```

### check_memory

Check if a model fits within a memory budget.

```python
from embodied_ai_architect.llm.graphs_tools import check_memory

result = check_memory(
    model_name="resnet50",
    hardware_name="Jetson-Orin-AGX",
    memory_budget_mb=512.0,
    batch_size=1
)
```

### full_analysis

Comprehensive analysis with optional constraint checking.

```python
from embodied_ai_architect.llm.graphs_tools import full_analysis

# Without constraint
result = full_analysis(
    model_name="resnet18",
    hardware_name="H100-SXM5-80GB"
)

# With constraint
result = full_analysis(
    model_name="resnet18",
    hardware_name="H100-SXM5-80GB",
    constraint_metric="latency",
    constraint_threshold=10.0,
    precision="FP16"
)
```

## Verdict-First Pattern

All tools return results in a verdict-first format:

1. **verdict**: `PASS | FAIL | PARTIAL | UNKNOWN`
   - PASS: Meets all constraints
   - FAIL: Does not meet constraints
   - UNKNOWN: Error or missing data

2. **confidence**: `high | medium | low`
   - high: Based on hardware allocation analysis
   - medium: Based on roofline estimates

3. **summary**: One-sentence explanation

4. **suggestions**: Actionable recommendations (on FAIL)

## Output Structure

```json
{
  "verdict": "PASS",
  "confidence": "high",
  "summary": "Latency of 0.4 meets 10.0 target with 96% headroom",
  "model_id": "ResNet-18",
  "hardware_id": "H100-SXM5-80GB",
  "batch_size": 1,
  "precision": "fp32",
  "metrics": {
    "latency_ms": 0.432,
    "throughput_fps": 2316.3,
    "energy_per_inference_mj": 97.148,
    "peak_memory_mb": 6.1
  },
  "roofline": {
    "bottleneck": "memory-bound",
    "utilization_pct": 9.3,
    "arithmetic_intensity": 23.6
  },
  "energy": {
    "compute_mj": 4.733,
    "memory_mj": 1.754,
    "static_mj": 90.66,
    "average_power_w": 225.0
  },
  "memory": {
    "weights_mb": 0.0,
    "activations_mb": 6.1,
    "fits_in_l2": true,
    "fits_in_device_memory": true
  },
  "constraint": {
    "metric": "latency",
    "threshold": 10.0,
    "actual": 0.43,
    "margin_pct": 95.7
  },
  "suggestions": []
}
```

## Supported Hardware

```python
from embodied_ai_architect.llm.graphs_tools import HARDWARE_CATALOG

# Categories:
# - datacenter_gpu: H100, A100, V100, L4, T4
# - edge_gpu: Jetson Orin AGX, Jetson Orin Nano
# - datacenter_cpu: Intel Xeon, AMD EPYC
# - tpu: TPU v4, Coral Edge TPU
# - accelerators: KPU, Hailo-8
# - automotive: TDA4VM, TDA4VL
```

## Supported Models

Common models include:
- ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
- MobileNet: mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large
- EfficientNet: efficientnet_b0 through efficientnet_b7
- ViT: vit_b_16, vit_b_32, vit_l_16
- YOLO: yolov8n, yolov8s, yolov8m

## Agent Integration

The tools are automatically available to the `ArchitectAgent`:

```python
from embodied_ai_architect.llm import ArchitectAgent

agent = ArchitectAgent()
response = agent.run(
    "Can ResNet-18 meet a 5ms latency target on the Jetson Orin AGX?"
)
```

The agent will use `check_latency` and interpret the verdict.

## Error Handling

On errors, tools return:

```json
{
  "verdict": "UNKNOWN",
  "error": "Error message",
  "traceback": "Full traceback..."
}
```
