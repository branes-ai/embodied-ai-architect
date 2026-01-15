---
title: MCP Tools Reference
description: Reference for Model Context Protocol tools.
---

Embodied AI Architect exposes tools via MCP for use with Claude and other LLM clients.

## Analysis Tools

### analyze_model_detailed

Perform detailed analysis using roofline modeling.

**Input:**
```json
{
  "model_name": "resnet18",
  "hardware_name": "H100-SXM5-80GB",
  "batch_size": 1,
  "precision": "FP16"
}
```

**Output:**
```json
{
  "model": "resnet18",
  "hardware": "H100-SXM5-80GB",
  "metrics": {
    "latency_ms": 0.82,
    "throughput_fps": 1219.5,
    "energy_mj": 0.45,
    "peak_memory_mb": 89.2
  },
  "bottleneck": {
    "type": "memory_bound",
    "compute_utilization": 12.3,
    "memory_utilization": 89.1
  }
}
```

### compare_hardware_targets

Compare model performance across hardware.

**Input:**
```json
{
  "model_name": "yolov8n",
  "hardware_targets": ["H100-SXM5-80GB", "Jetson-Orin-AGX", "Coral-Edge-TPU"]
}
```

### identify_bottleneck

Identify compute vs memory bottleneck.

**Input:**
```json
{
  "model_name": "resnet50",
  "hardware_name": "A100-SXM4-80GB"
}
```

### list_available_hardware

List supported hardware targets.

**Input:**
```json
{
  "category": "edge_gpu"
}
```

## Constraint Checking Tools

### check_latency

Check if model meets latency target.

**Input:**
```json
{
  "model_name": "yolov8n",
  "hardware_name": "Jetson-Orin-Nano",
  "latency_target_ms": 33
}
```

**Output:**
```json
{
  "verdict": "PASS",
  "confidence": "HIGH",
  "metrics": {
    "latency_ms": 28.5
  },
  "constraint": {
    "metric": "latency",
    "threshold": 33,
    "actual": 28.5,
    "margin_pct": 13.6
  }
}
```

### check_power

Check if model meets power budget.

**Input:**
```json
{
  "model_name": "yolov8n",
  "hardware_name": "Jetson-Orin-Nano",
  "power_budget_w": 15
}
```

### check_memory

Check if model fits in memory budget.

**Input:**
```json
{
  "model_name": "resnet50",
  "hardware_name": "Jetson-Orin-Nano",
  "memory_budget_mb": 4096
}
```

### full_analysis

Complete analysis with optional constraint.

**Input:**
```json
{
  "model_name": "yolov8n",
  "hardware_name": "Jetson-Orin-Nano",
  "constraint_metric": "latency",
  "constraint_threshold": 33
}
```

## Architecture Tools

### analyze_architecture

Analyze a complete pipeline on hardware.

**Input:**
```json
{
  "architecture_id": "drone_perception_v1",
  "hardware_id": "Jetson-Orin-Nano"
}
```

### check_scheduling

Check if all operators meet rate requirements.

**Input:**
```json
{
  "architecture_id": "drone_perception_v1",
  "hardware_id": "Jetson-Orin-Nano"
}
```

## Using with Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "embodied-ai": {
      "command": "embodied-ai",
      "args": ["mcp", "serve"]
    }
  }
}
```
