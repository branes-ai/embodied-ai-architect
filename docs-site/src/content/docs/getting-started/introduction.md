---
title: Introduction
description: Learn what Embodied AI Architect is and how it helps you design AI systems for the physical world.
---

Embodied AI Architect is a design environment for creating, evaluating, optimizing, and deploying autonomous agents. It bridges the gap between AI model development and real-world deployment on edge hardware.

## The Problem

Deploying AI models on edge devices is challenging:

- **Model-Hardware Mismatch**: A model that runs great in the cloud may be unusable on a Jetson Nano
- **Hidden Bottlenecks**: Is your model compute-bound or memory-bound? The answer determines your optimization strategy
- **Constraint Verification**: Does your model meet the 10ms latency requirement for a 100Hz control loop?
- **Hardware Selection**: Which of 50+ edge platforms is right for your use case?

## The Solution

Embodied AI Architect provides:

### 1. Roofline Analysis

Understand exactly how your model utilizes hardware resources using roofline modeling. Know whether you're compute-bound or memory-bound before you deploy.

### 2. Constraint Checking

Get clear PASS/FAIL verdicts for your deployment requirements:
- Latency targets (e.g., 10ms for real-time control)
- Power budgets (e.g., 5W for battery-powered drones)
- Memory limits (e.g., 2GB for edge devices)

### 3. Hardware Catalog

Compare 50+ hardware targets including:
- NVIDIA Jetson (Orin AGX, Orin NX, Orin Nano)
- Google Coral (Edge TPU)
- Intel OpenVINO targets
- Cloud GPUs (H100, A100, V100)
- Custom accelerators (KPU, Hailo)

### 4. Deployment Guidance

Generate deployment guides with:
- Quantization recommendations (FP16, INT8)
- Framework-specific optimizations (TensorRT, OpenVINO)
- Production configurations

## Architecture

The system consists of three main repositories:

```
embodied-schemas (shared data models)
       ↑              ↑
       │              │
   graphs      embodied-ai-architect
   (analysis)     (orchestration)
```

- **embodied-schemas**: Shared Pydantic schemas and hardware/model catalogs
- **graphs**: Roofline analysis, hardware simulation, performance modeling
- **embodied-ai-architect**: CLI, LLM integration, deployment tools

## Next Steps

- [Install the package](/getting-started/installation/)
- [Run through the quickstart](/getting-started/quickstart/)
- [Explore the hardware catalog](/catalog/hardware/)
