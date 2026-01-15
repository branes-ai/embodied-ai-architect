---
title: Introduction
description: Meet the Embodied AI Architect—your AI partner for designing differentiated embodied AI solutions.
---

The **Embodied AI Architect** is an agentic AI that partners with product architects and engineers to design embodied AI solutions that deliver capabilities commodity hardware cannot match.

## The Problem We Solve

Building breakthrough embodied AI products requires answering hard questions:

- **Build vs. Buy**: Should you use commercial hardware or invest in custom silicon?
- **Competitive Position**: How will your solution compare to competitors on performance, cost, and power?
- **Design Tradeoffs**: Which model architecture, quantization, and hardware combination meets your constraints?
- **Pre-Silicon Validation**: Can you validate custom hardware designs before committing to tape-out?

These questions require expertise across ML, hardware architecture, and systems engineering—plus access to data about hardware you may not have.

## What Makes Us Different

### 1. Agentic Design Partner

The Architect isn't just a tool—it's an AI that reasons about your requirements, explores the design space, and recommends solutions. Ask questions in natural language:

```
> I need to run a perception pipeline at 60Hz under 10W for a surgical robot.
> What are my options, and when does custom silicon make sense?
```

### 2. COTS + Custom Hardware

We analyze solutions across **commercial off-the-shelf** platforms (NVIDIA Jetson, Google Coral, Hailo, Intel) **and** custom AI accelerators:

- **50+ COTS platforms** with calibrated performance data
- **Pre-silicon modeling** for custom accelerators before tape-out
- **Comparative analysis** across your options and competitor systems

### 3. Competitive Intelligence

Our characterization methodology lets you predict performance, cost, and energy for systems you don't have access to:

- Estimate competitor hardware capabilities
- Generate quantitative competitive analysis
- Validate your differentiation before you build

### 4. Full Lifecycle Support

From concept through production:

| Phase | What the Architect Does |
|-------|------------------------|
| **Design** | Explore hardware options, model architectures, quantization strategies |
| **Analysis** | Roofline modeling, bottleneck identification, constraint checking |
| **Optimization** | Recommend optimizations with predicted impact |
| **Deployment** | Generate deployment configurations, quantization, runtime setup |
| **Validation** | Verify deployed performance matches predictions |

## Why Custom Matters

Tesla's FSD computer demonstrates the power of purpose-built solutions. Instead of using commodity hardware, Tesla designed a custom accelerator optimized for their neural networks—achieving capabilities no off-the-shelf hardware could deliver at their cost and power targets.

The Embodied AI Architect helps you determine when custom hardware delivers competitive advantage:

| Factor | COTS | Custom |
|--------|------|--------|
| Time to market | Fast | 18-24 months |
| NRE cost | Low | High |
| Unit cost at volume | Higher | Lower |
| Performance/watt | Good | Optimized |
| Differentiation | Limited | Unique |

## The Branes Platform

The Embodied AI Architect is the design interface to the **Branes Embodied AI Platform**:

```
┌─────────────────────────────────────────────────────────┐
│              Embodied AI Architect                       │
│         (Agentic Design Interface)                       │
├─────────────────────────────────────────────────────────┤
│  Hardware      │  Analysis      │  Deployment           │
│  Catalog       │  Engine        │  Automation           │
│  ─────────     │  ─────────     │  ─────────            │
│  COTS specs    │  Roofline      │  Quantization         │
│  Custom models │  Constraints   │  Runtime config       │
│  Calibrations  │  Predictions   │  Validation           │
└─────────────────────────────────────────────────────────┘
```

## Who Should Use This

- **Product Architects** exploring hardware options for new embodied AI products
- **Systems Engineers** optimizing perception pipelines for edge deployment
- **Hardware Teams** validating custom accelerator designs against application requirements
- **Technical Leadership** making build-vs-buy decisions with quantitative analysis

## Next Steps

- [Install the platform](/getting-started/installation/)
- [Run your first analysis](/getting-started/quickstart/)
- [Explore the hardware catalog](/catalog/hardware/)
