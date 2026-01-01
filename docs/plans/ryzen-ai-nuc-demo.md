# Ryzen AI NUC Demo Plan

Target Platforms

  | Platform       | CPU           | GPU                 | NPU          | TDP    |
  |----------------|---------------|---------------------|--------------|--------|
  | Ryzen 7 AI NUC | 8-core Zen 4  | Radeon 780M (12 CU) | XDNA 10 TOPS | 15-28W |
  | Ryzen 9 AI NUC | 8-core Zen 4+ | Radeon 780M (12 CU) | XDNA 16 TOPS | 35-54W |

## Target Application: PGN&C Pipeline

```
  Camera → Preprocess → Detect → Track → Scene Graph
    30Hz      GPU         NPU     CPU        CPU
                                    ↓
  IMU ──────────────────────→ State Est. → Planner → Controller
  100Hz                          CPU         CPU       CPU 100Hz
```

## 8-Week Implementation Plan

  | Phase                        | Weeks | Deliverables                            |
  |------------------------------|-------|-----------------------------------------|
  | 1. Platform Characterization | 1-2   | Hardware profiles, baseline benchmarks  |
  | 2. Operator Profiling        | 2-3   | Performance profiles for 15+ operators  |
  | 3. Reference Architecture    | 3-4   | pgnc_robot_v1 with 4 variants           |
  | 4. Operator Implementations  | 4-6   | Runnable YOLO (NPU), EKF, PID operators |
  | 5. Benchmark Harness         | 5-6   | Real execution with power measurement   |
  | 6. Demo Workflow             | 6-7   | Interactive agentic design session      |
  | 7. Documentation             | 7-8   | Report, demo video, polish              |

## Demo Workflow Preview

```text
  User: "Deploy a perception-control pipeline on Ryzen 7 AI NUC, 30fps, 25W"

  Agent: [analyze_architecture] → "48ms latency, only 20fps - below target"
  Agent: [identify_bottleneck] → "Detector on GPU takes 32ms (67%)"
  Agent: [suggest_optimizations] → "Move to NPU: 32ms → 12ms"
  Agent: [benchmark_architecture] → "Measured: 29.3ms, 34fps, 21.3W ✓"

  User: "Compare Ryzen 7 vs Ryzen 9"

  | Metric | Ryzen 7 | Ryzen 9 |
  |--------|---------|---------|
  | Throughput | 34 fps | 45 fps |
  | Power | 21W | 34W |
  | Efficiency | 1.6 fps/W | 1.3 fps/W |
```

## Key Technical Work

  1. Hardware catalog entries for both NUCs in embodied-schemas
  2. Real operator profiles measured on NPU/GPU/CPU
  3. Runnable operators: YOLO via ONNX Runtime + Ryzen AI EP
  4. Power monitoring via ryzen_smu or RAPL
  5. Architecture runner for real pipeline execution

## Success Criteria

  - [ ] Pipeline runs on both platforms
  - [ ] Estimates within 15% of measured
  - [ ] Optimization suggestions actually improve performance
  - [ ] Clear before/after demonstration
