# Phase 1: The "Digital Twin" Foundation

**Timeline**: Months 1–4
**Goal**: Establish a trusted simulation environment where software, hardware, and physics meet. The Agent is currently "passive," learning the correlation between embodied AI application code and energy consumption and latency.

---

## Overview

Phase 1 creates the foundational infrastructure that all subsequent phases depend on. Without an accurate, integrated simulation pipeline, the Agent cannot learn the relationships between software choices, hardware configurations, and physical outcomes.

The key deliverable is a **Co-Simulation Pipeline** that traces energy consumption from high-level drone maneuvers down to specific compute kernels.

---

## Milestone 1.1: The Coupled Simulation Pipeline

### User Story

> **As a Systems Engineer,**
>
> **I want** to run a complete drone mission in simulation and see exactly how much energy each software component (perception, planning, control) consumed on the target hardware,
>
> **So that** I can identify which components are the largest energy consumers and prioritize optimization efforts.

---

### Acceptance Criteria

| ID | Criterion Description | Key Metrics | Engineering Focus |
|:---|:---|:---|:---|
| **A/C 1.1.1** | **Simulator Integration** | The Co-Simulation Pipeline successfully executes a complete drone mission (takeoff → navigate → land) with synchronized physics (AirSim/Gazebo) and compute simulation (CSim). | **Integration:** Bi-directional data flow between flight simulator and compute simulator. |
| **A/C 1.1.2** | **Energy Traceability** | For any given mission segment (e.g., "obstacle avoidance maneuver"), the system produces a breakdown showing energy consumed by each software component (perception: X mJ, planning: Y mJ, control: Z mJ). | **Instrumentation:** Compute simulator accurately attributes energy to specific kernels/operators. |
| **A/C 1.1.3** | **Latency Correlation** | The system reports end-to-end latency for each control loop iteration, correlating sensor input timestamp to actuator command timestamp, with compute latency breakdown. | **Timing:** Accurate timestamping and latency measurement across simulation boundaries. |
| **A/C 1.1.4** | **Hardware Model Accuracy** | Energy estimates from CSim/ESim match empirical measurements on reference hardware (Jetson Orin) within **±15%** for representative workloads. | **Calibration:** Validation against physical hardware measurements. |

---

### Technical Deliverables

#### 1. Flight Simulator Integration

```
┌─────────────────────────────────────────────────────────────┐
│                    CO-SIMULATION PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    Sensor Data    ┌──────────────────┐   │
│  │              │ ────────────────► │                  │   │
│  │   AirSim /   │                   │  Perception +    │   │
│  │   Gazebo     │                   │  Planning +      │   │
│  │              │ ◄──────────────── │  Control Stack   │   │
│  │  (Physics)   │   Control Cmds    │                  │   │
│  └──────────────┘                   └────────┬─────────┘   │
│                                              │              │
│                                              ▼              │
│                                     ┌──────────────────┐   │
│                                     │  CSim + ESim     │   │
│                                     │  (HW Simulator)  │   │
│                                     │                  │   │
│                                     │  • Cycle counts  │   │
│                                     │  • Memory access │   │
│                                     │  • Energy est.   │   │
│                                     └──────────────────┘   │
│                                              │              │
│                                              ▼              │
│                                     ┌──────────────────┐   │
│                                     │  Metrics DB      │   │
│                                     │  (Time-series)   │   │
│                                     └──────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 2. Instrumentation Layer

| Component | Metrics Collected | Granularity |
|-----------|-------------------|-------------|
| Perception (YOLO, depth) | Latency, energy, memory BW | Per-inference |
| State Estimation (EKF, SLAM) | Latency, energy | Per-update |
| Planning (A*, RRT) | Latency, energy, iterations | Per-plan |
| Control (PID, MPC) | Latency, energy | Per-control-loop |

#### 3. Energy Attribution Model

```python
@dataclass
class EnergyBreakdown:
    """Energy breakdown for a mission segment."""
    segment_name: str
    duration_ms: float

    # Per-component energy (mJ)
    perception_mj: float
    state_estimation_mj: float
    planning_mj: float
    control_mj: float
    communication_mj: float
    static_power_mj: float

    # Derived metrics
    @property
    def total_mj(self) -> float:
        return (self.perception_mj + self.state_estimation_mj +
                self.planning_mj + self.control_mj +
                self.communication_mj + self.static_power_mj)

    @property
    def average_power_w(self) -> float:
        return self.total_mj / self.duration_ms  # mJ/ms = W
```

---

## Milestone 1.2: Baseline Profiling

### User Story

> **As a Data Scientist building the Agent,**
>
> **I want** a labeled dataset correlating software configurations to energy/latency outcomes across many flight scenarios,
>
> **So that** the Agent can learn the relationships between design choices and physical outcomes before attempting optimization.

---

### Acceptance Criteria

| ID | Criterion Description | Key Metrics | Engineering Focus |
|:---|:---|:---|:---|
| **A/C 1.2.1** | **Dataset Scale** | The baseline dataset contains at least **10,000 simulated flight minutes** across diverse mission profiles (hover, cruise, obstacle avoidance, landing). | **Scale:** Parallel simulation infrastructure to generate data efficiently. |
| **A/C 1.2.2** | **Configuration Coverage** | The dataset covers at least **50 distinct software configurations** varying: model precision (FP32/FP16/INT8), frame rates (10/15/30 Hz), model variants (YOLOv8n/s/m). | **Diversity:** Systematic exploration of software design space. |
| **A/C 1.2.3** | **Label Quality** | Each data point includes: mission outcome (success/failure/crash), energy breakdown by component, latency percentiles, and hardware utilization metrics. | **Labeling:** Automated annotation of simulation results. |
| **A/C 1.2.4** | **Correlation Discovery** | Analysis reveals at least **3 actionable correlations** (e.g., "INT8 quantization reduces perception energy by 40% with <2% accuracy loss on this mission type"). | **Analysis:** Statistical analysis and visualization of dataset. |

---

### Technical Deliverables

#### 1. Mission Profile Library

| Profile | Description | Key Stressors |
|---------|-------------|---------------|
| **Hover** | Stationary position hold | Low compute, high control precision |
| **Cruise** | Point-to-point flight, no obstacles | Moderate compute, energy efficiency focus |
| **Obstacle Avoidance** | Navigate through cluttered environment | High perception load, latency critical |
| **Search Pattern** | Systematic area coverage | Sustained compute, battery life focus |
| **Emergency Landing** | Rapid descent and landing | Reliability critical, degraded mode |

#### 2. Software Configuration Matrix

| Parameter | Values | Impact |
|-----------|--------|--------|
| **Detection Model** | YOLOv8n, YOLOv8s, YOLOv8m | Accuracy vs. latency |
| **Precision** | FP32, FP16, INT8 | Energy vs. accuracy |
| **Frame Rate** | 10, 15, 20, 30 Hz | Responsiveness vs. energy |
| **Resolution** | 320, 480, 640 px | Detection range vs. compute |
| **Planner** | A*, RRT, RRT* | Path quality vs. compute |
| **Control Rate** | 100, 250, 500 Hz | Stability vs. energy |

#### 3. Dataset Schema

```yaml
# Example dataset entry
mission_id: "sim_20260115_143022_001"
profile: "obstacle_avoidance"
duration_s: 120.5
outcome: "success"

software_config:
  detection_model: "yolov8n"
  precision: "int8"
  perception_rate_hz: 30
  input_resolution: 640
  planner: "astar"
  control_rate_hz: 250

hardware_config:
  platform: "jetson_orin_nano"
  power_mode: "15W"
  gpu_freq_mhz: 624

metrics:
  energy:
    total_mj: 45230
    perception_mj: 28450
    planning_mj: 2100
    control_mj: 8900
    static_mj: 5780
  latency:
    perception_p50_ms: 12.3
    perception_p95_ms: 18.7
    control_p50_ms: 1.2
    control_p95_ms: 2.1
    end_to_end_p95_ms: 24.5
  utilization:
    gpu_avg_pct: 78
    cpu_avg_pct: 45
    memory_bw_avg_pct: 62

annotations:
  bottleneck: "memory_bandwidth"
  optimization_opportunity: "reduce_perception_resolution"
  notes: "GPU underutilized due to memory bandwidth saturation"
```

#### 4. Analysis Dashboard

The dashboard displays:
- **Energy breakdown** by component (stacked bar chart)
- **Latency distributions** per component (box plots)
- **Configuration comparison** (parallel coordinates plot)
- **Correlation matrix** (software params vs. energy/latency)
- **Pareto frontier** (energy vs. accuracy trade-off)

---

## Risk Mitigation

| Risk | Probability | Severity | Mitigation |
|:---|:---:|:---:|:---|
| **Simulation Accuracy** | High | High | Implement empirical calibration with reference hardware; validate against Jetson Orin measurements |
| **Integration Complexity** | Medium | High | Start with simplified physics (no wind/noise); add complexity incrementally |
| **Data Quality** | Medium | Medium | Automated sanity checks; outlier detection; manual review of edge cases |
| **Simulation Speed** | Medium | Medium | Headless rendering; parallel execution; surrogate models for fast iteration |

---

## Success Definition

Phase 1 is complete when:

1. **Pipeline Operational**: A drone mission can be simulated end-to-end with energy attribution per component
2. **Accuracy Validated**: Energy estimates within ±15% of physical hardware measurements
3. **Dataset Ready**: 10,000+ flight minutes with 50+ configurations, fully labeled
4. **Insights Documented**: At least 3 actionable correlations identified and validated

---

## Dependencies

| Dependency | Source | Status |
|------------|--------|--------|
| AirSim or Gazebo | Open source | Available |
| CSim (Compute Simulator) | Branes internal | In development |
| ESim (Energy Simulator) | Branes internal | In development |
| Reference hardware (Jetson Orin) | NVIDIA | Available |
| Drone software stack | Open source + Branes | Available |

---

## Timeline

| Month | Focus | Key Deliverable |
|-------|-------|-----------------|
| **Month 1** | Simulator integration | AirSim ↔ CSim data flow working |
| **Month 2** | Instrumentation | Energy attribution per component |
| **Month 3** | Calibration | ±15% accuracy vs. physical hardware |
| **Month 4** | Dataset generation | 10,000 flight minutes collected |

---

## Transition to Phase 2

Phase 1 delivers:
- A trusted simulation environment
- A labeled dataset for Agent training
- Validated energy models

Phase 2 uses these to:
- Train the Agent to optimize software configurations
- Validate optimizations against the simulation
- Achieve measurable energy efficiency improvements
