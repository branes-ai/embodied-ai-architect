# Target System Architecture

This document describes the heterogeneous system architecture that the Embodied AI Architect targets for design exploration and optimization.

## System Composition Model

Embodied AI systems are not monolithic—they are **heterogeneous compositions** of a host processor and specialized accelerators working in concert:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Embodied AI System                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Host CPU                                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │   │
│  │  │     OS      │  │   Drivers   │  │  Application            │  │   │
│  │  │  - Kernel   │  │  - PCIe/USB │  │  Orchestration          │  │   │
│  │  │  - Sched    │  │  - DMA      │  │  - Operator dispatch    │  │   │
│  │  │  - Memory   │  │  - IRQ      │  │  - Data movement        │  │   │
│  │  │  - I/O      │  │  - Runtime  │  │  - Scheduling decisions │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│              ┌───────────────┼───────────────┐                         │
│              │               │               │                         │
│              ▼               ▼               ▼                         │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐              │
│  │  Accelerator  │  │  Accelerator  │  │  Accelerator  │   ...        │
│  │     (GPU)     │  │    (NPU)      │  │    (DSP)      │              │
│  │               │  │               │  │               │              │
│  │  - DNN infer  │  │  - INT8 ops   │  │  - Signal     │              │
│  │  - Parallel   │  │  - Low power  │  │    processing │              │
│  │    compute    │  │  - Edge AI    │  │  - Sensor     │              │
│  └───────────────┘  └───────────────┘  └───────────────┘              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Component Roles

### Host CPU

The CPU serves as the **system orchestrator**, responsible for:

| Function | Description |
|----------|-------------|
| **Operating System** | Process scheduling, memory management, file I/O, networking |
| **Device Drivers** | Hardware abstraction, DMA setup, interrupt handling |
| **Runtime Environment** | CUDA/TensorRT, OpenVINO, ONNX Runtime, vendor SDKs |
| **Application Orchestration** | Operator dispatch, dataflow coordination, timing control |
| **Control Logic** | Decision-making, state machines, non-accelerated operators |

The CPU is typically **not** the performance-critical path for inference, but it is **always** on the critical path for:
- Operator dispatch latency
- Data movement between accelerators
- Scheduling decisions
- I/O handling (sensors, actuators, network)

### Hardware Accelerators

Accelerators are **specialized compute engines** optimized for specific workloads:

| Accelerator Type | Optimized For | Power Profile | Example Parts |
|------------------|---------------|---------------|---------------|
| **GPU** | Parallel DNN inference, high throughput | High (50-400W) | H100, A100, RTX series |
| **Edge GPU** | Embedded DNN inference | Medium (15-60W) | Jetson Orin, AMD Ryzen AI |
| **NPU/TPU** | INT8/INT4 inference, efficiency | Low (5-15W) | Coral TPU, Hailo-8, Apple Neural Engine |
| **DSP** | Signal processing, audio, sensor fusion | Very low (1-5W) | Qualcomm Hexagon, TI C66x |
| **FPGA** | Custom dataflow, low latency | Variable | Xilinx Versal, Intel Agilex |
| **VPU** | Vision-specific ops | Low (2-10W) | Intel Movidius, OAK-D Myriad |

### Design Philosophy

The goal is **best-in-class performance AND energy efficiency** through:

1. **Right-sizing**: Match operator to accelerator capabilities
2. **Specialization**: Use purpose-built hardware for critical paths
3. **Heterogeneity**: Combine accelerators for complementary strengths
4. **Efficiency**: Minimize data movement, maximize accelerator utilization

## Scheduling Analysis Model

Our scheduling analysis must account for this heterogeneous architecture:

```python
class SystemComposition(BaseModel):
    """Complete system hardware configuration."""

    # Host processor
    host_cpu: CPUSpec
    os: OSSpec                       # Linux RT, QNX, FreeRTOS

    # Accelerator portfolio
    accelerators: list[AcceleratorBinding]

    # Interconnect characteristics
    host_to_accel_bandwidth_gbps: dict[str, float]
    accel_to_accel_direct: bool      # Can accelerators communicate directly?
    shared_memory: bool               # Unified memory architecture?

class AcceleratorBinding(BaseModel):
    """An accelerator instance in the system."""
    id: str                          # "gpu0", "npu0"
    hardware_id: str                 # Reference to HardwareEntry
    interface: str                   # "pcie_x16", "usb3", "integrated"

class SchedulingAnalysis(BaseModel):
    """Result of multi-rate scheduling analysis on heterogeneous system."""
    verdict: Verdict
    confidence: Confidence

    # Per-component analysis
    host_analysis: HostSchedulingResult
    accelerator_analysis: list[AcceleratorSchedulingResult]

    # System-level metrics
    system_utilization: SystemUtilization
    data_movement: DataMovementAnalysis

    # End-to-end timing
    critical_path: list[OperatorTiming]
    end_to_end_latency_ms: float

    # Power/thermal
    total_power_w: float
    thermal_feasibility: Verdict

    suggestions: list[str]

class HostSchedulingResult(BaseModel):
    """CPU scheduling analysis."""
    cpu_utilization: float           # 0-1, aggregate across cores
    per_core_utilization: list[float]

    # Orchestration overhead
    dispatch_latency_us: float       # Time to dispatch to accelerator
    context_switch_overhead_us: float
    driver_overhead_us: float

    # Rate feasibility for CPU-bound operators
    cpu_operator_rates: list[RateFeasibility]

    # OS scheduling
    scheduling_policy: str           # "SCHED_FIFO", "SCHED_RR", "CFS"
    priority_inversion_risk: bool

class AcceleratorSchedulingResult(BaseModel):
    """Per-accelerator scheduling analysis."""
    accelerator_id: str
    hardware_name: str

    utilization: float               # 0-1
    memory_utilization: float        # 0-1

    # Operators assigned to this accelerator
    assigned_operators: list[str]

    # Per-operator timing on this accelerator
    operator_timings: list[OperatorTiming]

    # Queuing analysis
    queue_depth: int
    avg_queue_wait_ms: float

    # Power state
    power_w: float
    power_state: str                 # "active", "idle", "sleep"

class DataMovementAnalysis(BaseModel):
    """Analysis of data transfers in the system."""

    # Host <-> Accelerator transfers
    host_to_accel_transfers: list[DataTransfer]
    accel_to_host_transfers: list[DataTransfer]

    # Accelerator <-> Accelerator (if applicable)
    accel_to_accel_transfers: list[DataTransfer]

    # Aggregate metrics
    total_bandwidth_used_gbps: float
    bandwidth_bottleneck: str | None  # Which link is saturated
    transfer_overhead_ms: float       # Total time in transfers

class DataTransfer(BaseModel):
    """A single data transfer between components."""
    source: str                      # "host", "gpu0", "npu0"
    destination: str
    size_mb: float
    latency_ms: float
    bandwidth_gbps: float
```

## Operator-to-Accelerator Mapping

A key analysis task is determining **which operator runs where**:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Operator Mapping                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Operator          Best Accelerator    Fallback     Reason      │
│  ─────────────────────────────────────────────────────────────  │
│  YOLO Detection    GPU / NPU           CPU          Parallel    │
│  Depth Estimation  GPU                 CPU          Conv-heavy  │
│  ByteTrack         CPU                 -            Sequential  │
│  Kalman Filter     CPU / DSP           -            Low compute │
│  PID Controller    CPU                 -            Control     │
│  Path Planner      CPU                 -            Branching   │
│  Preprocessing     GPU / VPU           CPU          Memory BW   │
│  NMS               CPU / GPU           -            Irregular   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Mapping Considerations

| Factor | Impact on Mapping |
|--------|-------------------|
| **Compute intensity** | High → Accelerator, Low → CPU |
| **Parallelism** | Data-parallel → GPU, Sequential → CPU |
| **Memory access pattern** | Regular → Accelerator, Irregular → CPU |
| **Precision requirements** | FP32 → GPU, INT8 → NPU, Mixed → Split |
| **Latency sensitivity** | Tight deadlines may prefer integrated accelerator |
| **Power budget** | Constrained → NPU/DSP, Unconstrained → GPU |
| **Data locality** | Minimize transfers, co-locate dependent ops |

## System-Level Optimization Targets

Our design exploration optimizes across these dimensions:

### 1. Latency Optimization

```
End-to-end latency = Σ(operator_latency) + Σ(transfer_latency) + dispatch_overhead
```

Optimization strategies:
- Operator fusion (reduce transfers)
- Pipelining (overlap compute and transfer)
- Accelerator selection (faster hardware)
- Precision reduction (faster inference)

### 2. Power Optimization

```
System power = CPU_active + Σ(accelerator_power) + memory_power + I/O_power
```

Optimization strategies:
- Right-size accelerators (NPU vs GPU)
- DVFS (dynamic voltage/frequency scaling)
- Operator batching (amortize wake-up)
- Sleep state management

### 3. Throughput Optimization

```
System throughput = min(operator_throughput) / pipeline_depth
```

Optimization strategies:
- Balance pipeline stages
- Parallelize independent operators
- Multi-stream execution on accelerators
- Reduce bottleneck operator latency

### 4. Energy Optimization

```
Energy per inference = Σ(power_i × time_i) for each active component
```

Optimization strategies:
- Minimize active time (faster accelerators)
- Use energy-efficient accelerators (NPU over GPU)
- Batch processing (amortize fixed costs)
- Quantization (less compute, less energy)

## Example: Drone Perception System

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Drone Perception - Jetson Orin Nano                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  ARM Cortex-A78AE (6 cores)                                      │   │
│  │  ├─ OS: Ubuntu 22.04 + RT patches                                │   │
│  │  ├─ Orchestration: Python application loop                       │   │
│  │  ├─ ByteTrack: runs on CPU cores 4-5                            │   │
│  │  └─ Kalman/Control: runs on CPU core 3                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Ampere GPU (1024 CUDA cores)                                    │   │
│  │  ├─ YOLOv8n detection: 18ms @ INT8                              │   │
│  │  ├─ Depth estimation: 12ms @ FP16                               │   │
│  │  └─ Preprocessing: 2ms (resize, normalize)                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  Data Flow:                                                             │
│  Camera(USB3) → CPU(decode) → GPU(preprocess+detect+depth)             │
│       → CPU(track+fuse) → CPU(control) → PWM(actuators)                │
│                                                                         │
│  Timing Budget (30fps = 33.3ms):                                        │
│  ├─ Camera capture: 1ms                                                │
│  ├─ Decode + transfer to GPU: 2ms                                      │
│  ├─ GPU inference (parallel): 18ms                                     │
│  ├─ Transfer to CPU: 1ms                                               │
│  ├─ Tracking: 4ms                                                      │
│  ├─ Control logic: 1ms                                                 │
│  └─ Total: 27ms (6ms headroom) ✓                                       │
│                                                                         │
│  Power Budget (15W TDP):                                                │
│  ├─ CPU: 3W                                                            │
│  ├─ GPU: 10W                                                           │
│  ├─ Memory/IO: 2W                                                      │
│  └─ Total: 15W (at limit) ⚠                                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Modeling Infrastructure Requirements

To support this heterogeneous system model, our infrastructure must:

### 1. Hardware Catalog Extensions

- **System-level entries**: Not just individual chips, but complete SoCs/boards
- **Accelerator portfolio**: List of accelerators available on each platform
- **Interconnect specs**: Bandwidth, latency between components
- **Power domains**: Which components share power rails

### 2. Operator Profiling

- **Per-accelerator profiles**: Same operator, different hardware
- **Transfer overhead**: Measured host↔accelerator latency
- **Dispatch overhead**: Driver and runtime costs

### 3. Scheduling Analysis

- **Multi-component utilization**: CPU + all accelerators
- **Data movement accounting**: Transfer times in critical path
- **Power aggregation**: Sum across active components
- **Thermal modeling**: Heat dissipation constraints

### 4. Optimization Recommendations

- **Operator placement**: Suggest better accelerator assignments
- **System upgrades**: Recommend different accelerator portfolio
- **Architecture changes**: Fuse operators, change precision, restructure pipeline

## Conclusion

The Embodied AI Architect must model and optimize **complete heterogeneous systems**, not isolated accelerators. The CPU orchestrates, accelerators compute, and our analysis must account for both—plus the data movement between them.

This system-level view enables:
- Accurate end-to-end latency prediction
- Realistic power/energy estimation
- Informed accelerator selection
- Optimization across the full stack
