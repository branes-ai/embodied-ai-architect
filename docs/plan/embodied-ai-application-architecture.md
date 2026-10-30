# Embodied AI Application Architecture

**Date**: 2025-11-02
**Status**: Architecture Analysis - Selected Option 3
**Next Step**: Implementation Planning

## Problem Statement

### Current Gap

The Embodied AI Architect currently supports analyzing and benchmarking **individual PyTorch models**. However, real embodied AI applications are **complete control loops** with heterogeneous operators:

```
Sensors → Preprocessing → DNN → State Estimation → Planning → Control → Actuators
                           ↓           ↓             ↓         ↓
                      (PyTorch)  (Kalman Filter)   (RRT*)    (PID)
```

**What we need:**
- Represent complete applications as computational pipelines
- Support heterogeneous operators (DNNs, filters, optimizers, controllers)
- Benchmark end-to-end: latency, accuracy, energy
- Enable transpilation to C++/Rust for deployment

### Key Requirements

1. **Application Representation**: Capture entire control loops, not just models
2. **Heterogeneous Operators**: DNNs (PyTorch), classical algorithms (Kalman, PID, RRT*), linear algebra (NumPy/SciPy)
3. **Benchmarking Framework**: Execute apps with recorded sensor data, measure end-to-end metrics
4. **Replay Capability**: Reproducible benchmarks with recorded sensor traces
5. **Transpilation Target**: Eventually generate C++/Rust production code
6. **Hardware Mapping**: Understand operator placement across heterogeneous hardware

---

## Architecture Options Evaluated

### Option 1: Python Application Framework with Computational Graph

**Concept**: Users write applications in Python. We trace execution to build computational graph.

```python
class AutonomousDroneApp(EmbodiedAIApp):
    def __init__(self):
        self.camera = Sensor("camera", shape=(3, 480, 640))
        self.imu = Sensor("imu", shape=(9,))
        self.object_detector = torch.load("yolov5.pt")
        self.kalman_filter = KalmanFilter(state_dim=6)
        self.pid_controller = PIDController(kp=1.0, ki=0.1, kd=0.01)
        self.motors = Actuator("motors", shape=(4,))

    @traced
    def step(self, dt: float):
        frame = self.camera.read()
        imu_data = self.imu.read()
        detections = self.object_detector(frame)
        measurements = self._extract_positions(detections)
        state = self.kalman_filter.update(measurements, imu_data)
        target = np.array([0, 0, 1])
        control = self.pid_controller.compute(state[:3], target)
        self.motors.write(control)
        return {"state": state, "control": control}
```

**Pros:**
- ✅ Familiar Python API
- ✅ Full expressiveness
- ✅ Automatic graph capture
- ✅ Easy to prototype

**Cons:**
- ❌ Python runtime required
- ❌ Complex tracing for control flow
- ❌ Need operator library

---

### Option 2: Declarative Pipeline Definition (YAML/JSON DSL)

**Concept**: Define applications declaratively. System loads and executes pipeline.

```yaml
application:
  name: "AutonomousDrone"

sensors:
  - name: "camera"
    shape: [3, 480, 640]
    rate: 30
  - name: "imu"
    shape: [9]
    rate: 100

operators:
  - id: "object_detection"
    type: "pytorch_model"
    model_path: "yolov5.pt"
    inputs: ["camera"]
    outputs: ["detections"]

  - id: "kalman_filter"
    type: "kalman_filter"
    state_dim: 6
    inputs: ["detections", "imu"]
    outputs: ["state_estimate"]

  - id: "pid_controller"
    type: "pid_controller"
    inputs: ["state_estimate"]
    outputs: ["control_signal"]

actuators:
  - name: "motors"
    inputs: ["control_signal"]
```

**Pros:**
- ✅ Language-agnostic
- ✅ Easy to validate/visualize
- ✅ Version control friendly
- ✅ Clear transpilation path

**Cons:**
- ❌ Less expressive
- ❌ Complex logic harder to express
- ❌ Debugging harder

---

### Option 3: Hybrid - Python Framework + Graph IR + DSL ⭐ SELECTED

**Concept**: Three-layer architecture combining best of both approaches.

```
┌─────────────────────────────────────────┐
│  Layer 1: Python Application Framework  │  ← User writes code here
│  - Rich operator library                │
│  - Type checking, IDE support           │
│  - Familiar Python patterns             │
|  - Natural Dev & Test workflow          |
└─────────────────────────────────────────┘
                  ↓ (trace/compile)
┌─────────────────────────────────────────┐
│  Layer 2: Graph IR (Intermediate Rep)   │  ← Analysis & optimization
│  - Computational graph                  │
│  - Graph optimizations                  │
│  - Hardware mapping                     │
│  - Profiling & analysis                 │
└─────────────────────────────────────────┘
                  ↓ (serialize/transpile)
┌─────────────────────────────────────────┐
│  Layer 3: Serialization & Code Gen      │  ← Deployment artifacts
│  - YAML/JSON for storage                │
│  - C++/Rust transpilation               │
│  - ONNX export                          │
└─────────────────────────────────────────┘
```

#### Layer 1: Python Application Framework

```python
from embodied_ai_architect.app import *

@embai_application(name="drone_navigation")
class DroneNav:
    def __init__(self):
        # Component declarations
        self.detector = PyTorchOperator("yolov5.pt")
        self.tracker = KalmanFilter(state_dim=6)
        self.planner = PathPlanner(algorithm="rrt_star")
        self.controller = PIDController(kp=1.0, ki=0.1, kd=0.01)

    @control_loop(frequency=30)  # 30 Hz control loop
    def navigate(self, sensors: SensorBundle) -> ActuatorBundle:
        # Perception
        detections = self.detector(sensors.camera)

        # State estimation
        state = self.tracker.predict()
        state = self.tracker.update(detections, sensors.imu)

        # Planning
        path = self.planner.plan(state.position, goal=[10, 10, 2])

        # Control
        waypoint = path.next_waypoint()
        control = self.controller.compute(state.position, waypoint)

        return ActuatorBundle(motors=control)
```

#### Layer 2: Graph IR

```python
class EmbodiedAIGraph:
    """Intermediate representation of embodied AI application."""

    def __init__(self):
        self.operators: Dict[str, Operator] = {}
        self.connections: List[Edge] = []
        self.schedule: ExecutionSchedule = None

    def optimize(self) -> "EmbodiedAIGraph":
        """Apply graph-level optimizations."""
        self.fuse_operators()
        self.eliminate_dead_code()
        self.schedule_operators()
        return self

    def profile(self, backend: Backend, dataset: Dataset) -> ProfilingResults:
        """Profile each operator subgraph on target backend."""
        # Returns latency/energy per operator subgraph

    def to_dsl(self) -> dict:
        """Serialize to declarative format."""

    def to_mlir(self) -> str:
        """Export to MLIR for IREE compilation."""

    def to_cpp(self) -> str:
        """Transpile to C++ code."""
```

#### Layer 3: DSL Format

```json
{
  "application": {
    "name": "drone_navigation",
    "version": "1.0.0",
    "control_loop_hz": 30
  },
  "graph": {
    "operators": [
      {
        "id": "detector",
        "type": "pytorch_model",
        "artifact": "models/yolov5.pt",
        "inputs": {"image": "sensors.camera"},
        "outputs": {"detections": "tensor"}
      },
      {
        "id": "tracker",
        "type": "kalman_filter",
        "state_dim": 6,
        "inputs": {
          "measurements": "detector.detections",
          "imu": "sensors.imu"
        },
        "outputs": {"state": "vector"}
      }
    ],
    "execution_schedule": [
      {"operator": "detector", "order": 1},
      {"operator": "tracker", "order": 2}
    ]
  }
}
```

#### Benchmarking System

```python
from embodied_ai_architect.benchmark import ApplicationBenchmark

app = DroneNav()  # Python instance
graph = app.compile()  # Get graph IR

benchmark = ApplicationBenchmark(
    application=graph,
    dataset=RosbagDataset("flight_001.bag"),
    ground_truth=GroundTruthLoader("flight_001_gt.json"),
    metrics=[
        LatencyMetric(percentiles=[50, 95, 99]),
        AccuracyMetric(iou_threshold=0.5),
        EnergyMetric(if_supported=True),
        ThroughputMetric()
    ]
)

results = benchmark.run(
    backends=[
        LocalCPUBackend(),
        CPUBackend(model="raspberry_pi"),
        KPUBackend(model="kpu_t64"),
        JetsonBackend(model="agx_orin"),
        SimulatorBackend(physics_engine="gazebo")
    ],
    iterations=1000
)

# Per-operator breakdown
# End-to-end metrics
# Backend comparison
# Hardware recommendations
report = results.generate_report()
```

**Framework -> Graph IR Benefits**

✅ **Best Developer Experience**: Python framework is familiar
✅ **Analyzable**: Graph IR enables optimization and profiling
✅ **Portable**: DSL format is language-agnostic
✅ **Clear Transpilation Path**: Python → IR → C++/Rust
✅ **Supports Optimization**: Graph-level passes
✅ **Gradual Migration**: Develop in Python, deploy in C++

**Tradeoffs:**
- Most complex to implement (3 representations)
- Requires maintaining consistency across layers
- Graph tracing can be complex for dynamic control flow

---

## Existing Infrastructure (Available)

Based on existing tooling, we have:

1. **Python Profiling/Tracing**: Can capture execution traces
2. **PyTorch FX & Dynamo**: Graph capture for PyTorch models
3. **torch.compile/MLIR/IREE**: C++ IR analysis and compilation
4. **Simulators**: Functional, performance, energy simulation
5. **Compiler/Runtime Modeler**: Hardware mapping analysis

**Integration Strategy:**
- Use PyTorch FX & Dynamo for DNN graph capture
- Bridge to torch.compile/MLIR/IREE for compilation
- Extend profiling to full applications
- Use simulators as benchmark backends
- Leverage hardware mapping models

---

## Implementation Phases

### Phase 1: Core Application Framework (3-4 weeks)

**Deliverables:**
- Python application framework (`embodied_ai_architect.app`)
- Base operator library (Sensor, Actuator, Operator)
- Control loop decorator and execution model
- Basic graph IR data structures

**Components:**
- `app/base.py`: Application base classes
- `app/decorators.py`: @embai_application, @control_loop
- `app/sensors.py`: Sensor abstraction
- `app/actuators.py`: Actuator abstraction
- `graph/ir.py`: Graph IR data structures

### Phase 2: Operator Library (2-3 weeks)

**Deliverables:**
- PyTorch operator wrapper
- Classical algorithm operators (Kalman, PID, etc.)
- NumPy/SciPy operator wrappers
- Operator type system and validation

**Operators:**
- `operators/pytorch_operator.py`: Wrap PyTorch models
- `operators/kalman_filter.py`: Kalman filter implementation
- `operators/pid_controller.py`: PID controller
- `operators/path_planners.py`: RRT*, A*, etc.
- `operators/base.py`: Operator interface

### Phase 3: Graph Capture (2 weeks)

**Deliverables:**
- Tracing infrastructure using PyTorch FX
- Graph construction from traced execution
- Data flow analysis
- Control flow handling

**Components:**
- `graph/tracer.py`: Application tracing
- `graph/builder.py`: Graph construction
- `graph/analysis.py`: Data/control flow analysis

### Phase 4: Benchmarking Framework (2-3 weeks)

**Deliverables:**
- Dataset loaders (ROS bags, custom formats)
- Metrics collection framework
- Benchmark orchestration
- Integration with existing simulators

**Components:**
- `benchmark/datasets.py`: Data loaders
- `benchmark/metrics.py`: Latency, accuracy, energy metrics
- `benchmark/harness.py`: Benchmark execution
- `benchmark/simulator_backend.py`: Bridge to simulators

### Phase 5: MLIR/IREE Integration (3-4 weeks)

**Deliverables:**
- Graph IR → MLIR conversion
- IREE compilation pipeline integration
- Hardware-specific optimizations
- Performance analysis feedback loop

**Components:**
- `codegen/mlir_exporter.py`: IR → MLIR
- `codegen/iree_compiler.py`: IREE integration
- `analysis/hardware_mapping.py`: Use existing compiler model

### Phase 6: DSL Serialization (1-2 weeks)

**Deliverables:**
- DSL format specification
- IR → DSL serialization
- DSL → IR deserialization
- Validation and schema

**Components:**
- `dsl/format.py`: DSL specification
- `dsl/serializer.py`: IR → DSL
- `dsl/loader.py`: DSL → IR

### Phase 7: C++/Rust Transpilation (Future - 4+ weeks)

**Deliverables:**
- C++ code generation from IR
- Runtime library for C++ apps
- Rust code generation (optional)

---

## Dependencies and Prerequisites

**External:**
- PyTorch (>= 2.0) - for FX graph capture
- MLIR/IREE - existing C++ IR infrastructure
- ROS (optional) - for ROS bag support
- NumPy/SciPy - for classical algorithms

**Internal:**
- Existing agents (ModelAnalyzer, Benchmark, etc.)
- Existing backend infrastructure
- Existing secrets management
- CLI framework

---

## Success Criteria

**Phase 1-2 Success:**
- Can write simple embodied AI app in Python
- Can execute app with synthetic sensor data
- Basic operators work (PyTorch, Kalman, PID)

**Phase 3-4 Success:**
- Can trace execution and build graph
- Can benchmark with recorded datasets
- Get per-operator latency breakdown

**Phase 5-6 Success:**
- Can export to MLIR
- Can compile with IREE
- Can serialize/deserialize via DSL

**Overall Success:**
- Can author drone navigation app in Python
- Can benchmark on multiple backends
- Can export to C++ for deployment
- Performance within 10% of hand-written C++

---

## Risk Assessment

**High Risk:**
- Graph tracing complexity for dynamic control flow
- MLIR integration may require significant glue code
- Operator library completeness

**Medium Risk:**
- Dataset format compatibility
- Performance overhead of Python layer
- Debugging across 3 representations

**Low Risk:**
- DSL serialization (well-understood)
- Basic operator implementation
- Integration with existing infrastructure

---

## Future Enhancements

- **Auto-tuning**: Automatic hyperparameter optimization
- **Multi-rate Scheduling**: Different operators at different frequencies
- **Distributed Execution**: Split operators across multiple devices
- **Formal Verification**: Prove safety properties
- **Hardware Co-design**: Optimize hardware and software together
- **Real-time Guarantees**: WCET analysis and scheduling

---

## References

- PyTorch FX: https://pytorch.org/docs/stable/fx.html
- MLIR: https://mlir.llvm.org/
- IREE: https://iree.dev/
- ROS bags: http://wiki.ros.org/rosbag

---

## Decision

**Selected Architecture: Option 3 (Hybrid)**

**Rationale:**
1. Leverages existing infrastructure (PyTorch FX, MLIR/IREE)
2. Best developer experience (Python)
3. Clear path to production (C++/Rust)
4. Enables comprehensive analysis and optimization
5. Flexible for future enhancements

**Next Steps:**
1. Create detailed implementation plan with milestones
2. Begin Phase 1: Core application framework
3. Develop reference example (autonomous drone navigation)
