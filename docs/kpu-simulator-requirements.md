# Stillwater KPU Simulator Requirements

This document specifies the interfaces and requirements for integrating a KPU simulator with the Embodied AI Architect deployment system.

## Overview

The deployment system expects the KPU simulator to provide three main components:

```
┌─────────────────────────────────────────────────────────────────┐
│                     Deployment System                            │
│                                                                  │
│   PyTorch Model ──► torch.compile ──► KPUCompiler ──► KPUProgram│
│         │                │                │              │       │
│         ▼                ▼                ▼              ▼       │
│     torch.nn          FX Graph        Graph Opts     Serialized  │
│     Module            Capture         Tiling         Program     │
│                                       Scheduling     + Weights   │
│                                                                  │
│   KPUProgram ──► KPURuntime ──► Execution Results + Metrics      │
└─────────────────────────────────────────────────────────────────┘
```

**Requirements:** PyTorch 2.0+ with torch.compile support

## Required Interfaces

### 1. KPUCompilerInterface

**Location:** `agents/deployment/targets/kpu/spec.py`

The compiler integrates with torch.compile to transform PyTorch models into executable KPU programs.

```python
class KPUCompilerInterface(ABC):
    @abstractmethod
    def compile(
        self,
        model_path: Path,
        config: KPUConfig,
        precision: KPUPrecision,
        calibration_data: Iterator[np.ndarray] | None = None,
    ) -> KPUProgram:
        """
        Compile PyTorch model to KPU program via torch.compile.

        MUST:
        - Register as a torch.compile backend
        - Receive FX graph from torch.compile
        - Lower FX operations to KPU operations
        - Apply graph optimizations (fusion, constant folding)
        - Determine tiling strategy for each operation
        - Allocate memory for tensors (respecting SRAM limits)
        - Schedule operations for execution
        - Quantize weights if precision is INT8/INT4

        RETURNS:
        - KPUProgram with all tensors, ops, and metadata populated
        - Program should be serializable and self-contained
        """
        pass

    @abstractmethod
    def validate_model(self, model_path: Path, config: KPUConfig) -> list[str]:
        """
        Quick validation without full compilation.
        Returns list of issues (empty if compatible).
        """
        pass

    @abstractmethod
    def estimate_memory(self, model_path: Path, config: KPUConfig, precision: KPUPrecision) -> dict[str, int]:
        """
        Estimate memory requirements.
        Returns: {"weights": N, "activations": N, "peak_sram": N, "peak_dram": N}
        """
        pass
```

#### Compiler Implementation Requirements

| Requirement | Description | Priority |
|-------------|-------------|----------|
| torch.compile Backend | Register as custom backend for torch.compile | P0 |
| FX Graph Lowering | Lower FX graph nodes to KPU operations | P0 |
| Operator Support | Support ops listed in `KPUConfig.native_ops` | P0 |
| Graph Fusion | Fuse Conv+BN+ReLU, MatMul+Add, etc. | P1 |
| Constant Folding | Fold constant expressions at compile time | P1 |
| Tiling | Tile operations to fit in L1/L2 SRAM | P0 |
| Memory Allocation | Assign tensors to memory hierarchy | P0 |
| Scheduling | Order ops for minimal memory footprint | P1 |
| INT8 Quantization | Calibrate and quantize weights/activations | P0 |
| Error Messages | Clear errors for unsupported ops/configs | P0 |

### 2. KPURuntimeInterface

**Location:** `agents/deployment/targets/kpu/spec.py`

The runtime executes compiled programs and returns results.

```python
class KPURuntimeInterface(ABC):
    @abstractmethod
    def load_program(self, program: KPUProgram) -> None:
        """
        Load program for execution.

        MUST:
        - Validate program compatibility with current config
        - Allocate memory for weights and activations
        - Load weights from program.weights_blob or external source
        - Initialize execution state
        """
        pass

    @abstractmethod
    def execute(
        self,
        inputs: dict[str, np.ndarray],
        mode: SimulationMode = SimulationMode.FUNCTIONAL,
    ) -> KPUExecutionResult:
        """
        Execute loaded program with inputs.

        MODES:
        - FUNCTIONAL: Fast execution, approximate timing
        - CYCLE_ACCURATE: Detailed timing with memory stalls
        - POWER_TRACE: Per-operation power tracking

        RETURNS:
        - outputs: Dictionary of output tensors (numpy arrays)
        - metrics: Performance metrics (cycles, time, power, memory)
        """
        pass

    @abstractmethod
    def get_input_specs(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Return [(name, shape, dtype), ...] for inputs."""
        pass

    @abstractmethod
    def get_output_specs(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Return [(name, shape, dtype), ...] for outputs."""
        pass

    @abstractmethod
    def unload_program(self) -> None:
        """Free resources and reset state."""
        pass
```

#### Runtime Implementation Requirements

| Requirement | Description | Priority |
|-------------|-------------|----------|
| Program Loading | Load serialized KPUProgram | P0 |
| Input Validation | Validate input shapes/dtypes | P0 |
| Execution | Execute ops in scheduled order | P0 |
| Output Collection | Return outputs as numpy arrays | P0 |
| Performance Metrics | Report cycles, time, power | P0 |
| Memory Metrics | Report SRAM/DRAM usage | P1 |
| Mode Support | Support FUNCTIONAL mode minimum | P0 |
| Cycle Accuracy | CYCLE_ACCURATE mode for detailed timing | P2 |

### 3. KPUSimulatorInterface (Optional)

**Location:** `agents/deployment/targets/kpu/spec.py`

For detailed cycle-accurate simulation with tracing.

```python
class KPUSimulatorInterface(ABC):
    @abstractmethod
    def reset(self) -> None: pass

    @abstractmethod
    def step(self, cycles: int = 1) -> None: pass

    @abstractmethod
    def run_until_complete(self) -> int: pass

    @abstractmethod
    def get_state(self) -> dict[str, Any]: pass

    @abstractmethod
    def get_power_trace(self) -> list[tuple[int, float]]: pass

    @abstractmethod
    def get_memory_trace(self) -> list[tuple[int, str, int, int]]: pass
```

## Data Models

### KPUConfig

Hardware configuration that the simulator must respect:

```python
@dataclass
class KPUConfig:
    name: str = "swkpu-v1"
    version: str = "1.0.0"

    memory: MemoryConfig  # SRAM/DRAM sizes and bandwidth
    compute: ComputeConfig  # Systolic array, clock, TOPS

    supported_precisions: list[KPUPrecision]  # INT8, FP16, etc.
    max_tensor_dims: int = 6
    max_batch_size: int = 16
    native_ops: list[str]  # Ops with hardware acceleration
```

### KPUProgram

Compiled program that flows from compiler to runtime:

```python
@dataclass
class KPUProgram:
    # Metadata
    name: str
    version: str
    source_model: str
    source_hash: str
    target_config: KPUConfig
    precision: KPUPrecision

    # Computation graph
    ops: list[KPUOp]  # Operations in scheduled order
    tensors: dict[str, KPUTensor]  # All tensors by ID

    # Input/output
    input_ids: list[str]
    output_ids: list[str]

    # Memory estimates
    peak_sram_bytes: int
    peak_dram_bytes: int
    weight_bytes: int
    activation_bytes: int

    # Performance estimates
    estimated_total_cycles: int
    estimated_total_energy_uj: float

    # Serialized weights
    weights_blob: bytes | None
```

### KPUOp

Single operation with tiling configuration:

```python
@dataclass
class KPUOp:
    id: str
    op_type: str  # "Conv", "MatMul", etc.

    input_ids: list[str]
    output_ids: list[str]
    attributes: dict[str, Any]

    # Tiling for this op
    tiling: TileConfig | None

    # Scheduling
    schedule_order: int
    fused_with: list[str]  # IDs of fused ops

    # Estimates
    estimated_cycles: int
    estimated_energy_pj: float
```

### TileConfig

Tiling configuration for an operation:

```python
@dataclass
class TileConfig:
    # Tile dimensions: {"N": 1, "C": 32, "H": 16, "W": 16}
    tile_sizes: dict[str, int]

    # Loop order: ["N", "H", "W", "C"]
    loop_order: list[str]

    # Optimizations
    double_buffer: bool = False
    prefetch_distance: int = 1
```

### KPUExecutionResult

Execution output from runtime:

```python
@dataclass
class KPUExecutionResult:
    success: bool
    error_message: str | None

    # Output tensors as numpy arrays
    outputs: dict[str, np.ndarray]

    # Performance metrics
    metrics: KPUPerformanceMetrics

    # Execution mode used
    mode: SimulationMode
```

### KPUPerformanceMetrics

Detailed performance metrics:

```python
@dataclass
class KPUPerformanceMetrics:
    # Timing
    total_cycles: int
    total_time_us: float

    # Cycle breakdown
    compute_cycles: int
    memory_stall_cycles: int
    other_cycles: int

    # Memory bandwidth
    dram_reads_bytes: int
    dram_writes_bytes: int
    sram_reads_bytes: int
    sram_writes_bytes: int

    # Cache stats
    l1_hits: int
    l1_misses: int
    l2_hits: int
    l2_misses: int

    # Power/energy
    average_power_watts: float
    total_energy_uj: float

    # Utilization
    compute_utilization: float  # 0-1
    memory_bandwidth_utilization: float  # 0-1

    # Per-op breakdown (optional)
    op_metrics: dict[str, dict[str, float]]
```

## Integration Points

### 1. Registering Custom Implementations

```python
from embodied_ai_architect.agents.deployment.targets.kpu import (
    StillwaterKPUTarget,
    KPUConfig,
)
from my_kpu_simulator import MyKPUCompiler, MyKPURuntime

# Create target with custom implementations
target = StillwaterKPUTarget(
    config=KPUConfig(
        name="swkpu-v2",
        version="2.0.0",
        # Custom config...
    ),
    compiler=MyKPUCompiler(),
    runtime=MyKPURuntime(),
)
```

### 2. Using in Deployment Agent

```python
from embodied_ai_architect.agents.deployment import DeploymentAgent
from embodied_ai_architect.agents.deployment.targets.kpu import StillwaterKPUTarget

# Register custom target
agent = DeploymentAgent()
agent.register_target("swkpu", StillwaterKPUTarget())

# Deploy PyTorch model
result = agent.execute({
    "model": "yolov8n.pt",  # PyTorch model (requires torch.compile)
    "target": "swkpu",
    "precision": "int8",
    "input_shape": [1, 3, 640, 640],
    "calibration_data": "./calibration_images",
})
```

### 3. Power Validation Integration

The deployment system will call runtime execution during validation to measure power:

```python
# During validation, the system:
# 1. Creates inference workload
# 2. Measures power via KPUPerformanceMetrics.average_power_watts
# 3. Compares to predicted power
# 4. Reports if within budget

config = ValidationConfig(
    power_validation=PowerConfig(
        enabled=True,
        power_budget_watts=5.0,  # KPU TDP constraint
        tolerance_percent=20.0,
    )
)
```

## Tiling Requirements

### Memory Hierarchy Constraints

```
Operation Input/Output Size
         │
         ▼
    ┌─────────────────┐
    │ Fits in L1 SRAM │ ◄── No tiling needed
    │   (256 KB)      │
    └────────┬────────┘
             │ No
             ▼
    ┌─────────────────┐
    │ Fits in L2 SRAM │ ◄── Tile to L1, stream from L2
    │    (2 MB)       │
    └────────┬────────┘
             │ No
             ▼
    ┌─────────────────┐
    │ Tile to L2,     │ ◄── Multi-level tiling
    │ stream from DRAM│
    └─────────────────┘
```

### Tiling Algorithm Requirements

1. **Convolution Tiling**
   - Tile output channels to fit in L1
   - Tile spatial dimensions (H, W) if needed
   - Consider kernel size in tile calculation
   - Handle padding at tile boundaries

2. **MatMul/GEMM Tiling**
   - Standard blocking: (M, N, K) tile sizes
   - Optimize for systolic array dimensions
   - Consider accumulator precision

3. **Memory Layout**
   - Support NCHW and NHWC layouts
   - Handle layout transforms at boundaries
   - Optimize for memory bandwidth

## Quantization Requirements

### INT8 Calibration Workflow

```
Calibration Dataset
        │
        ▼
┌───────────────────┐
│ Forward Pass      │ ◄── Run samples through FP32 model
│ (Collect Stats)   │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Histogram         │ ◄── Build activation histograms
│ Collection        │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Scale/ZP          │ ◄── Compute optimal scales
│ Computation       │     (MSE, entropy, percentile)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│ Weight            │ ◄── Quantize weights with scales
│ Quantization      │
└───────────────────┘
```

### Quantization Format

```python
# Per-tensor quantization
scale: float  # Q = R / scale
zero_point: int  # Q = R / scale + zero_point

# Dequantization: R = (Q - zero_point) * scale

# For INT8: Q in [-128, 127] or [0, 255]
```

## Validation Requirements

### Accuracy Validation

The deployment system validates by comparing outputs:

```python
# For each test sample:
baseline_output = pytorch_model(input)  # Reference PyTorch execution
kpu_output = kpu_runtime.execute(input)

# Compare
diff = np.abs(baseline_output - kpu_output).max()
passed = diff < tolerance  # Default 1% (0.01)
```

### Performance Validation

Required metrics from `KPUPerformanceMetrics`:

| Metric | Description | Required |
|--------|-------------|----------|
| total_cycles | End-to-end cycles | Yes |
| total_time_us | Wall clock time | Yes |
| compute_cycles | Pure compute cycles | Yes |
| memory_stall_cycles | Memory wait cycles | Yes |
| average_power_watts | Average power | Yes |
| total_energy_uj | Total energy | Yes |
| compute_utilization | Compute efficiency | Yes |

## Error Handling

### Compilation Errors

```python
class CompilationError(KPUError):
    """Base compilation error."""
    pass

class UnsupportedOpError(CompilationError):
    """Model contains unsupported operators."""
    def __init__(self, ops: list[str]):
        self.ops = ops
        super().__init__(f"Unsupported operators: {', '.join(ops)}")

class MemoryError(KPUError):
    """Model exceeds memory constraints."""
    def __init__(self, required: int, available: int, memory_type: str):
        ...
```

### Expected Error Messages

| Error Type | Example Message |
|------------|-----------------|
| Unsupported Op | "Unsupported operators: InstanceNorm, GroupNorm" |
| Memory Exceeded | "SRAM memory exceeded: 512KB required, 256KB available" |
| Invalid Input | "Input shape mismatch: expected (1,3,224,224), got (1,3,640,640)" |
| Quantization | "INT8 calibration requires at least 100 samples" |

## Testing Recommendations

### Unit Tests

1. **Compiler Tests**
   - Test each supported op in isolation
   - Test fusion patterns (Conv+BN+ReLU)
   - Test tiling for various input sizes
   - Test INT8 quantization accuracy

2. **Runtime Tests**
   - Test program loading/unloading
   - Test execution with random inputs
   - Verify output shapes
   - Test memory bounds

3. **Integration Tests**
   - End-to-end: PyTorch → torch.compile → execute → validate
   - Test common models (ResNet, YOLOv8, MobileNet)
   - Compare against PyTorch reference outputs

### Accuracy Benchmarks

| Model | Precision | Max Diff Target |
|-------|-----------|-----------------|
| ResNet-18 | FP16 | < 0.001 |
| ResNet-18 | INT8 | < 0.01 |
| YOLOv8n | FP16 | < 0.001 |
| YOLOv8n | INT8 | < 0.02 |
| MobileNetV2 | INT8 | < 0.01 |

## File Structure

```
src/embodied_ai_architect/agents/deployment/targets/kpu/
├── __init__.py         # Package exports
├── spec.py             # Interface definitions & data models
├── target.py           # DeploymentTarget implementation
├── compiler.py         # Real compiler implementation (your code)
├── runtime.py          # Real runtime implementation (your code)
└── simulator.py        # Optional cycle-accurate simulator (your code)
```

## Next Steps

1. **Review this specification** with the compiler/runtime team
2. **Implement `KPUCompilerInterface`** with basic op support
3. **Implement `KPURuntimeInterface`** with FUNCTIONAL mode
4. **Test integration** by replacing stub implementations
5. **Iterate on accuracy** until benchmarks pass
6. **Add CYCLE_ACCURATE mode** for detailed timing
7. **Implement power modeling** for energy estimation
