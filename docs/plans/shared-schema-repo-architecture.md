# Shared Schema Repository Architecture

## Decision Summary

**Decision**: Create a shared schema repository (`branes-ai/embodied-schemas`) that both `graphs` and `embodied-ai-architect` import.

**Rationale**: The schemas represent the contract between analysis tools (graphs) and orchestration/LLM layer (embodied-ai-architect). A shared package ensures:
- Single source of truth for data structures
- Independent versioning of schemas
- Clean dependency management
- A natural home for factual hardware/chip data

---

## Repository Structure

### New Repository: `branes-ai/embodied-schemas`

```
embodied-schemas/
├── README.md
├── pyproject.toml
├── src/
│   └── embodied_schemas/
│       ├── __init__.py
│       │
│       ├── # === SCHEMA MODELS === #
│       ├── hardware.py          # HardwareEntry, HardwareCapability, etc.
│       ├── models.py            # ModelEntry, ModelVariant, etc.
│       ├── sensors.py           # SensorEntry, CameraSpec, DepthSpec, etc.
│       ├── usecases.py          # UseCaseEntry, Constraint, SuccessCriterion
│       ├── benchmarks.py        # BenchmarkResult, BenchmarkConditions
│       ├── constraints.py       # ConstraintTier, PowerClass, LatencyTier
│       │
│       ├── # === FACTUAL DATA === #
│       ├── data/
│       │   ├── __init__.py
│       │   ├── hardware/
│       │   │   ├── nvidia/
│       │   │   │   ├── jetson_orin_agx.yaml
│       │   │   │   ├── jetson_orin_nano.yaml
│       │   │   │   ├── jetson_nano.yaml
│       │   │   │   └── ...
│       │   │   ├── qualcomm/
│       │   │   │   ├── rb5.yaml
│       │   │   │   ├── rb3_gen2.yaml
│       │   │   │   └── ...
│       │   │   ├── hailo/
│       │   │   │   ├── hailo8.yaml
│       │   │   │   ├── hailo8l.yaml
│       │   │   │   └── ...
│       │   │   ├── google/
│       │   │   │   ├── coral_edge_tpu.yaml
│       │   │   │   ├── coral_dev_board.yaml
│       │   │   │   └── ...
│       │   │   ├── intel/
│       │   │   │   ├── ncs2.yaml
│       │   │   │   └── ...
│       │   │   ├── amd/
│       │   │   │   └── ...
│       │   │   └── raspberry_pi/
│       │   │       ├── rpi4.yaml
│       │   │       ├── rpi5.yaml
│       │   │       └── ...
│       │   │
│       │   ├── chips/                 # Raw chip/SoC specifications
│       │   │   ├── nvidia/
│       │   │   │   ├── orin_soc.yaml  # Orin SoC specs (shared by AGX, NX, Nano)
│       │   │   │   ├── xavier_soc.yaml
│       │   │   │   └── ...
│       │   │   ├── qualcomm/
│       │   │   │   ├── qcs6490.yaml
│       │   │   │   ├── qcs8550.yaml
│       │   │   │   └── ...
│       │   │   └── ...
│       │   │
│       │   ├── models/
│       │   │   ├── detection/
│       │   │   │   ├── yolov8n.yaml
│       │   │   │   ├── yolov8s.yaml
│       │   │   │   ├── yolov8m.yaml
│       │   │   │   ├── rt_detr_l.yaml
│       │   │   │   └── ...
│       │   │   ├── segmentation/
│       │   │   │   └── ...
│       │   │   └── ...
│       │   │
│       │   ├── sensors/
│       │   │   ├── cameras/
│       │   │   ├── depth/
│       │   │   └── lidar/
│       │   │
│       │   ├── usecases/
│       │   │   ├── drone/
│       │   │   ├── quadruped/
│       │   │   ├── amr/
│       │   │   └── edge/
│       │   │
│       │   └── constraints/
│       │       ├── latency_tiers.yaml
│       │       ├── power_classes.yaml
│       │       └── environmental_ratings.yaml
│       │
│       ├── # === LOADERS === #
│       ├── loaders.py           # YAML loading with validation
│       └── registry.py          # Unified access to all data
│
└── tests/
    ├── test_hardware_schema.py
    ├── test_model_schema.py
    ├── test_data_validity.py    # Validate all YAML files
    └── ...
```

---

## Schema Contents

### Core Pydantic Models

From the knowledge-base-schema.md design:

| Module | Models |
|--------|--------|
| `hardware.py` | `HardwareEntry`, `HardwareCapability`, `PhysicalSpec`, `EnvironmentalSpec`, `PowerSpec`, `InterfaceSpec`, `PowerMode` |
| `models.py` | `ModelEntry`, `ModelVariant`, `AccuracyBenchmark`, `MemoryRequirements` |
| `sensors.py` | `SensorEntry`, `CameraSpec`, `DepthSpec`, `LidarSpec` |
| `usecases.py` | `UseCaseEntry`, `Constraint`, `SuccessCriterion`, `PerceptionRequirement` |
| `benchmarks.py` | `BenchmarkResult`, `LatencyMetrics`, `PowerMetrics`, `ThermalMetrics` |
| `constraints.py` | `LatencyTier`, `PowerClass`, `ConstraintCriticality` (enums + tier definitions) |

### Factual Data Categories

| Category | Contents | Source |
|----------|----------|--------|
| **Hardware Platforms** | Complete dev kits/boards with all specs | Datasheets |
| **Chips/SoCs** | Raw silicon specs (shared across platforms) | Vendor specs |
| **Models** | Architecture + accuracy + variants | Model zoos, papers |
| **Sensors** | Cameras, depth sensors, LiDAR specs | Datasheets |
| **Use Cases** | Constraint templates for domains | Domain expertise |
| **Constraints** | Tier definitions, classification rules | Design decisions |

---

## Dependency Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    embodied-schemas                              │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Pydantic   │  │  YAML Data   │  │     Loaders/         │  │
│  │   Models     │  │  (facts)     │  │     Registry         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                    ↑                           ↑
                    │                           │
        ┌───────────┴───────────┐   ┌──────────┴────────────┐
        │                       │   │                        │
        │      graphs           │   │  embodied-ai-architect │
        │                       │   │                        │
        │  - Analysis tools     │   │  - LLM orchestration   │
        │  - Roofline models    │   │  - Knowledge base      │
        │  - Simulators         │   │  - Tool wrappers       │
        │  - QA facilities      │   │  - CLI                 │
        │                       │   │                        │
        │  Uses schemas for:    │   │  Uses schemas for:     │
        │  - Tool return types  │   │  - KB storage          │
        │  - Hardware specs     │   │  - Tool I/O            │
        │  - Validation         │   │  - LLM responses       │
        └───────────────────────┘   └────────────────────────┘
```

---

## Package Configuration

### `pyproject.toml` for embodied-schemas

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "embodied-schemas"
version = "0.1.0"
description = "Shared schemas and factual data for embodied AI codesign"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Branes-ai Team"}
]
dependencies = [
    "pydantic>=2.0.0",
    "pyyaml>=6.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
"embodied_schemas.data" = ["**/*.yaml"]
```

### Integration in graphs

```toml
# graphs/pyproject.toml
[project]
dependencies = [
    # ... existing deps
    "embodied-schemas>=0.1.0",
]
```

### Integration in embodied-ai-architect

```toml
# embodied-ai-architect/pyproject.toml
[project]
dependencies = [
    # ... existing deps
    "embodied-schemas>=0.1.0",
]
```

---

## API Design

### Schema Import Pattern

```python
# Import schema models
from embodied_schemas import (
    HardwareEntry,
    ModelEntry,
    SensorEntry,
    UseCaseEntry,
    BenchmarkResult,
)

# Import enums
from embodied_schemas.constraints import (
    LatencyTier,
    PowerClass,
    HardwareType,
)
```

### Data Access Pattern

```python
from embodied_schemas.registry import Registry

# Load all data
registry = Registry.load()

# Access hardware
jetson_nano = registry.hardware.get("nvidia_jetson_nano_4gb")
all_nvidia = registry.hardware.find(vendor="NVIDIA")
edge_devices = registry.hardware.find(suitable_for="edge")

# Access models
yolov8s = registry.models.get("yolov8s")
detection_models = registry.models.find(type="object_detection")

# Access use cases
drone_obstacle = registry.usecases.get("drone_obstacle_avoidance")

# Check compatibility
compatible = registry.is_compatible(
    model_id="yolov8s",
    hardware_id="nvidia_jetson_nano_4gb",
    variant="fp16"
)
```

### Tool Output Pattern (in graphs)

```python
from embodied_schemas import BenchmarkResult
from embodied_schemas.hardware import LatencyMetrics, PowerMetrics

def analyze_model(model, hardware) -> BenchmarkResult:
    # ... run analysis ...
    return BenchmarkResult(
        model_id=model.id,
        hardware_id=hardware.id,
        latency=LatencyMetrics(
            mean_ms=45.2,
            std_ms=3.1,
            p95_ms=51.2,
        ),
        power=PowerMetrics(
            mean_watts=8.5,
            energy_per_inference_mj=385,
        ),
        verdict="FAIL",
        confidence="high",
    )
```

---

## Implementation Plan

### Phase 1: Repository Setup (Day 1)
1. Create `branes-ai/embodied-schemas` repo
2. Set up pyproject.toml with minimal dependencies
3. Create directory structure
4. Set up CI/CD for tests

### Phase 2: Core Schemas (Days 2-3)
1. Port Pydantic models from knowledge-base-schema.md
2. Implement `hardware.py`, `models.py`, `sensors.py`
3. Implement `usecases.py`, `benchmarks.py`, `constraints.py`
4. Write schema validation tests

### Phase 3: Data Loaders (Day 4)
1. Implement YAML loader with Pydantic validation
2. Create Registry class for unified access
3. Add query/filter methods
4. Test data loading

### Phase 4: Seed Data (Days 5-7)
1. Populate hardware data (start with 10 platforms)
2. Populate model data (start with YOLO, RT-DETR, MobileNet families)
3. Populate use case templates (5 core use cases)
4. Add constraint tier definitions

### Phase 5: Integration (Days 8-10)
1. Add `embodied-schemas` dependency to graphs
2. Update graphs tools to return schema types
3. Add `embodied-schemas` dependency to embodied-ai-architect
4. Migrate existing knowledge_base.py to use shared schemas
5. Update LLM tools to use registry

---

## Data Governance

### Adding New Hardware

1. Create YAML file in `data/hardware/{vendor}/{name}.yaml`
2. Fill all required fields from datasheet
3. Run validation: `pytest tests/test_data_validity.py`
4. Submit PR with datasheet link in commit message

### Schema Changes

1. Schema changes require version bump
2. Breaking changes require major version bump
3. Both graphs and embodied-ai-architect must update together for breaking changes
4. Use deprecation warnings for 1 minor version before removal

### Data Quality

- All YAML files validated against Pydantic schemas in CI
- Datasheet URLs required for hardware entries
- Last-updated timestamps required
- Benchmark results require reproducibility info (script, commit)

---

## Migration Path

### From Current embodied-ai-architect

1. `agents/hardware_profile/models.py` → `embodied_schemas/hardware.py`
2. `agents/hardware_profile/knowledge_base.py` → `embodied_schemas/data/hardware/`
3. Keep backward-compatible wrapper in embodied-ai-architect:

```python
# embodied_ai_architect/agents/hardware_profile/models.py
# Backward compatibility - re-export from shared schemas
from embodied_schemas.hardware import (
    HardwareProfile,
    HardwareCapability,
    HardwareType,
    # ... etc
)

__all__ = ["HardwareProfile", "HardwareCapability", "HardwareType", ...]
```

### From Current graphs

1. Identify all hardware-related data structures
2. Map to equivalent embodied-schemas types
3. Update tool return types to use shared schemas
4. Update tests

---

## Data Split: graphs/hardware_registry vs embodied-schemas

The `graphs` repository contains a comprehensive `hardware_registry/` with 43+ hardware profiles. This section defines what data stays in `graphs` vs. migrates to `embodied-schemas`.

### Current graphs/hardware_registry Structure

```
graphs/hardware_registry/
├── cpu/                    # 18 devices (Intel, AMD, ARM, Qualcomm, Ampere)
├── gpu/                    # 10 devices (NVIDIA datacenter + edge)
├── accelerator/            # 12 devices (TPU, Hailo, EdgeTPU, Stillwater KPU)
├── dsp/                    # 2 devices (Qualcomm Snapdragon)
└── boards/                 # 2 devices (Jetson dev kits)
    └── {device_id}/
        ├── spec.json                           # Hardware specifications
        └── calibrations/                       # Measured performance data
            ├── {power_mode}_{freq}MHz_{framework}.json
            └── {power_mode}_{freq}MHz_{framework}.log
```

### Data Classification

| Data Type | Description | Location |
|-----------|-------------|----------|
| **Datasheet specs** | Vendor-published facts (memory, TDP, cores) | `embodied-schemas` |
| **Power profiles** | Available modes and power limits | `embodied-schemas` |
| **Physical specs** | Weight, dimensions, form factor | `embodied-schemas` |
| **Environmental specs** | Temp range, IP rating | `embodied-schemas` |
| **Interface specs** | CSI, USB, PCIe counts | `embodied-schemas` |
| **ops_per_clock** | Derived calculation for roofline analysis | `graphs` |
| **theoretical_peaks** | Computed from ops_per_clock × frequency | `graphs` |
| **Calibration data** | Measured performance, efficiency curves | `graphs` |
| **Operation profiles** | GEMM, CONV, attention benchmarks | `graphs` |

### What Migrates to embodied-schemas

**From `spec.json` - the "datasheet facts":**

```yaml
# embodied-schemas/data/hardware/nvidia/jetson_orin_nano_gpu.yaml
id: jetson_orin_nano_gpu
name: Jetson Orin Nano GPU
vendor: NVIDIA
model: Orin Nano
hardware_type: gpu

capabilities:
  memory_gb: 8
  memory_bandwidth_gbps: 68
  compute_units: 512  # CUDA cores
  peak_tflops_fp16: 0.625

power:
  tdp_watts: 25
  power_modes:
    - name: "7W"
      power_watts: 7
      gpu_freq_mhz: 306
    - name: "15W"
      power_watts: 15
      gpu_freq_mhz: 612
    - name: "25W"
      power_watts: 25
      gpu_freq_mhz: 918
    - name: "MAXNSUPER"
      power_watts: 25
      gpu_freq_mhz: 1020

physical:
  form_factor: som
  weight_grams: 60
  dimensions_mm: [69.6, 45, 20]

environmental:
  operating_temp_c: [-25, 80]

interfaces:
  camera_csi: 2
  usb3: 3
  pcie_lanes: 4

suitable_for: [edge, robotics, drone]
last_updated: "2024-12-20"
```

### What Stays in graphs

**Roofline-specific derivations (`spec.json`):**

```json
{
  "base_id": "jetson_orin_nano_gpu",

  "ops_per_clock": {
    "fp32": 1024,
    "fp16": 2048,
    "int8": 4096
  },
  "ops_per_clock_notes": "512 CUDA cores × 2 ops/core for FP32",

  "theoretical_peaks": {
    "7W_306MHz": {"fp32_gflops": 313, "fp16_gflops": 626},
    "15W_612MHz": {"fp32_gflops": 626, "fp16_gflops": 1253},
    "25W_918MHz": {"fp32_gflops": 939, "fp16_gflops": 1879}
  }
}
```

**Calibration data (`calibrations/*.json`):**

```json
{
  "metadata": {
    "hardware_name": "jetson_orin_nano_gpu",
    "calibration_date": "2024-12-15T10:30:00Z",
    "power_mode": "15W",
    "framework": "pytorch"
  },
  "theoretical_peak_gflops": 1253,
  "best_measured_gflops": 1150,
  "best_efficiency": 0.92,
  "measured_bandwidth_gbps": 62.4,
  "bandwidth_efficiency": 0.92,
  "operation_profiles": {
    "blas3_gemm_fp16_1024x1024": {
      "measured_gflops": 1120,
      "efficiency": 0.89,
      "memory_bound": false,
      "compute_bound": true
    }
  }
}
```

### Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      embodied-schemas                            │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ HardwareEntry (datasheet specs)                           │  │
│  │ - id, vendor, model, architecture                         │  │
│  │ - capabilities: memory_gb, peak_bandwidth, compute_units  │  │
│  │ - power: tdp_watts, power_modes[]                         │  │
│  │ - physical, environmental, interfaces                     │  │
│  │ - suitable_for, target_applications                       │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              ↑
                              │ imports base specs
                              │
┌─────────────────────────────────────────────────────────────────┐
│                          graphs                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ RooflineProfile (analysis-specific)                       │  │
│  │ - base: HardwareEntry (from embodied-schemas)             │  │
│  │ - ops_per_clock: {fp32: 1024, fp16: 2048, int8: 4096}    │  │
│  │ - theoretical_peaks: {per power mode}                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ CalibrationProfile (measured data)                        │  │
│  │ - hardware_id: reference to base                          │  │
│  │ - power_mode, frequency, framework                        │  │
│  │ - measured_gflops, efficiency, bandwidth                  │  │
│  │ - operation_profiles: {gemm, conv, attention, ...}        │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Migration Strategy

**Phase 1: Parallel Operation**
1. Add base specs to `embodied-schemas` for hardware platforms
2. `graphs` continues using its current `spec.json` files
3. No breaking changes to `graphs` tools

**Phase 2: Reference Integration**
1. Update `graphs/hardware_registry/*/spec.json` to include `base_id` field
2. `graphs` imports `HardwareEntry` from `embodied-schemas` for base specs
3. Roofline tools compose: `base_specs + roofline_params + calibration`

**Phase 3: Cleanup**
1. Remove duplicated datasheet fields from `graphs/spec.json`
2. `graphs/spec.json` contains only: `base_id`, `ops_per_clock`, `theoretical_peaks`
3. All calibration data remains in `graphs/calibrations/`

### Field Mapping: graphs spec.json → embodied-schemas

| graphs field | embodied-schemas field | Notes |
|--------------|------------------------|-------|
| `id` | `id` | Same identifier |
| `vendor` | `vendor` | Direct mapping |
| `model` | `model` | Direct mapping |
| `device_type` | `hardware_type` | Enum mapping |
| `architecture` | `capabilities.architecture` | May need nesting |
| `compute_units` | `capabilities.compute_units` | Direct mapping |
| `memory_gb` | `capabilities.memory_gb` | Direct mapping |
| `peak_bandwidth_gbps` | `capabilities.memory_bandwidth_gbps` | Rename |
| `tdp_watts` | `power.tdp_watts` | Nested under power |
| `base_clock_mhz` | `power.power_modes[].gpu_freq_mhz` | Per mode |
| `power_profiles` | `power.power_modes` | Structure change |
| `ops_per_clock` | **stays in graphs** | Roofline-specific |
| `theoretical_peaks` | **stays in graphs** | Computed values |

### Example: Complete Integration

```python
# In graphs/src/graphs/hardware/registry/profile.py

from embodied_schemas import HardwareEntry
from embodied_schemas.registry import Registry

@dataclass
class RooflineProfile:
    """Hardware profile with roofline analysis parameters."""

    # Base specs from embodied-schemas
    base: HardwareEntry

    # Roofline-specific (from graphs/hardware_registry/*/spec.json)
    ops_per_clock: dict[str, int]
    theoretical_peaks: dict[str, dict[str, float]]

    # Calibrations (from graphs/hardware_registry/*/calibrations/)
    calibrations: dict[str, CalibrationData]

    @classmethod
    def load(cls, hardware_id: str) -> "RooflineProfile":
        # Load base from embodied-schemas
        registry = Registry.load()
        base = registry.hardware.get(hardware_id)

        # Load roofline params from graphs
        roofline_spec = load_roofline_spec(hardware_id)

        # Load calibrations from graphs
        calibrations = load_calibrations(hardware_id)

        return cls(
            base=base,
            ops_per_clock=roofline_spec["ops_per_clock"],
            theoretical_peaks=roofline_spec["theoretical_peaks"],
            calibrations=calibrations,
        )
```

### Rationale for This Split

1. **Datasheet specs are universal** - Any tool analyzing hardware needs memory, power, interfaces
2. **Roofline params are tool-specific** - `ops_per_clock` calculation is specific to roofline methodology
3. **Calibration is empirical** - Measured data belongs with the tools that generate and use it
4. **Clean dependency direction** - `graphs` depends on `embodied-schemas`, not vice versa
5. **Single source of truth** - Datasheet facts updated in one place, flow to all consumers

---

## Benefits

1. **Single Source of Truth** - One place for schema definitions and factual data
2. **Consistent Tool I/O** - graphs and embodied-ai-architect speak the same language
3. **Versioned Data** - Hardware specs, model data tracked in git with history
4. **Validation** - Pydantic ensures data quality at load time
5. **Extensible** - Easy to add new hardware vendors, model families, use cases
6. **Reusable** - Other projects can depend on embodied-schemas for the data

---

## Resolved Questions

1. **Chip vs Platform separation** - Should raw SoC specs (e.g., Orin SoC) be separate from platform specs (e.g., Jetson AGX Orin)?
   - **Answer**: Yes, `chips/` directory for raw silicon, `hardware/` for complete platforms

2. **Benchmark/Calibration data location** - Should measured benchmark results live in embodied-schemas or in the repo that generated them?
   - **Answer**: Calibration data and roofline-specific parameters stay in `graphs/hardware_registry/`. Only datasheet specs migrate to `embodied-schemas`. See "Data Split" section above.

3. **Model weights** - Should embodied-schemas track where to download model weights?
   - **Answer**: Yes, `source_url` field with download links, but no actual weights in repo

## Open Questions

1. **Precision type enumeration** - Should `embodied-schemas` define the canonical precision types (fp32, fp16, int8, etc.) or leave that to `graphs`?
   - **Tentative**: Define in `embodied-schemas` since it's needed for model variants and hardware capabilities

2. **Cross-repo ID consistency** - How do we ensure hardware IDs are consistent between `embodied-schemas` and `graphs/hardware_registry`?
   - **Tentative**: `graphs` uses `base_id` field to reference `embodied-schemas` entries; validation in CI

---

*Document created: December 2024*
*Updated: December 2024 - Added Data Split section for graphs/hardware_registry*
*Status: Architecture Decision - Approved*
*Completed: Created branes-ai/embodied-schemas repository with initial structure*
