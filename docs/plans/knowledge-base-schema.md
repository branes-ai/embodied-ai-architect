# Knowledge Base Schema Design

## Overview

This document defines the complete schema for the Embodied AI Codesign knowledge base. The schema extends the existing `HardwareProfile` infrastructure and adds Model, Sensor, UseCase, and Constraint catalogs with explicit relationships.

## Design Principles

1. **YAML files as source of truth** - Version-controlled, human-editable
2. **Pydantic models for validation** - Type safety and serialization
3. **Flat file hierarchy** - Easy to navigate and extend
4. **Explicit relationships** - Foreign keys by ID, not nested objects
5. **Verdict-ready fields** - Pre-computed thresholds for tool outputs

---

## 1. Directory Structure

```
knowledge_base/
├── hardware/
│   ├── _schema.yaml              # Schema documentation
│   ├── nvidia_jetson_orin.yaml
│   ├── nvidia_jetson_nano.yaml
│   ├── hailo_8.yaml
│   ├── google_coral.yaml
│   ├── qualcomm_rb5.yaml
│   └── ...
├── models/
│   ├── _schema.yaml
│   ├── yolov8n.yaml
│   ├── yolov8s.yaml
│   ├── yolov8m.yaml
│   ├── rt_detr_l.yaml
│   ├── mobilenet_v3.yaml
│   └── ...
├── sensors/
│   ├── _schema.yaml
│   ├── cameras/
│   │   ├── imx477.yaml
│   │   ├── ov5640.yaml
│   │   └── ...
│   ├── depth/
│   │   ├── intel_d435.yaml
│   │   ├── oak_d.yaml
│   │   └── ...
│   └── lidar/
│       ├── livox_mid360.yaml
│       └── ...
├── usecases/
│   ├── _schema.yaml
│   ├── drone_obstacle_avoidance.yaml
│   ├── drone_landing_zone.yaml
│   ├── quadruped_terrain.yaml
│   ├── amr_navigation.yaml
│   ├── edge_surveillance.yaml
│   └── ...
├── benchmarks/
│   ├── _schema.yaml
│   └── results/
│       ├── yolov8s_jetson_orin.yaml
│       ├── yolov8s_jetson_nano.yaml
│       └── ...
├── constraints/
│   ├── _schema.yaml
│   ├── latency_tiers.yaml
│   ├── power_classes.yaml
│   ├── accuracy_requirements.yaml
│   └── environmental_ratings.yaml
└── relationships/
    ├── model_hardware_compatibility.yaml
    ├── usecase_constraint_mappings.yaml
    └── sensor_interface_requirements.yaml
```

---

## 2. Hardware Schema

### Extends existing `HardwareProfile` with embodied AI fields

```yaml
# hardware/nvidia_jetson_nano.yaml
id: nvidia_jetson_nano_4gb
name: NVIDIA Jetson Nano
vendor: NVIDIA
model: Jetson Nano Developer Kit
hardware_type: gpu  # cpu, gpu, npu, tpu, fpga, dsp

# Existing capability fields (from current HardwareCapability)
capabilities:
  peak_tflops_fp32: 0.472
  peak_tflops_fp16: 0.944
  peak_tflops_int8: 1.888
  memory_gb: 4
  memory_bandwidth_gbps: 25.6
  compute_units: 128  # CUDA cores
  tdp_watts: 10
  typical_power_watts: 5

# NEW: Physical constraints (for embodied systems)
physical:
  weight_grams: 140  # Module only
  dimensions_mm: [100, 80, 29]  # L x W x H
  form_factor: som  # som, pcie, usb, standalone
  mounting: carrier_board  # carrier_board, direct, rack

# NEW: Environmental ratings
environmental:
  operating_temp_c: [-25, 80]  # [min, max]
  storage_temp_c: [-40, 85]
  humidity_percent: [5, 95]  # Non-condensing
  ip_rating: null  # IP65, IP67, etc. (null = none)
  vibration_g: 3.0  # Max sustained vibration
  shock_g: 30  # Max shock

# NEW: Power delivery
power:
  input_voltage_v: [5, 5]  # [min, max]
  power_modes:
    - name: 10W
      power_watts: 10
      gpu_freq_mhz: 921
      cpu_freq_mhz: 1479
    - name: 5W
      power_watts: 5
      gpu_freq_mhz: 640
      cpu_freq_mhz: 918
  battery_compatible: true

# NEW: Interfaces (for sensor integration)
interfaces:
  camera_csi: 1  # Number of CSI ports
  usb3: 4
  usb2: 0
  pcie: 1
  gpio: 40
  ethernet: 1gbps  # 1gbps, 10gbps, none
  wifi: false  # Onboard WiFi
  can_bus: false

# Software ecosystem
software:
  os: [linux, jetpack]
  frameworks: [pytorch, tensorflow, tensorrt, onnx, deepstream]
  quantization: [fp32, fp16, int8]
  sdk: jetpack

# Deployment classification
deployment:
  suitable_for: [edge, robotics, embedded, drone]
  target_applications: [vision, object_detection, tracking]

# Cost and availability
cost_usd: 149
availability: widely_available  # widely_available, limited, eol, announced
lifecycle_status: mature  # new, mature, legacy, eol

# Metadata
notes: "Entry-level edge AI platform, good for prototyping"
datasheet_url: "https://developer.nvidia.com/embedded/jetson-nano"
last_updated: 2024-12-01
```

### Hardware Pydantic Model

```python
from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class HardwareType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    NPU = "npu"
    TPU = "tpu"
    FPGA = "fpga"
    DSP = "dsp"

class FormFactor(str, Enum):
    SOM = "som"  # System on Module
    PCIE = "pcie"
    USB = "usb"
    STANDALONE = "standalone"
    RACK = "rack"

class PowerMode(BaseModel):
    name: str
    power_watts: float
    gpu_freq_mhz: Optional[int] = None
    cpu_freq_mhz: Optional[int] = None

class PhysicalSpec(BaseModel):
    weight_grams: Optional[float] = None
    dimensions_mm: Optional[List[float]] = None  # [L, W, H]
    form_factor: FormFactor
    mounting: str

class EnvironmentalSpec(BaseModel):
    operating_temp_c: List[float]  # [min, max]
    storage_temp_c: Optional[List[float]] = None
    humidity_percent: Optional[List[float]] = None
    ip_rating: Optional[str] = None
    vibration_g: Optional[float] = None
    shock_g: Optional[float] = None

class PowerSpec(BaseModel):
    input_voltage_v: List[float]  # [min, max]
    power_modes: List[PowerMode]
    battery_compatible: bool = False

class InterfaceSpec(BaseModel):
    camera_csi: int = 0
    usb3: int = 0
    usb2: int = 0
    pcie: int = 0
    gpio: int = 0
    ethernet: Optional[str] = None
    wifi: bool = False
    can_bus: bool = False
    i2c: int = 0
    spi: int = 0
    uart: int = 0

class HardwareEntry(BaseModel):
    """Complete hardware catalog entry for embodied AI."""
    id: str  # Unique identifier
    name: str
    vendor: str
    model: str
    hardware_type: HardwareType

    # Compute capabilities (existing schema)
    capabilities: HardwareCapability

    # NEW: Embodied AI fields
    physical: Optional[PhysicalSpec] = None
    environmental: Optional[EnvironmentalSpec] = None
    power: Optional[PowerSpec] = None
    interfaces: Optional[InterfaceSpec] = None

    # Software
    software: dict

    # Deployment
    deployment: dict

    # Cost
    cost_usd: Optional[float] = None
    availability: str = "widely_available"
    lifecycle_status: str = "mature"

    # Metadata
    notes: str = ""
    datasheet_url: Optional[str] = None
    last_updated: str
```

---

## 3. Model Schema

### Perception model catalog entry

```yaml
# models/yolov8s.yaml
id: yolov8s
name: YOLOv8s
family: yolov8
version: "8.0"
vendor: Ultralytics

# Architecture
architecture:
  type: object_detection  # object_detection, segmentation, pose, depth, tracking
  backbone: cspnet
  neck: pan
  head: decoupled
  params_millions: 11.2
  flops_billions: 28.6
  layers: 225

# Input specification
input:
  shape: [1, 3, 640, 640]  # NCHW
  dtype: float32
  normalization: [0, 1]  # Range
  channel_order: rgb

# Output specification
output:
  format: boxes_scores_classes
  max_detections: 300
  num_classes: 80  # COCO

# Accuracy benchmarks (on standard datasets)
accuracy:
  coco_val:
    map_50: 0.614
    map_50_95: 0.449
    dataset: coco_val2017
    methodology: ultralytics_default
  custom_drone:  # Optional domain-specific
    map_50: 0.58
    dataset: visdrone_val
    notes: "Lower due to small objects"

# Available variants and formats
variants:
  - name: fp32
    dtype: float32
    format: pytorch
    file_size_mb: 42.5
  - name: fp16
    dtype: float16
    format: onnx
    file_size_mb: 21.3
    accuracy_delta: -0.001  # Relative to fp32
  - name: int8
    dtype: int8
    format: tensorrt
    file_size_mb: 11.2
    accuracy_delta: -0.02
    calibration: coco_val_1000

# Memory requirements (runtime)
memory:
  weights_mb: 42.5
  peak_activation_mb: 180  # At batch=1
  workspace_mb: 50  # TensorRT workspace

# Known hardware compatibility
compatible_hardware:
  - nvidia_jetson_orin
  - nvidia_jetson_nano
  - hailo_8
  - google_coral  # Requires conversion

# Use case fit
suitable_for:
  - drone_obstacle_avoidance
  - amr_navigation
  - edge_surveillance

# Optimization notes
optimization_notes:
  tensorrt: "Use FP16 for best latency/accuracy"
  hailo: "Requires Hailo Model Zoo conversion"
  coral: "INT8 only, significant accuracy loss"

# Metadata
license: AGPL-3.0
source_url: "https://github.com/ultralytics/ultralytics"
paper_url: null
last_updated: 2024-12-01
```

### Model Pydantic Model

```python
class ModelType(str, Enum):
    OBJECT_DETECTION = "object_detection"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    POSE_ESTIMATION = "pose_estimation"
    DEPTH_ESTIMATION = "depth_estimation"
    CLASSIFICATION = "classification"
    TRACKING = "tracking"
    SLAM = "slam"

class ModelVariant(BaseModel):
    name: str
    dtype: str  # fp32, fp16, int8, int4
    format: str  # pytorch, onnx, tensorrt, tflite, hailo
    file_size_mb: float
    accuracy_delta: float = 0.0  # Relative to baseline
    calibration: Optional[str] = None

class AccuracyBenchmark(BaseModel):
    map_50: Optional[float] = None
    map_50_95: Optional[float] = None
    miou: Optional[float] = None
    accuracy: Optional[float] = None
    dataset: str
    methodology: Optional[str] = None
    notes: Optional[str] = None

class MemoryRequirements(BaseModel):
    weights_mb: float
    peak_activation_mb: float
    workspace_mb: float = 0

class ModelEntry(BaseModel):
    """Complete model catalog entry."""
    id: str
    name: str
    family: str
    version: str
    vendor: str

    # Architecture
    architecture: dict

    # I/O
    input: dict
    output: dict

    # Accuracy
    accuracy: dict[str, AccuracyBenchmark]

    # Variants
    variants: List[ModelVariant]

    # Memory
    memory: MemoryRequirements

    # Compatibility
    compatible_hardware: List[str]  # Hardware IDs
    suitable_for: List[str]  # Use case IDs

    # Metadata
    optimization_notes: dict = Field(default_factory=dict)
    license: str
    source_url: Optional[str] = None
    last_updated: str
```

---

## 4. Sensor Schema

```yaml
# sensors/cameras/imx477.yaml
id: sony_imx477
name: Sony IMX477
category: camera  # camera, depth, lidar, imu, gps

# Sensor specifications
specs:
  type: cmos
  resolution: [4056, 3040]  # [width, height]
  pixel_size_um: 1.55
  sensor_size_mm: [7.9, 6.3]
  max_fps: 60
  dynamic_range_db: 67
  sensitivity_iso: [100, 6400]

# Interface
interface:
  type: csi  # csi, usb, ethernet, i2c, spi
  lanes: 2
  data_rate_gbps: 1.0

# Optical
optics:
  lens_mount: c_mount
  fov_deg: null  # Depends on lens
  focus: manual  # manual, auto, fixed
  iris: manual

# Power
power:
  voltage_v: 3.3
  current_ma: 200
  power_watts: 0.66

# Physical
physical:
  weight_grams: 5
  dimensions_mm: [8.0, 6.3, 5.0]

# Environmental
environmental:
  operating_temp_c: [-30, 70]
  storage_temp_c: [-40, 85]

# Compatible hardware (by interface)
compatible_interfaces:
  - camera_csi  # Matches InterfaceSpec field names

# Use case fit
suitable_for:
  - drone_perception
  - quadruped_vision
  - edge_surveillance

# Cost
cost_usd: 50
availability: widely_available

# Metadata
datasheet_url: "https://..."
last_updated: 2024-12-01
```

### Depth Sensor Schema

```yaml
# sensors/depth/intel_d435.yaml
id: intel_d435
name: Intel RealSense D435
category: depth

specs:
  type: stereo  # stereo, tof, structured_light, lidar
  depth_resolution: [1280, 720]
  rgb_resolution: [1920, 1080]
  depth_fps: 90
  rgb_fps: 30
  range_m: [0.1, 10.0]  # [min, max]
  accuracy_percent: 2  # At 2m
  baseline_mm: 50  # Stereo baseline

interface:
  type: usb
  version: "3.0"
  data_rate_gbps: 5.0

power:
  voltage_v: 5.0
  current_ma: 700
  power_watts: 3.5

physical:
  weight_grams: 72
  dimensions_mm: [90, 25, 25]

environmental:
  operating_temp_c: [0, 35]
  indoor_outdoor: indoor  # indoor, outdoor, both

compatible_interfaces:
  - usb3

suitable_for:
  - amr_navigation
  - quadruped_terrain
  - indoor_robotics

cost_usd: 350
availability: widely_available
last_updated: 2024-12-01
```

---

## 5. Use Case Schema

```yaml
# usecases/drone_obstacle_avoidance.yaml
id: drone_obstacle_avoidance
name: Drone Obstacle Avoidance
category: drone  # drone, quadruped, biped, amr, edge

description: |
  Real-time obstacle detection and avoidance for UAV flight.
  Requires low-latency perception with high recall for safety.

# Platform constraints
platform:
  type: drone
  size_class: small  # micro, small, medium, large
  indoor_outdoor: outdoor

# Required perception capabilities
perception_requirements:
  tasks:
    - object_detection
    - depth_estimation
  target_classes:
    - person
    - vehicle
    - tree
    - building
    - wire  # Critical for drones
  detection_range_m: [1, 50]
  field_of_view_deg: 120  # Minimum horizontal FoV

# Hard constraints (MUST meet)
constraints:
  latency:
    max_ms: 33.3  # 30 FPS minimum
    tier: real_time
    criticality: safety_critical
  power:
    max_watts: 15
    tier: battery_powered
    criticality: hard
  accuracy:
    min_recall: 0.95  # High recall for safety
    min_precision: 0.70  # Tolerate false positives
    criticality: safety_critical
  memory:
    max_mb: 4000
    criticality: hard
  weight:
    max_grams: 200  # Compute + sensors
    criticality: medium
  cost:
    max_usd: 500
    criticality: soft

# Soft constraints (SHOULD meet)
preferences:
  - minimize_power
  - maximize_accuracy
  - minimize_weight

# Implied constraints (derived from above)
implied:
  - real_time  # From latency < 33ms
  - outdoor_rated  # From indoor_outdoor
  - low_power  # From battery_powered

# Recommended configurations
recommended_hardware:
  - id: nvidia_jetson_orin_nano
    notes: "Best balance of power and performance"
  - id: hailo_8
    notes: "Lower power, needs model conversion"

recommended_models:
  - id: yolov8n
    variant: int8
    notes: "Fastest, acceptable accuracy"
  - id: yolov8s
    variant: fp16
    notes: "Better accuracy, higher power"

recommended_sensors:
  - id: oak_d_lite
    notes: "Stereo depth + RGB in one unit"
  - id: sony_imx477
    with_lens: wide_angle
    notes: "Wide FoV for obstacle detection"

# Success criteria (for validation)
success_criteria:
  - metric: latency_ms
    target: 33.3
    operator: lt
  - metric: power_watts
    target: 15
    operator: lt
  - metric: recall
    target: 0.95
    operator: gte
  - metric: weight_grams
    target: 200
    operator: lt

# Metadata
last_updated: 2024-12-01
maintainer: "embodied-ai-team"
```

### Use Case Pydantic Model

```python
class ConstraintCriticality(str, Enum):
    SAFETY_CRITICAL = "safety_critical"
    HARD = "hard"
    MEDIUM = "medium"
    SOFT = "soft"

class Constraint(BaseModel):
    max_ms: Optional[float] = None  # For latency
    max_watts: Optional[float] = None  # For power
    max_mb: Optional[float] = None  # For memory
    max_grams: Optional[float] = None  # For weight
    max_usd: Optional[float] = None  # For cost
    min_recall: Optional[float] = None  # For accuracy
    min_precision: Optional[float] = None
    tier: Optional[str] = None
    criticality: ConstraintCriticality

class SuccessCriterion(BaseModel):
    metric: str
    target: float
    operator: str  # lt, lte, gt, gte, eq

class UseCaseEntry(BaseModel):
    """Complete use case template."""
    id: str
    name: str
    category: str  # drone, quadruped, biped, amr, edge
    description: str

    platform: dict
    perception_requirements: dict

    constraints: dict[str, Constraint]
    preferences: List[str]
    implied: List[str]

    recommended_hardware: List[dict]
    recommended_models: List[dict]
    recommended_sensors: List[dict]

    success_criteria: List[SuccessCriterion]

    last_updated: str
    maintainer: Optional[str] = None
```

---

## 6. Benchmark Results Schema

```yaml
# benchmarks/results/yolov8s_jetson_nano_fp16.yaml
id: yolov8s_jetson_nano_fp16_20241201
model_id: yolov8s
hardware_id: nvidia_jetson_nano_4gb
variant: fp16

# Test conditions
conditions:
  power_mode: 10W
  batch_size: 1
  input_shape: [1, 3, 640, 640]
  warmup_iterations: 50
  test_iterations: 1000
  ambient_temp_c: 25
  cooling: passive

# Latency results
latency:
  mean_ms: 45.2
  std_ms: 3.1
  min_ms: 42.0
  max_ms: 58.0
  p50_ms: 44.5
  p95_ms: 51.2
  p99_ms: 55.8

# Throughput
throughput:
  fps: 22.1
  samples_per_second: 22.1

# Memory
memory:
  model_mb: 21.3
  peak_mb: 890
  gpu_utilization_percent: 95

# Power
power:
  mean_watts: 8.5
  peak_watts: 10.2
  energy_per_inference_mj: 385

# Thermal
thermal:
  start_temp_c: 45
  end_temp_c: 68
  throttled: false

# Accuracy verification (optional)
accuracy:
  dataset: coco_val2017_subset_100
  map_50: 0.608
  expected_map_50: 0.612
  delta: -0.004

# Metadata
timestamp: 2024-12-01T14:30:00Z
benchmark_version: "1.0.0"
notes: "Passive cooling, 25C ambient"
reproducibility:
  script: "scripts/benchmark_model.py"
  commit: "abc123"
```

---

## 7. Constraint Ontology

```yaml
# constraints/latency_tiers.yaml
id: latency_tiers
name: Latency Tier Definitions

tiers:
  ultra_real_time:
    max_ms: 10
    fps_equivalent: 100
    use_cases:
      - high_speed_tracking
      - industrial_inspection
    description: "Sub-10ms, for high-speed control loops"

  real_time:
    max_ms: 33.3
    fps_equivalent: 30
    use_cases:
      - drone_obstacle_avoidance
      - quadruped_locomotion
      - amr_navigation
    description: "30+ FPS, standard real-time perception"

  interactive:
    max_ms: 100
    fps_equivalent: 10
    use_cases:
      - human_robot_interaction
      - gesture_recognition
    description: "Responsive but not real-time"

  batch:
    max_ms: 1000
    fps_equivalent: 1
    use_cases:
      - surveillance_archival
      - quality_inspection
    description: "Offline or near-real-time processing"

# Implication rules
implications:
  safety_critical:
    requires: [ultra_real_time, real_time]
    not_allowed: [batch]
  battery_powered:
    prefers: [real_time, interactive]
    avoids: [ultra_real_time]  # Power hungry
```

```yaml
# constraints/power_classes.yaml
id: power_classes
name: Power Budget Classifications

classes:
  ultra_low_power:
    max_watts: 2
    typical_source: coin_cell
    examples: [google_coral, ncs2]
    suitable_for: [iot, wearable, always_on]

  low_power:
    max_watts: 10
    typical_source: small_battery
    examples: [jetson_nano_5w, raspberry_pi]
    suitable_for: [drone, small_robot]

  medium_power:
    max_watts: 30
    typical_source: large_battery
    examples: [jetson_orin_nano, hailo_8]
    suitable_for: [quadruped, amr, large_drone]

  high_power:
    max_watts: 100
    typical_source: ac_power
    examples: [jetson_agx_orin, rtx_4090]
    suitable_for: [biped, vehicle, edge_server]

  datacenter:
    max_watts: 500
    typical_source: rack_power
    examples: [a100, h100, mi250x]
    suitable_for: [cloud, training]

# Platform power budgets
platform_defaults:
  micro_drone: 5
  small_drone: 15
  medium_drone: 50
  quadruped_small: 30
  quadruped_large: 100
  biped: 200
  amr_small: 50
  amr_large: 200
  edge_camera: 10
```

---

## 8. Relationship Mappings

```yaml
# relationships/model_hardware_compatibility.yaml
id: model_hardware_compatibility
description: "Which models work on which hardware with what performance"

mappings:
  - model_id: yolov8s
    hardware_id: nvidia_jetson_nano_4gb
    compatibility: full  # full, partial, unsupported
    variants_supported: [fp32, fp16]
    variants_unsupported: [int8]  # TensorRT INT8 needs more memory
    notes: "FP16 recommended, INT8 OOM on 4GB"

  - model_id: yolov8s
    hardware_id: nvidia_jetson_orin
    compatibility: full
    variants_supported: [fp32, fp16, int8]
    notes: "All variants work well"

  - model_id: yolov8s
    hardware_id: google_coral
    compatibility: partial
    variants_supported: [int8]
    requires_conversion: true
    conversion_tool: edgetpu_compiler
    accuracy_impact: high
    notes: "Requires Edge TPU compilation, significant accuracy loss"

  - model_id: yolov8s
    hardware_id: hailo_8
    compatibility: full
    variants_supported: [int8]
    requires_conversion: true
    conversion_tool: hailo_dataflow_compiler
    notes: "Use Hailo Model Zoo for optimized version"
```

```yaml
# relationships/usecase_constraint_mappings.yaml
id: usecase_constraint_mappings
description: "What constraints are implied by each use case category"

category_defaults:
  drone:
    implied_constraints:
      - battery_powered
      - weight_sensitive
      - outdoor_capable
    typical_latency_tier: real_time
    typical_power_class: low_power

  quadruped:
    implied_constraints:
      - battery_powered
      - vibration_resistant
      - outdoor_capable
    typical_latency_tier: real_time
    typical_power_class: medium_power

  biped:
    implied_constraints:
      - high_compute
      - indoor_outdoor
    typical_latency_tier: real_time
    typical_power_class: high_power

  amr:
    implied_constraints:
      - indoor_primary
      - safety_critical
    typical_latency_tier: real_time
    typical_power_class: medium_power

  edge:
    implied_constraints:
      - always_on
      - remote_deployment
    typical_latency_tier: interactive
    typical_power_class: low_power
```

---

## 9. Query Interface Design

The knowledge base should support these query patterns:

### Exact Lookups
```python
kb.get_hardware("nvidia_jetson_nano_4gb") → HardwareEntry
kb.get_model("yolov8s") → ModelEntry
kb.get_usecase("drone_obstacle_avoidance") → UseCaseEntry
```

### Filtered Queries
```python
kb.find_hardware(
    power_watts_max=15,
    memory_gb_min=4,
    form_factor="som"
) → List[HardwareEntry]

kb.find_models(
    type="object_detection",
    latency_ms_max=50,
    hardware_id="nvidia_jetson_nano_4gb"
) → List[ModelEntry]
```

### Relationship Queries
```python
kb.get_compatible_hardware(model_id="yolov8s") → List[HardwareEntry]
kb.get_compatible_models(hardware_id="nvidia_jetson_nano_4gb") → List[ModelEntry]
kb.get_constraints_for_usecase("drone_obstacle_avoidance") → ConstraintSet
```

### Benchmark Lookups
```python
kb.get_benchmark(
    model_id="yolov8s",
    hardware_id="nvidia_jetson_nano_4gb",
    variant="fp16"
) → BenchmarkResult | None
```

---

## 10. Implementation Plan

### Phase 1: Core Schema (Week 1-2)
1. Define Pydantic models in `src/embodied_ai_architect/knowledge_base/models/`
2. Create YAML loaders with validation
3. Implement `KnowledgeBase` class with exact lookups
4. Seed initial data: 5 hardware, 5 models, 3 use cases

### Phase 2: Relationships (Week 3)
1. Add relationship mapping support
2. Implement filtered queries
3. Add compatibility lookups
4. Create CLI for knowledge base inspection

### Phase 3: Benchmarks (Week 4)
1. Add benchmark result schema
2. Implement benchmark storage
3. Create feedback loop from `run_benchmark` tool
4. Validate benchmark data format

### Phase 4: Integration (Week 5)
1. Integrate with analysis tools
2. Update tool outputs to use verdicts
3. Add constraint satisfaction checking
4. Create knowledge base update CLI

---

## 11. Migration from Existing Code

The existing `HardwareProfile` and `HardwareCapability` classes in `agents/hardware_profile/models.py` should be:

1. **Extended, not replaced** - Add new fields as optional
2. **YAML-backed** - Move from `knowledge_base.py` function to YAML files
3. **ID-based** - Add unique `id` field for relationships

```python
# Compatibility layer
def get_default_hardware_profiles() -> List[HardwareProfile]:
    """Load hardware profiles from YAML knowledge base."""
    kb = KnowledgeBase.load("knowledge_base/")
    return [entry.to_legacy_profile() for entry in kb.hardware.values()]
```

---

*Document created: December 2024*
*Status: Schema Design Complete*
