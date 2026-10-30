# Embodied AI Codesign Agent: Subtask Enumeration

## Target Platforms

| Platform | Key Constraints | Primary Perception Needs |
|----------|-----------------|-------------------------|
| **Drones (UAVs)** | Weight, power, real-time | Obstacle avoidance, landing zone detection, tracking |
| **Quadrupeds** | Power, terrain adaptation | Ground plane estimation, obstacle detection, SLAM |
| **Bipeds** | Balance, human interaction | Full-body tracking, gesture recognition, scene understanding |
| **AMRs** | Navigation, safety | SLAM, obstacle detection, path planning |
| **Edge AI (static)** | Cost, power, deployment scale | Detection, classification, anomaly detection |

---

## 1. Knowledge Base Subtasks

### 1.1 Hardware Catalog

| Subtask | Description | Data Structure |
|---------|-------------|----------------|
| `KB-HW-001` | Catalog edge compute platforms (Jetson, Hailo, Coral, Qualcomm, etc.) | Hardware specs, benchmarks |
| `KB-HW-002` | Catalog accelerator modules (NPUs, GPUs, DSPs) | TOPS, power, interfaces |
| `KB-HW-003` | Define hardware capability ontology | Compute, memory, power, thermal, interfaces |
| `KB-HW-004` | Store validated benchmark results per hardware | Model × hardware → latency, power, accuracy |
| `KB-HW-005` | Track hardware availability, pricing, lifecycle status | Supply chain metadata |

### 1.2 Model Catalog

| Subtask | Description | Data Structure |
|---------|-------------|----------------|
| `KB-MOD-001` | Catalog perception models (YOLO, RT-DETR, SegFormer, etc.) | Architecture, params, FLOPs |
| `KB-MOD-002` | Catalog model variants and quantization levels | FP32, FP16, INT8, INT4 |
| `KB-MOD-003` | Store accuracy metrics per task/dataset | mAP, mIoU, accuracy |
| `KB-MOD-004` | Define model capability tags | detection, segmentation, tracking, depth, pose |
| `KB-MOD-005` | Track model optimization formats | ONNX, TensorRT, OpenVINO, TFLite, Hailo HEF |

### 1.3 Sensor Catalog

| Subtask | Description | Data Structure |
|---------|-------------|----------------|
| `KB-SEN-001` | Catalog camera sensors (resolution, FPS, interface) | Specs, power, cost |
| `KB-SEN-002` | Catalog depth sensors (stereo, ToF, structured light) | Range, accuracy, power |
| `KB-SEN-003` | Catalog LiDAR sensors | Points/sec, range, FoV, power |
| `KB-SEN-004` | Catalog IMUs and other proprioceptive sensors | Accuracy, drift, fusion requirements |
| `KB-SEN-005` | Define sensor-to-model compatibility matrix | What sensors feed which model types |

### 1.4 Use Case Templates

| Subtask | Description | Data Structure |
|---------|-------------|----------------|
| `KB-UC-001` | Define drone perception use cases | Obstacle avoidance, landing, tracking, inspection |
| `KB-UC-002` | Define quadruped perception use cases | Terrain, obstacles, navigation, manipulation |
| `KB-UC-003` | Define biped perception use cases | Human interaction, scene understanding, manipulation |
| `KB-UC-004` | Define AMR perception use cases | SLAM, navigation, safety, picking |
| `KB-UC-005` | Define static edge AI use cases | Surveillance, quality inspection, access control |
| `KB-UC-006` | Map use cases to constraint profiles | Which dimensions matter, typical thresholds |

### 1.5 Constraint Ontology

| Subtask | Description | Data Structure |
|---------|-------------|----------------|
| `KB-CON-001` | Define latency constraint taxonomy | Real-time tiers (1ms, 10ms, 33ms, 100ms) |
| `KB-CON-002` | Define power constraint taxonomy | Battery classes, thermal limits |
| `KB-CON-003` | Define accuracy constraint taxonomy | Safety-critical vs best-effort |
| `KB-CON-004` | Define cost constraint taxonomy | Consumer, industrial, military |
| `KB-CON-005` | Define environmental constraint taxonomy | IP rating, temp range, vibration |
| `KB-CON-006` | Create constraint implication rules | "outdoor drone" → IP65, wide temp range |

---

## 2. Analysis Tool Subtasks

### 2.1 Accuracy Analysis

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-ACC-001` | Analyze model accuracy on standard benchmarks | mAP, mIoU, precision, recall |
| `TOOL-ACC-002` | Estimate accuracy degradation from quantization | FP32 → INT8 accuracy delta |
| `TOOL-ACC-003` | Assess accuracy for specific use case conditions | Low light, motion blur, occlusion |
| `TOOL-ACC-004` | Compare model accuracy vs requirement threshold | PASS/FAIL with margin |

### 2.2 Latency Analysis

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-LAT-001` | Estimate inference latency on target hardware | ms per frame, with confidence |
| `TOOL-LAT-002` | Analyze end-to-end pipeline latency | Sensor → inference → output |
| `TOOL-LAT-003` | Model latency under thermal throttling | Sustained vs burst performance |
| `TOOL-LAT-004` | Compare latency vs requirement threshold | PASS/FAIL with margin |

### 2.3 Power/Energy Analysis

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-PWR-001` | Estimate inference power consumption | Watts during inference |
| `TOOL-PWR-002` | Calculate energy per inference | mJ per frame |
| `TOOL-PWR-003` | Project battery life for duty cycle | Hours of operation |
| `TOOL-PWR-004` | Analyze thermal envelope | TDP, cooling requirements |
| `TOOL-PWR-005` | Compare power vs budget threshold | PASS/FAIL with margin |

### 2.4 Memory Analysis

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-MEM-001` | Calculate model memory footprint | Weights, activations, workspace |
| `TOOL-MEM-002` | Analyze memory bandwidth requirements | GB/s needed |
| `TOOL-MEM-003` | Check fit on target hardware | Does it fit in RAM/VRAM |
| `TOOL-MEM-004` | Analyze multi-model memory sharing | Concurrent model memory budget |
| `TOOL-MEM-005` | Compare memory vs hardware capacity | PASS/FAIL with margin |

### 2.5 Cost Analysis

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-COST-001` | Estimate BOM cost for hardware config | Compute + sensors + peripherals |
| `TOOL-COST-002` | Analyze cost vs performance trade-offs | Pareto frontier |
| `TOOL-COST-003` | Project volume pricing | 1, 100, 10K, 100K unit pricing |
| `TOOL-COST-004` | Compare cost vs budget threshold | PASS/FAIL with margin |

### 2.6 Physical Constraint Analysis

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-PHY-001` | Analyze weight budget | Compute + sensors + structure |
| `TOOL-PHY-002` | Check form factor compatibility | Dimensions, mounting |
| `TOOL-PHY-003` | Verify environmental ratings | IP rating, temp, vibration |
| `TOOL-PHY-004` | Compare physical specs vs requirements | PASS/FAIL per dimension |

---

## 3. Recommendation Tool Subtasks

### 3.1 Hardware Recommendation

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-REC-HW-001` | Recommend compute platform for use case | Ranked list with rationale |
| `TOOL-REC-HW-002` | Recommend accelerator configuration | NPU/GPU/DSP allocation |
| `TOOL-REC-HW-003` | Identify hardware alternatives at different price points | Good/better/best options |
| `TOOL-REC-HW-004` | Flag hardware risks | EOL, supply chain, thermal |

### 3.2 Model Recommendation

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-REC-MOD-001` | Recommend perception model for task | Model + variant + quantization |
| `TOOL-REC-MOD-002` | Suggest model optimization path | Pruning, distillation, quantization |
| `TOOL-REC-MOD-003` | Identify model alternatives | Accuracy vs latency trade-offs |
| `TOOL-REC-MOD-004` | Recommend multi-model pipeline | Detection → tracking → reasoning |

### 3.3 Sensor Recommendation

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-REC-SEN-001` | Recommend sensor suite for use case | Camera, depth, LiDAR, IMU |
| `TOOL-REC-SEN-002` | Optimize sensor selection for constraints | Power, cost, weight |
| `TOOL-REC-SEN-003` | Validate sensor-compute compatibility | Interfaces, bandwidth |

### 3.4 Software Stack Recommendation

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-REC-SW-001` | Recommend OS and runtime | Linux, RTOS, specific distro |
| `TOOL-REC-SW-002` | Recommend middleware framework | ROS2, Isaac, custom |
| `TOOL-REC-SW-003` | Recommend inference runtime | TensorRT, ONNX Runtime, etc. |
| `TOOL-REC-SW-004` | Identify SDK/toolchain requirements | Vendor SDKs, compilers |

---

## 4. Synthesis Tool Subtasks

### 4.1 Trade-off Analysis

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-SYN-001` | Generate Pareto frontier for accuracy vs latency | Visualization + data |
| `TOOL-SYN-002` | Generate Pareto frontier for power vs performance | Visualization + data |
| `TOOL-SYN-003` | Identify constraint conflicts | "Cannot meet both X and Y" |
| `TOOL-SYN-004` | Suggest constraint relaxation options | "Relax latency by 20% to meet power" |

### 4.2 Configuration Generation

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-SYN-005` | Generate complete system configuration | HW + sensors + models + SW |
| `TOOL-SYN-006` | Validate configuration consistency | All components compatible |
| `TOOL-SYN-007` | Estimate system-level metrics | E2E latency, power, cost, weight |

### 4.3 Report Generation

| Subtask | Description | Output |
|---------|-------------|--------|
| `TOOL-SYN-008` | Generate executive summary | Key findings, recommendations |
| `TOOL-SYN-009` | Generate detailed analysis report | All metrics, evidence, trade-offs |
| `TOOL-SYN-010` | Generate comparison report | Multiple configurations side-by-side |
| `TOOL-SYN-011` | Export machine-readable results | JSON/YAML for downstream tools |

---

## 5. Decomposition & Planning Subtasks

### 5.1 Query Understanding

| Subtask | Description | Output |
|---------|-------------|--------|
| `PLAN-001` | Extract platform type from query | drone, quadruped, biped, AMR, edge |
| `PLAN-002` | Extract use case from query | obstacle avoidance, SLAM, tracking, etc. |
| `PLAN-003` | Extract explicit constraints from query | Latency < X, power < Y, cost < Z |
| `PLAN-004` | Infer implicit constraints from platform/use case | Real-time → latency, battery → power |

### 5.2 Dimension Prioritization

| Subtask | Description | Output |
|---------|-------------|--------|
| `PLAN-005` | Identify which analysis dimensions are relevant | latency, power, accuracy, memory, cost |
| `PLAN-006` | Rank dimensions by criticality for use case | Safety-critical first |
| `PLAN-007` | Identify dimension dependencies | "If memory fails, latency irrelevant" |

### 5.3 Tool Orchestration

| Subtask | Description | Output |
|---------|-------------|--------|
| `PLAN-008` | Map dimensions to analysis tools | latency → analyze_latency |
| `PLAN-009` | Determine parallel vs sequential execution | Independent dimensions in parallel |
| `PLAN-010` | Handle tool failures gracefully | Retry, fallback, partial results |
| `PLAN-011` | Synthesize multi-tool results into answer | Aggregate verdicts, identify blockers |

### 5.4 Iterative Refinement

| Subtask | Description | Output |
|---------|-------------|--------|
| `PLAN-012` | Detect when initial query is underspecified | Ask clarifying questions |
| `PLAN-013` | Propose alternatives when constraints conflict | Trade-off options |
| `PLAN-014` | Support "what-if" exploration | Re-run with modified constraints |

---

## 6. Validation & Benchmarking Subtasks

### 6.1 Benchmark Execution

| Subtask | Description | Output |
|---------|-------------|--------|
| `VAL-001` | Execute inference benchmark on local hardware | Measured latency, power |
| `VAL-002` | Execute inference benchmark on remote hardware (SSH) | Measured latency, power |
| `VAL-003` | Execute inference benchmark on cloud/K8s | Scaled benchmark results |
| `VAL-004` | Store benchmark results in knowledge base | Feedback loop |

### 6.2 Accuracy Validation

| Subtask | Description | Output |
|---------|-------------|--------|
| `VAL-005` | Run model on validation dataset | Measured accuracy |
| `VAL-006` | Compare measured vs claimed accuracy | Validation status |
| `VAL-007` | Test accuracy under edge conditions | Stress test results |

### 6.3 Integration Validation

| Subtask | Description | Output |
|---------|-------------|--------|
| `VAL-008` | Validate sensor-compute integration | Data flow verified |
| `VAL-009` | Validate end-to-end pipeline latency | Measured E2E timing |
| `VAL-010` | Validate thermal behavior under load | Sustained performance |

---

## 7. Implementation Priority

### Phase 1: Foundation (Core Analysis)
1. `KB-HW-001` - Hardware catalog (start with 5-10 platforms)
2. `KB-MOD-001` - Model catalog (start with YOLO, RT-DETR, MobileNet)
3. `KB-UC-001` through `KB-UC-005` - Use case templates
4. `TOOL-LAT-001` - Latency analysis tool
5. `TOOL-MEM-001` - Memory analysis tool
6. `TOOL-PWR-001` - Power analysis tool
7. `PLAN-001` through `PLAN-004` - Query understanding

### Phase 2: Recommendation
8. `TOOL-REC-HW-001` - Hardware recommendation
9. `TOOL-REC-MOD-001` - Model recommendation
10. `TOOL-SYN-005` - Configuration generation
11. `PLAN-005` through `PLAN-011` - Full orchestration

### Phase 3: Validation
12. `VAL-001` through `VAL-003` - Benchmark execution
13. `VAL-005`, `VAL-006` - Accuracy validation
14. Feedback loop to knowledge base

### Phase 4: Advanced
15. `KB-SEN-*` - Full sensor catalog
16. `TOOL-SYN-001` through `TOOL-SYN-004` - Trade-off analysis
17. `PLAN-012` through `PLAN-014` - Iterative refinement

---

## 8. Tool-to-Subtask Mapping

| Tool Name | Primary Subtasks | Dependencies |
|-----------|------------------|--------------|
| `analyze_latency` | TOOL-LAT-001 to TOOL-LAT-004 | KB-HW, KB-MOD |
| `analyze_memory` | TOOL-MEM-001 to TOOL-MEM-005 | KB-HW, KB-MOD |
| `analyze_power` | TOOL-PWR-001 to TOOL-PWR-005 | KB-HW, KB-MOD |
| `analyze_accuracy` | TOOL-ACC-001 to TOOL-ACC-004 | KB-MOD |
| `analyze_cost` | TOOL-COST-001 to TOOL-COST-004 | KB-HW, KB-SEN |
| `recommend_hardware` | TOOL-REC-HW-001 to TOOL-REC-HW-004 | All KB |
| `recommend_model` | TOOL-REC-MOD-001 to TOOL-REC-MOD-004 | KB-MOD, KB-UC |
| `generate_config` | TOOL-SYN-005 to TOOL-SYN-007 | All analysis tools |
| `run_benchmark` | VAL-001 to VAL-003 | Target hardware |

---

## 9. Success Criteria

The Agentic AI is "perfectly organized" when:

1. **Complete Coverage**: Every query type maps to available tools
2. **Accurate Verdicts**: Tools return correct PASS/FAIL with evidence
3. **Fast Execution**: Parallel analysis where possible, < 10s for typical queries
4. **Graceful Degradation**: Missing data → UNKNOWN, not hallucination
5. **Actionable Output**: Every FAIL includes a suggestion
6. **Knowledge Growth**: Benchmark results feed back into knowledge base

---

*Document created: December 2024*
*Status: Subtask Enumeration Complete*
