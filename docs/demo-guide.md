# Demo Guide

This document describes the current demo capabilities and how to run them.

## Prerequisites

```bash
# Install in development mode (required for all demos)
pip install -e ".[dev]"

# For Demo 3 (optimization loop), also install LangGraph
pip install -e ".[langgraph]"
```

## Demo Overview

| Demo | Script | What It Shows | Dependencies |
|------|--------|---------------|-------------|
| Demo 1 | `demo_soc_designer.py` | End-to-end SoC design pipeline with optimization loop | Core only |
| Demo 2 | `demo_dse_pareto.py` | Design-space exploration with Pareto front | Core only |
| Demo 3 | `demo_soc_optimizer.py` | Iterative power optimization loop | LangGraph |
| Demo 4 | `demo_kpu_rtl.py` | KPU micro-architecture + RTL generation | Core only |
| Demo 5 | `demo_hitl_safety.py` | Safety-critical design with HITL governance | Core only |
| Demo 6 | `demo_experience_cache.py` | Experience cache reuse across designs | Core only |
| Demo 7 | `demo_full_campaign.py` | Full multi-workload autonomous campaign | Core only |

There are also three utility examples (not demos):

| Example | Script | What It Shows | Dependencies |
|---------|--------|---------------|-------------|
| Simple Workflow | `simple_workflow.py` | Legacy Orchestrator pipeline with a PyTorch CNN | PyTorch |
| Remote Benchmark | `remote_benchmark_example.py` | SSH-based remote benchmarking | `paramiko` |
| K8s Scaling | `kubernetes_scaling_example.py` | Kubernetes horizontal scaling | `kubernetes` |

---

## Demo 1: Agentic SoC Designer

**Goal decomposition + full specialist pipeline + optimization loop.**

Runs a delivery drone SoC design through: Planner -> Dispatcher -> 6 Specialists -> Design Report. If PPA assessment produces any FAIL verdict, an optimization loop iteratively applies strategies (INT8 quantization, resolution reduction, etc.) until all constraints pass, then re-runs the critic for a final review.

### What happens

1. **Planning** — Decomposes the goal into a 6-task DAG
2. **Workload Analysis** — Estimates compute (GFLOPS) and memory from goal keywords
3. **Hardware Exploration** — Scores 6 hardware candidates against power/cost/latency constraints
4. **Architecture Composition** — Maps operators to hardware, generates IP blocks and interconnect
5. **PPA Assessment** — Estimates power/performance/area/cost with per-constraint PASS/FAIL verdicts
6. **Design Review** — Critic identifies risks, bottlenecks, and improvement opportunities
7. **Report Generation** — Structured design report with decision trail
8. **Optimization Loop** (if any FAIL) — Iteratively applies strategies until all verdicts PASS
9. **Final Review** — Critic re-evaluates the optimized design

### Usage

```bash
# Default: static plan, no API key needed
python examples/demo_soc_designer.py

# Use Claude LLM for goal decomposition (requires ANTHROPIC_API_KEY)
python examples/demo_soc_designer.py --llm

# Custom design goal
python examples/demo_soc_designer.py --goal "Design an SoC for autonomous mobile robot with SLAM"

# Override constraints
python examples/demo_soc_designer.py --power 3.0 --latency 20.0 --cost 50.0

# Control max optimization iterations
python examples/demo_soc_designer.py --max-iterations 3
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--llm` | off | Use Claude for goal decomposition |
| `--goal TEXT` | drone prompt | Custom design goal string |
| `--power FLOAT` | 5.0 | Max power budget (watts) |
| `--latency FLOAT` | 33.3 | Max latency (ms) |
| `--cost FLOAT` | 30.0 | Max BOM cost (USD) |
| `--max-iterations INT` | 5 | Max optimization iterations |

### Output sections

- **Constraints** — Input constraint summary
- **Task Graph** — DAG with agent assignments and dependencies
- **Workload Analysis** — Detected workloads with GFLOPS estimates
- **Hardware Candidates** — Ranked table with constraint verdicts
- **Selected Architecture** — Chosen compute paradigm and process node
- **IP Blocks** — Generated IP blocks with configurations
- **Interconnect** — Bus/NoC topology details
- **PPA Assessment** — Power/latency/area/cost with PASS/FAIL per constraint
- **Design Review** — Strengths, issues, recommendations from critic
- **Decision Trail** — Ordered audit log of all design decisions
- **Optimization Phase** (if FAIL) — Per-iteration strategy application with before/after PPA
- **Final PPA** (if optimized) — Post-optimization metrics with all-PASS verdicts
- **Final Design Review** (if optimized) — Critic re-assessment (ADEQUATE/STRONG)
- **Decision Trail (updated)** (if optimized) — Full trail including optimization steps

---

## Demo 2: Design-Space Exploration with Pareto Front

**Multi-hardware comparison with Pareto-optimal trade-off analysis.**

Explores a warehouse AMR design space across 6 hardware candidates, computing a Pareto front over power/latency/cost and identifying the knee point (best balanced design).

### What happens

1. **Workload Analysis** — MobileNetV2 + SLAM for warehouse navigation
2. **Hardware Exploration** — Scores 6 candidates against 15W/50ms/$100 constraints
3. **Design-Space Exploration** — Computes non-dominated Pareto front, identifies knee point
4. **Architecture Composition** — Maps workloads to the knee-point hardware
5. **PPA Assessment** — Constraint verdicts for the selected design
6. **Design Review** — Critic evaluation with trade-off analysis
7. **Report Generation** — Design report including Pareto front summary

### Usage

```bash
python examples/demo_dse_pareto.py
```

This demo uses a static plan and runs deterministically with no API key required.

### Output sections

- **Constraints** — Power (15W), latency (50ms), cost ($100)
- **Task Graph** — 7-task DAG with design_explorer between hw_explorer and architecture_composer
- **Pareto Front** — Non-dominated designs with power/latency/cost trade-offs
- **Knee Point** — Best balanced design (minimum normalized distance to origin)
- **PPA Assessment** — Per-constraint PASS/FAIL verdicts
- **Design Review** — Trade-off analysis and recommendations

---

## Demo 3: SoC Design Optimizer

**Iterative power convergence loop using LangGraph.**

The initial KPU design exceeds the 5W power budget (~6.3W). The optimizer iteratively applies strategies (INT8 quantization, resolution reduction) until the design passes all constraints.

### What happens

1. **Iteration 0** — Full pipeline runs; KPU selected but power ~6.3W -> FAIL
2. **Iteration 1** — Optimizer applies strategy (e.g., INT8 quantization) -> power reduced
3. **Iteration 2** — Further optimization -> power within budget -> PASS
4. **Convergence** — All constraint verdicts are PASS; loop terminates

### Usage

```bash
# Default: 5W budget, 10 max iterations
python examples/demo_soc_optimizer.py

# Tighter power budget (takes more iterations)
python examples/demo_soc_optimizer.py --power 4.0

# Control iteration limit
python examples/demo_soc_optimizer.py --max-iterations 5

# All options
python examples/demo_soc_optimizer.py --power 4.0 --latency 20.0 --cost 25.0 --max-iterations 15
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--power FLOAT` | 5.0 | Max power budget (watts) |
| `--latency FLOAT` | 33.3 | Max latency (ms) |
| `--cost FLOAT` | 30.0 | Max cost (USD) |
| `--max-iterations INT` | 10 | Max optimization iterations |

### Output sections

- **Optimization History** — Per-iteration PPA snapshot and verdict progression
- **Final PPA** — Converged power/latency/cost with verdicts
- **Design Decisions** — Rationale entries from the optimization process
- **Outcome** — CONVERGED (all PASS) or STOPPED (limit reached)

### Requires

LangGraph must be installed: `pip install -e ".[langgraph]"`

---

## Demo 4: KPU Micro-architecture + RTL Generation

**Full four-level flow: Outer Loop -> KPU Config -> Floorplan/Bandwidth -> RTL.**

Configures a KPU micro-architecture for a delivery drone perception workload, validates the physical layout and memory bandwidth, then generates RTL for all sub-components using template-based synthesis.

### What happens

1. **Workload Analysis** — Detection + tracking at 30fps
2. **Hardware Exploration** — Enumerate candidates under constraints
3. **Architecture Composition** — Select KPU compute engine
4. **KPU Configuration** — Size the checkerboard array (compute tiles, memory tiles, systolic arrays, SRAM hierarchy, DRAM controllers, NoC)
5. **Floorplan Validation** — Checkerboard pitch matching between compute and memory tiles (within +/-15%)
6. **Bandwidth Validation** — Data flow balance through DRAM -> L3 -> L2 -> L1 -> compute hierarchy
7. **RTL Generation** — Jinja2 templates for ~12 KPU sub-components, processed through EDA toolchain (Verilator lint -> Yosys synthesis -> Icarus simulation)
8. **PPA Assessment** — Aggregate synthesis cell counts -> area estimation via technology scaling
9. **Design Review** — Critic review of the complete design
10. **Report** — Final design report

### Usage

```bash
python examples/demo_kpu_rtl.py
```

This demo uses a static plan and has no CLI options. It runs deterministically with no API key required.

### Output sections

- **Constraints** — Power (5W), latency (33.3ms), cost ($30), area (100mm2)
- **Task Graph** — 10-task DAG with parallel floorplan + bandwidth validation
- **KPU Configuration** — Process node, checkerboard dimensions, systolic array size, SRAM hierarchy, DRAM technology, NoC topology
- **Floorplan Check** — Compute/memory tile dimensions, pitch ratios, total die area, feasibility
- **Bandwidth Check** — Per-link available/required bandwidth with utilization and bottleneck flags
- **RTL Generation** — Per-module synthesis cell counts (PASS/FAIL), total cell count
- **PPA Summary** — Aggregated power, area, latency, cost with verdicts

---

## Demo 5: Safety-Critical Design with HITL Governance

**Safety detection, redundancy injection, and human-in-the-loop approval gates.**

Designs a surgical robot SoC under IEC 62304 Class C safety requirements. The safety detector identifies safety-critical constraints and injects redundancy requirements (dual-lockstep CPU, ECC memory, watchdog timer). Governance gates flag safety decisions for human approval.

### What happens

1. **Safety Detection** — Detects `safety_critical=True` and IEC 62304 standard, injects redundancy requirements
2. **Workload Analysis** — Force-feedback control + haptic sensing at 1ms latency
3. **Hardware Exploration** — Candidates scored with safety compliance weighting
4. **Architecture Composition** — Incorporates dual-lockstep CPU, ECC, watchdog IP blocks
5. **PPA Assessment** — Constraint verdicts including safety overhead
6. **Design Review** — Critic flags safety gaps and compliance status
7. **Report Generation** — Design report with safety analysis and audit log

### Usage

```bash
python examples/demo_hitl_safety.py
```

This demo auto-approves safety decisions in non-interactive mode. No API key required.

### Output sections

- **Constraints** — Power (25W), latency (1ms), cost ($500), safety (IEC 62304 Class C)
- **Safety Analysis** — Detected standard, redundancy requirements, safety flags
- **Governance Audit** — Safety-critical decisions with approval status
- **PPA Assessment** — Per-constraint verdicts including safety overhead
- **Design Review** — Safety compliance assessment

---

## Demo 6: Experience Cache Reuse

**Cross-design knowledge transfer using episodic memory.**

Runs two design sessions back-to-back: first a delivery drone (Demo 1 pipeline), then an agricultural drone that retrieves and adapts the prior experience. Demonstrates warm-starting hardware selection from cached episodes.

### What happens

1. **First Run (Delivery Drone)** — Standard pipeline; episode saved to experience cache
2. **Second Run (Agricultural Drone)**:
   - **Experience Retrieval** — Searches cache, computes similarity to prior episodes
   - **Workload Analysis** — Crop detection at 15fps
   - **Hardware Exploration** — Candidates warm-started from prior episode (+15 score boost)
   - **Architecture → PPA → Critic → Report** — Standard pipeline

### Usage

```bash
python examples/demo_experience_cache.py
```

Uses an in-memory SQLite cache. No API key required.

### Output sections

- **Prior Experience** — Similarity score, matched episode, adapted constraints
- **Hardware Candidates** — Warm-started rankings showing prior experience influence
- **PPA Assessment** — Constraint verdicts for the agricultural drone
- **Comparison** — Side-by-side summary of delivery vs agricultural drone designs

---

## Demo 7: Full Autonomous Campaign

**Multi-workload quadruped robot with DSE, governance, and full evaluation.**

The most comprehensive demo: designs an SoC for a quadruped robot with 4 concurrent workloads (Visual SLAM, object detection, LiDAR processing, voice recognition). Exercises design-space exploration, multi-workload scheduling, governance with iteration limits, and the evaluation framework.

### What happens

1. **Workload Analysis** — Detects 4 workloads with concurrent scheduling, computes aggregate GFLOPS
2. **Hardware Exploration** — Scores candidates for multi-workload support
3. **Design-Space Exploration** — Pareto front with heterogeneous accelerator mapping
4. **Architecture Composition** — Multi-accelerator architecture with shared NoC
5. **PPA Assessment** — Power (15W), latency (50ms), cost ($50) constraint verdicts
6. **Design Review** — Critic evaluates multi-workload balance and resource utilization
7. **Report Generation** — Comprehensive design report with workload mapping

### Usage

```bash
python examples/demo_full_campaign.py
```

No API key required. Governance limits iterations to 10.

### Output sections

- **Multi-Workload Profile** — 4 workloads with individual GFLOPS, scheduling, and aggregate requirements
- **Pareto Front** — Trade-off analysis across power/latency/cost
- **Workload-to-Accelerator Mapping** — Which workloads run on which compute units
- **Governance Log** — Iteration tracking and decision audit
- **PPA Assessment** — Per-constraint verdicts under multi-workload load
- **Evaluation Scorecard** — 9-dimension composite score (target > 0.75)

---

## Utility Examples

### Simple Workflow

Legacy orchestrator pipeline that analyzes a PyTorch CNN model, profiles hardware, runs benchmarks, and generates a report.

```bash
python examples/simple_workflow.py
```

Requires PyTorch.

### Remote Benchmark

Demonstrates secure SSH-based benchmarking on remote machines.

```bash
# Setup: copy .env.example to .env and fill in SSH credentials
pip install 'embodied-ai-architect[remote]'
python examples/remote_benchmark_example.py
```

### Kubernetes Scaling

Demonstrates horizontal scaling of benchmarks across Kubernetes pods.

```bash
# Setup: deploy K8s resources and configure kubeconfig
pip install 'embodied-ai-architect[kubernetes]'
python examples/kubernetes_scaling_example.py
```

---

## Architecture Progression

The demos show the system's evolution across four phases:

```
Demo 1 (Phase 1+)    Pipeline + optimization loop: plan -> dispatch -> 6 specialists -> optimize -> PASS
Demo 2 (Phase 4)     + Pareto front design-space exploration
Demo 3 (Phase 2)     + Optimization loop with LangGraph (iterative convergence)
Demo 4 (Phase 3)     + KPU micro-arch sizing, floorplan, bandwidth, RTL generation
Demo 5 (Phase 4)     + Safety-critical detection, redundancy, HITL governance gates
Demo 6 (Phase 4)     + Experience cache for cross-design knowledge transfer
Demo 7 (Phase 4)     + Multi-workload, full autonomous campaign with evaluation
```

Each demo builds on the previous phases. Demo 1 works standalone and includes a built-in optimization loop that resolves failing PPA constraints. Demo 2 adds design-space exploration with Pareto analysis. Demo 3 provides an alternative LangGraph-based optimization loop. Demo 4 adds the full hardware design flow through RTL synthesis. Demos 5-7 add safety governance, episodic memory, and multi-workload autonomy with a 9-dimension evaluation framework.
