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
| Demo 1 | `demo_soc_designer.py` | End-to-end SoC design pipeline | Core only |
| Demo 3 | `demo_soc_optimizer.py` | Iterative power optimization loop | LangGraph |
| Demo 4 | `demo_kpu_rtl.py` | KPU micro-architecture + RTL generation | Core only |

There are also three utility examples (not demos):

| Example | Script | What It Shows | Dependencies |
|---------|--------|---------------|-------------|
| Simple Workflow | `simple_workflow.py` | Legacy Orchestrator pipeline with a PyTorch CNN | PyTorch |
| Remote Benchmark | `remote_benchmark_example.py` | SSH-based remote benchmarking | `paramiko` |
| K8s Scaling | `kubernetes_scaling_example.py` | Kubernetes horizontal scaling | `kubernetes` |

---

## Demo 1: Agentic SoC Designer

**Goal decomposition + full specialist pipeline.**

Runs a delivery drone SoC design through: Planner -> Dispatcher -> 6 Specialists -> Design Report.

### What happens

1. **Planning** — Decomposes the goal into a 6-task DAG
2. **Workload Analysis** — Estimates compute (GFLOPS) and memory from goal keywords
3. **Hardware Exploration** — Scores 6 hardware candidates against power/cost/latency constraints
4. **Architecture Composition** — Maps operators to hardware, generates IP blocks and interconnect
5. **PPA Assessment** — Estimates power/performance/area/cost with per-constraint PASS/FAIL verdicts
6. **Design Review** — Critic identifies risks, bottlenecks, and improvement opportunities
7. **Report Generation** — Structured design report with decision trail

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
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--llm` | off | Use Claude for goal decomposition |
| `--goal TEXT` | drone prompt | Custom design goal string |
| `--power FLOAT` | 5.0 | Max power budget (watts) |
| `--latency FLOAT` | 33.3 | Max latency (ms) |
| `--cost FLOAT` | 30.0 | Max BOM cost (USD) |

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

The demos show the system's evolution:

```
Demo 1 (Phase 1)     Single-pass pipeline: plan -> dispatch -> 6 specialists
Demo 3 (Phase 2)     + Optimization loop with LangGraph (iterative convergence)
Demo 4 (Phase 3)     + KPU micro-arch sizing, floorplan, bandwidth, RTL generation
```

Each demo builds on the previous phases. Demo 1 works standalone. Demo 3 adds the optimization outer loop. Demo 4 adds the full hardware design flow from micro-architecture configuration through RTL synthesis.
