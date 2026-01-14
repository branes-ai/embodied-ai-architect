# Embodied AI Architect - Complete Roadmap Summary

## Core Product Roadmap (12 Months)

---

## Phase 1: The "Digital Twin" Foundation

**Timeline**: Months 1-4
**Goal**: Establish trusted simulation environment where software, hardware, and physics meet

### Milestone 1.1: Coupled Simulation Pipeline

| A/C | Criterion | Metric |
|-----|-----------|--------|
| 1.1.1 | Simulator Integration | AirSim/Gazebo ↔ CSim bi-directional flow |
| 1.1.2 | Energy Traceability | Per-component breakdown (perception, planning, control) |
| 1.1.3 | Latency Correlation | End-to-end timing sensor → actuator |
| 1.1.4 | Hardware Model Accuracy | ±15% vs physical Jetson Orin |

### Milestone 1.2: Baseline Profiling

| A/C | Criterion | Metric |
|-----|-----------|--------|
| 1.2.1 | Dataset Scale | 10,000+ simulated flight minutes |
| 1.2.2 | Configuration Coverage | 50+ software configurations |
| 1.2.3 | Label Quality | Automated annotation (outcome, energy, latency) |
| 1.2.4 | Correlation Discovery | 3+ actionable insights |

**Monthly Breakdown**:

| Month | Deliverable |
|-------|-------------|
| 1 | Simulator integration working |
| 2 | Energy attribution per component |
| 3 | Calibration to ±15% accuracy |
| 4 | 10,000 flight minutes dataset |

---

## Phase 2: The "Software Architect" Agent

**Timeline**: Months 5-8
**Goal**: Agent actively optimizes software pipeline on fixed hardware

### Milestone 2.1: Autonomous Pipeline Search

| A/C | Criterion | Metric |
|-----|-----------|--------|
| 2.1.1 | Constraint Enforcement | Zero critical constraint violations |
| 2.1.2 | Energy Maximization | ≥20% improvement in Perf/Watt |
| 2.1.3 | Architecture Autonomy | 3+ distinct architectural changes per run |
| 2.1.4 | Failure Interpretation | Textual root cause + proposed correction |

### Milestone 2.2: Natural Language Constraints

**Example Translations**:

| User Input | Agent Action |
|------------|--------------|
| "Maximize flight time, BoM < $350" | Filter HW by cost, prioritize INT8, smaller cache |
| "Detect objects at 10m in any weather" | Prohibit aggressive quantization, add domain randomization |
| "Control loop ≥ 250 Hz" | Enforce <4ms latency, dedicate CPU core |
| "Operate at 60°C passive cooling" | Cap TDP at 5W, prioritize NPU over GPU |

---

## Phase 3: The "Hardware Co-Designer"

**Timeline**: Months 9-12
**Goal**: Agent designs custom hardware to fit optimized software

### Milestone 3.1: Hardware Parameter Tuning

| A/C | Criterion | Metric |
|-----|-----------|--------|
| 3.1.1 | Architectural Generation | Modify ≥2 HW params (cache, NPU array) |
| 3.1.2 | Co-Design Efficiency Uplift | +30% Perf/Watt vs Phase 2 baseline |
| 3.1.3 | Constraint-Aware Design | Respect power budget (e.g., 8W TDP) |
| 3.1.4 | Bottleneck Resolution | 40% reduction in primary bottleneck |

### Milestone 3.2: Sim-to-Real Golden Design

| A/C | Criterion | Metric |
|-----|-----------|--------|
| 3.2.1 | Deployment Success | Boot and complete mission on physical drone |
| 3.2.2 | Sim-to-Real Accuracy | Within ±10% of predicted energy |
| 3.2.3 | Efficiency Proof | ≥50% flight time increase vs baseline |
| 3.2.4 | Blueprint Package | Binary + HW config + BoM + validation report |

---

## Quarterly Summary

| Quarter | Focus | Success Metric |
|---------|-------|----------------|
| **Q1** | Build "Simulated Lab" | Energy model accuracy |
| **Q2** | Agent optimizes Software | Energy saved on fixed HW |
| **Q3** | Agent optimizes Hardware | Flight time improvement |
| **Q4** | Functional validation | SoC functional correctness |

---

## Team Structure (6 people, AI-leveraged)

| Role | Phase | Core Deliverable |
|------|-------|------------------|
| Lead Systems Architect | 1-3 | Constraint design, blueprint sign-off |
| Agentic AI/RL Engineer | 1-3 | Orchestrator Agent, reward functions |
| Simulation & Tooling Engineer | 1-3 | Co-Sim pipeline, log interpretation |
| Embedded/FPGA Specialist | 1-3 | Optimized kernels, physical deployment |
| Custom HW Synthesis Engineer | 3 | RTL/Verilog from Agent output |
| V&V Engineer | 3 | Sim-to-Real accuracy validation |

**Budget**: $1.5M - $2.0M (12 months)

---

## Supporting Infrastructure Roadmaps

### Documentation & Developer Experience (12 weeks)

| Phase | Timeline | Deliverables |
|-------|----------|--------------|
| 1. Foundation | Weeks 1-2 | Astro Starlight site, landing page, getting started |
| 2. Catalog | Weeks 3-4 | Auto-generated pages from embodied-schemas |
| 3. Tutorials | Weeks 5-8 | 7 step-by-step guides |
| 4. Reference | Weeks 9-10 | CLI, MCP tools, API docs |
| 5. Interactive | Weeks 11-12 | Hardware finder, compatibility matrix |

**Tech Stack**: Astro Starlight + `llms.txt` for AI discovery

### Knowledge Base (10 weeks)

| Phase | Timeline | Deliverables |
|-------|----------|--------------|
| 1. Foundation | Weeks 1-2 | Convert YAML → markdown chunks, vector index |
| 2. Analytical | Weeks 3-4 | graphs roofline integration, text-formatted tools |
| 3. Knowledge | Weeks 5-8 | 15 deployment guides, 10 optimization recipes |
| 4. Benchmarks | Weeks 9-10 | 30 measured model-hardware benchmark entries |

**Architecture**: LLM-native (text in context, not SQL)

---

## Risk Matrix

| Risk | Probability | Severity | Mitigation |
|------|:-----------:|:--------:|------------|
| Simulation Accuracy | High | High | Empirical calibration, Design Rule Checks |
| Agent Hallucination | High | High | Hard-coded DRC rejects invalid outputs |
| Long Iteration Cycles | Medium | High | Surrogate models for fast search |
| Sim-to-Real Gap | Medium | High | Domain randomization (wind, noise, drift) |
| Vendor Lock-in | Medium | Medium | HAL for ARM, RISC-V, FPGA backends |

---

## End State (Month 12)

The Embodied AI Architect delivers:

1. **Autonomous Design**: Agent generates optimized HW/SW configurations from natural language constraints
2. **Validated Efficiency**: 50%+ flight time improvement, verified on physical hardware
3. **Blueprint Package**: Deployable binary + HW config + BoM + validation report
4. **Documentation**: Complete user docs, tutorials, and LLM-integrated knowledge base

---

## Detailed Roadmap Documents

| Document | Description |
|----------|-------------|
| [roadmap.md](roadmap.md) | Main roadmap with methodology and budget |
| [roadmap-phase-1.md](roadmap-phase-1.md) | Phase 1 user stories and acceptance criteria |
| [roadmap-phase-2.md](roadmap-phase-2.md) | Phase 2 user stories and acceptance criteria |
| [roadmap-phase-3.md](roadmap-phase-3.md) | Phase 3 user stories and acceptance criteria |
| [roadmap-devteam.md](roadmap-devteam.md) | Team structure and AI-leverage strategy |
| [roadmap-documentation.md](roadmap-documentation.md) | Documentation site roadmap (Astro Starlight) |
| [roadmap-system-architecture.md](roadmap-system-architecture.md) | Technical architecture details |
| [../knowledge-base-architecture.md](../knowledge-base-architecture.md) | LLM-native knowledge base design |
