# Client Demo Guide

## Repository Assessment

### What This Repo Actually Is

A working **agentic SoC design automation platform** — AI agents that decompose a natural-language hardware goal (e.g., "design an SoC for a delivery drone") into a task DAG, then execute a pipeline of specialist agents that produce a concrete hardware design with PPA (Power/Performance/Area) metrics, constraint verdicts, and even RTL (Verilog) output. It's not scaffolding — there are ~10K lines in the graph-based agent system alone, 564 tests, and 7 runnable demos.

### Strongest Demo Narrative

The compelling story is: **"Describe what your robot needs to do in English → get a validated hardware design with RTL."** The system decomposes goals, explores hardware trade-offs, validates physical constraints, and generates synthesizable code — all autonomously.

---

## Recommended Demo Sequence

Run these three demos in order. All run locally, deterministically, with **no API key needed**, and complete in seconds.

### 1. Demo 1 — Agentic SoC Designer (the vision)

```bash
python examples/demo_soc_designer.py
```

**Shows:** Natural-language goal → 6-task DAG → workload analysis → hardware candidate ranking with PASS/FAIL verdicts → architecture selection → IP block generation → PPA assessment → critic review → **optimization loop that resolves failures** → final review with all PASS → full decision audit trail. This is the "aha" moment — AI designing hardware from a sentence, then fixing its own mistakes.

You can also show customizability live:

```bash
python examples/demo_soc_designer.py --goal "Design an SoC for autonomous mobile robot with SLAM" --power 3.0 --cost 50.0
```

### 2. Demo 4 — KPU RTL Generation (the depth)

```bash
python examples/demo_kpu_rtl.py
```

**Shows:** The full vertical — from goal all the way down to KPU micro-architecture sizing (systolic arrays, SRAM hierarchy, NoC topology), checkerboard floorplan validation with pitch matching, memory bandwidth balance checking through the hierarchy, and Jinja2-templated RTL generation for ~12 sub-components with cell counts. This is the "pre-silicon" differentiator that proves this isn't just analysis — it generates hardware.

### 3. Demo 7 — Full Autonomous Campaign (the breadth)

```bash
python examples/demo_full_campaign.py
```

**Shows:** A quadruped robot with 4 concurrent workloads (Visual SLAM, object detection, LiDAR, voice), Pareto-front design-space exploration, multi-workload scheduling, governance with iteration limits, and a 9-dimension evaluation scorecard. This demonstrates the system handling real complexity — multiple competing workloads, trade-off analysis, and self-assessment.

---

## Optional Add-ons (if time permits)

| Demo | What it adds | Requirement |
|------|-------------|-------------|
| **Demo 5** (`demo_hitl_safety.py`) | Surgical robot with IEC 62304 safety detection, redundancy injection, HITL governance gates | None |
| **Demo 6** (`demo_experience_cache.py`) | Cross-design knowledge transfer — second drone design warm-starts from first via episodic memory | None |
| **Demo 2** (`demo_dse_pareto.py`) | Warehouse AMR with Pareto front visualization and knee-point identification | None |
| **Simple Workflow** (`simple_workflow.py`) | Creates actual PyTorch CNN, benchmarks it, generates HTML report you can open in a browser | PyTorch |
| **Interactive Chat** (`branes chat`) | Claude-powered architect REPL — ask questions, analyze models interactively | `ANTHROPIC_API_KEY` |

---

## What's Real vs. What's Modeled

**Real implementation:**
- Task graph engine with DAG dependency resolution and topological sort
- 14 specialist agents with actual logic (not stubs)
- Hardware knowledge base with real specs (Jetson AGX Orin, A100, Coral, etc.)
- RTL template generation producing Verilog from Jinja2 templates
- EDA toolchain integration (Verilator lint, Yosys synthesis, Icarus sim) — with mock fallbacks when tools aren't installed
- 9-dimension evaluation framework with gold standards
- SQLite-backed experience cache with similarity matching
- Full drone perception prototype (YOLOv8 + ByteTrack + 3D scene graph)

**Modeled/estimated (not measured silicon):**
- PPA numbers are analytical estimates, not from actual synthesis runs
- RTL cell counts come from templates + scaling models, not place-and-route
- Hardware candidate scoring uses weighted heuristics against datasheet specs

This is the right level for an architecture exploration tool — it's accurate enough to make design decisions, which is the point.

---

## Demo Narrative Arc

The **progression from Demo 1 → Demo 4 → Demo 7** is the strongest narrative:

1. "Here's how we decompose a goal into a design — and when it fails, the system fixes itself" (Demo 1)
2. "And we go all the way to RTL" (Demo 4)
3. "At scale, with multiple workloads and self-evaluation" (Demo 7)

That progression shows vision, depth, and engineering maturity in under 2 minutes of total runtime.

---

## Screenshot Guide for Slides

Each screenshot below captures a key moment that tells a specific story. Use a dark terminal theme for visual impact (the output is plain text with alignment formatting that reads well in monospace).

### Demo 1 Screenshots (3 slides)

**Screenshot 1A — "Goal to DAG" (the pitch)**
Capture the banner through the task graph output:
```
========================================================================
  Agentic SoC Designer — Demo 1
========================================================================

  Goal: Design an SoC for a last-mile delivery drone that must: run a visual perception pipeline (...
  Mode: Static Plan
  ...
  Task graph (6 tasks, planned in 0.0s):

    [t1] Analyze perception workload (detection + tracking at 30fps)
         agent: workload_analyzer  deps: (none)
    [t2] Enumerate feasible hardware under 5W/​$30 constraints
         agent: hw_explorer  deps: t1
    ...
```
*Story: "You describe your robot in plain English. The system decomposes it into an agent task graph."*

**Screenshot 1B — "Hardware Trade-off Table" (the analysis)**
Capture the Hardware Candidates section:
```
--- Hardware Candidates ------------------------------------------------

  Rank  Name                         TDP    Cost  INT8 TOPS  Score  Verdict
  ----- ------------------------- ------ ------- ---------- ------  ----------
  1     Stillwater KPU              5.0W   25.0$       20.0   81.7  power:PASS cost:PASS
  2     Hailo-8                     2.5W   70.0$       26.0   67.5  power:PASS cost:FAIL
  3     Google Coral Edge TPU       2.0W   35.0$        4.0   59.0  power:PASS cost:FAIL
  4     NVIDIA Jetson Orin Nano    15.0W  199.0$       40.0   30.0  power:FAIL cost:FAIL
  5     AMD Ryzen AI (Xilinx NPU)  25.0W  150.0$       16.0   25.0  power:FAIL cost:FAIL
  6     Raspberry Pi 5              8.0W   80.0$        0.5   20.0  power:FAIL cost:FAIL
```
*Story: "Agents autonomously evaluate real hardware against your constraints. The system knows TDP, cost, TOPS for each candidate and gives PASS/FAIL verdicts."*

**Screenshot 1C — "FAIL → Optimize → PASS" (the resolution)**
Capture the Optimization Phase through the Final Design Review:
```
--- Optimization Phase -------------------------------------------------

  Initial: power=6.3W (FAIL), latency=4.5ms (PASS), cost=$25 (PASS)

  Iteration 1: Applied Smaller Model Variant (~35% power reduction)
    Power: 6.3W → 4.9W | Latency: 4.5ms → 3.6ms
    Verdicts: power:PASS  latency:PASS  cost:PASS

  Converged in 1 iteration — all constraints PASS

--- Final PPA (post-optimization) --------------------------------------
  Power......................... 4.9 W
  Latency....................... 3.6 ms
  Cost.......................... $25

  power                PASS
  latency              PASS
  cost                 PASS

--- Final Design Review ------------------------------------------------
  Assessment.................... ADEQUATE
```
*Story: "The system doesn't just report failures — it fixes them. Power exceeded 5W, so the optimizer applied a model variant strategy, brought power to 4.9W, and all constraints now pass. The critic upgrades its assessment from NEEDS_WORK to ADEQUATE."*

### Demo 4 Screenshots (3 slides)

**Screenshot 4A — "KPU Micro-architecture" (the depth)**
Capture the KPU Configuration section:
```
--- KPU Configuration --------------------------------------------------
  Process Node....................... 28nm
  Checkerboard....................... 2x2
  Systolic Array..................... 16x16
  L2/tile............................ 256KB
  L1/tile............................ 32KB
  Streamers/tile..................... 2
  L3/mem tile........................ 512KB
  Block Movers/mem tile.............. 1
  DRAM............................... LPDDR4X 2 ctrl
  NoC................................ mesh_2d 256-bit
```
*Story: "From a drone description, the system sizes a custom accelerator micro-architecture: systolic arrays, SRAM hierarchy, memory controllers, NoC topology."*

**Screenshot 4B — "Physical Validation" (the rigor)**
Capture Floorplan Check + Bandwidth Check together:
```
--- Floorplan Check ----------------------------------------------------
  Compute Tile....................... 1.88 x 1.88mm = 3.53mm2
  Memory Tile........................ 1.50 x 1.50mm = 2.26mm2
  Pitch Match W...................... 1.25 (MISMATCH)
  Pitch Match H...................... 1.25 (MISMATCH)
  Total Die Area..................... 95.2mm2
  Feasible........................... ** FAIL **

--- Bandwidth Check ----------------------------------------------------
  dram_to_l3......................... 25.6 avail, 1.5 req (6%)
  l3_to_l2........................... 64.0 avail, 1.1 req (2%)
  l2_to_l1........................... 64.0 avail, 0.5 req (1%)
  l1_to_compute...................... 256.0 avail, 0.2 req (0%)
  Balanced........................... PASS
```
*Story: "The system validates physical realizability — checkerboard pitch matching and bandwidth balance through the full memory hierarchy. It catches that the floorplan pitch is mismatched."*

**Screenshot 4C — "RTL Synthesis Results" (the deliverable)**
Capture the RTL Generation section:
```
--- RTL Generation -----------------------------------------------------
  mac_unit...........................    573 cells (PASS)
  compute_tile.......................      0 cells (FAIL)
  l1_skew_buffer.....................     24 cells (PASS)
  l2_cache_bank......................     22 cells (PASS)
  l3_tile............................     20 cells (PASS)
  noc_router.........................      0 cells (FAIL)
  dma_engine.........................     85 cells (PASS)
  block_mover........................    257 cells (PASS)
  streamer...........................    533 cells (PASS)
  memory_controller..................     19 cells (PASS)

  Total cells: 1533
```
*Story: "Real Verilog is generated from templates, linted with Verilator, and synthesized through Yosys. This is actual EDA toolchain output — 8 of 10 components synthesize successfully with real cell counts."*

### Demo 7 Screenshots (2 slides)

**Screenshot 7A — "Multi-Workload Profile" (the complexity)**
Capture the Multi-Workload Analysis + Pareto Front together:
```
--- Multi-Workload Analysis --------------------------------------------
  Total GFLOPS.................. 22.0
  Total Memory.................. 242.0 MB
  Workload Count................ 5

  Workload                  Model             GFLOPS Sched
  ------------------------- --------------- -------- ------------
  object_detection          YOLOv8               8.7 concurrent
  visual_slam               ORB-SLAM3            3.0 concurrent
  visual_perception         CNN-based            6.0 concurrent
  voice_recognition         Whisper-tiny         1.5 sequential
  lidar_processing          PointPillars         4.0 time_shared

--- Pareto Front Analysis ----------------------------------------------
  Design points................. 6
  Non-dominated................. 4
  Knee point.................... Stillwater KPU
    Power....................... 5.0W
    Latency..................... 5.7ms
    Cost........................ $25
```
*Story: "Real robots run multiple workloads concurrently. The system models scheduling (concurrent, sequential, time-shared), computes aggregate requirements, and finds the Pareto-optimal design across power/latency/cost."*

**Screenshot 7B — "All PASS + Decision Trail" (the confidence)**
Capture the PPA Assessment and Decision Trail:
```
--- PPA Assessment -----------------------------------------------------
  Power Watts................... 6.3 W
  Latency Ms.................... 5.7 ms
  Cost Usd...................... 25.0 $

    power           PASS
    latency         PASS
    cost            PASS

Overall...................... ALL PASS

--- Decision Trail -----------------------------------------------------
    1. [planner] Created task graph with 7 tasks
    2. [workload_analyzer] Completed task 'Analyze multi-workload'
    3. [hw_explorer] Completed task 'Enumerate hardware candidates'
    4. [design_explorer] Completed task 'Explore design space via Pareto front'
    5. [architecture_composer] Completed task 'Compose SoC architecture'
    6. [ppa_assessor] Completed task 'Assess PPA metrics'
    7. [critic] Completed task 'Review design'
    8. [report_generator] Completed task 'Generate comprehensive design report'
```
*Story: "Full audit trail of every design decision. Every step is traceable. This is the kind of accountability enterprise clients need."*

### Summary Slide Recommendations

| Slide | Screenshot | Key Message |
|-------|-----------|-------------|
| 1 | 1A: Goal → DAG | "Natural language to design plan" |
| 2 | 1B: HW table | "Autonomous hardware trade-off analysis" |
| 3 | 1C: FAIL → Optimize → PASS | "Self-healing design: detect failure, optimize, resolve" |
| 4 | 4A: KPU config | "Custom accelerator micro-architecture sizing" |
| 5 | 4B: Floorplan + BW | "Physical constraint validation" |
| 6 | 4C: RTL cells | "Real synthesizable Verilog output" |
| 7 | 7A: Multi-workload | "Complex multi-workload design-space exploration" |
| 8 | 7B: All PASS + trail | "Full audit trail, all constraints met" |

### Presentation Tips

- **Terminal font**: Use a clean monospace font at 14-16pt (e.g., JetBrains Mono, Fira Code)
- **Dark theme**: Dark background with light text screenshots are more visually striking in slides
- **Crop tightly**: Each screenshot should show exactly one section — don't try to fit the whole terminal
- **Timing**: Demo 1 runs instantly (<1s), Demo 7 runs instantly (<1s), Demo 4 takes ~90s (real EDA synthesis). Run Demo 4 last in live demos, or pre-capture the output
- **Live demo order**: Demo 1 → Demo 7 → Demo 4 (save the slow-but-impressive one for last)
- **Fallback**: If running live, have pre-captured terminal output ready in case of environment issues
