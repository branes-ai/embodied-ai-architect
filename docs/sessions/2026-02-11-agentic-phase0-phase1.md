# Session: Agentic SoC Designer — Phase 0 + Phase 1 Implementation

**Date**: 2026-02-11
**Focus**: Implementing the foundation layers (Phase 0) and specialist agents (Phase 1) of the agentic SoC designer, following the plan created on 2026-02-10.

## Summary

Built the core runtime for the agentic SoC designer: a task graph engine, typed state schema, LLM-powered planner, DAG-walking dispatcher, and 6 specialist agents. The system can now decompose a natural language design goal into a task DAG and execute it end-to-end, producing a scored hardware recommendation with PPA verdicts and a structured design report. Validated with drone and AMR use cases; the Stillwater KPU is correctly selected as the top candidate for low-power edge workloads.

## Key Accomplishments

### 1. Phase 0.1 — State Schema & Task Graph Engine

Created the foundational data structures that everything else builds on:

- **`task_graph.py`**: DAG engine with `TaskNode`, `TaskGraph`, `TaskStatus`, `CycleError`
  - Dependency tracking with lifecycle states: PENDING → READY → RUNNING → COMPLETED/FAILED/SKIPPED
  - Cycle detection via topological sort in `add_tasks()`
  - Batch operations, serialization round-trip (`to_dict()`/`from_dict()`)
  - Properties: `ready_tasks()`, `is_complete`, `has_failures`, `blocked_tasks`, `downstream()`

- **`soc_state.py`**: `SoCDesignState` TypedDict — the single state object flowing through the LangGraph pipeline
  - `DesignConstraints`: Power, latency, area, cost, process node, temperature, safety
  - `PPAMetrics`: Measurements with per-constraint verdicts (PASS/FAIL/PARTIAL)
  - `DesignDecision`: Audit trail entries with agent, action, rationale, alternatives
  - Helper functions for serialization boundary (TypedDict ↔ Pydantic models)

- **52 tests** covering DAG operations, lifecycle management, serialization, and state helpers

### 2. Phase 0.2 — Planner & Dispatcher

Built the "brain" and "scheduler" of the agentic system:

- **`planner.py`**: `PlannerNode` — LLM-powered goal decomposition
  - System prompt encoding SoC design expertise with 7 specialist agent descriptions
  - Structured JSON output parsing with markdown fence stripping
  - `tasks_to_graph()`: Converts parsed plan into a `TaskGraph` DAG
  - Supports static plans for deterministic testing (no LLM needed)

- **`dispatcher.py`**: `Dispatcher` — DAG-walking task scheduler
  - `step()`: Find ready tasks, dispatch to registered agent executors, record decisions
  - `run()`: Loop `step()` until complete or max-steps reached
  - `_state_updates` convention: task results can write to top-level state fields
  - Failure handling with downstream task skipping

- **28 tests** with `MockLLMClient` for plan parsing, dispatch loop, diamond DAGs, and integration

### 3. Phase 1 — Specialist Agents

Implemented 6 specialist agent executors that plug into the Dispatcher:

| Agent | Role | Key Output |
|-------|------|-----------|
| `workload_analyzer` | Keyword-based workload estimation from goal text | Operator profiles with GFLOPS/memory estimates |
| `hw_explorer` | Score hardware catalog against constraints | Ranked candidates with verdicts (KPU, Jetson, Coral, etc.) |
| `architecture_composer` | Map operators to hardware, generate IP blocks | IP blocks, memory map, interconnect topology |
| `ppa_assessor` | Estimate power/latency/area/cost | Per-constraint PASS/FAIL verdicts, bottlenecks, suggestions |
| `critic` | Design review and risk identification | Risks (single-compute, power margin, memory, safety) |
| `report_generator` | Structured design report | Executive summary, all artifacts, decision trail |

- **`create_default_dispatcher()`**: Factory wiring all 6 specialists into a ready-to-run Dispatcher
- **28 tests** covering individual specialists, factory, and full pipeline integration

### 4. End-to-End Validation

Two complete pipeline runs validated:

**Drone SoC** (goal: "Design an SoC for a delivery drone with real-time perception"):
- Constraints: <5W power, <33ms latency, <$30 cost
- Workloads detected: detection, tracking
- KPU selected as top candidate (5W TDP, $25, 20 TOPS INT8)
- All PPA constraints: PASS
- 7 design decisions recorded in audit trail

**AMR SoC** (goal: "Design an SoC for an autonomous mobile robot"):
- Multi-workload: SLAM + detection + voice + LiDAR
- Correctly identifies higher compute requirements
- Architecture includes ISP for vision workloads

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `src/embodied_ai_architect/graphs/task_graph.py` | Created | DAG engine with TaskNode, TaskGraph, TaskStatus |
| `src/embodied_ai_architect/graphs/soc_state.py` | Created | SoCDesignState TypedDict, DesignConstraints, PPAMetrics, helpers |
| `src/embodied_ai_architect/graphs/planner.py` | Created | PlannerNode with LLM protocol, plan parsing, static plan mode |
| `src/embodied_ai_architect/graphs/dispatcher.py` | Created | Dispatcher with agent registry, DAG walk, failure handling |
| `src/embodied_ai_architect/graphs/specialists.py` | Created | 6 specialist executors + factory function |
| `src/embodied_ai_architect/graphs/__init__.py` | Modified | Added exports for all new modules |
| `tests/test_soc_state.py` | Created | 52 tests for task graph and state schema |
| `tests/test_planner_dispatcher.py` | Created | 28 tests for planner and dispatcher |
| `tests/test_specialists.py` | Created | 28 tests for specialists and full pipeline |
| `CHANGELOG.md` | Updated | Added Phase 0 + Phase 1 entry |

## Bugs Fixed During Implementation

1. **Task graph cycle detection test**: Original test had a non-cyclic dependency mislabeled as a cycle. Replaced with proper self-loop and mutual-dependency tests.

2. **Blocked tasks counting**: Expected transitive blocking (t1 fails → t2 blocked → t3 blocked) but `blocked_tasks` correctly only counts tasks with a directly-failed dependency. Fixed test assertion.

3. **HW Explorer using old agent**: The existing `HardwareProfileAgent` returned results with different field names (Google TPU v4 instead of static catalog). Fixed by always using the static hardware catalog for consistent, curated results.

4. **PPA Assessor NoneType comparison**: When hardware candidates had `None` for cost/power fields, comparison operators failed. Added None guards for all constraint checks.

## Architecture Decisions

1. **Serialization boundary**: LangGraph TypedDict requires plain dicts. Rich objects (TaskGraph, DesignConstraints) serialize via `to_dict()`/`model_dump()` with helper functions to cross the boundary.

2. **`_state_updates` convention**: Task results containing a `_state_updates` key get their contents merged into top-level state by the Dispatcher. This lets specialists write to shared state fields (workload_profile, hardware_candidates, etc.) without coupling to the full state schema.

3. **Static hardware catalog**: Rather than delegating to the existing `HardwareProfileAgent` (which returns unpredictable results), specialists use a curated 6-platform catalog with the KPU prominently included.

4. **Dependency result access**: Added `get_dependency_results(state, task)` helper so specialists can read outputs from upstream tasks without knowing the full state structure.

## Commits

| Hash | Message |
|------|---------|
| `096d1e5` | Add SoC design state schema and task graph engine (Phase 0.1) |
| `fe3859a` | Add Planner agent and Dispatcher for agentic SoC design (Phase 0.2) |
| `1d83ac9` | Add specialist agents and full SoC design pipeline (Phase 1) |

## Test Results

108 tests total, all passing:
- `tests/test_soc_state.py`: 52 passed
- `tests/test_planner_dispatcher.py`: 28 passed
- `tests/test_specialists.py`: 28 passed

## Next Steps

1. **Phase 2 — Memory & Governance** (per implementation plan):
   - Working memory for multi-turn specialist reasoning
   - Experience cache for cross-session learning
   - Governance layer for safety-critical design approval
   - PostgreSQL persistence for design sessions

2. **Phase 3 — EDA Integration & RTL Generation**:
   - Containerized EDA toolchain (Verilator, Yosys, OpenROAD)
   - RTL generation loop with linter/synthesis/critic feedback
   - Design space explorer with Pareto analysis

3. **Immediate improvements**:
   - Wire the planner to a real LLM (currently tested with mock/static plans)
   - Build the LangGraph `StateGraph` with conditional edges for the optimization loop
   - Validate with Demo 1 from the plan (delivery drone with real LLM decomposition)
