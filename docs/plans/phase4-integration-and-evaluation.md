# Phase 4: Integration — Evaluation Harness, Missing Demos & Regression Suite

## Context

Phases 0-3 are complete (207 tests passing). The system has: task graph DAG, 12 specialist agents, LangGraph optimization loop, working memory, governance, experience cache, KPU micro-architecture sizing, floorplan/bandwidth validation, and template-based RTL generation.

**Phase 4 is the "Prove It Works" phase.** It adds:
- A 9-dimension evaluation framework to score agentic system quality
- 3 new specialists (Pareto explorer, safety detector, experience retriever)
- 4 missing demos (Demo 2, 5, 6, 7) — the master plan defines 7 demos as acceptance tests
- Regression infrastructure (golden traces, cost tracking)

**Success criteria:** Demo 7 autonomous with <3 human interventions; composite score >0.75; experience cache demonstrably helps; governance 100% compliant.

## Implementation Steps (18 steps, 5 work streams)

### Work Stream 1: Evaluation Framework (Steps 1-4)

**Step 1** — `graphs/evaluation.py` — Data models (Medium)
- `RunTrace`: captures demo execution (task_graph, ppa_metrics, iteration_history, tool_calls, audit_log, failures, recoveries, duration, cost_tokens, human_interventions)
- `GoldStandard`: expected behavior per demo (expected_task_graph, expected_ppa, governance_triggers, max_iterations)
- `DimensionScore`: single dimension result (dimension, score, weight, details)
- `Scorecard`: composite result (demo_name, dimensions list, composite_score, passed)

**Step 2** — `graphs/scoring.py` — 9 scoring functions (Large)
- `score_decomposition()` — task graph nodes/edges vs gold standard
- `score_ppa_accuracy()` — |estimated - expected| / expected, tiered (<10%=1.0, 10-25%=0.75, 25-50%=0.5)
- `score_exploration_efficiency()` — Pareto points / total designs
- `score_reasoning()` — keyword matching + structure check on rationale
- `score_convergence()` — monotonic improvement rate across iterations
- `score_governance()` — audit log coverage of expected governance triggers
- `score_tool_use()` — precision/recall vs expected tool calls
- `score_adaptability()` — recovery rate = recoveries / failures
- `score_efficiency()` — time and cost vs budgets
- Each takes `(RunTrace, GoldStandard) -> DimensionScore`
- Deps: Step 1

**Step 3** — `graphs/evaluator.py` — AgenticEvaluator class (Medium)
- `AgenticEvaluator(gold_standards, weights)` — default weights from master plan (DCS=15%, PPA=20%, exploration=10%, reasoning=15%, convergence=10%, governance=10%, tool_use=10%, adaptability=5%, efficiency=5%)
- `evaluate_run(trace) -> Scorecard` — runs all 9 scorers, computes weighted composite
- `evaluate_all(traces) -> dict[str, Scorecard]` — batch across demos
- `capture_run_trace(initial_state, final_state, demo_name) -> RunTrace` — extract trace from state diff
- Deps: Steps 1, 2

**Step 4** — `graphs/gold_standards.py` — Gold standards for all 7 demos (Medium)
- Hand-crafted `GoldStandard` objects per demo: expected task graphs, PPA targets, governance triggers, tool call expectations
- `ALL_GOLD_STANDARDS: dict[str, GoldStandard]`
- Deps: Step 1

### Work Stream 2: Missing Specialists (Steps 5-8)

**Step 5** — `graphs/pareto.py` — Pareto front explorer (Medium)
- `ParetoPoint(BaseModel)`: hardware_name, power, latency, cost, dominated, knee_point
- `compute_pareto_front(candidates, objectives) -> list[ParetoPoint]` — non-dominated sorting
- `identify_knee_point(front) -> ParetoPoint | None` — min normalized distance to origin
- `design_explorer(task, state) -> dict` — specialist agent, reads hw_candidates, computes Pareto, writes `pareto_results` to state

**Step 6** — `graphs/safety.py` + modify `graphs/governance.py` — Safety-critical detection (Medium)
- `safety_detector(task, state) -> dict` — detects safety_critical constraints, injects approval gates into governance, adds redundancy requirements (dual-lockstep CPU, ECC, watchdog)
- `GovernancePolicy` gains `safety_critical_actions` field
- `GovernanceGuard` gains `auto_detect_safety_critical()` and `flag_safety_decision()`

**Step 7** — `graphs/experience_specialist.py` — Experience retrieval (Medium)
- `experience_retriever(task, state) -> dict` — searches ExperienceCache, computes similarity, adapts prior experience for current constraints
- `compute_similarity(current, past) -> float` — use_case + platform + constraint overlap
- Warm-starts hardware_candidates from prior episode when similarity > threshold

**Step 8** — Modify `graphs/specialists.py` — Multi-workload support (Small)
- Enhance `_estimate_workload_from_goal()` with `scheduling` field per workload (concurrent/sequential/time_shared)
- Add `aggregate_workload_requirements()` for peak concurrent GFLOPS
- Enhance `_score_hardware()` to handle multi-workload scoring
- Add `map_workloads_to_accelerators()` for heterogeneous mapping

### Work Stream 3: Missing Demos (Steps 9-12)

**Step 9** — `examples/demo_dse_pareto.py` — Demo 2: 3-way hardware comparison (Medium)
- Warehouse AMR with MobileNetV2 + SLAM workload
- Static plan: workload → hw_explorer → design_explorer → architecture → PPA → critic → report
- Outputs Pareto front across power/latency/cost, identifies knee point
- Deps: Step 5

**Step 10** — `examples/demo_hitl_safety.py` — Demo 5: Surgical robot (Medium)
- Safety-critical surgical robot with IEC 62304 requirement, 1ms force-feedback latency
- Static plan: safety_detector → workload → hw_explorer → architecture → PPA → critic → report
- Simulates human approval (auto-approve in demo mode)
- Outputs audit log with safety decisions
- Deps: Step 6

**Step 11** — `examples/demo_experience_cache.py` — Demo 6: Cache reuse (Medium)
- Runs Demo 1 (delivery drone) first, saves to experience cache
- Then runs agricultural drone with experience_retriever as first task
- Asserts fewer dispatch steps in second run
- Deps: Step 7

**Step 12** — `examples/demo_full_campaign.py` — Demo 7: Quadruped robot (Large)
- 4 workloads: Visual SLAM, object detection, LiDAR, voice recognition
- 15W envelope, $50 BOM, 10K volume
- Uses outer loop (soc_graph) with governance (iteration limit=10)
- Exercises full pipeline including multi-workload and DSE
- Deps: Steps 5, 8

### Work Stream 4: Infrastructure (Steps 13-16)

**Step 13** — `graphs/trace.py` — RunTrace capture middleware (Medium)
- `TracingDispatcher(Dispatcher)` — wraps agent calls with timing/call logging
- `extract_trace_from_state()` — builds RunTrace from before/after state diff
- Deps: Step 1

**Step 14** — `graphs/golden_traces.py` — Golden trace storage (Small)
- `save_golden_trace()`, `load_golden_trace()` — JSON serialization
- `compare_traces() -> TraceComparison` — structural diff (task graph match, PPA regression, iteration regression)

**Step 15** — Modify `graphs/governance.py` — Cost tracking (Small)
- `CostTracker` class with `add_cost()`, `total_tokens`, `cost_by_agent`, `estimated_cost_usd()`
- Integrate into `GovernanceGuard.record()` for automatic accumulation
- `format_cost_report() -> str`

**Step 16** — Modify `graphs/soc_state.py` — State extensions (Small)
- New fields: `pareto_results`, `safety_analysis`, `prior_experience`, `cost_tracking`, `evaluation_scorecard`
- Update `create_initial_soc_state()` with defaults

### Work Stream 5: Registration & Tests (Steps 17-18)

**Step 17** — Modify `specialists.py` + `__init__.py` — Wire it all together (Small)
- Register `design_explorer`, `safety_detector`, `experience_retriever` in `create_default_dispatcher()`
- Export new public symbols from `__init__.py`
- Deps: Steps 5, 6, 7, 3, 13-16

**Step 18** — Tests (Large)
- `test_evaluation.py` — data model round-trips
- `test_scoring.py` — all 9 scoring functions with known inputs
- `test_evaluator.py` — AgenticEvaluator with synthetic traces
- `test_pareto.py` — Pareto front computation, knee point, specialist
- `test_safety.py` — safety detector, redundancy, governance gates
- `test_experience_specialist.py` — retriever, similarity, warm-start
- `test_golden_traces.py` — trace save/load/compare
- `test_cost_tracking.py` — CostTracker accumulation
- `test_demo_acceptance.py` — run all 7 demos, check composite > 0.75
- Deps: All prior steps

## Parallelization Strategy

**Group A (independent, launch first):** Steps 1, 5, 6, 7, 8, 14, 15, 16
**Group B (depends on A):** Steps 2, 4, 9, 10, 11, 13
**Group C (depends on A+B):** Steps 3, 12
**Sequential final:** Steps 17, 18

## Key Files to Modify
- `src/embodied_ai_architect/graphs/soc_state.py` — new state fields
- `src/embodied_ai_architect/graphs/governance.py` — safety + cost tracking
- `src/embodied_ai_architect/graphs/specialists.py` — multi-workload + registration
- `src/embodied_ai_architect/graphs/__init__.py` — exports

## Key Files to Create
- `graphs/evaluation.py`, `graphs/scoring.py`, `graphs/evaluator.py`, `graphs/gold_standards.py`
- `graphs/pareto.py`, `graphs/safety.py`, `graphs/experience_specialist.py`
- `graphs/trace.py`, `graphs/golden_traces.py`
- `examples/demo_dse_pareto.py`, `examples/demo_hitl_safety.py`, `examples/demo_experience_cache.py`, `examples/demo_full_campaign.py`
- 9 test files

## Verification
1. Run `pytest tests/` — all existing 207 tests still pass (backward compat)
2. Run each demo script individually: `python examples/demo_*.py`
3. Run `test_demo_acceptance.py` — composite score > 0.75 across all 7 demos
4. Check `docs/demo-guide.md` is updated with new demos
