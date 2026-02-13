# Phase 2: Optimization Loop, Memory, Governance, Persistence

## Context

Demo 1 runs end-to-end but the KPU design **fails power** (6.3W vs 5W budget). The system can't fix this — it has no optimization loop, no memory of what was tried, and no governance controls. Phase 2 closes these gaps by wrapping the existing Dispatcher in a LangGraph StateGraph that iterates until constraints are met.

**Two-level architecture:**
- **Outer loop** (new): LangGraph `StateGraph` — planner → dispatch → evaluate → [optimize → loop back | report → END]
- **Inner loop** (existing): `Dispatcher.run()` walks the TaskGraph DAG of specialist agents

This gives us: conditional edges for the optimization loop, `interrupt_before` for HITL, checkpointing for save/resume, and a clean separation between single-pass execution and iterative refinement.

```
                    +-----------+
               +--->|  planner  |---+
               |    +-----------+   |
               |                    v
         +-----+----+       +-----------+
         | optimize |       | dispatch  |  (runs Dispatcher.run() internally)
         +-----+----+       +-----+-----+
               ^                  |
               |            +-----v-----+
               |            | evaluate  |
               |            +-----+-----+
               |                  |
               |          +-------+------+
               |          |              |
               +---[FAIL/iterate]  [PASS/limit]
                                        |
                                  +-----v-----+
                                  |  report   |---> END
                                  +-----------+
```

## Implementation Steps (8 steps, strict dependency order)

### Step 1: Extend SoCDesignState

**Modify** `src/embodied_ai_architect/graphs/soc_state.py`

Add fields to `SoCDesignState`:
- `working_memory: dict` — WorkingMemoryStore serialized
- `optimization_history: list[dict]` — PPA snapshot per iteration
- `governance: dict` — GovernancePolicy serialized
- `audit_log: list[dict]` — AuditEntry list

Add helpers: `get_working_memory()`, `update_working_memory()`, `record_audit()`, `get_optimization_history()`

Update `create_initial_soc_state()` with optional `governance` param.

**Extend** `tests/test_soc_state.py` with tests for new fields/helpers.

### Step 2: Working Memory

**Create** `src/embodied_ai_architect/graphs/memory.py`

- `AgentWorkingMemory(BaseModel)`: agent_name, decisions_made, open_questions, constraints_discovered, things_tried (list of {description, outcome, iteration}), iteration_notes
- `WorkingMemoryStore(BaseModel)`: agents dict, `get_agent_memory()`, `record_attempt()`
- Serialized via `model_dump()`/constructor for TypedDict compatibility

**Create** `tests/test_memory.py`

### Step 3: Design Optimizer Specialist

**Create** `src/embodied_ai_architect/graphs/optimizer.py`

- `OPTIMIZATION_STRATEGIES` catalog: quantize_int8, reduce_resolution, clock_scaling, model_pruning, smaller_model — each with applicable_when conditions and power/latency reduction factors
- `design_optimizer(task, state)` — reads failing verdicts from `ppa_metrics`, filters applicable strategies excluding already-tried ones (from working memory), selects best, applies by modifying workload_profile/architecture/ip_blocks via `_state_updates`
- `_apply_strategy()` — mutates copies of state artifacts per strategy

**Modify** `src/embodied_ai_architect/graphs/specialists.py` — register `design_optimizer` in `create_default_dispatcher()`

**Create** `tests/test_optimizer.py`

### Step 4: Dispatcher Working Memory Integration

**Modify** `src/embodied_ai_architect/graphs/dispatcher.py`

- In `_dispatch_task()`: read agent's working memory before exec, record attempt after exec
- Backward compatible: existing agents work unchanged

**Extend** `tests/test_planner_dispatcher.py`

### Step 5: Governance Layer

**Create** `src/embodied_ai_architect/graphs/governance.py`

- `AuditEntry(BaseModel)`: timestamp, agent, action, input_summary, output_summary, cost_tokens, human_approved, iteration
- `GovernancePolicy(BaseModel)`: approval_required_actions, cost_budget_tokens, iteration_limit, require_human_approval_on_fail, fail_iteration_threshold
- `GovernanceGuard`: check_budget(), check_iteration_limit(), requires_approval(), record()

**Create** `tests/test_governance.py`

### Step 6: LangGraph Outer Loop (core of Phase 2)

**Create** `src/embodied_ai_architect/graphs/soc_graph.py`

`build_soc_design_graph(dispatcher, planner, governance, experience_cache, checkpointer, interrupt_at_evaluate)` → compiled StateGraph

Five nodes:
- `planner`: calls existing PlannerNode
- `dispatch`: calls Dispatcher.run() for full inner DAG; on iteration > 0, creates minimal re-eval plan (just ppa_assessor)
- `evaluate`: checks verdicts, governance limits, records optimization_history entry, sets `next_action`
- `optimize`: calls design_optimizer, increments iteration
- `report`: calls report_generator, saves experience episode

Routing: `evaluate → optimize` (FAIL + under limit) or `evaluate → report` (PASS or over limit)

Reference patterns: `experiments/langgraph/soc_optimizer/graph.py` (conditional edges, `graph.stream()`, `recursion_limit`), `graphs/pipelines/autonomy.py` (StateGraph construction)

**Create** `tests/test_soc_graph.py` — test single-pass PASS, optimization convergence, iteration limit, checkpointing

### Step 7: Experience Cache

**Create** `src/embodied_ai_architect/graphs/experience.py`

- `DesignEpisode(BaseModel)`: episode_id, goal, use_case, platform, constraints, architecture_chosen, hardware_selected, ppa_achieved, constraint_verdicts, outcome_score, iterations_used, key_decisions, lessons_learned, optimization_trace, problem_fingerprint
- `ExperienceCache`: SQLite-backed (`~/.embodied-ai/experience.db`), save(), load(), search_similar() (exact match on use_case, order by score), list_episodes(), fingerprint()
- Single table with JSON blob column for full episode data

**Create** `tests/test_experience.py` — uses `:memory:` SQLite

Integrate into `report` node in `soc_graph.py`.

### Step 8: Runner + Demo 3

**Create** `src/embodied_ai_architect/graphs/soc_runner.py`

- `SoCDesignRunner`: wraps compiled graph, provides `run()`, `resume()`, `get_state_history()`
- Sensible defaults, optional checkpointing via `langgraph-checkpoint-sqlite`

**Create** `examples/demo_soc_optimizer.py` — Demo 3: drone power optimization

```
Iteration 0: KPU selected, power 6.3W → FAIL
Iteration 1: Apply INT8 quantization → power ~5.1W → FAIL (still over, or barely)
Iteration 2: Reduce resolution 640→480 → power ~4.2W → PASS
Result: Converged in 2 optimization iterations.
```

**Modify** `src/embodied_ai_architect/graphs/__init__.py` — export new modules

**Modify** `pyproject.toml` — add `langgraph-checkpoint-sqlite` to langgraph extras

## Files Summary

| Action | File |
|--------|------|
| Modify | `src/embodied_ai_architect/graphs/soc_state.py` |
| Create | `src/embodied_ai_architect/graphs/memory.py` |
| Create | `src/embodied_ai_architect/graphs/optimizer.py` |
| Modify | `src/embodied_ai_architect/graphs/dispatcher.py` |
| Create | `src/embodied_ai_architect/graphs/governance.py` |
| Create | `src/embodied_ai_architect/graphs/soc_graph.py` |
| Create | `src/embodied_ai_architect/graphs/experience.py` |
| Create | `src/embodied_ai_architect/graphs/soc_runner.py` |
| Create | `examples/demo_soc_optimizer.py` |
| Modify | `src/embodied_ai_architect/graphs/__init__.py` |
| Modify | `src/embodied_ai_architect/graphs/specialists.py` |
| Modify | `pyproject.toml` |
| Create | `tests/test_memory.py` |
| Create | `tests/test_optimizer.py` |
| Create | `tests/test_governance.py` |
| Create | `tests/test_soc_graph.py` |
| Create | `tests/test_experience.py` |

## Scope Boundaries (NOT building)

- No PostgreSQL — SQLite only
- No embedding-based similarity search — exact match on use_case + fingerprint
- No LLM in the optimizer — deterministic strategy catalog only
- No RTL generation — that's Phase 3
- No Pareto front computation — optimization_history tracks PPA per iteration only
- No UI for time-travel — Python API only
- No real LLM token counting — AuditEntry.cost_tokens = 0 for deterministic agents

## Verification

1. `pytest tests/test_memory.py tests/test_optimizer.py tests/test_governance.py tests/test_experience.py -v` — unit tests
2. `pytest tests/test_soc_graph.py -v` — integration tests (optimization loop converges, iteration limit works)
3. `pytest tests/test_soc_state.py tests/test_planner_dispatcher.py tests/test_specialists.py -v` — existing tests still pass
4. `python examples/demo_soc_optimizer.py` — Demo 3 shows power convergence from 6.3W → PASS
5. `python examples/demo_soc_designer.py` — Demo 1 still works unchanged
