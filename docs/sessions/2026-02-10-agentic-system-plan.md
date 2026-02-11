# Session: Agentic System Architecture Assessment and Implementation Plan

**Date**: 2026-02-10
**Focus**: Honest assessment of current system, gap analysis, and comprehensive implementation plan for building a true agentic SoC designer

## Summary

Deep-dive assessment of whether the Embodied AI Architect qualifies as an "agentic system" (it does not), followed by creation of a comprehensive implementation plan to close the gap. The session produced a single planning document covering current state, gap analysis, specialist agent architecture, implementation roadmap, demo strategy, and evaluation framework.

## Key Accomplishments

### 1. Current State Assessment

Thoroughly explored the codebase to produce an honest architectural assessment:

- **Orchestrator** (`src/embodied_ai_architect/orchestrator.py`): Hardcoded 4-step sequential pipeline. No decision-making, no dependency graph, no conditional branching.
- **ArchitectAgent** (`src/embodied_ai_architect/llm/agent.py`): LLM tool-use loop with max 10 iterations. No planning, no memory persistence, no task decomposition.
- **Agents** (`src/embodied_ai_architect/agents/`): Stateless functions with `execute(input) -> AgentResult`. No reasoning, no state carryover, no learning.
- **LangGraph Pipeline** (`src/embodied_ai_architect/graphs/`): Perception pipeline with conditional routing. Not a design exploration system.

**Conclusion**: The system is a useful tool pipeline with an LLM chat wrapper, but lacks the architectural primitives (planning, task graphs, memory, governance) required for agentic behavior.

### 2. Gap Analysis

Identified 12 capability gaps across 3 severity levels:

**Critical (6)**: Goal decomposition, task graph/DAG, orchestration/dispatch, agent memory, design space exploration, optimization loop

**High (4)**: Governance/accountability, constraint system, domain knowledge depth, iterative optimization

**Medium (2)**: Validation/verification, feedback learning

### 3. CrewAI vs. Modern Approaches Analysis

Evaluated CrewAI's specialist agent model (RTL Designer, SoC Floorplanner, QA Test Writer, PPA Assessor) against LangGraph state graph approach:

- **CrewAI strengths**: Intuitive role-based agents that mirror real chip design teams
- **CrewAI weaknesses**: No native cycles/loops, no checkpointing, no HITL primitives, implicit state passing
- **Recommendation**: Adopt the specialist agent *concept* from CrewAI, implement as LangGraph nodes with typed state, checkpointing, and conditional edges
- **Key insight**: Specialization is about tools and prompts, not about the framework. An "RTL Architect Agent" is a LangGraph node with a specialized prompt, scoped tools, and its own working memory.

### 4. Implementation Plan

Created 4-phase, ~24-week implementation roadmap:

| Phase | Weeks | Key Deliverables |
|-------|-------|-----------------|
| Phase 0: Foundation | 1-3 | SoCDesignState schema, TaskGraph engine, Planner agent, Dispatcher |
| Phase 1: Specialists | 4-8 | 7 specialist agents (workload analyzer, HW explorer, architecture composer, PPA assessor, design space explorer, RTL architect, critic) |
| Phase 2: Memory + Gov | 9-12 | Working memory, experience cache, governance layer, PostgreSQL persistence |
| Phase 3: EDA + RTL | 13-20 | Containerized EDA toolchain (Verilator, Yosys, OpenROAD), RTL generation loop |

### 5. Demo Strategy

Designed 7 demo prompts progressing from simple to complex:

1. **Goal Decomposition**: Delivery drone SoC with power/latency/cost constraints
2. **Design Space Exploration**: 3-way hardware comparison with Pareto front
3. **Iterative Optimization**: Fix a 7.2W -> 5W power budget violation
4. **Multi-Agent RTL**: 32-bit MAC unit with linter/synthesis/critic loop
5. **HITL Governance**: Safety-critical surgical robot (IEC 62304 Class C)
6. **Experience Cache**: Agricultural drone reusing delivery drone knowledge
7. **Full End-to-End**: Quadruped warehouse robot with 4 AI workloads

### 6. Evaluation Framework

Defined 9 evaluation dimensions with concrete metrics:

1. Task Decomposition Completeness (DCS) -- gold-standard comparison
2. PPA Estimation Accuracy -- % error vs ground truth
3. Exploration Efficiency -- Pareto points per design evaluated
4. Reasoning Quality -- rationale scoring by human expert
5. Convergence Behavior -- monotonic improvement rate
6. Governance Compliance -- pass/fail checklist
7. Tool Use Accuracy -- precision and recall vs ideal sequence
8. Adaptability -- failure recovery rate
9. Session Efficiency -- time, cost, human interventions

Composite scorecard with weighted dimensions. Success criteria: composite > 0.75, PPA within 25%, design time < 50% of manual.

## Files Changed

| File | Action | Description |
|------|--------|-------------|
| `docs/plans/agentic-system-implementation-plan.md` | Created | Master planning document (1037 lines) |
| `CHANGELOG.md` | Updated | Added entry for this session |
| `docs/sessions/2026-02-10-agentic-system-plan.md` | Created | This session log |

## Prior Documents Reviewed

The following existing documents were reviewed and built upon:

- `docs/plans/agentic-framework-assessment.md` -- LangGraph SoC optimization loop design
- `docs/plans/agentic-tool-architecture.md` -- Tool granularity and verdict-first schema
- `docs/plans/agentic-ai-dynamics.md` -- Tool selection strategy
- `docs/plans/crew-ai/crew-ai-option-claude.md` -- CrewAI implementation sketch
- `docs/plans/langgraph-migration-plan.md` -- Migration architecture
- `docs/plans/target-system-architecture.md` -- Heterogeneous system model
- `docs/plans/prompt-test-suite-architecture.md` -- Test framework
- `docs/plans/roadmap.md` -- Product roadmap
- `docs/plans/self-reflection.md` -- Agent self-awareness limitations

## Decisions Made

1. **LangGraph over CrewAI** for orchestration runtime (reaffirmed from prior sessions)
2. **Specialist agents as graph nodes**, not CrewAI crew members
3. **Task graph engine** as the foundation layer (must be built first)
4. **Planner agent** as the "brain" that decomposes goals into DAGs
5. **Experience cache** for cross-session learning (not just in-memory history)
6. **Governance layer** is non-negotiable for safety-critical applications
7. **Evaluation framework** must be built alongside the system, not after

## Next Steps

1. Begin Phase 0: Define and implement the `SoCDesignState` TypedDict
2. Implement `TaskGraph` with dependency tracking and status management
3. Build the Planner agent with goal decomposition capability
4. Implement the Dispatcher to replace the hardcoded Orchestrator
5. Validate with Demo 1 (delivery drone SoC goal decomposition)
