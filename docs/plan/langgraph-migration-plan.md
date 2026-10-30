# LangGraph Migration Plan for SoC Designer

## Overview

This document outlines the migration from the current sequential `Orchestrator` pattern to a LangGraph-based state machine for SoC optimization workflows.

## Why LangGraph Over CrewAI

| Aspect | CrewAI | LangGraph |
|--------|--------|-----------|
| **Execution Model** | Role-based agents in sequence/hierarchy | Explicit state graph with conditional edges |
| **State Management** | Implicit via context passing | Explicit `TypedDict` state with full control |
| **Cycles/Loops** | Awkward (Flows try to add this) | Native `add_conditional_edges` |
| **Checkpointing** | Not built-in | PostgreSQL/Redis checkpointers |
| **Human-in-the-loop** | Manual implementation | `interrupt_before` built-in |
| **Long-running jobs** | Poor support | Async + checkpointing designed for this |
| **Debugging** | Limited | "Time travel" state inspection |

## Architecture

### Current Architecture
```
Orchestrator.process()
    → ModelAnalyzer.execute()
    → HardwareProfile.execute()
    → Benchmark.execute()
    → ReportSynthesis.execute()
```

### Target Architecture (LangGraph)
```
StateGraph[SoCDesignState]
    ┌─────────────┐
    │  architect  │ ←──────────────┐
    └──────┬──────┘                │
           ↓                       │
    ┌─────────────┐                │
    │   linter    │ ───(fail)──────┤
    └──────┬──────┘                │
           ↓ (pass)                │
    ┌─────────────┐                │
    │  synthesis  │ ───(fail)──────┤
    └──────┬──────┘                │
           ↓ (pass)                │
    ┌─────────────┐                │
    │ validation  │ ───(fail)──────┤
    └──────┬──────┘                │
           ↓ (pass)                │
    ┌─────────────┐                │
    │   critic    │ ───(iterate)───┘
    └──────┬──────┘
           ↓ (success)
         [END]
```

## State Schema

```python
class SoCDesignState(TypedDict):
    # Design artifacts
    rtl_code: str                         # Current Verilog source
    top_module: str                       # Top-level module name
    testbench: Optional[str]              # Testbench for validation

    # Constraints and metrics
    constraints: dict                     # Target PPA
    baseline_metrics: Optional[dict]      # Original design metrics
    current_metrics: Optional[dict]       # Latest synthesis results

    # Optimization state
    iteration: int                        # Current iteration count
    max_iterations: int                   # Safety limit
    history: List[dict]                   # Log of all changes

    # Routing control
    next_action: Literal[
        "architect", "lint", "synthesize",
        "validate", "critique", "sign_off", "abort"
    ]

    # Error tracking
    last_error: Optional[str]
    lint_errors: List[str]
    synthesis_errors: List[str]
```

## Node Implementations

### 1. Architect Node
- Uses LLM to analyze current design and propose optimizations
- Strategies: pipelining, resource sharing, logic restructuring
- Outputs modified RTL code

### 2. Linter Node (Fast-Fail)
- Uses Verilator `--lint-only` for sub-second feedback
- Catches syntax errors before expensive synthesis
- Routes back to architect on failure

### 3. Synthesis Node
- Uses Yosys for gate-level synthesis
- Extracts PPA metrics (area, cell count)
- Stores baseline on first run

### 4. Validation Node
- Uses Icarus Verilog for functional simulation
- Ensures optimized design is functionally correct
- Catches semantic errors LLM might introduce

### 5. Critic Node
- Analyzes PPA metrics against constraints
- Determines if design meets targets or needs iteration
- Routes to architect for another attempt or to END on success

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Add `langgraph` to dependencies
- [ ] Create `src/embodied_ai_architect/langgraph/` package
- [ ] Implement `SoCDesignState` schema
- [ ] Port existing tools to work as LangGraph nodes

### Phase 2: Core Graph (Week 3-4)
- [ ] Implement the 5 core nodes
- [ ] Build state graph with conditional routing
- [ ] Add basic CLI command
- [ ] Test with RISC-V ALU example

### Phase 3: Production Features (Week 5-6)
- [ ] Add PostgreSQL checkpointing
- [ ] Implement human-in-the-loop breakpoints
- [ ] Add WebSocket streaming for progress
- [ ] Create experience cache

### Phase 4: Integration (Week 7-8)
- [ ] Integrate with existing CLI
- [ ] Add visualization (LangSmith or custom)
- [ ] Containerize EDA tools
- [ ] Documentation

## Test Case: RISC-V ALU

The initial test case is a simple ALU with known optimization opportunities:

```verilog
module alu_core (
    input [31:0] a, b, c, d,
    output [31:0] result
);
    // Deep combinational path - will fail timing at high freq
    assign result = (a * b) + (c * d);
endmodule
```

Expected optimization: Pipeline the multipliers to meet timing.

## Dependencies

```toml
[project.optional-dependencies]
langgraph = [
    "langgraph>=0.2.0",
    "langchain-anthropic>=0.3.0",
    "langgraph-checkpoint-postgres>=0.1.0",
]
```

## Related Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Agentic Framework Assessment](./agentic-framework-assessment.md)
- [systars HDL Library](https://github.com/stillwater-sc/systars) - Reusable HDL components
