# Session: SoC Optimizer Recursion Fix and PPA Analysis

**Date**: 2025-12-23
**Focus**: LangGraph SoC Optimizer debugging and improvements

## Summary

Fixed the GraphRecursionError that was causing the optimization loop to fail after 5 iterations. Verified the full LLM-powered optimization workflow works. Identified key gaps in the current scoring approach that need to be addressed.

## Problem: GraphRecursionError

The optimization loop was hitting LangGraph's default recursion limit of 25 after running through multiple iterations with syntax error retries.

### Root Cause

Each iteration through the optimization loop involves 5 nodes:
- architect → linter → synthesis → validation → critic

When the LLM generates code with syntax errors, a retry loop occurs:
- architect → linter (FAIL) → architect → linter (FAIL) → architect → linter (pass) → ...

This consumes 2 node invocations per syntax error without incrementing the iteration counter.

With 5 iterations × 5 nodes = 25 minimum, plus syntax retries, the limit was exceeded.

### Fix

Added dynamic recursion_limit to the LangGraph config in `graph.py`:

```python
recursion_limit = (max_iterations + 1) * 6 + 10
config = {
    "configurable": {"thread_id": initial_state["project_id"]},
    "recursion_limit": recursion_limit,
}
```

Formula rationale:
- `max_iterations + 1`: Account for baseline + optimization iterations
- `× 6`: 5 nodes per iteration plus buffer for linter retries
- `+ 10`: Additional buffer for safety

## Verification

Tested with various scenarios:

| Test | Max Iter | Area Target | Result |
|------|----------|-------------|--------|
| Mock mode | 3 | 4000 | ABORT (no LLM changes) |
| LLM mode | 5 | 4000 | SUCCESS (1 iteration, 33% reduction) |
| LLM mode | 5 | 4000 | SUCCESS (3 iterations, 73.5% reduction) |

## Issue Identified: Functional Correctness Gap

The 73.5% reduction run revealed a critical problem:

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Area | 4109 cells | 1087 cells |
| Tests Passed | 21 | 18 |

Claude reduced area by removing functionality that broke 3 test cases. The current critic only checks `area_cells > max_area` but doesn't verify functional equivalence.

### Required Improvements

1. **Functional Correctness Gate**: Critic should verify `tests_passed >= baseline_tests_passed` before signing off

2. **Composite PPA Scoring**: Currently only area is checked. Need:
   - Area (cells or µm²)
   - Timing (critical path delay, WNS)
   - Power (dynamic + leakage estimation)

3. **Weighted Score Function**: Allow trade-offs between PPA dimensions based on constraints

## Files Modified

- `experiments/langgraph/soc_optimizer/graph.py` - Added recursion_limit to config

## Next Steps

1. Add test count validation to critic node
2. Implement composite PPA scoring function
3. Consider retry limits per iteration for syntax errors
4. Investigate why some runs produce better LLM output quality than others

## Observations

LLM output quality varies significantly between runs:
- Some runs: incremental improvements (4109 → 4082 → 4049)
- Some runs: aggressive valid optimizations (4109 → 2754)
- Some runs: aggressive invalid optimizations (breaks tests)

The non-deterministic nature of LLM outputs makes reliable optimization challenging. Need guardrails to prevent functional regressions.
