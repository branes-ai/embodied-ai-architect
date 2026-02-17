# Session: Complexity-Based Synthesis Timeout Design

**Date:** 2026-02-17
**Commits:** `8c9938b`, `cb01d30`

## Summary

Redesigned the Yosys synthesis timeout strategy. The previous hardcoded 30s timeout silently fell back to mock synthesis for any non-trivial design, producing heuristic PPA metrics that could be 10x off from real synthesis. Replaced with a 3-tier complexity-based estimator, configurable override at every layer, and explicit mock-vs-real result tagging.

## What Was Built

### Complexity-Based Timeout Estimator (`eda_tools/synthesis.py`)

`RTLSynthesisTool.estimate_timeout()` classifies RTL by three complexity signals:

| Signal | Source |
|--------|--------|
| Line count | `len(rtl_source.splitlines())` |
| Always blocks | `rtl_source.count("always")` |
| Inferred memories | `reg [...] name [...]` pattern matches |

**Timeout tiers:**

| Tier | Criteria | Timeout |
|------|----------|---------|
| Simple | <100 lines, <5 always blocks, no memories | 30s |
| Medium | <500 lines, no inferred memories | 120s |
| Complex | Everything else | 300s |

### Configurable Timeout Threading

Timeout is configurable at every layer, defaulting to `AUTO_TIMEOUT` (-1) which triggers the complexity estimator:

| Layer | Parameter | Default |
|-------|-----------|---------|
| `RTLSynthesisTool(timeout=)` | `timeout` | `AUTO_TIMEOUT` |
| `EDAToolchain(synth_timeout=)` | `synth_timeout` | `AUTO_TIMEOUT` |
| `RTLLoopConfig(synth_timeout=)` | `synth_timeout` | `-1` |

### Explicit Mock Fallback Annotation

`SynthesisResult` gained a `source` field (`"yosys"` or `"mock"`) so callers can distinguish real PPA from heuristic estimates. Mock results carry reason-specific warnings:

- `"Using mock synthesis (reason: yosys_not_found)"`
- `"Using mock synthesis (reason: timeout_120s)"`
- `"Metrics are heuristic estimates -- real synthesis timed out"`

## Files Modified

| File | Change |
|------|--------|
| `src/embodied_ai_architect/graphs/eda_tools/synthesis.py` | `AUTO_TIMEOUT` sentinel, `estimate_timeout()`, `_resolve_timeout()`, `source` field on `SynthesisResult`, `reason` parameter on `_mock_synthesis()`, logging |
| `src/embodied_ai_architect/graphs/eda_tools/toolchain.py` | `synth_timeout` parameter threaded to `RTLSynthesisTool` |
| `src/embodied_ai_architect/graphs/eda_tools/__init__.py` | Export `AUTO_TIMEOUT` |
| `src/embodied_ai_architect/graphs/rtl_loop.py` | `synth_timeout` field on `RTLLoopConfig`, passed to `EDAToolchain` |

## Test Results

All 17 existing EDA/RTL tests pass (no new tests added — existing `test_eda_tools.py`, `test_rtl_loop.py`, `test_rtl_integration.py` cover the changed paths through mock fallback).

## Design Rationale

The previous approach had two problems:
1. **Silent data corruption**: 30s timeout + mock fallback meant medium/large designs got fake PPA numbers with no indication they were fake.
2. **One-size-fits-all**: A simple counter and a full cache controller shouldn't share the same timeout.

The experiment copies (`experiments/langgraph/`, `experiments/crew-ai/`) were intentionally left at their hardcoded 120s since they are independent prototypes.
