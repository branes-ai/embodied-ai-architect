# RTL Synthesis Timeout Guide

The Yosys synthesis timeout is **complexity-adaptive by default** and **configurable at every layer**. Results are tagged with their source so callers can distinguish real PPA from heuristic estimates.

## Timeout Tiers

When no explicit timeout is set, `RTLSynthesisTool.estimate_timeout()` classifies the RTL and picks a tier:

| Tier | Criteria | Timeout |
|------|----------|---------|
| Simple | <100 lines, <5 always blocks, no inferred memories | 30s |
| Medium | <500 lines, no inferred memories | 120s |
| Complex | Everything else (large designs, memories) | 300s |

"Inferred memories" means `reg [N:0] name [M:0]` patterns that Yosys expands during the `memory` pass.

## Usage

### Direct — `RTLSynthesisTool`

```python
from embodied_ai_architect.graphs.eda_tools import RTLSynthesisTool

# Default: auto-detect timeout from RTL complexity
tool = RTLSynthesisTool(process_nm=7)
result = tool.run(rtl_source, "mac_unit")

# Fixed timeout for known-large designs
tool = RTLSynthesisTool(process_nm=7, timeout=300)
result = tool.run(rtl_source, "cache_controller")
```

### Through `EDAToolchain`

```python
from embodied_ai_architect.graphs.eda_tools import EDAToolchain

# Auto timeout (default)
toolchain = EDAToolchain(process_nm=28)

# Override for a session with large designs
toolchain = EDAToolchain(process_nm=7, synth_timeout=300)

result = toolchain.synthesize(rtl_source, "my_module")
```

### Through `RTLLoopConfig`

```python
from embodied_ai_architect.graphs.rtl_loop import run_rtl_loop, RTLLoopConfig

# Auto timeout (default)
config = RTLLoopConfig(process_nm=7)

# Fixed timeout
config = RTLLoopConfig(process_nm=7, synth_timeout=300)

result = run_rtl_loop("my_accelerator", rtl_source, config)
```

### Inspecting the Estimated Timeout

```python
from embodied_ai_architect.graphs.eda_tools import RTLSynthesisTool

timeout = RTLSynthesisTool.estimate_timeout(rtl_source)
print(f"Would use {timeout}s timeout")
```

## Detecting Mock Results

When Yosys is missing or times out, synthesis falls back to a heuristic estimator. The result is tagged so you can tell:

```python
result = toolchain.synthesize(rtl_source, "my_module")

if result["source"] == "mock":
    print("Heuristic estimate — not real synthesis")
    print("Reason:", result["warnings"])
elif result["source"] == "yosys":
    print("Real Yosys synthesis")
```

The `source` field is always one of:

| Value | Meaning |
|-------|---------|
| `"yosys"` | Real Yosys synthesis completed within the timeout |
| `"mock"` | Heuristic fallback — Yosys not installed or timed out |

Mock results include descriptive warnings:

- `"Using mock synthesis (reason: yosys_not_found)"` — Yosys binary not on `$PATH`
- `"Using mock synthesis (reason: timeout_120s)"` — Yosys killed after 120s
- `"Metrics are heuristic estimates — real synthesis timed out"` — appended on timeout

## The `AUTO_TIMEOUT` Sentinel

All layers default to `AUTO_TIMEOUT` (-1), which triggers the complexity estimator. You can import it explicitly:

```python
from embodied_ai_architect.graphs.eda_tools import AUTO_TIMEOUT

# These are equivalent:
tool = RTLSynthesisTool(process_nm=7)
tool = RTLSynthesisTool(process_nm=7, timeout=AUTO_TIMEOUT)
```

Pass any positive integer to override:

```python
tool = RTLSynthesisTool(process_nm=7, timeout=600)  # 10 minutes
```

## Mock Estimator Accuracy

The mock heuristic counts `always` blocks, `assign` statements, and operators to estimate cell count. It is intentionally coarse — expect 2-10x error vs real synthesis. Always check `result["source"]` before making PPA-based decisions.
