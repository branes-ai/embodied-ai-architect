# Session: Manufacturing Cost Model + Process Node Optimizer

**Date:** 2026-02-18
**Parent Commit:** `c8eded5`

## Problem

The SoC design optimizer could not meaningfully optimize cost because:

1. **Cost was a static BOM lookup**: `cost_usd = selected_hw.get("cost_usd", 0)` — the catalog price of a COTS accelerator chip, not the custom SoC being designed.
2. **No strategy could reduce cost**: `smaller_model` listed `"cost"` in `applicable_when` but only reduced GFLOPS — `cost_usd` never changed.
3. **Process node was not an optimizer dimension**: moving from 28nm to 40nm would massively reduce cost (cheaper wafers, lower NRE) at the expense of larger area and more power — a real engineering tradeoff the optimizer could not explore.

## Solution

### Manufacturing Cost Model (`manufacturing.py`)

Replaced the static BOM lookup with a physics-based cost model:

```
total_unit_cost = die_cost + package + test + NRE/volume
where:
  die_cost     = wafer_cost / (dies_per_wafer × yield)     [variable]
  NRE/volume   = (mask_set + design_nre + test_nre) / vol  [fixed, amortized]
```

Key components:
- **Dies per wafer**: Circular packing formula for 300mm wafer (3mm edge exclusion)
- **Yield**: Murphy's model `Y = ((1 - e^(-D·A)) / (D·A))²` — more conservative than Poisson
- **NRE**: Three components — mask sets, design NRE (EDA/engineering/IP/verification), test NRE (probe cards, qualification)
- **Process economics database**: 16 nodes from 2nm to 180nm with wafer cost, defect density, and all three NRE components

### NRE Decomposition

The original plan had mask-set-only NRE, which was unrealistically cheap. Full NRE at 28nm:

| Component | Cost | /unit at 100K |
|-----------|------|---------------|
| Mask sets | $2M | $20 |
| Design NRE | $5M | $50 |
| Test NRE | $500K | $5 |
| **Total NRE** | **$7.5M** | **$75** |

This makes cost fail the $30 constraint at 100K volume on 28nm — forcing the optimizer to explore process-node tradeoffs or volume changes.

### Process Node Optimizer Strategies

Two new strategies in `optimizer.py`:

| Strategy | Applies when | Effect |
|----------|-------------|--------|
| `shrink_process_node` | power, latency, area FAIL | Move to next smaller node (e.g. 28→22nm) — lower power/area, higher NRE |
| `grow_process_node` | cost FAIL | Move to next larger node (e.g. 28→40nm) — lower NRE, larger die |

Both use `get_adjacent_nodes()` in `technology.py` to traverse `_AVAILABLE_NODES`. Dynamic strategy naming (`grow_process_node_28`) allows multiple applications (28→40→65).

Removed `"cost"` from `smaller_model.applicable_when` since reducing GFLOPS cannot reduce manufacturing cost.

### Failure Analysis in Demo Output

Replaced generic "STOPPED (limit reached)" with actionable diagnostics:
- Per-constraint failure analysis (e.g., "NRE is 90% of unit cost")
- Break-even volume calculation
- Specific suggestions (increase volume, relax constraint, use COTS)

## Cost Model Tradeoffs

Example: 26.4mm² die, 100K volume, same logic at different nodes:

| Node | Die cost | NRE/unit | Total | Constraint ($30) |
|------|----------|----------|-------|-------------------|
| 28nm | $1.67 | $75.00 | $78 | FAIL |
| 40nm | $3.09 | $43.00 | $48 | FAIL |
| 65nm | $7.63 | $22.00 | $31 | FAIL (barely) |
| 28nm @ 1M | $1.67 | $7.50 | $11 | PASS |
| 40nm @ 1M | $3.09 | $4.30 | $9 | PASS |

Key insight: at typical embedded volumes (100K), NRE dominates. Volume is the primary lever, not process node.

## Files Changed

| File | Change |
|------|--------|
| `src/.../graphs/manufacturing.py` | **NEW** — cost model: `PROCESS_ECONOMICS` (16 nodes), `ManufacturingCostBreakdown`, `dies_per_wafer()`, `murphy_yield()`, `estimate_manufacturing_cost()` |
| `src/.../graphs/technology.py` | Added `get_adjacent_nodes()` |
| `src/.../graphs/soc_state.py` | Added `cost_breakdown: Optional[dict]` to `PPAMetrics` |
| `src/.../graphs/specialists.py` | Replaced `cost_usd = selected_hw.get("cost_usd", 0)` with `estimate_manufacturing_cost()` |
| `src/.../graphs/optimizer.py` | Added `shrink_process_node` / `grow_process_node` strategies, `"constraints"` handling in `_apply_strategy()`, dynamic naming, removed `"cost"` from `smaller_model` |
| `examples/demo_soc_designer.py` | Cost breakdown display, process node strategy labels, failure analysis |
| `examples/demo_dse_pareto.py` | Cost breakdown display |
| `examples/demo_soc_optimizer.py` | Cost breakdown display, convergence failure analysis with break-even volume |
| `examples/demo_kpu_rtl.py` | Cost breakdown display |
| `examples/demo_hitl_safety.py` | Cost breakdown display |
| `examples/demo_full_campaign.py` | Cost breakdown display |
| `tests/test_manufacturing.py` | **NEW** — 14 tests (dies_per_wafer, murphy_yield, cost estimation, volume scaling, node coverage) |
| `tests/test_specialists.py` | 3 new tests: cost_breakdown_present, cost_uses_manufacturing_model, volume_affects_cost |
| `tests/test_optimizer.py` | 4 new tests: grow/shrink strategies, constraint updates, dynamic naming |

## Test Results

- 76/76 passed (manufacturing + specialists + optimizer + acceptance)
- 514/514 full suite passed (0 failures; 39 pre-existing errors from optional deps)
