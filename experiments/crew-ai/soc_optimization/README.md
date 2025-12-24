# SoC Optimization Agent Evaluation Demo

An example demonstrating trajectory coverage evaluation for LLM-based hardware design optimization.

## Quick Start

```bash
# Run the workflow
python workflow.py
```

## What's Included

| File | Purpose |
|------|---------|
| `designs/alu_baseline.v` | Original RTL (deliberately suboptimal ALU) |
| `designs/alu_tb.v` | Testbench for functional verification |
| `designs/constraints.sdc` | Timing constraints |
| `tools.py` | EDA tool wrappers (lint, synthesis, simulation) |
| `pipeline.py` | Agent workflow with mock LLM responses |
| `trajectory_eval.py` | Trajectory analysis and evaluation |
| `workflow.py` | Complete workflow demonstration script |

## What It Demonstrates

1. **Buggy LLM run**: Simulates common LLM failure (breaks shifter functionality)
2. **Correct LLM run**: Shows successful optimization
3. **Trajectory analysis**: Both runs score 100/100 on trajectory (same process)
4. **Outcome difference**: Buggy run fails validation, correct run passes

## Key Insight

  - Trajectory evaluation catches **process** issues (wrong tools, bad order).

  - Outcome evaluation catches **result** issues (broken functionality).

You need BOTH to trust an LLM-based optimization system.

## Optional: Real EDA Tools

The demo uses mock tools by default. For real synthesis/simulation, install:

```bash
# Yosys (synthesis)
apt install yosys

# Icarus Verilog (simulation)
apt install iverilog
```

The tools will automatically use real EDA when available.
