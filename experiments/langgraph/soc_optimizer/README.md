# LangGraph SoC Optimizer

A LangGraph-based optimization loop for SoC (System-on-Chip) design exploration.

## Overview

This experiment implements the optimization loop described in the [LangGraph Migration Plan](../../../docs/plans/langgraph-migration-plan.md). It demonstrates:

1. **State Machine Architecture**: Explicit graph with conditional routing
2. **Tool Integration**: Yosys synthesis, Verilator/Icarus simulation
3. **Iterative Optimization**: Architect → Lint → Synthesis → Validate → Critique loop
4. **LLM Integration**: Optional Claude-powered RTL architect

## Architecture

```
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

## Quick Start

```bash
# Navigate to experiment directory
cd experiments/langgraph/soc_optimizer

# Run with mock mode (no LLM, just tools)
python workflow.py

# Run with LLM (requires ANTHROPIC_API_KEY)
export ANTHROPIC_API_KEY=your-key-here
python workflow.py --with-llm

# Run simple loop (no LangGraph dependency)
python workflow.py --simple
```

## Requirements

### Required (EDA Tools)
- **Yosys**: For synthesis (`apt install yosys` or OSS-CAD-SUITE)
- **Verilator** or **Icarus Verilog**: For simulation

### Optional (Full LangGraph)
```bash
pip install langgraph langchain-anthropic
```

## Files

```
soc_optimizer/
├── state.py          # SoCDesignState schema
├── graph.py          # LangGraph assembly
├── workflow.py       # Main runner script
├── nodes/            # LangGraph nodes
│   ├── architect.py  # LLM-powered RTL optimization
│   ├── linter.py     # Fast syntax checking
│   ├── synthesis.py  # Yosys synthesis
│   ├── validation.py # Functional simulation
│   └── critic.py     # PPA analysis & routing
├── tools/            # EDA tool wrappers
│   ├── lint.py       # Verilator/Yosys lint
│   ├── synthesis.py  # Yosys synthesis
│   └── simulation.py # Icarus Verilog simulation
└── designs/          # Test designs
    ├── riscv_alu.sv  # RISC-V ALU (optimization target)
    └── riscv_alu_tb.sv # Testbench
```

## Test Design: RISC-V ALU

The test design is a RISC-V-style ALU with intentional optimization opportunities:

1. **Deep combinational logic**: Multiplier creates long critical path
2. **Unshared operations**: Add/subtract could share hardware
3. **Non-pipelined shifter**: Barrel shifter optimization potential

### systars Integration

The ALU includes a MAC (Multiply-Accumulate) operation compatible with
[systars](https://github.com/stillwater-sc/systars) PE components:

```verilog
// MAC: D = A * B + C (systars PE dataflow)
assign mac_result = mul_result[WIDTH-1:0] + accumulator;
```

## Command Line Options

```
python workflow.py --help

Options:
  --with-llm        Use Claude for architect node
  --simple          Use simple loop (no LangGraph)
  --max-iter N      Maximum iterations (default: 5)
  --max-area N      Maximum area constraint in cells
  --target-clock F  Target clock period in ns (default: 1.0)
  --work-dir DIR    Working directory for outputs
  --quiet           Reduce output verbosity
  --no-testbench    Skip functional validation
```

## Example Output

```
============================================================
LangGraph SoC Optimizer Demo
============================================================
Mode: LangGraph
LLM: Mock mode
Max iterations: 5
============================================================

Loaded RTL: 4521 characters
Loaded testbench: 3842 characters
Work directory: /path/to/optimization_run

[architect] → lint
[linter] → synthesize
[synthesis] → validate
[validation] → critique
[critic] → sign_off

=== SoC Optimization State ===
Project: opt_riscv_alu_20241223_150000
Module: riscv_alu
Iteration: 0/5
Next Action: sign_off
Baseline: 1247 cells
Current: 1247 cells

============================================================
OPTIMIZATION RESULTS
============================================================

Status: SUCCESS
Iterations: 0

Baseline Metrics:
  Area: 1247 cells
  Wires: 2494

Final Metrics:
  Area: 1247 cells
  Wires: 2494

  Area Change: +0.0%
============================================================
```

## Extending

### Adding New Optimization Strategies

Edit `nodes/architect.py` to add strategies to `ARCHITECT_SYSTEM_PROMPT`:

```python
ARCHITECT_SYSTEM_PROMPT = """
...
5. **New Strategy**: Description of optimization
   - When to apply
   - Implementation approach
"""
```

### Adding New Constraints

Edit `state.py` to add fields to `DesignConstraints` and update
`nodes/critic.py` to check them.

### Using systars Designs

Generate Verilog from systars and use as input:

```bash
# In systars repo
just gen-pe

# Copy generated Verilog
cp gen/pe.v /path/to/soc_optimizer/designs/systars_pe.sv
```

## Related Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangGraph Migration Plan](../../../docs/plans/langgraph-migration-plan.md)
- [Agentic Framework Assessment](../../../docs/plans/agentic-framework-assessment.md)
- [systars HDL Generator](https://github.com/stillwater-sc/systars)
