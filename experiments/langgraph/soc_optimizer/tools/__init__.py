"""
EDA Tool Wrappers for SoC Optimization.

These tools wrap real EDA operations (Yosys, Verilator, Icarus Verilog)
and provide structured input/output for LangGraph nodes.

Tools:
- RTLLintTool: Parse and lint Verilog/SystemVerilog
- RTLSynthesisTool: Synthesize to gate-level netlist
- SimulationTool: Run functional simulation with testbench
"""

from .lint import RTLLintTool, LintResult
from .synthesis import RTLSynthesisTool, SynthesisResult
from .simulation import SimulationTool, SimulationResult

__all__ = [
    "RTLLintTool",
    "LintResult",
    "RTLSynthesisTool",
    "SynthesisResult",
    "SimulationTool",
    "SimulationResult",
]
