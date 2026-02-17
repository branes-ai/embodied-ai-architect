"""EDA tool wrappers for RTL lint, synthesis, and simulation.

Ported from experiments/langgraph/soc_optimizer/tools/ with enhancements:
- Process-aware area scaling via technology.py
- Unified EDAToolchain facade
- 3-tier fallback: real tool → alternative tool → mock

All tools work without any EDA tools installed (mock fallback).
"""

from embodied_ai_architect.graphs.eda_tools.lint import RTLLintTool, LintResult
from embodied_ai_architect.graphs.eda_tools.synthesis import (
    AUTO_TIMEOUT,
    RTLSynthesisTool,
    SynthesisResult,
)
from embodied_ai_architect.graphs.eda_tools.simulation import SimulationTool, SimulationResult
from embodied_ai_architect.graphs.eda_tools.toolchain import EDAToolchain

__all__ = [
    "AUTO_TIMEOUT",
    "RTLLintTool",
    "LintResult",
    "RTLSynthesisTool",
    "SynthesisResult",
    "SimulationTool",
    "SimulationResult",
    "EDAToolchain",
]
