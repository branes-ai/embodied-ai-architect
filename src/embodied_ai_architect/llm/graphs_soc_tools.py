"""SoC study tools from branes-ai/graphs, for the LLM orchestrator.

graphs#269 built an SoC study framework: it composes IP blocks at a process
node, schedules a pipeline workload's mission profiles on the result, rolls
up power, and sweeps designs with bound-aware Pareto reports. graphs exposes
it as three MCP tools (``graphs.mcp.soc_tools``):

- list_soc_designs: designs, nodes, efficiency tables, studies, profiles
- analyze_soc:      one SoC design on one mission profile
- sweep_soc_study:  a study, with bound-aware Pareto / union-of-regimes

Unlike ``graphs_tools``, which hand-writes schemas around graphs' Python API,
these adapters take the definitions *from* graphs and route calls through
``graphs.mcp.server.execute_mcp_tool``, so the schemas cannot drift from the
implementation and errors come back as JSON rather than exceptions.

The tools report lower bounds, open (null) feasibility and withheld TOPS/W
where graphs' inputs have gaps -- their descriptions say so, and
``confidence_summary.limited_by`` names each gap. When the installed graphs
predates the SoC tools, this module offers none.
"""

from __future__ import annotations

from typing import Any, Callable

try:
    from graphs.mcp.server import execute_mcp_tool
    from graphs.mcp.soc_tools import SOC_HANDLERS, soc_tool_definitions

    HAS_GRAPHS_SOC = True
except ImportError:
    HAS_GRAPHS_SOC = False
    execute_mcp_tool = None
    SOC_HANDLERS = {}
    soc_tool_definitions = None


def get_graphs_soc_tool_definitions() -> list[dict[str, Any]]:
    """The SoC tools' definitions as graphs states them, or none."""
    return soc_tool_definitions() if HAS_GRAPHS_SOC else []


def _executor(name: str) -> Callable[..., str]:
    def run(**kwargs: Any) -> str:
        return execute_mcp_tool(name, kwargs)

    run.__name__ = name
    run.__doc__ = f"Run graphs' {name} MCP tool; returns its JSON result."
    return run


def create_graphs_soc_tool_executors() -> dict[str, Callable[..., str]]:
    """One executor per SoC tool, each returning graphs' JSON string."""
    if not HAS_GRAPHS_SOC:
        return {}
    return {name: _executor(name) for name in SOC_HANDLERS}
