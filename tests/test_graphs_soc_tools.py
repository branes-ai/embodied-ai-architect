"""graphs' SoC study tools as orchestrator tools (graphs#269 PR 4.4 follow-up)."""

import json

import pytest

from embodied_ai_architect.llm import graphs_soc_tools
from embodied_ai_architect.llm.graphs_soc_tools import (
    HAS_GRAPHS_SOC,
    create_graphs_soc_tool_executors,
    get_graphs_soc_tool_definitions,
)

SOC_TOOLS = {"list_soc_designs", "analyze_soc", "sweep_soc_study"}

needs_graphs_soc = pytest.mark.skipif(
    not HAS_GRAPHS_SOC, reason="installed graphs predates the SoC MCP tools"
)


def test_without_graphs_soc_there_are_no_tools(monkeypatch):
    monkeypatch.setattr(graphs_soc_tools, "HAS_GRAPHS_SOC", False)
    assert graphs_soc_tools.get_graphs_soc_tool_definitions() == []
    assert graphs_soc_tools.create_graphs_soc_tool_executors() == {}


@needs_graphs_soc
def test_definitions_are_graphs_own():
    """Taken from graphs, not re-written here, so they cannot drift."""
    from graphs.mcp.soc_tools import soc_tool_definitions

    assert get_graphs_soc_tool_definitions() == soc_tool_definitions()
    assert {d["name"] for d in get_graphs_soc_tool_definitions()} == SOC_TOOLS


@needs_graphs_soc
def test_the_orchestrator_offers_and_can_run_them():
    from embodied_ai_architect.llm.tools import create_tool_executors, get_tool_definitions

    names = [d["name"] for d in get_tool_definitions()]
    assert SOC_TOOLS <= set(names)
    assert all(names.count(n) == 1 for n in SOC_TOOLS)
    assert SOC_TOOLS <= set(create_tool_executors())


@needs_graphs_soc
def test_analyze_soc_returns_graphs_json_with_bounds():
    run = create_graphs_soc_tool_executors()["analyze_soc"]
    out = json.loads(run(design="orin_class_reference", profile="far flight"))
    assert out["die"]["area_is_lower_bound"] is True
    assert out["summary"]["feasible"] is False
    assert out["confidence_summary"]["limited_by"]


@needs_graphs_soc
def test_errors_come_back_as_json_not_exceptions():
    run = create_graphs_soc_tool_executors()["analyze_soc"]
    out = json.loads(run(design="nope", profile="far flight"))
    assert out["tool"] == "analyze_soc" and "no SoC design" in out["error"]


@needs_graphs_soc
def test_list_soc_designs():
    out = json.loads(create_graphs_soc_tool_executors()["list_soc_designs"]())
    assert "orin_class_reference" in {d["id"] for d in out["designs"]}


def test_tool_names_are_unique():
    """The Messages API rejects a tool list with a repeated name (400,
    'Tool names must be unique'), which disabled every tool whenever graphs
    was installed: base and graphs both define list_available_hardware."""
    from embodied_ai_architect.llm.tools import create_tool_executors, get_tool_definitions

    names = [d["name"] for d in get_tool_definitions()]
    assert len(names) == len(set(names))
    assert set(names) <= set(create_tool_executors())
