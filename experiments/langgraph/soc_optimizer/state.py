"""
SoC Design State Schema for LangGraph optimization loops.

The state persists across all nodes in the graph and tracks:
- Design artifacts (RTL code, testbenches)
- Constraints and metrics (PPA targets)
- Optimization history for learning
- Routing control for conditional edges
"""

from typing import TypedDict, List, Optional, Literal, Any
from dataclasses import dataclass, field, asdict
from enum import Enum


class ActionType(str, Enum):
    """Possible next actions in the optimization loop."""
    ARCHITECT = "architect"      # Generate/modify RTL
    LINT = "lint"               # Syntax check
    SYNTHESIZE = "synthesize"   # Run EDA synthesis
    VALIDATE = "validate"       # Functional simulation
    CRITIQUE = "critique"       # Analyze PPA results
    SIGN_OFF = "sign_off"       # Success - exit loop
    ABORT = "abort"             # Failure - exit loop


@dataclass
class PPAMetrics:
    """Power, Performance, Area metrics from EDA tools."""
    # Timing
    wns_ps: Optional[float] = None          # Worst Negative Slack (picoseconds)
    tns_ps: Optional[float] = None          # Total Negative Slack
    clock_period_ns: Optional[float] = None # Achieved clock period

    # Area
    area_cells: Optional[int] = None        # Cell count
    area_um2: Optional[float] = None        # Physical area in um^2
    num_wires: Optional[int] = None         # Wire count
    cell_breakdown: Optional[dict] = None   # Cell type counts

    # Power (optional, requires more advanced tools)
    power_mw: Optional[float] = None        # Total power
    leakage_mw: Optional[float] = None      # Leakage power
    dynamic_mw: Optional[float] = None      # Dynamic power

    # Synthesis metadata
    critical_path: Optional[str] = None     # Critical path description

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class DesignConstraints:
    """Target constraints for the SoC design."""
    target_clock_ns: float = 1.0            # Target clock period in ns
    max_area_cells: Optional[int] = None    # Maximum cell count
    max_area_um2: Optional[float] = None    # Maximum physical area
    max_power_mw: Optional[float] = None    # Maximum power

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class OptimizationStep:
    """Record of a single optimization step."""
    iteration: int
    agent: str                              # Which agent acted
    action: str                             # What action was taken
    reasoning: Optional[str] = None         # LLM reasoning (if any)
    metrics_before: Optional[dict] = None   # Metrics before change
    metrics_after: Optional[dict] = None    # Metrics after change
    success: bool = True
    error: Optional[str] = None
    duration_ms: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


class SoCDesignState(TypedDict):
    """
    The persistent state for an SoC optimization run.

    This state flows through all nodes in the LangGraph and is
    updated by each node based on its analysis or transformations.
    """

    # =========== Design Artifacts ===========
    rtl_code: str                           # Current Verilog source
    top_module: str                         # Top-level module name
    testbench: Optional[str]                # Testbench for validation

    # =========== Constraints and Metrics ===========
    constraints: dict                       # Target PPA (DesignConstraints)
    baseline_metrics: Optional[dict]        # Original design metrics (PPAMetrics)
    current_metrics: Optional[dict]         # Latest synthesis results (PPAMetrics)

    # =========== Optimization State ===========
    iteration: int                          # Current iteration count
    max_iterations: int                     # Safety limit to prevent infinite loops
    history: List[dict]                     # Log of all OptimizationSteps

    # =========== Routing Control ===========
    next_action: str                        # ActionType - controls graph routing

    # =========== Error Tracking ===========
    last_error: Optional[str]               # Most recent error message
    lint_errors: List[str]                  # Accumulated lint errors
    synthesis_errors: List[str]             # Accumulated synthesis errors
    validation_errors: List[str]            # Accumulated validation errors

    # =========== Metadata ===========
    project_id: Optional[str]               # Unique ID for checkpointing
    created_at: Optional[str]               # ISO timestamp

    # =========== systars Integration (optional) ===========
    systars_config: Optional[dict]          # SystolicConfig if using systars


def create_initial_state(
    rtl_code: str,
    top_module: str,
    constraints: DesignConstraints,
    testbench: Optional[str] = None,
    max_iterations: int = 10,
    project_id: Optional[str] = None,
    systars_config: Optional[dict] = None,
) -> SoCDesignState:
    """Create an initial state for a new optimization run."""
    from datetime import datetime

    return SoCDesignState(
        # Design artifacts
        rtl_code=rtl_code,
        top_module=top_module,
        testbench=testbench,

        # Constraints
        constraints=constraints.to_dict(),
        baseline_metrics=None,
        current_metrics=None,

        # Optimization state
        iteration=0,
        max_iterations=max_iterations,
        history=[],

        # Routing
        next_action=ActionType.LINT.value,

        # Errors
        last_error=None,
        lint_errors=[],
        synthesis_errors=[],
        validation_errors=[],

        # Metadata
        project_id=project_id or f"opt_{top_module}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        created_at=datetime.now().isoformat(),

        # systars
        systars_config=systars_config,
    )


def state_summary(state: SoCDesignState) -> str:
    """Generate a human-readable summary of the current state."""
    lines = [
        f"=== SoC Optimization State ===",
        f"Project: {state.get('project_id', 'unknown')}",
        f"Module: {state['top_module']}",
        f"Iteration: {state['iteration']}/{state['max_iterations']}",
        f"Next Action: {state['next_action']}",
    ]

    if state.get('baseline_metrics'):
        bm = state['baseline_metrics']
        lines.append(f"Baseline: {bm.get('area_cells', 'N/A')} cells")

    if state.get('current_metrics'):
        cm = state['current_metrics']
        lines.append(f"Current: {cm.get('area_cells', 'N/A')} cells")

        # Calculate improvement
        if state.get('baseline_metrics') and cm.get('area_cells'):
            baseline = state['baseline_metrics'].get('area_cells', 0)
            current = cm['area_cells']
            if baseline > 0:
                improvement = (baseline - current) / baseline * 100
                lines.append(f"Improvement: {improvement:+.1f}%")

    if state.get('last_error'):
        lines.append(f"Last Error: {state['last_error']}")

    return "\n".join(lines)
