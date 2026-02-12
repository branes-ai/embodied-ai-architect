"""High-level runner for the SoC design optimization loop.

Wraps the compiled LangGraph StateGraph with sensible defaults,
optional checkpointing, and convenient run/resume methods.

Usage:
    from embodied_ai_architect.graphs.soc_runner import SoCDesignRunner

    runner = SoCDesignRunner()
    result = runner.run(
        goal="Design a drone SoC: <5W, <33ms, <$30",
        use_case="delivery_drone",
        platform="drone",
        constraints=DesignConstraints(max_power_watts=5.0, max_latency_ms=33.3),
    )
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from embodied_ai_architect.graphs.governance import GovernanceGuard, GovernancePolicy
from embodied_ai_architect.graphs.planner import PlannerNode
from embodied_ai_architect.graphs.soc_state import (
    DesignConstraints,
    SoCDesignState,
    create_initial_soc_state,
    get_iteration_summary,
)
from embodied_ai_architect.graphs.specialists import create_default_dispatcher

logger = logging.getLogger(__name__)


class SoCDesignRunner:
    """High-level interface for running the SoC design optimization loop.

    Wraps build_soc_design_graph with sensible defaults and provides
    run(), resume(), and get_state_history() methods.
    """

    def __init__(
        self,
        static_plan: Optional[list[dict[str, Any]]] = None,
        llm: Any = None,
        governance: Optional[GovernancePolicy] = None,
        experience_db_path: Optional[str] = None,
        checkpointer: Any = None,
        recursion_limit: int = 50,
    ) -> None:
        """Initialize the runner.

        Args:
            static_plan: Pre-built plan for deterministic mode.
            llm: LLM client for dynamic planning. One of static_plan/llm required.
            governance: Governance policy for iteration/budget limits.
            experience_db_path: Path to experience SQLite DB. None = default location.
            checkpointer: LangGraph checkpointer for save/resume.
            recursion_limit: LangGraph recursion limit for the graph.
        """
        self._static_plan = static_plan
        self._llm = llm
        self._governance_policy = governance
        self._experience_db_path = experience_db_path
        self._checkpointer = checkpointer
        self._recursion_limit = recursion_limit
        self._compiled_graph = None
        self._state_history: list[dict[str, Any]] = []

    def _build_graph(self) -> Any:
        """Build and compile the LangGraph StateGraph."""
        from embodied_ai_architect.graphs.soc_graph import build_soc_design_graph

        # Create planner
        if self._static_plan is not None:
            planner = PlannerNode(static_plan=self._static_plan)
        elif self._llm is not None:
            planner = PlannerNode(llm=self._llm)
        else:
            raise ValueError("Either static_plan or llm must be provided")

        # Create dispatcher
        dispatcher = create_default_dispatcher()

        # Create governance
        governance = None
        if self._governance_policy is not None:
            governance = GovernanceGuard(self._governance_policy)

        # Create experience cache
        experience_cache = None
        if self._experience_db_path is not None:
            from embodied_ai_architect.graphs.experience import ExperienceCache

            experience_cache = ExperienceCache(db_path=self._experience_db_path)

        return build_soc_design_graph(
            dispatcher=dispatcher,
            planner=planner,
            governance=governance,
            experience_cache=experience_cache,
            checkpointer=self._checkpointer,
        )

    def run(
        self,
        goal: str,
        constraints: Optional[DesignConstraints] = None,
        use_case: str = "",
        platform: str = "",
        max_iterations: int = 20,
        session_id: Optional[str] = None,
        governance_dict: Optional[dict] = None,
    ) -> SoCDesignState:
        """Run a complete SoC design session.

        Args:
            goal: Natural language design objective.
            constraints: Design constraints.
            use_case: Application type.
            platform: Platform type.
            max_iterations: Maximum optimization iterations.
            session_id: Optional session identifier.
            governance_dict: Optional governance policy dict.

        Returns:
            Final SoCDesignState after optimization completes.
        """
        state = create_initial_soc_state(
            goal=goal,
            constraints=constraints,
            use_case=use_case,
            platform=platform,
            max_iterations=max_iterations,
            session_id=session_id,
            governance=governance_dict,
        )

        graph = self._build_graph()
        self._compiled_graph = graph

        config = {"recursion_limit": self._recursion_limit}
        if self._checkpointer is not None:
            config["configurable"] = {"thread_id": state.get("session_id", "default")}

        logger.info("Starting SoC design session: %s", state.get("session_id"))
        result = graph.invoke(state, config=config)

        self._state_history.append(dict(result))
        logger.info("Session complete: %s", get_iteration_summary(result))

        return result

    def get_state_history(self) -> list[dict[str, Any]]:
        """Get the history of final states from all runs."""
        return list(self._state_history)
