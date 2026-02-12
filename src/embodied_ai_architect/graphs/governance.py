"""Governance layer for the SoC design optimization loop.

Provides budget tracking, iteration limits, approval gates, and audit logging.
The GovernanceGuard is called by the outer loop (soc_graph) to enforce policy
before each action.

Usage:
    from embodied_ai_architect.graphs.governance import GovernancePolicy, GovernanceGuard

    policy = GovernancePolicy(iteration_limit=10, cost_budget_tokens=100000)
    guard = GovernanceGuard(policy)

    if guard.check_iteration_limit(current_iteration=5):
        print("Within iteration budget")

    guard.record(agent="optimizer", action="apply INT8", iteration=1)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class AuditEntry(BaseModel):
    """Record of a single action taken during design optimization."""

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    agent: str
    action: str
    input_summary: str = ""
    output_summary: str = ""
    cost_tokens: int = 0
    human_approved: bool = False
    iteration: int = 0


class GovernancePolicy(BaseModel):
    """Configuration for design loop governance.

    Controls what requires approval, budget limits, and iteration bounds.
    """

    approval_required_actions: list[str] = Field(
        default_factory=list,
        description="Actions requiring human approval (e.g. 'deploy', 'tape_out')",
    )
    cost_budget_tokens: int = Field(
        default=0,
        description="Total LLM token budget for the session (0 = unlimited)",
    )
    iteration_limit: int = Field(
        default=10,
        description="Maximum optimization iterations allowed",
    )
    require_human_approval_on_fail: bool = Field(
        default=False,
        description="If True, require human approval when design fails constraints after limit",
    )
    fail_iteration_threshold: int = Field(
        default=3,
        description="After this many failing iterations, escalate to human",
    )


class GovernanceGuard:
    """Enforces governance policy during design optimization.

    Tracks cumulative token cost, checks iteration and budget limits,
    and determines when human approval is required.
    """

    def __init__(self, policy: Optional[GovernancePolicy] = None) -> None:
        self.policy = policy or GovernancePolicy()
        self._total_cost_tokens: int = 0
        self._audit_entries: list[AuditEntry] = []

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GovernanceGuard:
        """Create a GovernanceGuard from a serialized policy dict."""
        if not data:
            return cls()
        return cls(policy=GovernancePolicy(**data))

    def check_budget(self, additional_tokens: int = 0) -> bool:
        """Check if the token budget allows the next action.

        Returns True if within budget (or budget is unlimited).
        """
        if self.policy.cost_budget_tokens <= 0:
            return True  # unlimited
        return (self._total_cost_tokens + additional_tokens) <= self.policy.cost_budget_tokens

    def check_iteration_limit(self, current_iteration: int) -> bool:
        """Check if the current iteration is within the allowed limit.

        Returns True if within limit.
        """
        return current_iteration < self.policy.iteration_limit

    def requires_approval(self, action: str) -> bool:
        """Check if an action requires human approval.

        Args:
            action: The action name to check.

        Returns:
            True if the action requires approval per policy.
        """
        return action in self.policy.approval_required_actions

    def should_escalate_to_human(self, consecutive_failures: int) -> bool:
        """Check if we should escalate to human review.

        Args:
            consecutive_failures: Number of consecutive failing iterations.

        Returns:
            True if policy says to escalate after this many failures.
        """
        if not self.policy.require_human_approval_on_fail:
            return False
        return consecutive_failures >= self.policy.fail_iteration_threshold

    def record(
        self,
        agent: str,
        action: str,
        iteration: int = 0,
        input_summary: str = "",
        output_summary: str = "",
        cost_tokens: int = 0,
        human_approved: bool = False,
    ) -> AuditEntry:
        """Record an action in the audit trail.

        Args:
            agent: Name of the agent performing the action.
            action: Description of the action taken.
            iteration: Current optimization iteration.
            input_summary: Brief summary of the input.
            output_summary: Brief summary of the output.
            cost_tokens: Number of LLM tokens consumed (0 for deterministic).
            human_approved: Whether the action was human-approved.

        Returns:
            The created AuditEntry.
        """
        entry = AuditEntry(
            agent=agent,
            action=action,
            iteration=iteration,
            input_summary=input_summary,
            output_summary=output_summary,
            cost_tokens=cost_tokens,
            human_approved=human_approved,
        )
        self._audit_entries.append(entry)
        self._total_cost_tokens += cost_tokens
        return entry

    @property
    def total_cost_tokens(self) -> int:
        """Total tokens consumed so far."""
        return self._total_cost_tokens

    @property
    def audit_entries(self) -> list[AuditEntry]:
        """All audit entries recorded."""
        return list(self._audit_entries)
