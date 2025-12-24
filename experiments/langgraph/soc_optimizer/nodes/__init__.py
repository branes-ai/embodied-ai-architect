"""
LangGraph Nodes for SoC Optimization.

Each node is a function that takes SoCDesignState and returns state updates.
Nodes are connected via conditional edges based on the 'next_action' field.

Node Types:
- architect: LLM-powered RTL optimization
- linter: Fast syntax checking
- synthesis: EDA tool execution
- validation: Functional simulation
- critic: PPA analysis and routing decisions
"""

from .architect import create_architect_node
from .linter import create_linter_node
from .synthesis import create_synthesis_node
from .validation import create_validation_node
from .critic import create_critic_node

__all__ = [
    "create_architect_node",
    "create_linter_node",
    "create_synthesis_node",
    "create_validation_node",
    "create_critic_node",
]
