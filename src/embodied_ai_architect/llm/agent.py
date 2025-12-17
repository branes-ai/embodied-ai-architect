"""Agentic loop for the Embodied AI Architect.

This module implements a Claude Code-style interactive agent that can
reason about user requests and call tools to accomplish tasks.
"""

import json
from typing import Any, Callable

from .client import LLMClient, LLMResponse
from .tools import get_tool_definitions, create_tool_executors


SYSTEM_PROMPT = """\
You are the Embodied AI Architect, an expert assistant for designing and deploying
AI systems on embedded and edge hardware. You help users:

- Analyze neural network models to understand their structure and requirements
- Recommend optimal hardware targets (Jetson, Coral Edge TPU, FPGA, cloud GPU, etc.)
- Benchmark model performance on different backends
- Optimize deployments for constraints like power, latency, and memory

## Your Approach

When a user asks about deploying a model:
1. First use analyze_model to understand the model's characteristics
2. Consider their constraints (power budget, latency requirements, cost)
3. Use recommend_hardware to suggest appropriate hardware
4. Offer to run_benchmark if they want concrete performance numbers

## Communication Style

- Be direct and technical - users are engineers
- Show your reasoning when making recommendations
- Cite specific numbers from tool results (parameters, TOPS, latency)
- If you don't have enough information, ask clarifying questions
- When tools return errors, explain what went wrong and suggest fixes

## Available Tools

You have access to tools for:
- Model analysis (layer types, parameters, memory requirements)
- Hardware recommendation (matching models to hardware based on constraints)
- Benchmarking (measuring actual inference performance)
- File exploration (listing and reading files)

Use these tools to gather data before making recommendations. Don't guess -
verify with actual measurements when possible.
"""


class ArchitectAgent:
    """Interactive agent for Embodied AI architecture design.

    Implements a tool-use loop similar to Claude Code, where the LLM
    can call tools and reason about results.
    """

    def __init__(
        self,
        llm: LLMClient | None = None,
        tools: dict[str, Callable] | None = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        """Initialize the agent.

        Args:
            llm: LLM client (creates default if not provided)
            tools: Tool executors (creates defaults if not provided)
            max_iterations: Maximum tool-use iterations per turn
            verbose: Print detailed execution info
        """
        self.llm = llm or LLMClient()
        self.tools = tools or create_tool_executors()
        self.tool_definitions = get_tool_definitions()
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.messages: list[dict[str, Any]] = []

    def reset(self) -> None:
        """Clear conversation history."""
        self.messages = []

    def run(
        self,
        user_input: str,
        on_tool_start: Callable[[str, dict], None] | None = None,
        on_tool_end: Callable[[str, str], None] | None = None,
        on_thinking: Callable[[str], None] | None = None,
    ) -> str:
        """Process user input and return response.

        Implements the agentic loop:
        1. Send user message to LLM
        2. If LLM requests tool calls, execute them
        3. Send tool results back to LLM
        4. Repeat until LLM produces final response

        Args:
            user_input: User's message
            on_tool_start: Callback when tool execution starts (name, args)
            on_tool_end: Callback when tool execution ends (name, result)
            on_thinking: Callback for LLM's intermediate thoughts

        Returns:
            Final response text from the LLM
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_input})

        iterations = 0
        while iterations < self.max_iterations:
            iterations += 1

            # Get LLM response
            response = self.llm.chat(
                messages=self.messages,
                tools=self.tool_definitions,
                system=SYSTEM_PROMPT,
            )

            # If there's text and we're thinking, emit it
            if response.text and on_thinking and response.has_tool_calls:
                on_thinking(response.text)

            if response.has_tool_calls:
                # Add assistant message with tool use
                self.messages.append({
                    "role": "assistant",
                    "content": self._format_tool_use_content(response),
                })

                # Execute each tool and collect results
                tool_results = []
                for tool_call in response.tool_calls:
                    if on_tool_start:
                        on_tool_start(tool_call.name, tool_call.args)

                    result = self._execute_tool(tool_call.name, tool_call.args)

                    if on_tool_end:
                        on_tool_end(tool_call.name, result)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_call.id,
                        "content": result,
                    })

                # Add tool results
                self.messages.append({
                    "role": "user",
                    "content": tool_results,
                })
            else:
                # No tool calls - we have the final response
                self.messages.append({
                    "role": "assistant",
                    "content": response.text,
                })
                return response.text

        # Hit max iterations
        return (
            "I've reached the maximum number of tool calls for this turn. "
            "Please let me know if you'd like me to continue."
        )

    def _execute_tool(self, name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return its result.

        Args:
            name: Tool name
            args: Tool arguments

        Returns:
            Tool result as string
        """
        if name not in self.tools:
            return f"Error: Unknown tool '{name}'"

        try:
            executor = self.tools[name]
            result = executor(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

    def _format_tool_use_content(self, response: LLMResponse) -> list[dict[str, Any]]:
        """Format response content including tool use blocks.

        Args:
            response: LLM response

        Returns:
            Content list for message
        """
        content = []

        if response.text:
            content.append({"type": "text", "text": response.text})

        for tool_call in response.tool_calls:
            content.append({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.name,
                "input": tool_call.args,
            })

        return content

    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for debugging.

        Returns:
            Formatted conversation summary
        """
        lines = []
        for msg in self.messages:
            role = msg["role"].upper()
            content = msg["content"]

            if isinstance(content, str):
                preview = content[:100] + "..." if len(content) > 100 else content
                lines.append(f"{role}: {preview}")
            elif isinstance(content, list):
                # Tool use or tool results
                types = [c.get("type", "unknown") for c in content]
                lines.append(f"{role}: [{', '.join(types)}]")

        return "\n".join(lines)
