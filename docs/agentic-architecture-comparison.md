# Agentic AI Architecture Comparison

This document compares three approaches to building agentic AI systems, analyzed in the context of the Embodied AI Architect project.

## Overview

| Approach | Location | Pattern | LLM Support |
|----------|----------|---------|-------------|
| Custom Chatbot | `src/embodied_ai_architect/llm/` | Manual agentic loop | **LLM-agnostic** (swap client) |
| LangGraph | `experiments/langgraph/soc_optimizer/` | Declarative state machine | **LLM-agnostic** (via LangChain) |
| Claude Agent SDK | External dependency | Framework-managed loop | **Claude-only** |

---

## 1. Custom Chatbot (Our Implementation)

**Location**: `src/embodied_ai_architect/llm/`

### Architecture

```
User Input → LLMClient → Response (tool calls?) → Execute → Feed results → Loop
```

### Key Components

- **LLMClient** (`client.py`): Wraps LLM API with tool use support
- **ArchitectAgent** (`agent.py`): Implements the agentic loop
- **Tools** (`tools.py`): Tool definitions and executors

### How It Works

```python
# agent.py - simplified
while iteration < max_iterations:
    response = llm.chat(messages, tools, system)
    if response.tool_calls:
        for tool in response.tool_calls:
            result = execute_tool(tool)
            add_to_messages(result)
    else:
        return response.text
```

### Characteristics

| Aspect | Details |
|--------|---------|
| **Code size** | ~300 LOC |
| **Routing** | LLM-driven (Claude decides which tools) |
| **State** | Message history list |
| **Tool execution** | Sequential, implemented by us |
| **Parallelism** | None (tools run one at a time) |
| **Checkpointing** | Not implemented |
| **LLM support** | Agnostic - swap `LLMClient` implementation |

### LLM Agnosticism

The current implementation uses Anthropic's API, but the abstraction is clean:

```python
class LLMClient:
    def chat(self, messages, tools, system) -> LLMResponse:
        # Currently calls Anthropic API
        # Could be swapped for OpenAI, Gemini, DeepSeek, etc.
```

To support multiple LLMs:
1. Create provider-specific clients (e.g., `GeminiClient`, `DeepSeekClient`)
2. Normalize tool call/result formats across providers
3. Use a factory or config to select provider

### Strengths

- Full control over the loop
- Easy to understand and debug
- LLM-agnostic with minor refactoring
- Integrated with existing agent system

### Limitations

- Manual tool execution
- No built-in parallelism
- No checkpointing/resumability
- Context window limits with long conversations

---

## 2. LangGraph (Experiment)

**Location**: `experiments/langgraph/soc_optimizer/`

### Architecture

```
StateGraph(TypedDict)
    ├─ Nodes: architect, linter, synthesis, validation, critic
    └─ Conditional Edges: next_action field routes to next node
```

### Key Components

- **State** (`state.py`): `SoCDesignState` TypedDict with all workflow data
- **Graph** (`graph.py`): Node and edge definitions
- **Nodes** (`nodes/`): Specialist functions for each step

### How It Works

```python
# graph.py - simplified
workflow = StateGraph(SoCDesignState)
workflow.add_node("linter", linter_node)
workflow.add_node("architect", architect_node)
workflow.add_conditional_edges(
    "linter",
    lambda s: s["next_action"],
    {ActionType.ARCHITECT: "architect", ActionType.SYNTHESIZE: "synthesis"}
)
```

Each node:
1. Reads from shared state
2. Performs its operation (calls tools directly)
3. Updates state and sets `next_action`
4. Returns updated fields

### Characteristics

| Aspect | Details |
|--------|---------|
| **Code size** | ~1000+ LOC |
| **Routing** | Explicit conditional edges |
| **State** | TypedDict with `next_action` control field |
| **Tool execution** | Nodes call tools directly |
| **Parallelism** | Can parallelize independent nodes |
| **Checkpointing** | Built-in via LangGraph |
| **LLM support** | Agnostic via LangChain integrations |

### LLM Agnosticism

LangGraph inherits LangChain's LLM abstraction:

```python
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic

# Swap LLM by changing the import and initialization
llm = ChatOpenAI(model="gpt-4")  # or
llm = ChatGoogleGenerativeAI(model="gemini-pro")  # or
llm = ChatAnthropic(model="claude-3-sonnet")
```

### Strengths

- Declarative graph structure (visualizable)
- Built-in checkpointing and persistence
- Native LLM agnosticism via LangChain
- Scalable to distributed workers
- History tracking built into state

### Limitations

- More complex setup
- Specialized for defined workflows (less open-ended)
- State can become large
- Steeper learning curve

---

## 3. Claude Agent SDK

**External package**: `claude-agent-sdk` (Python/TypeScript)

### Architecture

```
query(prompt, tools, agents) → SDK manages loop → Async message stream
```

### How It Works

```python
from claude_agent_sdk import query, ClaudeAgentOptions

async for message in query(
    prompt="Review this codebase",
    options=ClaudeAgentOptions(
        allowed_tools=["Read", "Glob", "Grep", "Bash", "Task"],
        agents={
            "security": AgentDefinition(
                description="Security reviewer",
                tools=["Read", "Grep"],
                model="opus"
            )
        }
    )
):
    print(message.result)
```

### Characteristics

| Aspect | Details |
|--------|---------|
| **Code size** | ~10 LOC to start |
| **Routing** | LLM-driven (SDK-managed) |
| **State** | Sessions with persistence |
| **Tool execution** | SDK handles built-in tools |
| **Parallelism** | Native subagent parallelism |
| **Checkpointing** | Sessions across invocations |
| **LLM support** | **Claude-only** |

### LLM Lock-in

**The Claude Agent SDK only supports Claude models.** You cannot use:
- OpenAI (GPT-4, o1)
- Google Gemini
- DeepSeek
- Any other LLM provider

This is by design - the SDK leverages Claude-specific features:
- Tool use protocol
- Extended thinking
- Vision capabilities
- Anthropic-optimized prompting

### Strengths

- Minimal boilerplate
- Production-ready built-in tools
- Native subagent orchestration
- Hooks for validation/audit
- Session persistence

### Limitations

- **Claude-only** - significant vendor lock-in
- Less control over the loop
- Opaque internals
- Requires Anthropic API access

---

## Comparison Matrix

| Aspect | Custom Chatbot | LangGraph | Claude Agent SDK |
|--------|----------------|-----------|------------------|
| **LLM Agnostic** | Yes (with refactor) | Yes (native) | **No** |
| **Who drives routing** | LLM | Explicit edges | LLM (SDK-managed) |
| **Tool execution** | Manual | Node-direct | SDK-managed |
| **State management** | Message history | TypedDict | Sessions |
| **Parallelism** | Manual | Node-level | Subagents |
| **Checkpointing** | None | Built-in | Sessions |
| **Multi-agent** | No | Nodes as specialists | Native subagents |
| **Setup complexity** | Medium | High | Low |
| **Control level** | Full | Full | Declarative |
| **Vendor lock-in** | None | None | High (Anthropic) |

---

## Recommendations

### For LLM-Agnostic Requirements

If LLM portability is a priority, **avoid the Claude Agent SDK** for core functionality.

**Recommended approach**:

1. **Use LangGraph for specialized workflows** (RTL optimization, benchmarking pipelines)
   - Native LLM abstraction via LangChain
   - Checkpointing and persistence
   - Well-suited for iterative refinement

2. **Extend the custom chatbot for interactive use**
   - Refactor `LLMClient` into a provider-agnostic interface
   - Create adapters for each LLM provider
   - Normalize tool calling conventions

### Provider-Agnostic LLM Interface

```python
# Proposed abstraction
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list, tools: list, system: str) -> LLMResponse:
        """Send messages and get response with potential tool calls."""
        pass

class AnthropicProvider(LLMProvider):
    def chat(self, messages, tools, system):
        # Anthropic-specific implementation
        ...

class OpenAIProvider(LLMProvider):
    def chat(self, messages, tools, system):
        # OpenAI-specific implementation (translate tool format)
        ...

class GeminiProvider(LLMProvider):
    def chat(self, messages, tools, system):
        # Gemini-specific implementation
        ...
```

### Hybrid Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM Provider Abstraction Layer                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │  Anthropic  │ │   OpenAI    │ │   Gemini    │  ...       │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌──────────────────────────┐    ┌──────────────────────────┐
│   Custom Chatbot Agent   │    │   LangGraph Workflows    │
│   (Interactive reasoning)│    │   (Specialized pipelines)│
└──────────────────────────┘    └──────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Execution Layer                      │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐            │
│  │   Agents    │ │  Backends   │ │   Registry  │            │
│  └─────────────┘ └─────────────┘ └─────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

---

## When to Use Each

| Use Case | Recommendation |
|----------|----------------|
| Interactive user-facing chat | Custom chatbot |
| Specialized optimization loops | LangGraph |
| Need LLM portability | Custom chatbot or LangGraph |
| Rapid prototyping with Claude | Claude Agent SDK |
| Production with vendor lock-in acceptable | Claude Agent SDK |
| Multi-LLM evaluation/comparison | Custom chatbot with provider abstraction |

---

## References

- Custom chatbot: `src/embodied_ai_architect/llm/`
- LangGraph experiment: `experiments/langgraph/soc_optimizer/`
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Claude Agent SDK](https://docs.anthropic.com/en/docs/agents-and-tools/claude-agent-sdk)
- [LangChain LLM Integrations](https://python.langchain.com/docs/integrations/llms/)
