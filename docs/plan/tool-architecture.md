# Tool Architecture

The tool registration uses two distinct registration patterns:
   1. Orchestrator Agent Registration
   2. LLM Tool Registration

## Orchestrator Agent Registration
In src/embodied_ai_architect/orchestrator.py, agents are registered manually via a dictionary:
```python
  orchestrator = Orchestrator()
  orchestrator.register_agent(ModelAnalyzerAgent())
  orchestrator.register_agent(HardwareProfileAgent())
```
The orchestrator stores agents by name (agent.name → BaseAgent) and has a hard-coded execution order in process() that checks for agents by string name:
```python
  if "ModelAnalyzer" in self.agents:
      # execute it
  if "HardwareProfile" in self.agents:
      # execute it
```
## LLM Tool Registration

In `src/embodied_ai_architect/llm/tools.py`, there is a dual registration pattern:

  - get_tool_definitions() - Returns a list of JSON schemas for the Claude API
  - create_tool_executors() - Returns a dict mapping tool names → callable functions

These must be kept in sync manually. The ArchitectAgent binds them at init time.

## How Would You Know You Have the Right Tools?

Currently: You wouldn't - there's no validation. The system has these gaps:

  | Issue                      | Consequence                                 |
  |----------------------------|---------------------------------------------|
  | No completeness checking   | Missing agents are silently skipped         |
  | No dependency declarations | Agents don't declare what they need         |
  | Manual sync required       | Tool definitions and executors can drift    |
  | Hard-coded workflow order  | Can't reason about what's needed for a task |

## What Would "Right Set of Tools" Mean?

To answer this properly for an LLM-driven system, you'd need:

  - Task Decomposition - The LLM must understand what subtasks are required
  - Tool Capability Matching - A way to match tool capabilities to subtasks
  - Gap Detection - Identify when no tool covers a required capability

The current system relies on:
  - The LLM's training to know what tools exist (from get_tool_definitions())
  - The LLM returning an error message if it calls an unknown tool
  - Human judgment when setting up the orchestrator

There's no programmatic way to validate "I have the tools needed to answer this prompt" before execution. The system discovers missing tools at runtime when a call fails.

Would you like me to explore potential solutions for tool coverage validation?

## Subtask Decomposition

Subtask decomposition is the foundational layer for an Agentic system. Without a clear model of how tasks break down, tool organization is premature.

The design space for subtask decomposition:

| Approach           | How It Works                          | Trade-offs                            |
|--------------------|---------------------------------------|---------------------------------------|
| LLM-driven         | LLM reasons about task → emits plan   | Flexible but opaque, hard to validate |
| Template workflows | Predefined DAGs for known task types  | Predictable but rigid                 |
| Goal-oriented      | Declare end state, planner finds path | Powerful but complex to implement     |
| Hybrid             | LLM selects/parameterizes templates   | Balance of flexibility and structure  |

**Key Design Decisions**

  1. Granularity - What constitutes an atomic subtask? Is "analyze model" atomic or does it decompose further?
  2. Dependency representation - How do subtasks declare what they need from prior steps? (data flow vs control flow)
  3. Replanning - Can the decomposition change mid-execution based on intermediate results?
  4. Observability - How does the system (and user) see the current plan and progress?

**Starting Point Questions**

Before sketching a design:

  - Are the task types relatively bounded (model analysis, hardware matching, benchmarking) or open-ended?
  - Should the decomposition be explicit to the user (showing a plan) or implicit (just works)?
  - How important is determinism - same prompt → same decomposition?

## Tool design

**Question**: if we take the example of analysis on the applicability of a particular DNN to solve a particular embodied perception problem, we
can analyze it in many different dimensions. For example, is the DNN applicable to the problem because it fits the accuracy, the energy constraint,
the performance constraint, the cost constraint, the memory constraint, etc.. So identifying the analysis task as a task needs to be much more
refined. And from a tool's architecture perspective, we can create one 'analysis' tool that we specialize through command line arguments, or we can
build tools that are very task specific, so by name. We can also create an analysis tool that tries to figure out the context and adapt, or one that
simply does it all and provides a massive output report. Of all these options, what is the best and most productive way to make the LLM tool calls so
the plan execution is fast, accurate, and robust?


The Four Tool Design Options

| Design                | Speed                      | Accuracy               | Robustness                            | LLM Cognitive Load                     |
|-----------------------|----------------------------|------------------------|---------------------------------------|----------------------------------------|
| Generic + args        | Medium                     | Lower (arg mistakes)   | Fragile (wrong args = wrong analysis) | High (must learn arg taxonomy)         |
| Task-specific by name | Fast (parallel)            | High (clear semantics) | High (partial success possible)       | Medium (more tools, but obvious names) |
| Context-adaptive      | Slow (figures out context) | Unpredictable          | Low (magic = hard to debug)           | Low (but opaque)                       |
| Do-everything         | Slow (wasted compute)      | High but noisy         | Medium (all or nothing)               | Low (but output floods context)        |

The Core Tension

  The LLM needs to:
  1. Select the right tool(s) for the task
  2. Parameterize correctly
  3. Interpret the output
  4. Compose multiple results into an answer

Each design stresses different parts:

  - Generic + args: Easy selection (one tool), hard parameterization
  - Task-specific: Harder selection (many tools), easy parameterization
  - Do-everything: Trivial selection, impossible interpretation (output overload)

Best option is Dimension-Specific Tools that match to a very specific subtask and name the tool such that this specificity is clear. For example, a tool naming taxonomy for the analysis subtask for specific constraints could look like this:
```text
  analyze_accuracy(model, dataset, threshold?)
  analyze_energy(model, hardware, power_budget?)
  analyze_latency(model, hardware, target_ms?)
  analyze_memory(model, hardware, memory_limit?)
  analyze_cost(model, hardware, budget?)
```

## Why this wins:

  1. Speed - Tools are independent, can run in parallel. Only invoke what's needed.
  2. Accuracy - Tool name matches intent. "Is energy okay?" → analyze_energy. No argument taxonomy to learn.
  3. Robustness - If analyze_cost fails, you still have accuracy/latency/memory results. Clear error attribution.
  4. Composability - The LLM can reason: "User asked about edge deployment → need energy + memory + latency, not cost"
  5. Bounded output - Each tool returns focused results. No context pollution.

## The Key Insight

The LLM is good at semantic matching (goal → tool name) but struggles with complex parameterization (goal → correct arg combination for generic tool).
Design tools so selection is the hard part, not parameterization.

## What This Means for Subtask Decomposition

For a user prompt like this:
```
  **User**: "Can YOLOv8s run on Jetson Nano for drone obstacle avoidance?"
```

If tools are dimension-specific, subtask decomposition becomes:

Decomposition:
  1. What constraints matter for this use case?
     → latency (real-time), energy (battery), memory (edge device)
  2. Map to tools:
     → analyze_latency, analyze_energy, analyze_memory
  3. Execute (parallel)
  4. Synthesize: all pass? → yes. any fail? → explain which and why

The decomposition is now about identifying which dimensions matter, not figuring out how to parameterize a mega-tool.
 
## Tool output characteristics

Before we design the subtask decomposition, we also need to refine what the best output is for the tool. What type of prose or data will create the
  best value for the LLM plan executor to deduce that the subtask has been completed with high accuracy.

The output format determines whether the LLM can confidently conclude "this subtask succeeded" or is left guessing. 
What makes tool output effective for LLM plan execution:

What the LLM Executor Needs to Determine

  1. Did this succeed? - Binary/categorical signal, not buried in prose
  2. What's the evidence? - Quantitative data that justifies the verdict
  3. How confident? - Can I trust this result or need verification?
  4. What's next? - If failed, what action would help?

Output Format Comparison

  - Prose-heavy (bad for LLM reasoning):
```text
  The model appears to have reasonable latency characteristics on the
  Jetson Nano. Inference times should be acceptable for most real-time
  applications, around 45ms per frame which is quite good.

  Problems: "Reasonable," "acceptable," "quite good" are subjective. The LLM must guess if this meets requirements.
```
  - Data-only (bad for interpretation):
```json
  {"latency_ms": 45.2, "std_dev": 3.1, "hardware": "jetson_nano"}
```

Problems: Is 45.2ms a pass or fail? The LLM must know that 30fps requires <33.3ms. That's domain knowledge it may not reliably apply.

### Verdict-first structured 

Verdict-first structured output is the best for communicating results, with confidence, metrics, and output assessments:
```json
  {
    "verdict": "FAIL",
    "confidence": "high",
    "metric": {
      "name": "latency",
      "measured": 45.2,
      "required": 33.3,
      "unit": "ms"
    },
    "gap": "+35.7%",
    "evidence": "Measured over 100 inference runs, std_dev 3.1ms",
    "suggestion": "Consider YOLOv8n (~22ms) or apply TensorRT optimization"
  }
```

The Principle: Pre-Digested Judgment

  The tool should do the domain reasoning, not the LLM. The tool knows:
  - What threshold applies (30fps → 33.3ms)
  - How to interpret the measurement
  - What alternatives exist

The LLM should receive a verdict it can trust, not raw data it must interpret.

Proposed Output Schema
```json
  {
    "verdict": "PASS | FAIL | PARTIAL | UNKNOWN",
    "confidence": "high | medium | low",
    "summary": "One sentence: what was checked, what was found",
    "metric": {
      "name": "string",
      "measured": "number",
      "required": "number | null",
      "unit": "string",
      "margin": "+/- percentage from requirement"
    },
    "evidence": "Brief description of how this was determined",
    "suggestion": "If not PASS, what action would help (or null)"
  }
```

Why This Enables Accurate Plan Execution

  | LLM Executor Need     | How Schema Addresses It                   |
  |-----------------------|-------------------------------------------|
  | Subtask complete?     | verdict field - no parsing required       |
  | Can I trust it?       | confidence field - know when to verify    |
  | Compose results?      | Uniform schema across all dimension tools |
  | Report to user?       | summary field - human-readable one-liner  |
  | Recover from failure? | suggestion field - actionable next step   |

The Execution Pattern

  1. LLM calls tools
```
  Ttool call: analyze_latency(yolov8s, jetson_nano, requirement=33.3)

  Tool returns: {"verdict": "FAIL", "confidence": "high", ...}
```

  2. LLM reasons:
    - verdict is FAIL → subtask complete but constraint not met
    - confidence is high → no need to re-verify
    - suggestion exists → can offer alternative to user

The LLM doesn't interpret 45.2ms. It reads "FAIL" and acts accordingly.

## Open Question

Should required come from the tool call (explicit) or from context the tool infers?

  - Explicit: analyze_latency(model, hw, required_ms=33.3) - LLM must know the threshold
  - Inferred: Tool knows "drone obstacle avoidance" implies real-time → 30fps → 33.3ms

The explicit approach keeps tools simple but pushes domain reasoning to the LLM. The inferred approach requires tools to understand use-case context.
