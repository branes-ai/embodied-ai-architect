# Agentic Frameworks

Designing an Agentic System-on-Chip (SoC) Designer platform requires a framework that can handle high-stakes iterative loops, complex state management (storing RTL snippets, synthesis logs, and PPA metrics), and robust tool integration for EDA (Electronic Design Automation).

Based on current 2024-2025 benchmarks and the specific needs of SoC optimization (Power, Performance, Area - PPA), here is an assessment of the leading frameworks and new SOTA tools.

---

## 1. Comparison of Frameworks for SoC Design

| Framework | Core Architecture | Best Use Case for SoC | Suitability for Optimization Cycles |
| --- | --- | --- | --- |
| **LangGraph** | State Machine (Cycles) | **Highest.** Ideal for "Design-Verify-Optimize" loops where state must persist across steps. | **Excellent.** Built-in support for cycles and "Human-in-the-loop" for design sign-off. |
| **CrewAI** | Role-Based (Teams) | Good for mimicking a design team (e.g., Architect, RTL Coder, Verification Eng). | **Moderate.** Better for linear tasks; optimization loops can become messy without a graph. |
| **AutoGen** | Conversational (Event-driven) | Multi-agent "debates" on architectural trade-offs or floorplanning strategies. | **High.** Strong at self-correction and code debugging via agent-to-agent chatter. |
| **Claude Agent SDK** | Lightweight / Computer Use | Automating GUI-based EDA tools or simple scripting tasks. | **Low-Moderate.** Powerful for specific tool interaction but lacks robust multi-agent orchestration. |

### Why LangGraph is likely your winner:

SoC design is rarely linear. You will need to branch based on synthesis results (e.g., if Timing Slack < 0, go to "Logic Restructuring"; else, go to "Power Gating"). LangGraph’s **Stateful Graph** allows you to treat the entire SoC project as a living object that gets updated by different specialist agents (Synthesis Agent, Timing Agent, etc.) with full "Time Travel" debugging to revert to a previous design state if an optimization fails.

---

## 2. State-of-the-Art (SOTA) Tools & Frameworks You Should Consider

Beyond the "Big Three," several 2025-era frameworks are better suited for the **heavy engineering** aspect of SoC design:

### A. PydanticAI (The "Engineering" Choice)

If your SoC platform needs strict data validation (e.g., ensuring an agent doesn't pass a "string" where a "voltage value" is expected), **PydanticAI** is the emerging favorite. It uses Pydantic for "Structured Output" and "Type Safety," which is critical when agents are generating configuration files for tools like Synopsys or Cadence.

### B. VeriMaAS (Domain Specific)

Recently published research (2025) highlights **VeriMaAS**, a multi-agent framework specifically designed for **RTL code generation**. It integrates formal verification feedback directly into the agent’s reasoning loop.

* **Key Feature:** It uses "Verification Feedback" as a primary reward signal for the agent, rather than just "LLM-based reflection."

### C. Microsoft Agent Framework (MAF) / AutoGen v0.4

Microsoft recently unified AutoGen with Semantic Kernel. The new version (v0.4) is **asynchronous and event-driven**, making it much better at handling long-running EDA simulations that might take hours to return a result to the agent.

---

## 3. Recommended Agentic Architecture for SoC Optimization

### The "Optimizer-Evaluator" Pattern

For SoC PPA (Power, Performance, Area) tuning, the most effective pattern is the **Evaluator-Optimizer**:

1. **Planner Agent:** Suggests a change (e.g., "Increase FIFO depth to improve throughput").
2. **Tool Executor:** Scripts the EDA tool (Verilator, Yosys, or Vivado) to run a simulation.
3. **Analyst Agent:** Parses the log files and extracts PPA metrics.
4. **Critic Agent:** Compares results against requirements. If not met, it sends the logs back to the Planner with "Why it failed" context.

---

## 4. Essential Tools to Integrate

To make this work, your agents need more than just an LLM; they need a **Hardware Toolbelt**:

* **Logic Synthesis:** Yosys (Open Source) or Synopsys Design Compiler (Commercial).
* **Simulation/Verification:** Verilator or Icarus Verilog.
* **PPA Extraction:** OpenRoad (for open-source physical design metrics).
* **Sandboxed Execution:** E2B or Docker containers to run HDL compilations safely.

## LangGraph State Schema

To implement an SoC optimization loop, we need a architecture that treats the chip design as a **State Object**. This ensures that as an agent tweaks a bus width or a clock gate, the metrics (Power, Performance, Area) are tracked and compared against the "Best Known Configuration."

### 1. The LangGraph State Schema

In LangGraph, the `State` is a typed dictionary that persists throughout the cycle. For an SoC designer, your state should look like this:

```python
from typing import TypedDict, List

class SoCState(TypedDict):
    rtl_code: str              # The current Verilog/Chisel source
    constraints: dict          # Timing, Power, and Area targets
    current_metrics: dict      # Latest results from EDA tools
    history: List[dict]        # Log of previous iterations for "backtracking"
    next_action: str           # Routing logic (e.g., "optimize", "verify", "finish")

```

---

### 2. The Agentic Optimization Workflow

The graph operates as a continuous loop. Unlike a standard chatbot, this system thrives on **negative feedback** (e.g., "Slack violation detected; re-run logic synthesis").

#### Phase A: The Architect (Planner)

The Architect agent analyzes the `constraints` and `current_metrics`. If the power consumption is too high, it proposes an architectural change (e.g., "Implement fine-grained clock gating on the ALU").

#### Phase B: The Tool Integrator (Executor)

This agent is strictly for **Tool-Use**. It doesn't "think"—it acts.

* **Action:** It writes the updated RTL to a file.
* **Action:** it triggers a headless EDA container (like OpenROAD or Synopsys) to run synthesis.
* **SOTA Tip:** Use **E2B (Executive to Box)** or a dedicated **Kubernetes Pod** for this, as HDL compilation is computationally expensive.

#### Phase C: The Critic (Analyst)

This agent parses the raw `.rpt` or `.log` files from the tools. It extracts the  (Worst Negative Slack), Total Power, and Cell Area.

* If the metrics meet the requirements: **Terminate and Export GDSII.**
* If metrics are worse than before: **Revert to previous state and try a different strategy.**

---

### 3. SOTA Multi-Agent Design Patterns for SoC

To make this platform robust, I recommend implementing these specific patterns:

| Pattern | Implementation | Benefit for SoC |
| --- | --- | --- |
| **Self-Reflection** | The LLM reviews its own Verilog before synthesis. | Catches "hallucinated" syntax errors before wasting 30 mins on a tool run. |
| **Multi-Agent Debate** | Agent A (Performance) vs. Agent B (Power). | Automatically finds the Pareto frontier for SoC trade-offs. |
| **Human-in-the-loop (HITL)** | LangGraph's `interrupt` feature. | Allows a human architect to "approve" a floorplan change before the agent proceeds to routing. |

---

### 4. Integration with State-of-the-Art EDA Tools

Your "Tools" should not just be `print("running bash script")`. You should use Python wrappers that allow the agents to interact with the design data directly:

* **PyVerilog:** For parsing and modifying the AST (Abstract Syntax Tree) of the RTL.
* **SiliconCompiler:** A modular "build system" for hardware that acts as the "LLM-to-EDA" bridge.
* **Verible:** For automated linting and formatting to ensure the LLM-generated code is readable.

## LangGraph Optimization Node

To implement this, we utilize LangGraph’s ability to route the flow based on data. The "Optimization Node" acts as the brain that looks at the raw output from your EDA tools and decides if the design is improving or if it needs a radical architectural pivot.

We need to specifically parses a synthesis report to decide the next design step.

### 1. The Optimization Node Logic

In this example, the agent doesn't just "see" the report; it parses it into a structured format to compare against the **Global State**. If the **Worst Negative Slack (WNS)** is negative, the graph is programmed to loop back to the "Logic Repair" agent.

```python
import json
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# Define our structured PPA (Power, Performance, Area) report
def optimization_node(state: SoCState):
    llm = ChatOpenAI(model="gpt-4o")
    
    # 1. Extract metrics from the 'current_metrics' provided by the Tool Node
    metrics = state['current_metrics']
    wns = metrics.get("wns", 0.0)
    power = metrics.get("total_power", 0.0)
    
    # 2. Logic to determine the next design step
    if wns < 0:
        # TIMING VIOLATION: Route to Timing Optimizer
        instruction = f"Timing failed with WNS: {wns}. Suggest gate-sizing or pipeline depth increase."
        return {"next_action": "fix_timing", "history": state['history'] + [instruction]}
    
    elif power > state['constraints']['max_power']:
        # POWER VIOLATION: Route to Power Optimizer
        instruction = f"Power limit exceeded ({power}W). Suggest clock-gating or voltage scaling."
        return {"next_action": "optimize_power", "history": state['history'] + [instruction]}
    
    else:
        # SUCCESS: Move to Final Sign-off
        return {"next_action": "sign_off"}


```

---

### 2. The Feedback Loop Visualization

The power of LangGraph here is the **Conditional Edge**. You define a router that checks the `next_action` and physically moves the "state" to the correct specialist agent.

### 3. Key Components for your SoC Platform

To make this "Agentic," you need to bridge the gap between the LLM and the physical design constraints.

| Component | SOTA Integration | Role in SoC Design |
| --- | --- | --- |
| **Logic Parser** | **PyVerilog** | Allows the agent to "edit" specific modules without rewriting the whole file. |
| **Sandbox** | **E2B Code Interpreter** | A secure cloud environment to run `iverilog` or `Vivado` scripts. |
| **Memory** | **Vector DB (Chroma/Pinecone)** | Stores previous "Fixes" (e.g., "Last time we had this timing slack, we increased the buffer size"). |
| **Human Gate** | **LangGraph Breakpoints** | Stops the loop to ask the Lead Engineer: "The agent wants to add a pipeline stage, which increases latency. Allow?" |

---

### 4. Handling Long-Running EDA Tasks

SoC synthesis can take 10 minutes to 10 hours. LangGraph is superior here because it supports **Persistent Checkpointing**.

1. The agent submits a synthesis job to a high-performance computing (HPC) cluster.
2. The LangGraph state "sleeps" (is saved to a database).
3. Once the HPC job finishes, a webhook "wakes up" the graph, injects the synthesis report, and the agent continues its analysis.

## Critic Agent

We will need a **system prompt** for the "Critic Agent" that teaches it how to interpret a **Standard Delay Format (.SDF)** file or a **Synthesis Timing Report**.

To turn an LLM into a competent SoC Critic, the prompt must shift from general reasoning to **structural data analysis**. In the SoC domain, the agent needs to act like a Senior Static Timing Analysis (STA) Engineer.

Below is a specialized system prompt designed for a **Critic Agent** within your LangGraph workflow. It focuses on identifying the root cause of failures rather than just reporting them.

---

### 1. System Prompt: The SoC Timing & PPA Critic

> **Role:** You are the Lead STA & PPA Analyst Agent for an Agentic SoC Design Platform.
> **Input:** You will receive a raw Synthesis Timing Report (typically from tools like Synopsys Design Compiler, Cadence Genus, or Yosys) and a Power Report.
> **Objective:** > 1. Parse the **Worst Negative Slack (WNS)** and **Total Negative Slack (TNS)**.
> 2. Identify the **Critical Path**. Locate the startpoint and endpoint (e.g., Flip-Flop to Flip-Flop).
> 3. Determine the **Bottleneck Type**:
> * *Logic Depth:* Too many gates between registers.
> * *Fan-out:* One signal driving too many loads.
> * *Congestion:* Physical routing issues.
> 

> 4. **Reject or Pass:** If WNS < 0, you MUST reject the design and provide a "Root Cause Prescription" to the Architect Agent.
> 
> 
> **Constraint:** Do not suggest generic fixes. Reference specific module names and signal paths found in the report.

In technical documentation and LaTeX, variables like **WNS** (Worst Negative Slack) are often wrapped in delimiters that can sometimes get stripped.

### Why **WNS** is the Critical Pivot

In SoC design, the **Worst Negative Slack** represents the difference between the time a signal *must* arrive to be captured by a clock edge and the time it *actually* arrives.

* **Positive Slack ():** The signal arrived early. The design is "timing clean."
* **Zero Slack ():** The signal arrived exactly on time (the "Goldilocks" zone).
* **Negative Slack ():** The signal arrived too late. The chip will fail to function at that clock frequency.

---

### The Logic for the Critic's "If" Clause

In an Agentic workflow, the "Condition" in the code isn't just a boolean; it is a **Routing Trigger**. Here is how that logic is handled inside the LangGraph node:

```python
# The missing logic in the Critic node
if wns < 0:
    # DESIGN FAILURE: We must iterate.
    state['next_action'] = "RE-ARCHITECT"
    state['failure_report'] = "Setup violation on path X. Slack is -150ps."
else:
    # DESIGN SUCCESS: Proceed to Physical Design / GDSII
    state['next_action'] = "SIGN_OFF"

```

### Why we use WNS instead of "Total Power" as the primary gate:

1. **Non-Negotiability:** Timing is a "hard" constraint. A chip that is 10% over power targets can still be sold or throttled; a chip with negative slack is essentially a "brick" (it won't boot).
2. **Optimization Conflict:** Usually, fixing **WNS** (by adding buffers or increasing gate sizes) *increases* power. The Critic must balance these. If it sees **WNS** is barely passing but power is way over, it might still reject the design to find a more efficient architecture.

---

### 2. How the Critic Evaluates the Path

When the Critic parses a report, it looks for the "Path Histogram." If the timing fails, the Critic uses the following logic to "prescribe" a fix to the next agent in your LangGraph:

| If the Critic sees... | Root Cause | Prescription to Architect |
| --- | --- | --- |
| **High Logic Depth (>20 gates)** | Combinational bloat. | "Insert a pipeline register at [Module X] to break the path." |
| **High Fan-out (>30)** | Signal degradation. | "Insert a buffer tree or replicate the driver for signal [Signal_Name]." |
| **High Leakage Power** | Wrong Vt cell selection. | "Swap non-critical path cells from LVT to RVT/HVT." |
| **Negative Setup Slack** | Clock frequency too high. | "Reduce frequency or optimize the specific critical path logic." |

---

### 3. Integrating the "Prescription" into LangGraph

To make this actionable, the Critic shouldn't just return text; it should return **structured metadata** that the next node (the Architect) can use to modify the RTL.

```python
# Example of the Critic's structured output
{
    "status": "REJECT",
    "metrics": {"wns": -120.5, "unit": "ps"},
    "bottleneck_path": "u_alu/inst_mult/out_reg",
    "recommendation": "PIPELINE_INSERTION",
    "target_module": "alu_multiplier.v"
}

```

### 4. SOTA Tip: The "Small-Model" Verifier

For a production SoC platform, you can use a smaller, cheaper model (like **Llama-3-8B** or **Claude 3 Haiku**) as a "Pre-Critic." It can quickly check for syntax errors or obvious "Linter" violations (using Verible) before you send the design to a larger model like GPT-4o or Claude 3.5 Sonnet for the expensive architectural optimization.

---

## RTL Architect Agent

We now have the Critic's feedback, and we would like to feed that into the RTL Architect Agent to improve the RTL. To ensure the **RTL Architect Agent** doesn't hallucinate or break the hierarchy of a multi-million gate SoC, we must move away from "Write the whole file" prompts and toward **Targeted Refactoring**.

The Architect must act like a surgical tool, using the **Critic's Prescription** to modify specific RTL blocks. We can structure the pipeline by using a targeted prompt that articulates the constraints.

---

### 1. The "Surgical" RTL Architect Prompt

> **Role:** You are a Senior RTL Architect specializing in Verilog/SystemVerilog optimization.
> **Task:** You will receive a **Failure Prescription** from the Critic and the **Source Code** of the failing module. Your goal is to apply a specific micro-architectural transformation to resolve timing or power violations.
> **Transformation Strategies:**
> 1. **Pipelining:** Insert `reg` stages to break long combinational paths.
> 2. **Logic Flattening:** Reduce nested `if-else` or `case` statements into parallel logic.
> 3. **Gate Sizing/Strength:** (For Physical Design agents) modify cell instances.
> 4. **Register Retiming:** Move a register across a combinational boundary.
> 
> 
> **Constraint:** You MUST maintain the original module interface (port list). You are only permitted to modify the internal `always` blocks and `assign` statements.

---

### 2. Implementation: AST-Based Modification vs. String Replacement

In a SOTA SoC platform, the agent shouldn't just "guess" where to put code. You should use a **Search-and-Replace** or **Diff-patch** mechanism.

By using **PyVerilog**, your agent can identify the exact line where a high-fanout signal is declared and insert a "Buffer Tree" or "Pipeline Register" precisely at that node in the Abstract Syntax Tree (AST).

---

### 3. The Architect's "Chain of Thought" Logic

When the Architect receives a failure, it follows this decision tree:

* **Step 1: Impact Analysis.** If I add a pipeline stage to fix timing, will I break the cycle-accurate behavior of the bus? (If yes, notify the **Protocol Agent** to update the testbench).
* **Step 2: Localized Edit.** Locate the critical path signal (e.g., `data_path_out`).
* **Step 3: Implementation.** Rewrite the logic. For example, converting a deep nested operation into two cycles:

```verilog
// BEFORE (Deep logic failing timing)
assign out = (a * b) + (c * d) + (e * f);

// AFTER (Architect inserts pipeline stage)
always @(posedge clk) begin
    prod1 <= a * b;
    prod2 <= c * d;
    prod3 <= e * f;
    out   <= prod1 + prod2 + prod3;
end

```

---

### 4. Integration into LangGraph: The "Rollback" Mechanism

Because LLMs can occasionally introduce syntax errors, your LangGraph must have a **Linter Loop**.

1. **Architect** modifies the code.
2. **Linter Node** (using `verilator --lint-only`) checks for syntax errors.
3. **If Linter fails:** The error log is sent back to the **Architect** for an immediate fix.
4. **If Linter passes:** The code is sent to **Synthesis** for the next optimization cycle.

### Why this works for SoC:

This mimics the real-world engineering flow. You don't send code to the "expensive" synthesis tool until it has passed the "cheap" linter.

**Would you like me to provide the "Linter Node" Python code that catches Verilog syntax errors and feeds them back to the Architect?**

## A Fast Linter

This **Linter Node** is the "Fast-Fail" mechanism of your platform. In SoC design, running a full synthesis to find a missing semicolon or a mismatched bit-width is a waste of  hours and API tokens.

By using an open-source tool like **Verilator**, you provide the Agent with a sub-second feedback loop.

---

### 1. The Linter Node Implementation

This Python function acts as a LangGraph node. It executes a shell command to lint the generated Verilog and captures `stderr`. If errors exist, it routes the state back to the Architect; otherwise, it proceeds to the expensive EDA tools.

```python
import subprocess

def linter_node(state: SoCState):
    # 1. Write the Architect's code to a temporary file
    with open("temp_design.v", "w") as f:
        f.write(state['rtl_code'])
    
    # 2. Run Verilator in lint-only mode
    # --lint-only: checks syntax/style without generating executable code
    result = subprocess.run(
        ["verilator", "--lint-only", "-Wall", "temp_design.v"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        # SYNTAX ERROR DETECTED
        error_msg = f"Linting failed. Please fix these errors:\n{result.stderr}"
        return {
            "next_action": "RE-ARCHITECT", 
            "history": state['history'] + [{"role": "linter", "content": error_msg}]
        }
    
    # SUCCESS: Design is syntactically sound
    return {"next_action": "SYNTHESIZE"}

```

---

### 2. Why Verilator is the SOTA Choice for Agents

Unlike commercial tools, Verilator is extremely "vocal." It doesn't just say "Error"; it provides **contextual warnings** that are goldmines for an LLM Architect:

* **WIDTH warnings:** "Operator ASSIGNW expects 8 bits on the Assign RTL, but lookup gave 7 bits."
* **UNOPTFLAT:** Identifying combinational loops that would cause simulation to hang.
* **UNUSED:** Identifying signals that increase area without providing logic value.

---

### 3. The "Self-Healing" Design Loop

By connecting this node in LangGraph, you create a **Self-Healing Loop**. The Architect learns from its mistakes in real-time.

| Iteration | Architect Action | Linter Result | State Transition |
| --- | --- | --- | --- |
| **1** | Inserts pipeline register. | `Error: 'prod1' not declared.` | Loop back to Architect. |
| **2** | Declares `reg [15:0] prod1`. | `Warning: Signal 'prod1' width mismatch.` | Loop back to Architect. |
| **3** | Corrects bit-width to 32. | `Clean` | Move to Synthesis Node. |

---

### 4. Advanced: Multi-File Dependency Handling

In a real SoC, your agent might be modifying a sub-module that depends on a `header.vh` or a package. To handle this, your Linter Node should:

1. **Search the workspace** for include paths.
2. **Mount the entire design directory** into the E2B sandbox or Docker container.
3. **Pass `-I<path>` flags** to the linter automatically.

---

## SoC Designer Orchestration loop

This is the "blueprint" of your SoC Designer. By using LangGraph, we define the **Control Flow**—the set of rules that governs when the design moves forward to synthesis and when it gets kicked back for a re-design.

### 1. The SoC Orchestrator Graph

This code defines the state machine. It uses **Conditional Edges** to route the design based on the `next_action` determined by your Linter and Critic agents.

```python
from langgraph.graph import StateGraph, END

# Initialize the Graph
workflow = StateGraph(SoCState)

# Define the Nodes (The agents we've built)
workflow.add_node("architect", architect_node)
workflow.add_node("linter", linter_node)
workflow.add_node("synthesis", synthesis_tool_node) # Runs EDA tools
workflow.add_node("critic", critic_node)

# Define the Edge Logic
workflow.set_entry_point("architect")

# 1. After Architecting, always Lint
workflow.add_edge("architect", "linter")

# 2. Conditional Routing from Linter
workflow.add_conditional_edges(
    "linter",
    lambda x: x["next_action"],
    {
        "RE-ARCHITECT": "architect",  # Syntax error? Go back.
        "SYNTHESIZE": "synthesis"    # Clean? Run EDA tools.
    }
)

# 3. After Synthesis, always run the Critic
workflow.add_edge("synthesis", "critic")

# 4. Conditional Routing from Critic (The PPA Loop)
workflow.add_conditional_edges(
    "critic",
    lambda x: x["next_action"],
    {
        "RE-ARCHITECT": "architect",  # Timing/Power fail? Redesign.
        "sign_off": END               # Requirements met!
    }
)

# Compile the Graph
app = workflow.compile()

```

---

### 2. Visualizing the Iterative Optimization Loop

The resulting graph creates a "Double Loop" system. The inner loop handles syntax and basic errors (Linter), while the outer loop handles complex physical design constraints (PPA).

---

### 3. Implementation Challenges & Solutions

When building this for a real-world SoC, you will encounter three primary hurdles. Here is how to handle them using SOTA techniques:

| Challenge | SOTA Solution |
| --- | --- |
| **Token Limits:** RTL files for a full SoC are too large for an LLM context. | **RAG for Hardware:** Index the SoC's IP library and only "Check Out" the specific module the Critic identifies as the bottleneck. |
| **Hallucinated Ports:** Agent adds a port that doesn't exist in the top-level. | **Stitcher Agent:** Use a dedicated agent whose only job is to manage `instantiations` and ensure wire-connectivity remains valid. |
| **EDA Latency:** Synthesis takes 30+ minutes. | **Async Checkpointing:** Use LangGraph’s `checkpointer` (PostgreSQL/Redis) to save the state so the agent can "sleep" while the server farms run the job. |

---

### 4. Human-in-the-loop (The "Sign-off" Gate)

In professional silicon design, you rarely want an AI to finalize a GDSII (the final chip layout) without a human looking at it. You can add a "Human Review" node before the `END` of your graph:

```python
# Insert a breakpoint for human verification
workflow.add_node("human_review", human_approval_node)
workflow.add_edge("critic", "human_review")
workflow.add_edge("human_review", END)

# Use compile(interrupt_before=["human_review"]) to pause the graph

```
---

## Agentic Tool-Belt API

The Tool-Belt would be the set of Python functions that allow the agents to interact with your specific EDA environment (e.g., launching a Docker container with OpenROAD or connecting to a Synopsys license server).

To build a production-grade **Agentic Tool-Belt**, we need to move away from simple local script execution and toward a **scalable, containerized EDA infrastructure**. Since EDA tools often require specific OS environments (usually RHEL/CentOS/Ubuntu) and heavy dependencies, your agents should interact with them through an abstraction layer.

Here is the design for the **SoC Tool-Belt API** that connects your LangGraph agents to the physical hardware tools.

---

### 1. The Containerized Execution Strategy

Instead of installing EDA tools on the same machine as your LLM framework, use a "Sidecar" pattern. The agent sends a **JSON payload** to a runner, which spins up a short-lived container.

**The API should provide three core capabilities:**

1. **File Workspace Management:** Syncing RTL, constraints (), and libraries ().
2. **Tool Execution:** Launching the process (e.g., `yosys`, `openroad`, or `vivado`).
3. **Result Extraction:** Parsing binary or log outputs into JSON for the agent.

---

### 2. SOTA Implementation: The `EDARunner` Class

This Python class acts as the "hands" for your agents. It handles the messy details of shell commands and filesystem management.

```python
import docker
import os

class EDARunner:
    def __init__(self, workspace_dir: str):
        self.client = docker.from_env()
        self.workspace = workspace_dir

    def run_synthesis(self, top_module: str):
        """Launches an OpenROAD/Yosys container to synthesize the RTL."""
        # Define the shell command to be run inside the container
        cmd = f"yosys -p 'read_verilog {top_module}.v; synth -top {top_module}; write_verilog synth_out.v'"
        
        container = self.client.containers.run(
            image="openroad/openlane", # SOTA Open-source EDA image
            command=["bash", "-c", cmd],
            volumes={self.workspace: {'bind': '/work', 'mode': 'rw'}},
            working_dir="/work",
            detach=False
        )
        return container.decode('utf-8')

    def extract_ppa(self, log_path: str):
        """Utility to parse logs into a dictionary the Critic Agent can read."""
        # Logic to grep/parse Slack, Area, and Power metrics
        metrics = {"wns": -0.5, "area": 4500.0, "power": 0.12} 
        return metrics

```

---

### 3. Essential Toolset Integrations

For a comprehensive SoC platform, your Tool-Belt should support these four categories:

| Tool Category | Recommended Tool | Agent Use Case |
| --- | --- | --- |
| **Logic Verification** | **Verilator** | Syntax checking and cycle-accurate functional testing. |
| **Formal Verification** | **SymbiYosys** | Proving that an optimization didn't change the logic's intent. |
| **Physical Design** | **OpenROAD** | Extracting real-world timing and area after floorplanning. |
| **PDK (Foundry Data)** | **Sky130 / ASAP7** | Providing the agent with the "rules" of the silicon (gate delays, etc.). |

---

### 4. Handling State Persistence (The "Checkpoint" Tool)

One often overlooked "tool" is a **Design Version Control** system. When an agent experiments with a design, it needs to be able to "undo."

* **The Tool:** Git or a simple S3-based snapshotting system.
* **The Agentic Action:** Before the `Architect` modifies a file, the system creates a `checkpoint_id`. If the `Critic` sees the metrics get worse, it triggers the `RollbackTool(checkpoint_id)`.

---

### 5. Final Architecture Overview

Your platform is now structured as follows:

1. **LangGraph:** Manages the high-level logic and "who speaks when."
2. **LLM Agents:** Act as the Architect, Linter, and Critic.
3. **Tool-Belt API:** Interfaces with Dockerized EDA environments.
4. **Foundry PDKs:** Provides the physical constraints the agents must respect.

### Next Steps for your Project

You now have the framework, the nodes, the logic, and the tool-belt. **Would you like to focus on the "Foundry Integration" next—how to provide the agent with the specific timing libraries (.lib) so it understands the performance limits of a specific technology node (like 7nm or 130nm)?**

## The Infrastructure Layer

Up to this point, we’ve discussed the "Brain" (Agents) and the "Tools" (EDA), but an **Agentic SoC Designer** isn't just a Python script; it is a **distributed system**. In a professional environment, you cannot run synthesis on the same machine that hosts your LangGraph state.

The architecture we are missing is the **Three-Tier Orchestration Model**:

---

## The 3-Tier Agentic SoC Architecture

### 1. The Control Plane (State & Reasoning)

This is where **LangGraph** lives. It acts as the "Server Room."

* **State Store:** A database (PostgreSQL/Redis) that saves the "DNA" of your SoC. Every version of the RTL and every synthesis report is stored here.
* **Orchestrator:** Manages the turn-taking between agents. It ensures that if the "Architect" is thinking, the "Critic" is waiting.

### 2. The Execution Plane (Sandboxed Workers)

Agents are notoriously "messy"—they write files, delete directories, and run untrusted code.

* **Worker Nodes:** These are ephemeral Docker containers (or E2B environments). When an agent wants to run a tool, the Control Plane spins up a worker, mounts the specific RTL module, and executes the command.
* **The Bridge:** A FastAPI or gRPC layer that translates "Agent Requests" into "Shell Commands."

### 3. The Data Plane (The Silicon Knowledge Base)

This is the "Library" the agents reference.

* **PDK (Process Design Kit):** The physical rules from the foundry (TSMC, Intel, etc.).
* **IP Repository:** A library of "Known Good" modules (CPUs, SRAM, PCIe controllers) that the agent can pull from.
* **Vector Database:** A "Memory" of past design failures. If the agent fixed a timing violation in an ALU last week, it should retrieve that solution instead of reinventing it.

---

## Visualizing the System Architecture

---

## The "Missing Link": The Design Manager

In a manual flow, a human uses a tool like **Make** or **Python scripts** to manage the design. In your platform, you need a **Design Manager Service** that sits between the Agent and the PDK.

| Component | Responsibility | Why it’s needed |
| --- | --- | --- |
| **API Gateway** | Auth & Token usage tracking. | Limits costs and secures the license-heavy EDA tools. |
| **Telemetry** | Tracking "Optimization Convergence." | Visualizes if the agent is actually making the chip better or just "looping" in circles. |
| **Filesystem Sync** | Moving large `.v` and `.gds` files. | Ensures the "Critic" is looking at the exact same file the "Architect" just wrote. |

---

## How it looks in production

Imagine a user types: *"Optimize this RISC-V core for 1GHz on Sky130."*

1. **Control Plane** parses the intent and initializes the **State Object**.
2. **Architect Agent** requests the RTL from the **Data Plane**.
3. **Execution Plane** spins up a container to run the initial synthesis.
4. **The Result** is piped back to the **Control Plane**, updating the state.
5. **LangGraph** triggers the next node based on the results.

In summary, the high-level architectural split, Control vs. Execution vs. Data, decomposes the problem into a set of well-defined actitivities that each have their own life cycle.

## Hardware RAG

To build a robust **Data Plane** for an SoC platform, we have to solve the "context window" problem. Modern SoC designs involve thousands of files; you cannot simply feed the entire codebase into an LLM.

Instead, we use a **Hardware-Aware RAG (Retrieval-Augmented Generation)** system. This acts as the platform's "Long-Term Memory."

---

## 1. The Data Plane Architecture

The Data Plane is responsible for giving the agent "Vision" into the foundry rules and "Memory" of the existing IP blocks.

### A. The Structural Index (RTL & IP)

Instead of treating code as plain text, we index it by **Module Hierarchy**.

* **The Tool:** Use a Verilog parser (like **PyVerilog**) to create a "Map" of the design.
* **The Vector DB:** Store code snippets indexed by function (e.g., "Arbiter," "FIFO," "Crossbar").
* **Why:** When the Critic says "The Arbiter is the bottleneck," the Architect can query the Data Plane for: *"Find all alternative Arbiter implementations in our library with lower logic depth."*

### B. The Constraint & Rule Index (PDK/SDC)

Foundry manuals (PDKs) are thousands of pages of PDFs.

* **The Content:** Metal pitch, cell delay tables, and design rule check (DRC) requirements.
* **The RAG Use Case:** When the agent suggests a specific metal layer for routing, the Data Plane provides the specific spacing rules for that layer to ensure the agent's suggestion is physically possible.

---

## 2. Implementing "Circuit-Aware" Retrieval

Standard RAG often fails on code because it doesn't understand connectivity. To fix this, your Data Plane should use **Graph-Link Retrieval**.

| Retrieval Type | How it Works | Use Case in SoC |
| --- | --- | --- |
| **Vector Search** | Finds "Semantically similar" code. | "Find me a low-power counter." |
| **Graph Search** | Follows the signal paths (Netlist). | "Show me every module connected to the `sys_rst_n` signal." |
| **Metadata Filtering** | Filters by PPA metrics. | "Show me the smallest 32-bit multiplier we have." |

---

## 3. The Knowledge Graph (The "Missing Link")

A SOTA architecture often overlays a **Knowledge Graph** on top of the Vector DB. This graph stores the relationships between components.

If the **Architect Agent** wants to change the clock frequency, it queries the Knowledge Graph to see which other modules share that clock domain. This prevents the agent from fixing one module while accidentally breaking the timing of ten others.

---

## 4. The "Experience" Cache (Optimization Logs)

This is the most powerful part of the Data Plane. You should store every successful "Optimization Loop" in a **Trace Database**.

* **Input:** "Slack was -200ps in a 64-bit adder."
* **Action:** "Architect added a pipeline stage at the carry-lookahead logic."
* **Outcome:** "Success: Slack improved to +10ps."

Next time an agent encounters a similar timing failure, it retrieves this "Success Trace" and applies the same strategy, drastically reducing the number of expensive synthesis cycles.

### Next Step

Would you like to see the **Metadata Schema** for this "Experience Cache"? This would define exactly what data the agents should save after every loop to ensure the platform gets smarter over time.

## Experience Cache

To make your SoC Designer platform truly "agentic," it needs to learn from its own mistakes. Without an **Experience Cache**, your agents will repeat the same costly EDA mistakes (like trying to pipeline a path that is already congested) every time they start a new project.

Here is the metadata schema for the **Optimization Trace Database**, designed to turn "failures" into "training data."

---

## 1. The Optimization Trace Schema

This schema captures the "State Delta." It doesn't just store the code; it stores the **Reasoning** and the **Result**.

| Field | Type | Description |
| --- | --- | --- |
| `problem_fingerprint` | **Vector/Hash** | An embedding of the failing path (e.g., "Critical path through 64-bit Floating Point Multiplier"). |
| `initial_metrics` | **JSON** | WNS, TNS, Power, and Area before the change. |
| `agent_hypothesis` | **Text** | The Architect's reasoning (e.g., "Inserting a register at the stage-2 partial product summation"). |
| `transformation_diff` | **Code Diff** | The exact Verilog change applied. |
| `post_tool_metrics` | **JSON** | The new PPA results after synthesis/STA. |
| `outcome_score` | **Float** | A normalized score ( for fix,  for worsening timing). |

---

## 2. Using the Cache in the LangGraph Loop

When the **Architect Agent** starts a task, it performs a "Pre-flight Retrieval." It searches the Experience Cache for similar `problem_fingerprints`.

### The Workflow:

1. **Search:** Architect finds a 90% match for a "Carry-Save Adder timing violation."
2. **Retrieve:** It pulls the `transformation_diff` that worked in the past.
3. **Adapt:** It applies that logic to the current specific module name and signal naming convention.
4. **Execute:** If it works again, the "Weight" of that solution in the Data Plane increases.

---

## 3. Visualizing the Full Platform Architecture

Now that we have all the pieces—**Control, Execution, and Data**—here is how they look as a single integrated SoC CAD system.

---

## 4. Why this is the "Agentic" SOTA

Most current AI-for-Chip-Design tools are "Copilots" (autocomplete). Your platform is a **Self-Optimizing Agent** because:

* **It closes the loop:** It doesn't just suggest code; it proves the code works via the Execution Plane.
* **It has a memory:** Through the Data Plane and Experience Cache, it doesn't solve the same problem twice.
* **It handles complexity:** By using LangGraph, it can manage the non-linear "Back-and-Forth" of real silicon engineering.

---

## Final Assessment of the Tech Stack

* **Orchestration:** **LangGraph** (Mandatory for the cycles/state management).
* **Intelligence:** **GPT-4o or Claude 3.5 Sonnet** (Highest reasoning for RTL logic).
* **Execution:** **Docker/Kubernetes + OpenROAD/Verilator** (The open-source SOTA EDA stack).
* **Data Plane:** **ChromaDB or Weaviate** (For the Vector/Experience Cache).

## The Core Shell

To get to a working "shell," you need to build the **Agentic Runtime Environment**. This is the wrapper that allows LangGraph to exist not just as a script, but as a persistent service that can manage long-running EDA jobs and handle state interruptions.

The "shell" consists of the **Graph Runtime**, the **State Checkpointer**, and the **Asynchronous Event Loop**.

---

## 1. The Core Shell Structure: The "Persistent Runtime"

In SoC design, you cannot use a standard "Stateless" API. If a synthesis job takes 4 hours, the shell must be able to save the entire agent's "thought process" to a database and resume it later.

### The Checkpointer Implementation

Using **LangGraph’s Checkpointer**, you create a "Time-Travel" shell. This allows you to pause the design process, let a human review the RTL, and then "resume" the agent from exactly where it left off.

```python
from langgraph.checkpoint.postgres import PostgresSaver # SOTA for production shells

# The "Shell" Database configuration
connection_string = "postgresql://user:pass@localhost:5432/soc_design_db"
checkpointer = PostgresSaver.from_conn_string(connection_string)

# Compiling the shell with persistence
app = workflow.compile(checkpointer=checkpointer)

# Running a specific project (thread_id)
config = {"configurable": {"thread_id": "SoC_Project_Alpha_001"}}
app.invoke(initial_state, config)

```

---

## 2. The "Human-in-the-loop" Shell Pattern

SoC design is too high-stakes for 100% autonomy. The "shell" must support **Breakpoints**. This is a specific LangGraph feature that halts execution before "Expensive" or "Risky" nodes (like submitting a tape-out file or running a $10k cloud synthesis job).

### How to trigger a Breakpoint in the shell:

```python
# We interrupt the shell before the Synthesis node to let a human check the RTL
app = workflow.compile(
    checkpointer=checkpointer,
    interrupt_before=["synthesis"] 
)

# The shell will stop here. You can then use:
# app.update_state(config, {"rtl_code": "manually_fixed_code.v"}) 
# To "nudge" the agent before it proceeds.

```

---

## 3. The "Tool-Call" Event Loop (The Shell's Hands)

The shell needs a way to communicate with your EDA Docker containers. Since EDA tools are CLI-heavy, the shell must manage **Streaming Logs**. You don't want the agent to wait 20 minutes for a result; you want it to "tail" the log file to see if the design is failing early.

### The Async Execution Shell:

```python
async def run_eda_shell(state: SoCState):
    # This node launches a background process
    process = await anyio.open_process(["yosys", "script.ys"])
    
    # The shell streams output back to the UI/Monitor
    async for line in process.stdout:
        print(f"[EDA LOG]: {line}") 
    
    return {"current_metrics": parsed_results}

```

---

## 4. The "Dashboard" Shell (The Operator Interface)

Since this is an "Agentic SoC Designer platform," the shell needs a UI. You shouldn't just look at a terminal. A working shell usually integrates **LangGraph Studio** or a custom React frontend that visualizes the Graph.

### What the Operator Shell displays:

* **The Design Graph:** Which agent is currently "active"?
* **The State Inspector:** What is the current  and  at this exact moment?
* **The History Explorer:** Allow the user to "Rewind" the design to 2 hours ago if the agent went down a bad optimization path.

---

## 5. Summary: The Working Shell Stack

To build this shell today, your "Workday 1" stack should be:

1. **FastAPI:** To provide a REST API for the shell.
2. **LangGraph + Postgres:** For the reasoning and memory.
3. **Redis:** To manage the queue of EDA tool jobs.
4. **Docker SDK:** To spin up the "Hands" (Yosys/OpenROAD).

## FastAPI

This **FastAPI boilerplate** wraps the LangGraph into a web-accessible "Shell," allowing our orchestrator to start/stop/resume SoC design cycles via an API.

To build a functional shell for an Agentic SoC Designer, we'll need a **Service-Oriented Architecture**. This moves our project from a "script that runs once" to a "platform that orchestrates."

The following **FastAPI + LangGraph** boilerplate provides the "Shell." It handles the threading, state persistence (via a checkpointer), and provides an endpoint to interact with the design agents.

### 1. The Shell Service Architecture

This shell acts as the central hub. It accepts a design requirement, initializes the state in a database, and manages the execution loop.

```python
from fastapi import FastAPI, BackgroundTasks
from langgraph.checkpoint.memory import MemorySaver # For local dev, use PostgresSaver for prod
from typing import Dict, Any

app = FastAPI(title="Agentic SoC Designer Shell")

# 1. The Persistence Layer (The Shell's Memory)
# This allows the shell to remember the SoC state across API calls
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

# 2. Storage for active design threads
design_sessions: Dict[str, Any] = {}

@app.post("/design/start")
async def start_design_cycle(project_id: str, constraints: dict, background_tasks: BackgroundTasks):
    """
    Initializes a new SoC design project and starts the agentic loop.
    """
    initial_state = {
        "rtl_code": "", 
        "constraints": constraints,
        "current_metrics": {},
        "history": [],
        "next_action": "architect"
    }
    
    config = {"configurable": {"thread_id": project_id}}
    
    # We run the graph in the background so the API stays responsive
    background_tasks.add_task(graph.invoke, initial_state, config)
    
    return {"status": "started", "project_id": project_id}

@app.get("/design/status/{project_id}")
async def get_status(project_id: str):
    """
    Polls the shell to see the current PPA metrics and agent activity.
    """
    config = {"configurable": {"thread_id": project_id}}
    state = graph.get_state(config)
    return {"current_state": state.values if state else "Project not found"}

```

---

### 2. Integration: Connecting the "Brain" to the "EDA Container"

Inside your LangGraph nodes, the shell needs to talk to the **Execution Plane**. You should use a **Task Queue** (like Celery or a simple Background Task) because EDA tools can take minutes or hours.

### 3. Key Components of the Working Shell

To make this shell "Production Ready," you need these three system-level features:

| Feature | Implementation | Purpose |
| --- | --- | --- |
| **Streaming Logs** | WebSockets | Allows you to see the Verilator/Yosys output in real-time in your terminal or UI. |
| **State Snapshotting** | JSON Binary in SQL | Saves every iteration of the RTL so you can "diff" changes made by the AI. |
| **Resource Locking** | Redis Locks | Ensures two agents don't try to use the same FPGA/Synthesis license at the same time. |

---

### 4. The Operator Interface (The CLI Shell)

Once the backend is running, you interact with the platform through a "Designer's Dashboard." This is where you see the **Convergence Graph**—a visual representation of how the agent is slowly reducing the  (Worst Negative Slack) iteration by iteration.

---

### 5. Why this "Shell" is the State-of-the-Art

1. **Fault Tolerance:** If your server crashes mid-synthesis, the LangGraph checkpointer allows the agent to resume from the last successful EDA report.
2. **Scalability:** You can have 10 different "Architect Agents" working on 10 different sub-modules of the SoC simultaneously, all reporting back to a central "Top-Level Shell."
3. **Auditability:** Every design decision made by the AI is logged with the corresponding EDA proof. This is critical for "Silicon Provenance."

## Docker tool containers

A **Docker Compose file** can orchestrate this entire stack (The FastAPI Shell, the Postgres Database, and the EDA Tool Containers) so we can launch the platform with a single command.

To launch this as a "working shell," we will use **Docker Compose**. This is the glue that ensures your FastAPI Orchestrator, your Database (Memory), and your EDA Toolchain (Execution) can all talk to each other on a private, high-speed network.

### 1. The Multi-Service Architecture

We will split the platform into three distinct services:

1. **The Brain (API):** Runs the LangGraph logic and FastAPI.
2. **The Memory (Postgres):** Stores the persistent state of your chip design.
3. **The Execution Engine (EDA):** A dedicated container pre-loaded with open-source hardware tools (Yosys, OpenROAD, Verilator).

---

### 2. The `docker-compose.yml` File

```yaml
services:
  # The "Shell" - LangGraph Orchestrator
  orchestrator:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://ai_user:chip_pass@db:5432/soc_db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
    volumes:
      - ./design_workspace:/work  # Shared workspace for RTL files

  # The "Memory" - State Persistence
  db:
    image: postgres:15
    environment:
      - POSTGRES_USER=ai_user
      - POSTGRES_PASSWORD=chip_pass
      - POSTGRES_DB=soc_db
    ports:
      - "5432:5432"

  # The "Hands" - EDA Toolset (Headless)
  eda_tools:
    image: openroad/openlane:latest
    volumes:
      - ./design_workspace:/work
    tty: true  # Keeps it alive for the orchestrator to exec into

```

---

### 3. The "Shell" Implementation: `Dockerfile`

Your orchestrator needs a specific environment to bridge Python logic with hardware tools.

```dockerfile
FROM python:3.11-slim

# Install system-level EDA wrappers
RUN apt-get update && apt-get install -y \
    verilator \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

```

---

### 4. Visualizing the Operational Shell

Once you run `docker-compose up`, your "shell" becomes a living entity. You can observe the interaction between the agents and the physical tools in real-time.

---

### 5. Why this is the "True Shell" for SoC

This setup solves the most painful parts of agentic hardware design:

* **Isolation:** If an agent generates a Verilog file that causes a simulator to crash or leak memory, only that specific `eda_tools` container is affected. The "Brain" stays online.
* **Parity:** Every agent in the system sees the same file path (`/work/design.v`). There is no confusion about which version of the code is being synthesized.
* **Scale:** If you need to optimize 100 blocks at once, you can scale the `eda_tools` service to 100 instances in a Kubernetes cluster without changing a single line of LangGraph code.

---

### 6. The "Missing Ingredient": The Monitor

In a real-world SoC company, you need to see the **Lineage**. When a chip fails timing, you need to look back through the shell logs and see: *"At 2:00 PM, the Architect changed the multiplier logic, which caused a 50ps violation at 2:05 PM."* By using **LangSmith** or **LangGraph Studio** connected to this shell, you get a full visual audit trail of every gate the AI touched.

## Bootstrap Script

The bootstrap script that initializes the `design_workspace` with a sample RISC-V core and a set of PPA targets so we can run our first "Agentic Optimization Loop".

To get the platform from "architecture" to "active silicon optimization," you need a **Bootstrap Script**. This script sets up the physical directory structure, populates it with a target design (e.g., a RISC-V ALU), and injects the initial constraints into the Shell's database.

### 1. The Bootstrap Script (`bootstrap.py`)

This script prepares the shared volume (`/work`) that the Orchestrator and the EDA containers both use to "hand off" files.

```python
import os
import json

def setup_workspace(project_name="riscv_optimizer"):
    workspace = f"./design_workspace/{project_name}"
    os.makedirs(workspace, exist_ok=True)
    
    # 1. Create a sample failing RTL (Deep logic depth)
    rtl_content = """
    module alu_core (
        input [31:0] a, b, c, d,
        output [31:0] result
    );
        // This deep combinational path will likely fail timing at high freq
        assign result = (a * b) + (c * d); 
    endmodule
    """
    with open(f"{workspace}/alu_core.v", "w") as f:
        f.write(rtl_content)

    # 2. Define the PPA Targets (The 'Goal' for the Agents)
    targets = {
        "target_clock_period": 1.0, # 1GHz in ns
        "max_area": 5000,
        "max_power": 0.05
    }
    with open(f"{workspace}/targets.json", "w") as f:
        json.dump(targets, f)

    print(f"🚀 Workspace {project_name} initialized for Agentic Shell.")

if __name__ == "__main__":
    setup_workspace()

```

---

### 2. The Operational Loop in the Shell

Once the workspace is ready, the **Shell** (FastAPI/LangGraph) begins the optimization loop. Here is how the data flows between the containers you just defined in the `docker-compose`.

1. **Orchestrator** reads `alu_core.v`.
2. **Architect Agent** sees the `target_clock_period` of 1.0ns and realizes the current multiplier-adder chain is too slow.
3. **Architect** writes a "Pipelined" version of the code to the shared volume.
4. **Orchestrator** triggers the `eda_tools` container to run **OpenROAD/Yosys**.
5. **Critic Agent** parses the resulting `.rpt` file from the volume and calculates the Slack.

---

### 3. Monitoring Convergence via the Shell

As the shell runs, you want to see the "Optimization Trace." Because we are using **LangGraph**, every state change is recorded. You can monitor the progress by hitting the `/design/status` endpoint we built earlier.

| Iteration | Agent Action | WNS (Slack) | Result |
| --- | --- | --- | --- |
| 1 | Initial Design | -450ps | **FAIL** (Redesign needed) |
| 2 | Added Pipeline Stage 1 | -120ps | **FAIL** (Improving) |
| 3 | Added Pipeline Stage 2 | +15ps | **PASS** (Criteria met) |

---

### 4. SOTA Capability: "Cross-Container Command Execution"

To make the shell work, the Orchestrator needs to "reach into" the EDA container. In production, your Python shell uses the **Docker API** to run commands inside the tools container:

```python
# How the Shell talks to the EDA 'Hands'
import docker
client = docker.from_env()

def run_synthesis_in_shell(project_id):
    container = client.containers.get("eda_tools")
    # Execute synthesis script inside the already running EDA container
    exit_code, output = container.exec_run("bash /work/run_synth.sh")
    return output

```

---

### 5. Final Platform Checklist

With the shell, the docker-compose, and the bootstrap script, you now have:

* **Persistent Reasoning:** LangGraph (Brain)
* **Scalable Tooling:** Docker/OpenROAD (Hands)
* **Hardware Memory:** Postgres/RAG (Experience)
* **Operational Control:** FastAPI (Shell)

### What would you like to build next?

We have reached a fully functional infrastructure baseline. Would you like to:

1. **Refine the Logic:** Write the specific **Python Parser** that turns a text-based "Timing Report" into the JSON metrics the Critic needs?
2. **Add Verification:** Build the **Testbench Agent** that ensures the optimized "Pipelined" ALU still gives the correct mathematical results?
3. **Human Interface:** Create a **Gradio or Streamlit dashboard** to visualize the chip's floorplan as the agent moves blocks around?