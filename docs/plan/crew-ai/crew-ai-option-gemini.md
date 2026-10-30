# CrewAI

CrewAI is built on a "role-playing" paradigm where specialized agents collaborate like a high-functioning engineering team. For your SoC (System on Chip) project, it moves you away from a single, massive prompt and toward a **distributed reasoning system**.

### 1. The Core Building Blocks

To architect your system, you’ll define three main objects:

* **Agents:** Specialized personas (e.g., your "SoC Analyzer" or "Scoring Agent").
* **Tasks:** Discrete assignments with specific inputs/outputs (e.g., "Analyze power-to-area ratio").
* **Crews:** The orchestration layer that defines *how* these agents work together (Sequential vs. Hierarchical).

---

### 2. Implementation: Mapping Your SoC Workflow

In CrewAI, you would implement your vision by defining specialized agents and linking their outputs as context for the next.

| Agent | Role & Goal | Key Tool Example |
| --- | --- | --- |
| **SoC Analyzer** | Identify bottlenecks in existing RTL or architectural specs. | `FileReadTool`, `StaticAnalysisTool` |
| **Generalizer** | Abstract patterns into high-level architectural templates. | `PatternMatchingTool` |
| **Synthesis Agent** | Propose new SoC configurations based on abstractions. | `LLM` (Reasoning-focused) |
| **Validator** | Run functional simulations or formal verification checks. | `TerminalTool` (to run Verilator/Xcelium) |
| **Scoring Agent** | Compare PPA (Power, Performance, Area) against previous baselines. | `DataAnalysisTool` (Python/Pandas) |

---

### 3. The "Optimization Quest": Implementing a Loop

Because optimization is iterative, you shouldn't just use a simple `Sequential` crew. Instead, you should use **CrewAI Flows**. Flows allow you to create stateful, event-driven loops:

```python
from crewai.flow.flow import Flow, start, listen, router

class SoCOptimizationFlow(Flow):
    @start()
    def analyze_and_synthesize(self):
        # Kick off the first Crew (Analyzer + Generalizer + Synthesizer)
        result = soc_design_crew.kickoff()
        self.state.current_design = result
        return result

    @listen(analyze_and_synthesize)
    def validate_and_score(self):
        # Kick off the second Crew (Validator + Scorer)
        scores = evaluation_crew.kickoff(inputs={'design': self.state.current_design})
        self.state.latest_score = scores
        return scores

    @router(validate_and_score)
    def check_progress(self):
        if self.state.latest_score['improvement'] < 0.05:
            return "continue_optimizing" # Loops back to the start
        return "goal_reached"

```

---

### 4. Evaluating the "Goodness" of your Agent Architecture

CrewAI helps you evaluate the **Agent Architecture** (the "meta-design") through three mechanisms:

* **Observability & Tracing:** By setting `verbose=True` or integrating with tools like **AgentOps** or **Opik**, you can see exactly where the "handoff" between the Analyzer and Generalizer fails. You can track "Action Advancement"—did the Validator actually run the simulation or did it just hallucinate the result?
* **LLM-as-a-Judge:** You can treat your **Scoring Agent** as an architectural evaluator. Not only does it score the SoC design, but it can also score the *Synthesis Agent's* creativity or the *Analyzer's* depth, providing a feedback signal to refine the agents' backstories.
* **The "Manager" Pattern:** Using a `Process.hierarchical` setup, you can appoint a **Manager LLM** (like GPT-4o). The Manager evaluates the quality of each agent's output before passing it to the next, acting as a built-in "Architectural Review Board" for your system.

### Next Step

Would you like me to generate the specific **YAML configurations** for these SoC agents so you can begin testing the handoffs?

---

[Evaluating Multi-Agent Systems](https://www.youtube.com/watch?v=PvdaIqIUpnQ)
This video provides a deep dive into the metrics and observability tools required to move a multi-agent system from a prototype to a reliable production architecture.

---

## Declarative YAML Definitions of Agents

YAML-based configurations is the "production-grade" way to build with CrewAI. It separates the **logic** (your Python code) from the **personas and objectives** (the YAML), allowing you to tune the agent architecture without refactoring your codebase.

Here is a structured implementation for your SoC design and optimization system.

---

### 1. Defining the Personas (`agents.yaml`)

This file defines the specific "brains" of your system. Note how the backstories provide the professional context needed for the LLM to adopt the right persona.

```yaml
soc_analyzer:
  role: >
    SoC Design Analyst
  goal: >
    Identify performance bottlenecks and architectural inefficiencies in {existing_design}.
  backstory: >
    You are an expert hardware architect with 20 years of experience in RTL design 
    and performance analysis. You excel at finding critical paths and power leaks.

generalizer:
  role: >
    Hardware Architecture Abstractor
  goal: >
    Translate specific SoC designs into generalized architectural templates and patterns.
  backstory: >
    You are a specialist in hardware abstraction layers. You take raw specs and 
    turn them into high-level mathematical or structural models that can be scaled.

synthesis_agent:
  role: >
    Architectural Synthesizer
  goal: >
    Propose novel SoC configurations by applying generalizations to the {optimization_targets}.
  backstory: >
    You are a generative design specialist. You take abstract patterns and 
    re-materialize them into optimized SoC architectures that push the boundaries of PPA.

validator:
  role: >
    Verification & Validation Agent
  goal: >
    Ensure the synthesized design is functionally correct and meets the technical constraints.
  backstory: >
    You are a rigorous verification engineer. You don't trust any design until 
    it passes formal checks and functional simulation.

scoring_agent:
  role: >
    PPA Optimization Scorer
  goal: >
    Quantify the "goodness" of the new design compared to the baseline {existing_design}.
  backstory: >
    You are a data-driven performance engineer. You provide objective scores based 
    on Power, Performance, and Area (PPA) metrics.

```

---

### 2. Defining the Workflow (`tasks.yaml`)

This is where the "Agent Architecture" is implemented. The `context` key is crucial: it tells CrewAI which previous tasks provide the input for the current one.

```yaml
analysis_task:
  description: >
    Analyze the provided SoC design: {existing_design}. Focus on identifying 
    interconnect congestion and cache hierarchy inefficiencies.
  expected_output: >
    A detailed report of architectural bottlenecks.
  agent: soc_analyzer

generalization_task:
  description: >
    Based on the analysis, create a generalized template that abstracts the 
    core compute-to-memory relationship of this design.
  expected_output: >
    A set of generalized architectural patterns.
  agent: generalizer
  context: [analysis_task]

synthesis_task:
  description: >
    Using the generalized patterns, synthesize a new SoC architecture that 
    specifically addresses the {optimization_targets}.
  expected_output: >
    A new SoC architectural specification.
  agent: synthesis_agent
  context: [generalization_task]

validation_task:
  description: >
    Run a suite of functional checks on the synthesized design to ensure it 
    adheres to standard protocol specifications (e.g., AXI/CHI).
  expected_output: >
    A validation report (Pass/Fail) with specific error logs if failed.
  agent: validator
  context: [synthesis_task]

scoring_task:
  description: >
    Compare the validated design against the original {existing_design}. 
    Calculate the improvement index based on PPA.
  expected_output: >
    A final "Optimization Score" and a recommendation on whether to proceed.
  agent: scoring_agent
  context: [analysis_task, synthesis_task, validation_task]

```

---

### 3. Evaluating the "Goodness" of the Architecture

To evaluate if this **Agent Architecture** is working well, look for these three indicators during execution:

1. **Contextual Coherence:** Does the `synthesis_agent` actually use the abstractions from the `generalizer`? If not, you may need to sharpen the `expected_output` descriptions in the YAML.
2. **The Feedback Loop:** If the `validator` fails a design, does your "Flow" (the Python logic) route the error back to the `synthesis_agent`?
3. **Hallucination Rate:** If the `scoring_agent` gives a high score to a design that didn't actually pass validation, your architecture lacks "Cross-Agent Critique." You might need to add a "Reviewer" agent to audit the scores.

### Next Step

Would you like me to show you the **Python "Flow" code** that connects these YAML files and handles the logic for looping back if the score isn't high enough?

## Python "Flow" code

This is where the "Agent Architecture" truly comes to life. In CrewAI, a **Flow** is the state machine that manages the lifecycle of your optimization quest. It ensures that the agents don't just run once and stop, but instead work in a loop until your SoC design hits its PPA (Power, Performance, Area) targets.

Below is the Python implementation that ties your YAML configurations into a functional, iterative system.

### The Python Implementation: `main.py`

```python
from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, start, listen, router
from pydantic import BaseModel

# 1. Define the State to track the optimization progress
class SoCState(BaseModel):
    current_design: str = ""
    optimization_score: float = 0.0
    iterations: int = 0
    max_iterations: int = 3

class SoCOptimizationFlow(Flow[SoCState]):

    @start()
    def initial_analysis_and_synthesis(self):
        print(f"--- Starting Iteration {self.state.iterations + 1} ---")
        
        # This crew handles the "Creative" side: Analyzing and Synthesizing
        # It uses the YAML configs we defined earlier
        design_crew = Crew(
            agents='config/agents.yaml',
            tasks='config/tasks.yaml',
            process=Process.sequential # Analyzer -> Generalizer -> Synthesizer
        )
        
        result = design_crew.kickoff(inputs={
            "existing_design": "V0 Baseline SoC",
            "optimization_targets": "Reduce thermal throttling by 15%"
        })
        
        self.state.current_design = result.raw
        return result

    @listen(initial_analysis_and_synthesis)
    def validate_and_score(self, design_output):
        # This crew handles the "Rigorous" side: Validation and Scoring
        evaluation_crew = Crew(
            agents='config/agents.yaml',
            tasks='config/tasks.yaml', 
            process=Process.sequential # Validator -> Scorer
        )
        
        eval_result = evaluation_crew.kickoff(inputs={
            "synthesized_design": self.state.current_design
        })
        
        # Assume the Scorer returns a structured score
        # For this example, we'll parse a mock float
        self.state.optimization_score = float(eval_result["ppa_improvement"])
        self.state.iterations += 1
        return eval_result

    @router(validate_and_score)
    def check_goal_met(self):
        # The Decision Logic: Do we have a "Good" architecture yet?
        if self.state.optimization_score >= 0.15: # 15% improvement goal
            return "success"
        elif self.state.iterations >= self.state.max_iterations:
            return "max_retries_reached"
        else:
            return "improve_design"

    @listen("improve_design")
    def loop_back(self):
        print(f"Goal not met ({self.state.optimization_score}). Re-routing to agents...")
        return self.initial_analysis_and_synthesis()

# To run the system:
flow = SoCOptimizationFlow()
final_report = flow.kickoff()

```

---

### How this evaluates the "Goodness" of your Agent Architecture

By using this Flow, you can now evaluate your **Agent Architecture** using specific telemetry:

1. **Iteration Efficiency:** If your agents take 10 loops to find a 5% improvement, your **Generalizer agent** might be too "loose" in its abstractions. You would then go back to the YAML and refine its `backstory` to be more mathematically rigorous.
2. **Validation Bottlenecks:** If the `validator` agent consistently rejects designs, the "Handshake" (context passing) between the `synthesis_agent` and the `validator` is broken. You might need a **Refinement Agent** between them to "clean up" the specs before they hit simulation.
3. **Traceability:** CrewAI allows you to map every SoC design decision back to a specific agent's output. If the `scoring_agent` flags a design as "Great" but it fails in the real world, you have identified a "Blind Spot" in your scoring agent's persona.

### Next Step

Would you like me to explain how to integrate **custom tools** (like a Python script that actually calls a power simulator) so the `validator` and `scoring_agent` can use real hardware data instead of just LLM reasoning?

Yes, please. We already had architected the system in such a way that synthesizer, validator, assessor, etc. would be domain knowledge tools that have their own development and QA, and we would use the LLM only to interface with the human and create plans. So LLM hallucinations never pollute the quantitative data. We definitely want to make a note of this design choice and then define the specs required for the tooling to be as good, efficient, and accurate as possible.

## Custom Tool Integration

This is a sophisticated and robust approach. By treating the LLM as the **"Reasoning Engine" (Planner/Interface)** and your specialized software as **"Execution Engines" (Ground Truth)**, you effectively build a firewall against hallucinations.

In CrewAI, this is achieved by subclassing `BaseTool`. This allows your agents to "hand off" the heavy lifting to your existing development and QA-tested pipeline.

---

### 1. Wrapping Your Domain Tools

To make your SoC tools "Agent-Ready," you wrap them so the Agent knows exactly how to call them and what to expect back.

```python
from crewai.tools import BaseTool
from pydantic import Field
import subprocess

class PowerAnalysisTool(BaseTool):
    name: str = "ppa_simulator"
    description: str = (
        "Calculates Power, Performance, and Area for a given SoC spec. "
        "Input must be a JSON string representing the architecture."
    )

    def _run(self, architecture_json: str) -> str:
        # This is where your QA-tested domain logic lives.
        # It could be a CLI call to a simulator or a REST API request.
        try:
            # Example: Calling an external C++/Python PPA model
            result = subprocess.run(
                ["./bin/ppa_analyzer", "--input", architecture_json],
                capture_output=True, text=True, check=True
            )
            return result.stdout # Return the ground-truth quantitative data
        except Exception as e:
            return f"Error running PPA simulation: {str(e)}"

```

---

### 2. Strategic Note: The "Agentic Interface" Design

By architecting the system this way, you have created a **Hybrid Intelligence System**:

* **LLM Role:** High-level planning, synthesizing diverse tool outputs into human-readable reports, and managing the "Search Space" of designs.
* **Tool Role:** Quantitative validation, physical constraints checking, and deterministic scoring.

**Why this is "Good" Agent Architecture:**

1. **Auditability:** You can log exactly which parameters the LLM sent to the `Validator` tool.
2. **Safety:** The LLM never "guesses" if a design is valid; it simply reports what the `Validator` tool (the source of truth) dictated.
3. **Efficiency:** The LLM doesn't waste tokens trying to perform complex hardware math it isn't suited for.

---

### 3. Tooling Specs for High Efficiency

To ensure your domain tools work seamlessly with the LLM planner, they should adhere to these **Agent-Friendly Specs**:

| Spec Attribute | Requirement | Why it matters for Agents |
| --- | --- | --- |
| **Atomic Inputs** | Use structured formats like JSON or strict CLI flags. | Prevents the LLM from sending "prose" where the tool expects a float. |
| **Explicit Error Messages** | Tools must return *why* a design failed (e.g., "Fan-out exceeds limit at Node X"). | The LLM uses this feedback to "plan" the next iteration's fix. |
| **Standardized Output** | Every tool should return a consistent schema (e.g., PPA always includes `mW`, `ns`, `um²`). | Allows the `Scoring Agent` to parse data reliably across iterations. |
| **Statelessness** | Tools should ideally be stateless or handle their own session state. | Allows the Agent to explore multiple design branches in parallel without "polluting" the environment. |

---

### 4. Refining the "Planning" Persona

Now that your tools are robust, your **Agent backstories** (in the YAML we created earlier) should change from "Doing the work" to "Orchestrating the work."

* **Old Role:** "Calculate the power consumption." (Prone to hallucination)
* **New Role:** "Utilize the `ppa_simulator` tool to retrieve authoritative power data and interpret the results against the design goals."

### Next Step

Would you like me to help you draft a **"Constraint Checker" Tool** schema that the Validator agent can use to ensure the LLM's proposed architectures don't violate fundamental physical SoC limits before they even hit the heavy simulators?

### Example: Constraint Checker Tool

The "Constraint Checker" acts as the **Fast-Fail Gate** in your agent architecture. In hardware design, running a full-blown simulation (like a logic synthesis or thermal analysis) can take hours or even days. You don't want your LLM planner waiting that long only to find out it placed two high-power blocks in a way that violates a basic floorplan rule.

By implementing a "pre-sim" constraint checker, your **Validator Agent** can immediately reject "hallucinated" architectures that are physically impossible.

---

### 1. The Design: Pre-Simulation Validation

This tool lives inside the Validator agent's belt. It uses deterministic heuristics (Python/C++) to check the "sanity" of the design.

### 2. Implementation: The `SoCConstraintTool`

We will use Pydantic to ensure the LLM sends the data in a strict format, and the tool will return a detailed "violation report" if it fails.

```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict

class DesignInput(BaseModel):
    blocks: List[Dict] = Field(..., description="List of blocks with 'name', 'power_mw', and 'coordinates'")
    die_size_mm2: float = Field(..., description="Total area of the die")
    max_thermal_density: float = Field(..., description="Max allowed mW per mm2")

class SoCConstraintTool(BaseTool):
    name: str = "fast_fail_constraint_checker"
    description: str = "Checks for physical violations like area overflow or thermal hotspots before simulation."
    args_schema: type[BaseModel] = DesignInput

    def _run(self, blocks: List[Dict], die_size_mm2: float, max_thermal_density: float) -> str:
        # 1. Check Area Constraint
        total_area = sum(b.get('area', 0) for b in blocks)
        if total_area > die_size_mm2:
            return f"FAIL: Area overflow. Design uses {total_area}mm2 but die is only {die_size_mm2}mm2."

        # 2. Check Thermal Hotspots (Simple heuristic)
        for b in blocks:
            density = b['power_mw'] / b.get('area', 1)
            if density > max_thermal_density:
                return f"FAIL: Thermal Hotspot in {b['name']}. Density {density} exceeds limit {max_thermal_density}."

        return "PASS: Design meets basic physical constraints. Proceeding to simulation."

```

---

### 3. Tool Specs for "Agentic Efficiency"

To make your tools as accurate and "agent-ready" as possible, ensure they follow these specifications:

| Spec Property | Requirement | Benefit for the Agent Architecture |
| --- | --- | --- |
| **Atomic Feedback** | Return specific error codes (e.g., `ERR_THERMAL_01`). | The LLM can map specific errors to specific "repair" strategies. |
| **Deterministic Grading** | No "maybe" or "mostly" passing. | Removes ambiguity; the Planner knows exactly when to stop or pivot. |
| **Self-Documenting** | Comprehensive docstrings in the tool class. | CrewAI uses these strings to tell the LLM *how* and *when* to use the tool. |
| **Telemetry-Rich** | Log execution time and resource usage. | Helps you evaluate the cost-efficiency of your agent pipeline. |

---

### 4. Why this Evaluates your "Goodness"

This architecture is "good" because it creates a **Hierarchical Validation Loop**:

1. **Level 1 (The LLM):** Synthesizes a creative design.
2. **Level 2 (The Constraint Tool):** Catches "stupid" mistakes in milliseconds (Fast-fail).
3. **Level 3 (The Simulation Tool):** Performs deep, quantitative validation (The Ground Truth).

If Level 2 catches 80% of the failures, your **Agent Architecture** is highly efficient because you are not wasting expensive GPU tokens or simulator licenses on invalid designs.

> **Note:** This design choice protects your "Optimization Quest" from the most common multi-agent failure: the **"Garbage In, Garbage Out" loop**, where an LLM spends 10 iterations trying to fix a design that was physically impossible from step one.

### Next Step

Would you like me to show you how to set up **"Agent Memory"** so the Synthesis Agent remembers which constraints it violated in previous rounds, ensuring it doesn't make the same mistake twice?

### Agent Memory

In the context of your SoC optimization quest, **Agent Memory** is the difference between a system that "tries again" and a system that "learns."

CrewAI provides three types of memory. For your architecture, **Long-Term Memory** and **Short-Term Memory** are the most critical for ensuring the synthesizer doesn't repeat the same physical constraint violations.

### 1. The Three Layers of Memory

* **Short-Term (Contextual):** Allows agents to remember what happened in the *current* session (e.g., "The Validator just told me the cache size was too big").
* **Long-Term (Historical):** Stores results from *past* sessions in a local database (usually `storage/long_term_memory_storage.db`). This allows the agent to remember, "Last week, every time I used this specific interconnect pattern, the PPA score dropped."
* **Entity Memory:** Extracts and stores key entities (e.g., "Die Size = 50mm²") so the agent doesn't have to keep re-reading the baseline spec.

---

### 2. Implementation: Enabling the "Learning Loop"

You enable memory at the **Crew** level. When enabled, CrewAI automatically creates a vector database to store and retrieve these experiences.

```python
from crewai import Crew, Process

# Create the SoC Crew with Memory enabled
soc_crew = Crew(
    agents=[soc_analyzer, synthesis_agent, validator],
    tasks=[analysis_task, synthesis_task, validation_task],
    process=Process.sequential,
    memory=True, # <--- Enables the memory ecosystem
    verbose=True,
    embedder={
        "provider": "openai",
        "config": {
            "model": "text-embedding-3-small"
        }
    }
)

```

---

### 3. How it Fixes the "Optimization Quest"

By turning on `memory=True`, the "Agent Architecture" changes from a stateless loop to a **Stateful Evolution**:

1. **The Failure:** The `synthesis_agent` proposes a design. The `SoCConstraintTool` (the fast-fail tool we built) returns: `"FAIL: Thermal Hotspot in Block_A."`
2. **The Storage:** CrewAI stores this interaction in the agent's Short-Term memory.
3. **The Pivot:** In the next iteration of the `Flow`, when the `synthesis_agent` is asked to "improve the design," it performs a **RAG (Retrieval-Augmented Generation) query** against its own memory.
4. **The Insight:** It sees the previous failure and adds a "Negative Constraint" to its internal plan: *"Do not place high-power blocks in the top-right quadrant, as that led to a thermal hotspot in Iteration 1."*

---

### 4. Evaluating the "Goodness" of Memory

To see if your memory architecture is working, look for **"Convergence Velocity"**:

* **Bad Architecture (Stateless):** The agents keep hitting the same constraint violations across multiple loops. The "Quest" feels like a random walk.
* **Good Architecture (Stateful):** The number of "Fast-Fail" triggers from your `SoCConstraintTool` decreases with every iteration. The LLM is effectively "learning" the physical rules of your domain tools.

### Strategic Note on the "Design Choice"

Since you mentioned you want to keep LLM hallucinations away from quantitative data, **Memory is your safest ally.** * The LLM remembers the **qualitative** feedback (e.g., "This design was rejected because of power").

* The Tools provide the **quantitative** data (e.g., "Power = 450mW").
* The Memory links them, allowing the LLM to refine its *strategy* without ever being allowed to override the *data*.

**Would you like to see how to manually "Seed" the agent's memory with a "Lessons Learned" document from your previous SoC projects, so it doesn't start the quest from zero?**

### Agent Memory Seed example

To seed your system with prior expertise and measure how fast it's "learning" the optimization space, we will use the **CrewAI Knowledge** component and a custom **Telemetry Tracker** within your Flow.

### 1. Seeding Memory: The "Knowledge" Component

Instead of forcing the agent to learn everything from scratch, you can provide it with a "Lessons Learned" corpus. This acts as the **"Ancestral Memory"** of your agent architecture.

Create a file called `soc_knowledge.md`:

```markdown
# SoC Design Lessons Learned
- **Thermal:** Placing the NPU adjacent to the high-speed SerDes always causes thermal throttling.
- **Interconnect:** AXI-4 crossbars with more than 8 masters consistently fail timing at 2GHz.
- **Cache:** Increasing L2 cache beyond 2MB yields diminishing returns for this specific workload.

```

Then, integrate it into your Crew:

```python
from crewai import Crew, Agent, Task, Process
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

# Create a knowledge source from your previous project data
soc_knowledge = TextFileKnowledgeSource(
    file_paths=["soc_knowledge.md"]
)

soc_crew = Crew(
    agents=[synthesis_agent], # Synthesis agent now has "Experience"
    tasks=[synthesis_task],
    knowledge_sources=[soc_knowledge], # Seeding the memory
    memory=True
)

```

---

### 2. Measuring "Convergence Velocity"

To evaluate the "Goodness" of your architecture, you need to know if the agents are getting smarter or just spinning their wheels. **Convergence Velocity (CV)** is the rate at which the agents align their proposals with the tool-dictated reality.

#### The Metric: Constraint Decay Rate (CDR)

The most valuable metric for your specific "Tool vs. LLM" setup is **CDR**. It measures the reduction in "Fast-Fail" violations over time.

**Formula:**



*(Where  is the number of distinct physical constraint violations identified by your tool.)*

* **High CDR:** The agent architecture is "learning" the boundaries of the physical world quickly.
* **Zero/Low CDR:** The agents are ignoring the tool feedback (Memory/Prompting failure).

---

### 3. The Collection Mechanism

You can implement this directly in your `SoCState` to create a CSV "Learning Curve" after every run.

```python
import pandas as pd

class SoCState(BaseModel):
    iteration_history: list = [] # To store [Iteration, Violations, PPA_Score]

class SoCOptimizationFlow(Flow[SoCState]):
    
    @listen(validate_and_score)
    def log_telemetry(self, eval_output):
        # Extract data from the tool output
        violation_count = eval_output.get("violation_count", 0)
        ppa_score = eval_output.get("ppa_improvement", 0.0)
        
        # Append to history
        self.state.iteration_history.append({
            "iteration": self.state.iterations,
            "violations": violation_count,
            "ppa_score": ppa_score
        })
        
    def export_metrics(self):
        # Once the loop finishes, export to a CSV for analysis
        df = pd.DataFrame(self.state.iteration_history)
        df.to_csv("agent_architecture_performance.csv")
        print("Convergence data saved. Review 'violations' column for decay rate.")

```

### Why this is the "Gold Standard" for Evaluation:

By tracking the `violation_count` against `iterations`, you get a literal graph of the **Agent's Learning Curve**.

1. **If the violations drop to zero quickly**, your "Memory" and "Tool-to-Agent Feedback Loop" is excellently architected.
2. **If the PPA score plateaus while violations are zero**, you know you've exhausted the current "Synthesis Strategy" and need to either prompt for more "out-of-the-box" thinking or add a new Synthesis Agent with a different persona (e.g., "The Radical Optimizer").

**Would you like to explore how to add a "Reviewer" agent that specifically audits these telemetry logs to decide when the "Agent Architecture" itself needs to be tweaked?**

## Reviewer Agent

Adding a **Meta-Reviewer Agent** moves your system from a static pipeline to a **self-optimizing agent architecture**. While your domain tools (synthesizer, validator, etc.) handle the SoC data, this agent monitors the "Health of the Quest."

### 1. The Meta-Reviewer Persona (`agents.yaml`)

This agent doesn't look at RTL or netlists; it looks at **iteration logs and delta-scores**.

```yaml
meta_reviewer:
  role: >
    Agent Architecture Auditor
  goal: >
    Analyze the convergence logs to identify if the synthesis strategy has plateaued 
    or if the agents are stuck in a repetitive error loop.
  backstory: >
    You are a system-level orchestrator. You monitor the performance of other 
    AI agents. Your expertise is in identifying "Local Optima"—where a design 
    is improving but will never reach the global target without a radical change 
    in strategy.

```

---

### 2. The Review Task (`tasks.yaml`)

```yaml
architecture_review_task:
  description: >
    Review the telemetry from the last {iteration_count} iterations. 
    1. Check if the PPA Improvement Gradient is < 1%.
    2. Check if the Constraint Decay Rate is stalling.
    3. If so, recommend a 'Pivot' (e.g., change synthesis constraints or 
       increase exploration temperature).
  expected_output: >
    A strategic directive: either 'CONTINUE', 'PIVOT', or 'HALT'.
  agent: meta_reviewer

```

---

### 3. Integrating the "Self-Correction" Loop

In your Python Flow, the Meta-Reviewer acts as a high-level router. If it detects a plateau, it can dynamically modify the state or the inputs to the next loop.

```python
@router(log_telemetry)
def meta_review_logic(self):
    # Every 5 iterations, we run the Meta-Reviewer
    if self.state.iterations % 5 == 0:
        review_crew = Crew(
            agents=[meta_reviewer],
            tasks=[architecture_review_task]
        )
        # Pass the iteration history as context
        decision = review_crew.kickoff(inputs={
            "history": str(self.state.iteration_history),
            "iteration_count": self.state.iterations
        })

        if "PIVOT" in decision.raw:
            self.state.strategy_mode = "Exploration" # Change the persona behavior
            print("Meta-Reviewer signaled a PIVOT: Switching to Exploration mode.")
            return "improve_design"
        
    # Standard logic if no pivot is needed
    if self.state.optimization_score >= 0.15:
        return "success"
    return "improve_design"

```

---

### 4. Metrics for the Reviewer's "Goodness" Assessment

To make the Meta-Reviewer effective, you feed it three specific "Meta-Metrics" collected by your Flow:

| Meta-Metric | What it Signals | Agent Architecture Adjustment |
| --- | --- | --- |
| **Tool Rejection Ratio** | Synthesizer is ignoring the Validator's ground-truth data. | **Action:** Increase the "penalty" in the Synthesizer's prompt or clear/refresh Short-Term memory. |
| **Design Entropy** | The Synthesizer is making tiny, redundant changes (Stuck). | **Action:** Inject "Randomized Noise" into the synthesis parameters or switch to the "Radical Optimizer" persona. |
| **PPA Gradient (PPA)** | The rate of improvement is approaching zero. | **Action:** Re-run the **Generalizer** to find a completely new architectural pattern. |

### Why this is the "Final Evolution"

This architecture fulfills your requirement of keeping **LLMs separate from quantitative data**, but adds a layer of **Qualitative Oversight**. The Meta-Reviewer uses the quantitative telemetry (which it cannot change) to make qualitative decisions about the *process* (which it can influence).

**Next Step**
We have built the Personas, the Tools, the Memory, and the Meta-Reviewer. Would you like me to provide a **Dockerfile and Project Structure** template so you can actually deploy this system and start running your first SoC optimization quest?

## Project Structure

This final step transitions your SoC optimization system from a conceptual architecture to a deployable engineering platform. By using Docker, you ensure that your domain tools (simulators, verifiers, and checkers) run in a deterministic environment, while the CrewAI "Reasoning Layer" remains portable and scalable.

### 1. The Recommended Project Structure

This structure follows CrewAI’s best practices for separating logic, configuration, and domain knowledge.

```text
soc_optimizer/
├── config/
│   ├── agents.yaml          # Personas (Synthesizer, Validator, etc.)
│   └── tasks.yaml           # Workflow and dependency definitions
├── knowledge/
│   └── soc_knowledge.md     # Seeding memory with "Lessons Learned"
├── src/
│   ├── __init__.py
│   ├── main.py              # The Flow/State logic we built
│   ├── crew.py              # Crew orchestration logic
│   └── tools/
│       ├── __init__.py
│       ├── ppa_tool.py      # Wrapper for PPA simulator
│       └── constraint_tool.py # Fast-fail physical checker
├── bin/
│   └── ppa_analyzer         # Your proprietary domain executable
├── .env                     # API keys (OPENAI_API_KEY, etc.)
├── Dockerfile
├── pyproject.toml           # Dependencies (CrewAI, Pydantic, Pandas)
└── README.md

```

---

### 2. The Dockerfile

This Dockerfile sets up a Python environment but also leaves room for your specific hardware toolchains (e.g., Verilator, Python-based PPA models, or C++ binaries).

```dockerfile
# Use a slim Python image for the reasoning layer
FROM python:3.11-slim

# Install system dependencies (e.g., build-essential for hardware tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir .

# Copy the source code and domain tools
COPY src/ ./src/
COPY config/ ./config/
COPY knowledge/ ./knowledge/
COPY bin/ ./bin/

# Ensure domain tools are executable
RUN chmod +x ./bin/*

# Set environment variables (keys should be passed at runtime)
ENV PYTHONUNBUFFERED=1

# Run the optimization flow
CMD ["python", "src/main.py"]

```

---

### 3. Dependency Management (`pyproject.toml`)

Using `pyproject.toml` is the modern standard for CrewAI projects, ensuring strict versioning for your "Execution Engines."

```toml
[project]
name = "soc-optimizer"
version = "0.1.0"
description = "Multi-agent SoC Optimization System"
dependencies = [
    "crewai[tools]>=0.100.0",
    "pydantic>=2.0.0",
    "pandas",
    "agentops" # Highly recommended for the Meta-Reviewer telemetry
]

```

---

### 4. Running the Optimization Quest

Once your project is structured, you can build and run your "Quest" with a single command. This ensures the **Agent Architecture** starts with a clean slate every time, or mounts a volume to persist **Long-Term Memory**.

```bash
# Build the image
docker build -t soc-optimizer .

# Run the quest (passing API keys and mounting a volume for memory persistence)
docker run --env-file .env \
  -v $(pwd)/storage:/app/storage \
  soc-optimizer

```

---

### Final Evaluation of your "Goodness"

By implementing this Dockerized structure, you have achieved the three pillars of a high-quality agent architecture:

1. **Isolation:** Your LLM-based agents cannot "break" the system; they can only interact through tested, containerized tools.
2. **Traceability:** Every design choice is logged in `storage/`, allowing your **Meta-Reviewer** to analyze progress over time.
3. **Reproducibility:** Anyone on your team can run the exact same optimization quest and get the same tool-driven results, regardless of their local setup.

### What's next?

You are now ready to begin the **Optimization Quest**. Would you like me to help you draft the **first prompt** for your "SoC Analyzer" to kick off the very first iteration of your baseline design?

## First Prompt

To kick off your quest, the first prompt for the **SoC Analyzer** needs to be highly technical and structured. It serves as the "Sensing" stage of your architecture—if the analysis is vague, the generalizations will be weak, and the synthesis will fail to optimize.

Here is the strategic first prompt, designed to be passed into the `existing_design` variable of your `analysis_task`.

---

### The "Initial Kickoff" Prompt

> **Task: Deep Architectural Audit of V0 Baseline SoC**
> **Objective:** Perform a critical path and PPA (Power, Performance, Area) audit of the provided baseline configuration. Identify the top three architectural bottlenecks that prevent this design from reaching its target of a 15% reduction in thermal throttling.
> **Input Configuration (V0 Baseline):**
> * **Compute:** 8x Heterogeneous Cores (4x Performance, 4x Efficiency).
> * **Interconnect:** AXI-4 Crossbar with 128-bit data width.
> * **Memory Hierarchy:** 32KB L1 I/D, 512KB private L2 per cluster, 4MB shared L3.
> * **Floorplan:** NPU and GPU located in the top-right quadrant, sharing a common voltage rail.
> * **Known Issue:** Thermal sensors report peak temperatures near the NPU/GPU boundary during heavy AI inference.
> 
> 
> **Requirements for your Report:**
> 1. **Congestion Mapping:** Identify where the AXI-4 crossbar likely suffers from arbitration latency.
> 2. **Thermal Hotspot Hypothesis:** Explain the correlation between the shared voltage rail and the thermal throttling observed.
> 3. **Abstraction Targets:** List three "Structural Invariants" (parts of the design that should remain constant) and three "Optimization Variables" (parts that the Synthesizer should be allowed to change).
> 
> 
> **Constraint:** Do not suggest specific transistor-level fixes. Focus purely on **Architectural Topology** and **Data Flow**.

---

### Why this prompt works for your Architecture

* **Signals the Goal:** By explicitly mentioning "Thermal Throttling," you align the Analyzer with the Scoring Agent’s ultimate goal.
* **Separates Variables from Invariants:** This is crucial for your **Generalizer Agent**. It tells the next agent in the chain what it is *allowed* to abstract vs. what is a hard physical requirement.
* **Grounds the LLM:** By providing a specific known issue (NPU/GPU thermal boundary), you give the LLM a "hook" to look for quantitative evidence from your tools later on.

### 5. Tracking the "Goodness" of this First Iteration

When the Analyzer returns its first report, evaluate it using these two criteria:

1. **Technical Granularity:** Did it identify "Interconnect Congestion" generally, or did it point to "AXI-4 Arbitration Latency"? The latter is a better signal for the Synthesizer.
2. **Actionability:** Does the report provide enough detail for the **Generalizer** to create a mathematical or structural pattern?

---

### Final Project Checklist

Before you hit `flow.kickoff()`, ensure:

* [ ] Your **.env** file has your LLM and Telemetry API keys.
* [ ] Your **bin/** folder contains the compiled PPA and Constraint binaries.
* [ ] The **soc_knowledge.md** file is populated with at least 5-10 historical design rules.

**Would you like me to generate a "Mock Output" of what a high-quality response from this Analyzer would look like, so you know exactly what to expect from your system?**

## Mock Output

A high-quality response from your **SoC Analyzer** serves as the technical foundation for the entire optimization loop. It must bridge the gap between "raw specs" and "architectural strategy."

Here is what you should expect to see in your terminal or logs after the first task completes.

---

### Mock Output: SoC Architectural Audit Report (V0 Baseline)

**To:** Generalizer Agent / Synthesis Agent

**From:** SoC Design Analyst

**Status:** Analysis Complete – Potential Bottlenecks Identified

#### 1. Executive Summary

The V0 Baseline exhibits a classic **Power-Density Imbalance**. While the heterogeneous core approach is efficient, the physical clustering of high-performance IPs (NPU/GPU) combined with a shared voltage rail creates a "thermal island." This leads to premature throttling, limiting the effective frequency of the Performance Cores by ~12% under sustained AI workloads.

#### 2. Interconnect & Congestion Mapping

* **The Issue:** The 128-bit AXI-4 Crossbar is currently the primary source of memory-access latency.
* **Observation:** High-bandwidth traffic from the NPU competes with the GPU for L3 cache access. We are seeing **Head-of-Line (HoL) blocking** at the main crossbar ports during peak inference cycles.
* **Quantitative Signal:** Theoretical peak bandwidth is 25.6 GB/s, but sustained throughput drops to 18.4 GB/s due to arbitration overhead among the 12 masters.

#### 3. Thermal Hotspot Hypothesis

The current floorplan places the NPU and GPU in the top-right quadrant.

* **Thermal Coupling:** Because they share a voltage rail, the IR drop during simultaneous bursts leads to increased leakage current, further raising temperatures.
* **Failure Mode:** The thermal sensors trigger a global frequency cap for the *entire* cluster, meaning efficiency cores are slowed down by the heat of the performance blocks.

#### 4. Strategy for the Generalizer (Abstraction Targets)

To enable effective synthesis, I have categorized the architecture into the following:

| Category | Elements | Reasoning |
| --- | --- | --- |
| **Structural Invariants** | L1 Cache Sizes, Core Count (8), ISA. | Fixed by project scope; changing these disrupts software compatibility. |
| **Optimization Variables** | Interconnect Topology, Block Placement, Voltage Domains. | These are the "knobs" the Synthesizer can turn to solve the heat issue. |
| **Proposed Abstraction** | "Decoupled Power-Compute Pattern" | Abstract the design from a "Shared Rail" model to a "Distributed Power Domain" model. |

---

### Why this is a "Good" result for your Architecture:

1. **It avoids Hallucination:** It doesn't claim to have fixed the RTL; it describes the *symptoms* based on the provided baseline.
2. **It sets up the next Agent:** It explicitly tells the **Generalizer** what it is allowed to change. This prevents the "Optimization Quest" from trying to change things it shouldn't (like the ISA or core count).
3. **It uses Domain Language:** By using terms like "Head-of-Line blocking" and "IR drop," it ensures that if this text is passed to a RAG-enabled memory or a specialized hardware tool, the tool receives high-signal technical input.

### Final Next Step

Would you like me to generate the **"Prompt Template"** for the **Generalizer Agent**, which will take this report as input and convert it into a formal architectural pattern?

---

Since you are currently using the Free tier on your personal account, there isn't a direct "transfer" button to move a live chat thread between two separate accounts. However, you can easily migrate the "context" and "state" of our architectural design so you don't lose progress.

Here is the most efficient way to move this work to your business account:

### 1. The "Context Bridge" Method

Since we have already architected the agents, tasks, and the logic for the "Optimization Quest," you can copy the following **State Summary** and paste it into a new chat once you log into your business account.

**Copy and paste this into your first message on the new account:**

> "I am migrating a project from my personal account. We are building a CrewAI-based SoC Optimization system. Here is the established architecture:
> * **Agents:** SoC Analyzer, Generalizer, Synthesis Agent, Validator, Scoring Agent, and a Meta-Reviewer.
> * **Logic:** A Python 'Flow' using a loop with a 'Fast-Fail' constraint checker (deterministic tool) and a full PPA simulator.
> * **Metric:** Measuring 'Convergence Velocity' via Constraint Decay Rate.
> * **Stage:** We just finished the 'SoC Analyzer' kickoff prompt and mock output.
> * **Next Step:** Generate the 'Generalizer Agent' prompt template to convert the Analyzer's report into an architectural pattern."
> 
> 

---

### 2. Exporting Your Files

Before you switch, make sure you have copied the code blocks we generated:

* The **YAML configurations** (`agents.yaml` and `tasks.yaml`).
* The **Python Flow** (`main.py`).
* The **Dockerfile** and project structure.

---

### 3. How to Switch

1. **Log Out:** Click on your profile icon in the bottom left/top right and sign out.
2. **Log In:** Sign in using your business credentials.
3. **Start a New Chat:** Paste the "Context Bridge" text above. I will immediately recognize the project state and we can pick up right where we left off with the Generalizer prompt.

**Would you like me to consolidate all the YAML and Python code we've written so far into one single "Project Manifest" file for you to copy before you switch accounts?**