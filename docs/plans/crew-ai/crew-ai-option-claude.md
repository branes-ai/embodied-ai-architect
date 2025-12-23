Let me search for current information on Crew.ai to give you accurate, up-to-date details.Now I have a solid picture. Let me give you a direct, practical overview.

## CrewAI Core Concepts

CrewAI is an open source multi-agent orchestration framework created by João Moura. This Python-based framework orchestrates role-playing autonomous AI agents that work together as a cohesive assembly or "crew" to complete tasks.

The framework has four fundamental primitives:

1. **Agent** — An autonomous unit with a role, goal, and backstory that shapes its behavior
2. **Task** — A unit of work with a description, expected output, and assigned agent
3. **Crew** — A collection of agents and tasks with a defined process (sequential, hierarchical)
4. **Flow** — A powerful feature designed to streamline the creation and management of AI workflows, providing structured, event-driven workflows that connect multiple tasks, manage state, and control the flow of execution

### Two Orchestration Modes

Crews provide autonomous agent collaboration, ideal for tasks requiring flexible decision-making and dynamic interaction. Flows offer precise, event-driven control, ideal for managing detailed execution paths and secure state management.

For your SoC optimization pipeline, **Flows** are likely the better fit. You need deterministic, reproducible pipelines where the output of one stage feeds the next—not free-form agent conversation.

---

## Mapping to Your SoC System

Your pipeline:
```
Analyzer → Abstractor → Generalizer → Validator → Scorer
```

This is fundamentally a **sequential flow with potential feedback loops** (if validation fails, you might want to re-synthesize). Here's a concrete skeleton:

```python
from crewai import Agent, Task, Crew, Process
from crewai.flow.flow import Flow, start, listen, router
from pydantic import BaseModel
from typing import Optional

# ─────────────────────────────────────────────────────────────
# State: What gets passed through the pipeline
# ─────────────────────────────────────────────────────────────
class SoCOptimizationState(BaseModel):
    original_design: str = ""
    abstracted_repr: Optional[dict] = None
    synthesized_design: Optional[dict] = None
    validation_result: Optional[dict] = None
    score: Optional[float] = None
    iteration: int = 0
    max_iterations: int = 5


# ─────────────────────────────────────────────────────────────
# Agents
# ─────────────────────────────────────────────────────────────
design_analyzer = Agent(
    role="SoC Design Analyzer",
    goal="Extract structural and behavioral characteristics from existing RTL/netlist",
    backstory="Expert in hardware analysis with deep knowledge of timing, area, power trade-offs",
    verbose=True,
    allow_delegation=False,
)

abstractor = Agent(
    role="Design Abstractor",
    goal="Transform concrete design details into canonical intermediate representation",
    backstory="Specializes in creating technology-agnostic representations suitable for synthesis",
    verbose=True,
    allow_delegation=False,
)

generalizer = Agent(
    role="Architecture Synthesizer",
    goal="Generate novel SoC architectures that improve on the abstracted input",
    backstory="Creative architect who explores the design space for PPA improvements",
    verbose=True,
    allow_delegation=False,
)

validator = Agent(
    role="Functional Validator",
    goal="Verify that synthesized designs are functionally correct",
    backstory="Rigorous verification engineer who catches corner cases",
    verbose=True,
    allow_delegation=False,
)

scorer = Agent(
    role="Optimization Scorer",
    goal="Quantify improvement in PPA metrics relative to baseline",
    backstory="Metrics-driven analyst who provides objective progress assessment",
    verbose=True,
    allow_delegation=False,
)


# ─────────────────────────────────────────────────────────────
# Flow: Orchestrating the pipeline
# ─────────────────────────────────────────────────────────────
class SoCOptimizationFlow(Flow[SoCOptimizationState]):

    @start()
    def analyze_design(self):
        task = Task(
            description=f"Analyze the following SoC design:\n{self.state.original_design}",
            expected_output="Structured analysis including hierarchy, critical paths, resource utilization",
            agent=design_analyzer,
        )
        crew = Crew(agents=[design_analyzer], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        # Store intermediate result
        self.state.abstracted_repr = {"raw_analysis": str(result)}
        return result

    @listen(analyze_design)
    def abstract_design(self, analysis_result):
        task = Task(
            description=f"Abstract the following analysis into canonical form:\n{analysis_result}",
            expected_output="Technology-agnostic IR suitable for architecture exploration",
            agent=abstractor,
        )
        crew = Crew(agents=[abstractor], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.abstracted_repr = {"ir": str(result)}
        return result

    @listen(abstract_design)
    def synthesize_architecture(self, abstraction):
        task = Task(
            description=f"Generate improved architecture from:\n{abstraction}",
            expected_output="New SoC design specification with expected improvements",
            agent=generalizer,
        )
        crew = Crew(agents=[generalizer], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.synthesized_design = {"design": str(result)}
        return result

    @listen(synthesize_architecture)
    def validate_design(self, new_design):
        task = Task(
            description=f"Validate functional correctness of:\n{new_design}",
            expected_output="Validation report: PASS/FAIL with details",
            agent=validator,
        )
        crew = Crew(agents=[validator], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        self.state.validation_result = {"report": str(result)}
        return result

    @router(validate_design)
    def route_on_validation(self, validation_result):
        """Branch based on validation outcome."""
        if "PASS" in str(validation_result).upper():
            return "score_design"
        elif self.state.iteration < self.state.max_iterations:
            self.state.iteration += 1
            return "synthesize_architecture"  # Retry synthesis
        else:
            return "score_design"  # Give up, score what we have

    @listen("score_design")
    def score_design(self):
        task = Task(
            description=f"Score the design:\n{self.state.synthesized_design}\nBaseline:\n{self.state.original_design}",
            expected_output="Numerical score and breakdown of PPA improvements",
            agent=scorer,
        )
        crew = Crew(agents=[scorer], tasks=[task], process=Process.sequential)
        result = crew.kickoff()
        # Parse score from result (you'd want structured output here)
        self.state.score = 0.0  # Placeholder
        return result


# ─────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    flow = SoCOptimizationFlow()
    result = flow.kickoff(inputs={"original_design": "<your RTL or spec here>"})
    print(f"Final state: {flow.state}")
```

---

## Critical Evaluation: Where CrewAI Helps and Where It Doesn't

### What CrewAI provides:
1. **Role encapsulation** — Agents have personas that can constrain/guide LLM behavior
2. **State management** — The flow state maintains context between steps and stores both inputs and outputs
3. **Routing/branching** — The `@router` decorator enables conditional paths (essential for your validate→retry loop)
4. **Observability** — Traces, testing with LLM-as-a-Judge techniques, and training with human feedback to monitor agent decisions, debug issues, and continuously improve performance

### What CrewAI will NOT solve for you:

1. **Domain-specific validation logic** — Your validator agent will only be as good as:
   - The prompts you write
   - The tools you provide (e.g., an actual RTL simulator, formal verification engine)
   - CrewAI can call external tools, but you must integrate them

2. **Determinism** — LLM outputs are inherently stochastic. Even with Flows, repeated runs may produce different results.

3. **Grounding in actual hardware semantics** — An LLM-based "generalizer" may hallucinate architectures that are physically unrealizable or violate timing constraints. You need hard constraints enforced outside the LLM.

---

## Evaluating the Agent Architecture Itself

This is the important part you asked about. You need to distinguish between:

| Level | What You're Evaluating | Methods |
|-------|------------------------|---------|
| **LLM quality** | Does each agent produce sensible outputs? | LLM-as-judge, human review, unit tests with golden outputs |
| **Task decomposition** | Are the tasks correctly scoped? Does the abstractor output what the generalizer needs? | Interface contracts (Pydantic schemas), integration tests |
| **Flow topology** | Is the pipeline structure correct? Are branches/loops handled properly? | Simulation with synthetic inputs, coverage of all paths |
| **System-level** | Does the entire system improve designs over baseline? | End-to-end metrics on a test corpus |

### Practical Approaches

**1. Structured Outputs with Pydantic**

Force agents to return structured data so you can validate inter-agent contracts:

```python
from pydantic import BaseModel

class AbstractedDesign(BaseModel):
    modules: list[str]
    interconnect_graph: dict
    timing_constraints: dict
    power_budget: float

# In your task:
task = Task(
    description="...",
    expected_output="JSON matching AbstractedDesign schema",
    agent=abstractor,
    output_pydantic=AbstractedDesign,  # CrewAI validates against this
)
```

If the agent produces garbage, the Pydantic validation fails—you catch integration bugs early.

**2. LLM-as-Judge for Quality**

LLM-as-a-judge is a general technique where you use an LLM to approximate human labeling. When you ask an LLM to assess qualities like "faithfulness to source," "correctness," or "helpfulness," you define what these terms mean in the evaluation prompt.

Implement a separate evaluator agent (not part of the main flow) that grades outputs:

```python
evaluator = Agent(
    role="Architecture Critic",
    goal="Assess whether synthesized designs are realistic and improve on baseline",
    backstory="Senior architect with 20 years of SoC experience",
)

def evaluate_design(original: str, synthesized: str) -> dict:
    task = Task(
        description=f"""
        Original: {original}
        Synthesized: {synthesized}
        
        Score on 1-10 for:
        1. Functional plausibility
        2. PPA improvement likelihood
        3. Implementability
        
        Provide scores and justification.
        """,
        expected_output="JSON with scores and reasoning",
        agent=evaluator,
    )
    crew = Crew(agents=[evaluator], tasks=[task])
    return crew.kickoff()
```

**Caveat:** LLMs assigning number scores isn't the best idea, since LLMs have weak understanding of the difference between a 7.2 vs 7.45. Use coarse buckets (1-3-5-7-10) or categorical labels (poor/acceptable/good/excellent).

**3. Ground Truth Test Corpus**

Build a dataset of:
- Known-good designs (baseline)
- Expected abstractions
- Hand-verified synthesized improvements

Run your pipeline on this corpus and measure:
- Abstraction fidelity (does it capture the right features?)
- Synthesis novelty vs. realism trade-off
- Validation false-positive/negative rate
- Score correlation with actual PPA metrics (if you have RTL synthesis results)

**4. A/B Testing Flow Topologies**

CrewAI lets you swap orchestration strategies. Try:
- Sequential vs. hierarchical (manager agent orchestrates others)
- Different agent decompositions (combine abstractor+generalizer into one?)
- Different LLMs per agent role

Measure end-to-end quality and cost (tokens, latency) for each variant.

---

## Honest Assessment

**CrewAI is useful here for:**
- Rapid prototyping of the agent topology
- State management across pipeline stages
- Routing logic (validation retry loops)
- Plugging in observability/tracing

**CrewAI is insufficient for:**
- Guaranteeing correctness (you need external formal methods/simulators)
- Preventing LLM hallucinations about hardware
- Providing domain-specific evaluation metrics

For a production SoC optimization system, I'd recommend:
1. Use CrewAI for orchestration and experimentation
2. Integrate real EDA tools (synthesis, simulation, timing analysis) as CrewAI "tools"
3. Treat LLM agents as heuristic generators, not oracles—always verify outputs with deterministic checks
4. Build a domain-specific evaluation harness outside CrewAI that measures actual PPA on synthesized designs

Want me to elaborate on tool integration patterns or evaluation harness design?

## Tool Integration Patterns

Now I have the full picture. Let me walk you through the tool integration patterns, specifically framed for your SoC optimization context.

---

## Tool Integration Patterns for SoC Optimization

### Fundamental Tool Structure

To create a personalized tool, inherit from BaseTool and define the necessary attributes, including the args_schema for input validation, and the _run method.

There are two approaches:

**1. Class-based (BaseTool)** — More control, better for complex tools with state or configuration
**2. Decorator-based (@tool)** — Quick and lightweight

For EDA tool integration, you almost always want the class-based approach because you need:
- Configuration (paths, license servers, timeout settings)
- State (connection handles, cached results)
- Structured error handling

---

### Pattern 1: Wrapping CLI-Based EDA Tools

Most EDA tools (synthesis, simulation, timing analysis) are invoked via command line. Here's a robust pattern:

```python
import subprocess
import json
import tempfile
import hashlib
from pathlib import Path
from typing import Type, Optional
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


class SynthesisInput(BaseModel):
    """Input schema for RTL synthesis."""
    rtl_source: str = Field(..., description="Verilog/SystemVerilog source code")
    top_module: str = Field(..., description="Top-level module name")
    target_freq_mhz: float = Field(default=100.0, description="Target frequency in MHz")
    optimization_goal: str = Field(
        default="balanced",
        description="One of: area, speed, power, balanced"
    )


class SynthesisResult(BaseModel):
    """Structured output from synthesis."""
    success: bool
    area_um2: Optional[float] = None
    timing_slack_ns: Optional[float] = None
    power_mw: Optional[float] = None
    critical_path: Optional[str] = None
    warnings: list[str] = []
    errors: list[str] = []
    raw_log: str = ""


class RTLSynthesisTool(BaseTool):
    """
    Wraps an RTL synthesis tool (e.g., Yosys, Genus, Design Compiler).
    
    This tool invokes the actual EDA binary and parses results.
    The agent sees structured output, not raw logs.
    """
    name: str = "rtl_synthesis"
    description: str = (
        "Synthesize RTL (Verilog/SystemVerilog) to gate-level netlist. "
        "Returns area, timing, and power estimates. Use this to validate "
        "that a proposed design is synthesizable and to get PPA metrics."
    )
    args_schema: Type[BaseModel] = SynthesisInput
    
    # Configuration - set at instantiation, not by agent
    synth_binary: str = "/opt/eda/yosys/bin/yosys"
    liberty_file: str = "/opt/pdk/sky130/sky130_fd_sc_hd__tt_025C_1v80.lib"
    timeout_seconds: int = 300
    work_dir: Path = Path("/tmp/synthesis_runs")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.work_dir.mkdir(parents=True, exist_ok=True)
    
    def _run(
        self,
        rtl_source: str,
        top_module: str,
        target_freq_mhz: float = 100.0,
        optimization_goal: str = "balanced"
    ) -> str:
        """Execute synthesis and return structured result as JSON string."""
        
        # Create isolated run directory (reproducibility)
        run_hash = hashlib.sha256(rtl_source.encode()).hexdigest()[:12]
        run_dir = self.work_dir / f"run_{run_hash}"
        run_dir.mkdir(exist_ok=True)
        
        # Write RTL to file
        rtl_path = run_dir / f"{top_module}.v"
        rtl_path.write_text(rtl_source)
        
        # Generate synthesis script
        script = self._generate_synth_script(
            rtl_path, top_module, target_freq_mhz, optimization_goal
        )
        script_path = run_dir / "synth.ys"
        script_path.write_text(script)
        
        # Execute
        result = self._execute_synthesis(script_path, run_dir)
        
        # Return as JSON for the agent to parse
        # CrewAI tools return strings; the agent interprets them
        return result.model_dump_json(indent=2)
    
    def _generate_synth_script(
        self, rtl_path: Path, top: str, freq: float, goal: str
    ) -> str:
        """Generate Yosys synthesis script."""
        # Map goal to Yosys optimization flags
        opt_flags = {
            "area": "-area",
            "speed": "-fast",
            "power": "",  # Yosys doesn't have direct power opt
            "balanced": ""
        }
        
        return f"""
# Auto-generated synthesis script
read_verilog {rtl_path}
hierarchy -check -top {top}
proc; opt; fsm; opt; memory; opt
techmap; opt
abc -liberty {self.liberty_file} {opt_flags.get(goal, '')}
stat -liberty {self.liberty_file}
write_json {rtl_path.parent / 'netlist.json'}
"""
    
    def _execute_synthesis(self, script: Path, run_dir: Path) -> SynthesisResult:
        """Run synthesis binary and parse output."""
        try:
            proc = subprocess.run(
                [self.synth_binary, "-s", str(script)],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=run_dir
            )
            
            # Parse Yosys output for metrics
            return self._parse_yosys_output(proc.stdout, proc.stderr, proc.returncode)
            
        except subprocess.TimeoutExpired:
            return SynthesisResult(
                success=False,
                errors=[f"Synthesis timed out after {self.timeout_seconds}s"],
                raw_log=""
            )
        except FileNotFoundError:
            return SynthesisResult(
                success=False,
                errors=[f"Synthesis binary not found: {self.synth_binary}"],
                raw_log=""
            )
        except Exception as e:
            return SynthesisResult(
                success=False,
                errors=[f"Unexpected error: {str(e)}"],
                raw_log=""
            )
    
    def _parse_yosys_output(
        self, stdout: str, stderr: str, returncode: int
    ) -> SynthesisResult:
        """Extract metrics from Yosys output."""
        warnings = []
        errors = []
        
        # Parse warnings/errors
        for line in stderr.splitlines():
            if "Warning:" in line:
                warnings.append(line.strip())
            elif "Error:" in line:
                errors.append(line.strip())
        
        if returncode != 0:
            return SynthesisResult(
                success=False,
                warnings=warnings,
                errors=errors or ["Synthesis failed with non-zero exit code"],
                raw_log=stdout[-2000:]  # Truncate for agent context
            )
        
        # Extract metrics from "stat" output
        # Real implementation would use regex or structured output
        area = self._extract_metric(stdout, r"Chip area.*?:\s*([\d.]+)")
        
        return SynthesisResult(
            success=True,
            area_um2=area,
            timing_slack_ns=None,  # Would need STA tool
            power_mw=None,         # Would need power analysis
            warnings=warnings,
            errors=[],
            raw_log=stdout[-2000:]
        )
    
    def _extract_metric(self, text: str, pattern: str) -> Optional[float]:
        import re
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None
```

**Key points:**
- **Structured input/output** via Pydantic — agents can't pass garbage
- **Configuration separate from runtime args** — license paths, timeouts are not LLM-decided
- **Error isolation** — exceptions don't crash the pipeline; they return error objects
- **Truncated logs** — raw EDA output can be huge; give the agent enough to reason, not 50MB of log

---

### Pattern 2: Caching for Expensive Operations

Synthesis/simulation runs are expensive. Tools can optionally implement a cache_function to fine-tune caching behavior. This function determines when to cache results based on specific conditions.

```python
import hashlib
import json
from pathlib import Path
from functools import lru_cache


class CachedSynthesisTool(RTLSynthesisTool):
    """Synthesis tool with persistent caching."""
    
    cache_dir: Path = Path("/var/cache/soc_optimization/synthesis")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _run(
        self,
        rtl_source: str,
        top_module: str,
        target_freq_mhz: float = 100.0,
        optimization_goal: str = "balanced"
    ) -> str:
        # Compute cache key from inputs
        cache_key = self._compute_cache_key(
            rtl_source, top_module, target_freq_mhz, optimization_goal
        )
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        # Check cache
        if cache_file.exists():
            return cache_file.read_text()
        
        # Run actual synthesis
        result = super()._run(rtl_source, top_module, target_freq_mhz, optimization_goal)
        
        # Cache successful results only
        result_obj = SynthesisResult.model_validate_json(result)
        if result_obj.success:
            cache_file.write_text(result)
        
        return result
    
    def _compute_cache_key(self, rtl: str, top: str, freq: float, goal: str) -> str:
        """Deterministic cache key from inputs."""
        content = json.dumps({
            "rtl_hash": hashlib.sha256(rtl.encode()).hexdigest(),
            "top": top,
            "freq": freq,
            "goal": goal,
            "liberty": self.liberty_file,  # Include PDK in cache key!
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:24]
```

**Critical detail:** Include PDK/library versions in the cache key. Otherwise, you'll return stale results after a PDK update.

---

### Pattern 3: Async Tools for Long-Running Jobs

CrewAI supports asynchronous tools, allowing you to implement tools that perform non-blocking operations like network requests, file I/O, or other async operations without blocking the main execution thread.

For jobs that take minutes/hours (full SoC synthesis, gate-level simulation):

```python
import asyncio
import aiofiles
from crewai.tools import BaseTool


class AsyncSimulationTool(BaseTool):
    """
    Runs gate-level simulation asynchronously.
    
    Submits job to a compute cluster and polls for completion.
    """
    name: str = "gate_level_simulation"
    description: str = (
        "Run gate-level simulation with SDF timing. "
        "Returns pass/fail and coverage metrics. "
        "Warning: This can take 10-60 minutes for large designs."
    )
    
    job_server_url: str = "http://job-server.internal:8080"
    poll_interval_seconds: int = 30
    max_wait_seconds: int = 3600
    
    async def _run(self, netlist_path: str, testbench: str, sdf_path: str) -> str:
        """Submit simulation job and wait for results."""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            # Submit job
            job_id = await self._submit_job(session, netlist_path, testbench, sdf_path)
            
            # Poll until complete or timeout
            elapsed = 0
            while elapsed < self.max_wait_seconds:
                status = await self._check_job(session, job_id)
                
                if status["state"] == "completed":
                    return json.dumps(status["result"])
                elif status["state"] == "failed":
                    return json.dumps({"success": False, "error": status["error"]})
                
                await asyncio.sleep(self.poll_interval_seconds)
                elapsed += self.poll_interval_seconds
            
            return json.dumps({
                "success": False,
                "error": f"Simulation timed out after {self.max_wait_seconds}s"
            })
    
    async def _submit_job(self, session, netlist: str, tb: str, sdf: str) -> str:
        async with session.post(
            f"{self.job_server_url}/jobs",
            json={"netlist": netlist, "testbench": tb, "sdf": sdf}
        ) as resp:
            data = await resp.json()
            return data["job_id"]
    
    async def _check_job(self, session, job_id: str) -> dict:
        async with session.get(f"{self.job_server_url}/jobs/{job_id}") as resp:
            return await resp.json()
```

---

### Pattern 4: Tool Composition — Pipelines Within Tools

Sometimes you need multiple EDA steps as a single atomic operation. Wrap them:

```python
class FullPPAAnalysisTool(BaseTool):
    """
    Runs complete PPA analysis: synthesis → STA → power.
    
    Returns unified PPA report. This is more reliable than having
    the agent orchestrate 3 separate tool calls.
    """
    name: str = "full_ppa_analysis"
    description: str = (
        "Complete Power-Performance-Area analysis. "
        "Takes RTL, returns comprehensive PPA metrics. "
        "More accurate than running synthesis alone."
    )
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Compose sub-tools
        self.synth_tool = CachedSynthesisTool()
        self.sta_tool = StaticTimingAnalysisTool()
        self.power_tool = PowerAnalysisTool()
    
    def _run(self, rtl_source: str, top_module: str, constraints_sdc: str) -> str:
        # Step 1: Synthesis
        synth_result = json.loads(self.synth_tool._run(rtl_source, top_module))
        if not synth_result.get("success"):
            return json.dumps({
                "success": False,
                "stage": "synthesis",
                "error": synth_result.get("errors")
            })
        
        netlist_path = synth_result["netlist_path"]
        
        # Step 2: Static Timing Analysis
        sta_result = json.loads(self.sta_tool._run(netlist_path, constraints_sdc))
        if not sta_result.get("success"):
            return json.dumps({
                "success": False,
                "stage": "sta",
                "error": sta_result.get("errors"),
                "partial": {"synthesis": synth_result}
            })
        
        # Step 3: Power Analysis
        power_result = json.loads(self.power_tool._run(
            netlist_path, 
            sta_result["switching_activity"]
        ))
        
        # Unified report
        return json.dumps({
            "success": True,
            "area_um2": synth_result["area_um2"],
            "timing": {
                "wns_ns": sta_result["worst_negative_slack"],
                "tns_ns": sta_result["total_negative_slack"],
                "critical_path": sta_result["critical_path"]
            },
            "power": {
                "total_mw": power_result["total_mw"],
                "dynamic_mw": power_result["dynamic_mw"],
                "leakage_mw": power_result["leakage_mw"]
            }
        })
```

**Why this matters:** If you let the LLM orchestrate three tool calls, it might:
- Pass wrong file paths between steps
- Misinterpret intermediate results
- Forget to run one of the steps

Composed tools enforce the correct sequence.

---

### Pattern 5: Guardrails — Tools That Refuse Bad Inputs

LLMs will try to synthesize garbage. Catch it early:

```python
import re
from crewai.tools import BaseTool


class GuardedSynthesisTool(RTLSynthesisTool):
    """Synthesis tool with input validation guardrails."""
    
    # Patterns that indicate problematic RTL
    FORBIDDEN_PATTERNS = [
        (r"initial\s+begin", "Initial blocks not synthesizable"),
        (r"#\d+", "Delays not synthesizable"),
        (r"\$display", "System tasks not synthesizable"),
        (r"\$random", "Random functions not synthesizable"),
    ]
    
    MAX_RTL_SIZE_BYTES = 1_000_000  # 1MB limit
    
    def _run(
        self,
        rtl_source: str,
        top_module: str,
        target_freq_mhz: float = 100.0,
        optimization_goal: str = "balanced"
    ) -> str:
        # Guardrail 1: Size check
        if len(rtl_source.encode()) > self.MAX_RTL_SIZE_BYTES:
            return json.dumps({
                "success": False,
                "errors": [f"RTL exceeds size limit ({self.MAX_RTL_SIZE_BYTES} bytes)"],
                "guardrail": "size_limit"
            })
        
        # Guardrail 2: Basic syntax check
        if not self._has_module_definition(rtl_source, top_module):
            return json.dumps({
                "success": False,
                "errors": [f"No module '{top_module}' found in RTL"],
                "guardrail": "module_missing"
            })
        
        # Guardrail 3: Non-synthesizable construct detection
        for pattern, message in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, rtl_source):
                return json.dumps({
                    "success": False,
                    "errors": [message],
                    "guardrail": "non_synthesizable_construct",
                    "hint": "Remove simulation-only constructs before synthesis"
                })
        
        # Guardrail 4: Frequency sanity check
        if not (1.0 <= target_freq_mhz <= 10000.0):
            return json.dumps({
                "success": False,
                "errors": [f"Target frequency {target_freq_mhz} MHz out of range [1, 10000]"],
                "guardrail": "invalid_frequency"
            })
        
        # All checks passed — run actual synthesis
        return super()._run(rtl_source, top_module, target_freq_mhz, optimization_goal)
    
    def _has_module_definition(self, rtl: str, module_name: str) -> bool:
        pattern = rf"module\s+{re.escape(module_name)}\s*[\(#]"
        return bool(re.search(pattern, rtl))
```

---

### Pattern 6: Connecting Tools via State (Flow Integration)

When tools need to share artifacts (netlists, reports), use the Flow state as the coordination mechanism:

```python
from crewai.flow.flow import Flow, start, listen
from pydantic import BaseModel
from pathlib import Path


class DesignArtifacts(BaseModel):
    """Tracks all generated artifacts through the pipeline."""
    rtl_path: Optional[Path] = None
    netlist_path: Optional[Path] = None
    sdf_path: Optional[Path] = None
    sta_report_path: Optional[Path] = None
    power_report_path: Optional[Path] = None
    
    # Metrics (populated incrementally)
    area_um2: Optional[float] = None
    timing_slack_ns: Optional[float] = None
    power_mw: Optional[float] = None


class SoCOptimizationState(BaseModel):
    original: DesignArtifacts = DesignArtifacts()
    synthesized: DesignArtifacts = DesignArtifacts()
    improvement_pct: Optional[float] = None


class SoCFlow(Flow[SoCOptimizationState]):
    """Flow that threads artifacts through tools via state."""
    
    def __init__(self):
        super().__init__()
        # Instantiate tools once
        self.synth = CachedSynthesisTool()
        self.sta = StaticTimingAnalysisTool()
        self.power = PowerAnalysisTool()
    
    @start()
    def synthesize_original(self):
        """Synthesize original design, store artifacts in state."""
        result = json.loads(self.synth._run(
            rtl_source=self.state.original.rtl_path.read_text(),
            top_module="top"
        ))
        
        if result["success"]:
            self.state.original.netlist_path = Path(result["netlist_path"])
            self.state.original.area_um2 = result["area_um2"]
        
        return result
    
    @listen(synthesize_original)
    def run_sta_original(self, synth_result):
        """Run STA using netlist from state."""
        if not self.state.original.netlist_path:
            return {"skipped": True, "reason": "No netlist from synthesis"}
        
        result = json.loads(self.sta._run(
            netlist_path=str(self.state.original.netlist_path),
            constraints_sdc="constraints.sdc"
        ))
        
        if result["success"]:
            self.state.original.timing_slack_ns = result["worst_negative_slack"]
            self.state.original.sta_report_path = Path(result["report_path"])
        
        return result
    
    # ... continue pattern for synthesized design ...
    
    @listen("final_comparison")
    def compute_improvement(self):
        """Compute improvement percentage from state."""
        orig = self.state.original
        synth = self.state.synthesized
        
        if orig.area_um2 and synth.area_um2:
            area_improvement = (orig.area_um2 - synth.area_um2) / orig.area_um2 * 100
            self.state.improvement_pct = area_improvement
        
        return {
            "area_improvement_pct": self.state.improvement_pct,
            "original": orig.model_dump(),
            "synthesized": synth.model_dump()
        }
```

---

### Pattern 7: MCP Integration for Existing Tool Servers

If you have EDA tools exposed via Model Context Protocol (MCP) servers:

CrewAI Tools supports the Model Context Protocol (MCP). It gives you access to thousands of tools from the hundreds of MCP servers out there built by the community.

```python
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters


# If you have an MCP server wrapping your EDA tools
serverparams = StdioServerParameters(
    command="/opt/eda-mcp-server/bin/eda-server",
    args=["--tools", "synth,sta,power"],
    env={"EDA_LICENSE": "27000@license-server"}
)

async def create_eda_crew():
    async with MCPServerAdapter(serverparams) as tools:
        # tools is now a list of CrewAI-compatible tools
        # Each MCP tool becomes a BaseTool instance
        
        eda_agent = Agent(
            role="EDA Engineer",
            goal="Analyze and optimize SoC designs",
            tools=tools,  # MCP tools injected here
        )
        
        # ... create tasks and crew ...
```

This is useful if your organization has already built tool servers.

---

## Tool Architecture Summary for Your SoC System

```
┌─────────────────────────────────────────────────────────────────┐
│                        Flow State                               │
│   (DesignArtifacts, metrics, iteration count)                   │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Analyzer    │      │ Generalizer │      │ Validator   │
│ Agent       │      │ Agent       │      │ Agent       │
└─────────────┘      └─────────────┘      └─────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Tool Layer                              │
├─────────────┬─────────────┬─────────────┬──────────────┬────────┤
│ RTLParser   │ Synthesis   │ STA         │ Simulation   │ Power  │
│ Tool        │ Tool        │ Tool        │ Tool         │ Tool   │
│ (linting,   │ (Yosys,     │ (OpenSTA,   │ (Verilator,  │(Joules,│
│  hierarchy) │  Genus)     │  PrimeTime) │  VCS)        │ Voltus)│
└─────────────┴─────────────┴─────────────┴──────────────┴────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Caching Layer                                │
│   (Content-addressed cache, PDK-versioned keys)                 │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EDA Binaries / Compute                       │
│   (Local execution, LSF/Slurm cluster, cloud runners)           │
└─────────────────────────────────────────────────────────────────┘
```

### Critical Implementation Advice

1. **Never let agents generate file paths** — tools should manage their own working directories
2. **Validate tool outputs structurally** — return Pydantic models, catch malformed results before agents see them
3. **Cache aggressively** — synthesis is expensive; cache on (RTL hash, PDK version, constraints hash)
4. **Timeout everything** — EDA tools hang; never wait indefinitely
5. **Log tool inputs/outputs** — you'll need this for debugging when agents make weird decisions
6. **Version your tools separately from agents** — tool behavior changes should be tracked independently

Want me to continue with the evaluation harness design next?

## Evaluating Harnass

Now I have the full context. Let me give you a comprehensive evaluation harness design.

---

## Evaluation Harness Design for SoC Optimization Agents

Evaluating multi-agent systems is fundamentally harder than evaluating single LLM calls. You have multiple interacting components, emergent behaviors, and both intermediate and final outputs to assess.

### Evaluation Taxonomy

Before building anything, you need clarity on **what** you're evaluating:

| Level | Question | Methods | Frequency |
|-------|----------|---------|-----------|
| **Unit** | Does each agent produce sensible outputs in isolation? | Golden datasets, LLM-as-judge | Every code change |
| **Integration** | Do agents hand off correctly? Are contracts honored? | Schema validation, interface tests | Every code change |
| **Trajectory** | Did the system take reasonable steps to reach the answer? | Trajectory matching, tool call audits | Weekly / per-experiment |
| **End-to-End** | Does the system actually improve SoC designs? | Ground-truth PPA comparison | Per release |
| **Operational** | Is the system fast, cheap, and reliable? | Latency, cost, error rate | Continuous |

---

### Part 1: CrewAI's Built-in Testing (and Its Limitations)

With crewAI, you can easily test your crew and evaluate its performance using the built-in testing capabilities. The CLI command `crewai test` runs your crew for a specified number of iterations and provides detailed performance metrics.

```bash
# Basic usage
crewai test -n 5 -m gpt-4o

# Output shows per-task scores and execution times
```

**What it provides:**
- Per-task scores (1-10) from LLM-as-judge
- Execution time tracking
- Token usage metrics

**Limitations for your use case:**
1. **Generic scoring** — the judge doesn't understand SoC design quality
2. **No ground truth** — it judges "did this look good?" not "is this functionally correct?"
3. **No tool output validation** — synthesis failures aren't reflected in scores
4. **Single-run variance** — LLM outputs vary; you need statistical aggregation

You'll need a custom harness.

---

### Part 2: Custom Evaluation Harness Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Evaluation Orchestrator                           │
│  (Runs test cases, collects results, computes metrics, generates reports)  │
└─────────────────────────────────────────────────────────────────────────────┘
         │                    │                    │                    │
         ▼                    ▼                    ▼                    ▼
┌─────────────┐      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
│ Unit Evals  │      │ Integration │      │ Trajectory  │      │ End-to-End  │
│             │      │ Evals       │      │ Evals       │      │ Evals       │
└─────────────┘      └─────────────┘      └─────────────┘      └─────────────┘
         │                    │                    │                    │
         ▼                    ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Test Case Repository                             │
│  (Designs, expected outputs, golden trajectories, baseline PPA metrics)    │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Results Database                               │
│  (Run history, scores, traces, diffs — for regression detection)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### Part 3: Test Case Repository

Your evaluation is only as good as your test corpus. For SoC optimization:

```python
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from enum import Enum


class DesignComplexity(str, Enum):
    TRIVIAL = "trivial"      # Counter, simple FSM
    SIMPLE = "simple"        # UART, SPI controller
    MODERATE = "moderate"    # RISC-V core, DMA engine
    COMPLEX = "complex"      # Full SoC subsystem


class TestCase(BaseModel):
    """A single test case for the SoC optimization pipeline."""
    
    id: str
    name: str
    description: str
    complexity: DesignComplexity
    
    # Input
    rtl_path: Path
    constraints_sdc: Path
    top_module: str
    
    # Ground truth (from running actual EDA tools offline)
    baseline_ppa: "PPAMetrics"
    
    # Expected behaviors (for trajectory evaluation)
    expected_tools_called: list[str]  # e.g., ["rtl_synthesis", "sta", "power"]
    expected_analysis_keywords: list[str]  # Terms the analyzer should identify
    
    # Optional: known-good optimized version
    golden_optimized_rtl: Optional[Path] = None
    golden_optimized_ppa: Optional["PPAMetrics"] = None
    
    # Constraints on valid outputs
    must_preserve_interface: bool = True
    max_acceptable_area_increase_pct: float = 10.0
    min_timing_slack_ns: float = 0.0


class PPAMetrics(BaseModel):
    """Power-Performance-Area metrics from EDA tools."""
    area_um2: float
    wns_ns: float  # Worst negative slack
    tns_ns: float  # Total negative slack
    power_total_mw: float
    power_dynamic_mw: float
    power_leakage_mw: float
    critical_path_stages: int
    
    def improvement_over(self, baseline: "PPAMetrics") -> dict:
        """Compute percentage improvements."""
        return {
            "area_pct": (baseline.area_um2 - self.area_um2) / baseline.area_um2 * 100,
            "power_pct": (baseline.power_total_mw - self.power_total_mw) / baseline.power_total_mw * 100,
            "timing_improved": self.wns_ns >= baseline.wns_ns,
        }


class TestCorpus:
    """Manages the test case collection."""
    
    def __init__(self, corpus_dir: Path):
        self.corpus_dir = corpus_dir
        self._cases: dict[str, TestCase] = {}
        self._load_cases()
    
    def _load_cases(self):
        for case_dir in self.corpus_dir.iterdir():
            if case_dir.is_dir() and (case_dir / "case.json").exists():
                case = TestCase.model_validate_json(
                    (case_dir / "case.json").read_text()
                )
                self._cases[case.id] = case
    
    def get_by_complexity(self, complexity: DesignComplexity) -> list[TestCase]:
        return [c for c in self._cases.values() if c.complexity == complexity]
    
    def get_smoke_tests(self) -> list[TestCase]:
        """Fast subset for CI."""
        return self.get_by_complexity(DesignComplexity.TRIVIAL)[:3]
    
    def get_full_suite(self) -> list[TestCase]:
        return list(self._cases.values())
```

**Critical: Building the corpus**

1. **Start with open-source designs** — OpenCores, RISC-V cores, OpenTitan modules
2. **Run actual EDA flows offline** — Generate ground-truth PPA for each design
3. **Include adversarial cases** — Designs that are already optimal, designs with bugs, designs that can't meet timing
4. **Version the corpus** — Tie test cases to PDK versions

---

### Part 4: Unit Evaluation — Testing Agents in Isolation

Each agent should be testable independently:

```python
import json
import pytest
from crewai import Agent, Task, Crew
from pydantic import BaseModel


class AnalyzerOutput(BaseModel):
    """Expected structure from the analyzer agent."""
    modules: list[str]
    hierarchy_depth: int
    estimated_complexity: str  # "low", "medium", "high"
    critical_signals: list[str]
    optimization_opportunities: list[str]


class TestAnalyzerAgent:
    """Unit tests for the design analyzer agent."""
    
    @pytest.fixture
    def analyzer(self):
        return Agent(
            role="SoC Design Analyzer",
            goal="Extract structural characteristics from RTL",
            backstory="Expert in hardware analysis",
            verbose=False,
        )
    
    @pytest.fixture
    def simple_rtl(self) -> str:
        return """
        module counter #(parameter WIDTH=8) (
            input clk, rst,
            output reg [WIDTH-1:0] count
        );
            always @(posedge clk or posedge rst)
                if (rst) count <= 0;
                else count <= count + 1;
        endmodule
        """
    
    def test_identifies_module_name(self, analyzer, simple_rtl):
        """Analyzer should identify the top module."""
        task = Task(
            description=f"Analyze this RTL:\n{simple_rtl}",
            expected_output="JSON with modules list",
            agent=analyzer,
            output_pydantic=AnalyzerOutput,
        )
        crew = Crew(agents=[analyzer], tasks=[task], verbose=False)
        result = crew.kickoff()
        
        output = AnalyzerOutput.model_validate_json(result.raw)
        assert "counter" in output.modules
    
    def test_identifies_parameterization(self, analyzer, simple_rtl):
        """Analyzer should note parameterized designs."""
        task = Task(
            description=f"Analyze this RTL and identify parameters:\n{simple_rtl}",
            expected_output="JSON with optimization opportunities",
            agent=analyzer,
            output_pydantic=AnalyzerOutput,
        )
        crew = Crew(agents=[analyzer], tasks=[task], verbose=False)
        result = crew.kickoff()
        
        output = AnalyzerOutput.model_validate_json(result.raw)
        # Should identify WIDTH as tunable
        assert any("WIDTH" in opp or "parameter" in opp.lower() 
                   for opp in output.optimization_opportunities)
    
    def test_handles_malformed_rtl(self, analyzer):
        """Analyzer should gracefully handle garbage input."""
        garbage = "this is not verilog at all {{{}}}"
        task = Task(
            description=f"Analyze this RTL:\n{garbage}",
            expected_output="JSON or error message",
            agent=analyzer,
        )
        crew = Crew(agents=[analyzer], tasks=[task], verbose=False)
        result = crew.kickoff()
        
        # Should not crash, should indicate error
        assert "error" in result.raw.lower() or "invalid" in result.raw.lower()


class TestGeneralizerAgent:
    """Unit tests for the architecture synthesizer agent."""
    
    @pytest.fixture
    def generalizer(self):
        return Agent(
            role="Architecture Synthesizer",
            goal="Generate improved SoC architectures",
            backstory="Creative architect exploring design space",
            verbose=False,
        )
    
    def test_preserves_interface(self, generalizer):
        """Generated designs must preserve I/O interface."""
        original_interface = {
            "inputs": ["clk", "rst", "data_in[7:0]"],
            "outputs": ["data_out[7:0]", "valid"]
        }
        
        task = Task(
            description=f"""
            Generate an optimized version of a design with this interface:
            {json.dumps(original_interface)}
            
            The optimized design MUST have the same ports.
            """,
            expected_output="Verilog module with same interface",
            agent=generalizer,
        )
        crew = Crew(agents=[generalizer], tasks=[task], verbose=False)
        result = crew.kickoff()
        
        # Parse output and verify interface preservation
        for port in original_interface["inputs"] + original_interface["outputs"]:
            port_name = port.split("[")[0]  # Strip bus notation
            assert port_name in result.raw, f"Missing port: {port_name}"
    
    def test_does_not_hallucinate_features(self, generalizer):
        """Generated designs shouldn't add impossible features."""
        task = Task(
            description="""
            Optimize this simple 8-bit adder for area.
            Constraints: No multiplication, no memory, synchronous only.
            """,
            expected_output="Optimized Verilog",
            agent=generalizer,
        )
        crew = Crew(agents=[generalizer], tasks=[task], verbose=False)
        result = crew.kickoff()
        
        # Check for forbidden constructs
        forbidden = ["$random", "initial begin", "#10", "* ", "memory", "ram", "rom"]
        for pattern in forbidden:
            assert pattern.lower() not in result.raw.lower(), \
                f"Hallucinated forbidden construct: {pattern}"
```

---

### Part 5: Integration Evaluation — Contract Verification

Test that agent outputs satisfy the contracts expected by downstream agents:

```python
from typing import Any
import json


class ContractViolation(Exception):
    """Raised when an agent's output violates the expected contract."""
    pass


class ContractValidator:
    """Validates inter-agent data contracts."""
    
    @staticmethod
    def validate_analyzer_to_abstractor(analyzer_output: str) -> dict:
        """
        Contract: Analyzer → Abstractor
        
        Analyzer must produce:
        - modules: list[str] (non-empty)
        - hierarchy: dict with parent-child relationships
        - signals: dict mapping signal names to properties
        """
        try:
            data = json.loads(analyzer_output)
        except json.JSONDecodeError:
            raise ContractViolation("Analyzer output is not valid JSON")
        
        required_keys = ["modules", "hierarchy", "signals"]
        for key in required_keys:
            if key not in data:
                raise ContractViolation(f"Missing required key: {key}")
        
        if not isinstance(data["modules"], list) or len(data["modules"]) == 0:
            raise ContractViolation("modules must be non-empty list")
        
        return data
    
    @staticmethod
    def validate_abstractor_to_generalizer(abstractor_output: str) -> dict:
        """
        Contract: Abstractor → Generalizer
        
        Abstractor must produce:
        - ir: Intermediate representation (dict)
        - constraints: Timing/area constraints to preserve
        - interface: Port definitions (must not be modified)
        """
        try:
            data = json.loads(abstractor_output)
        except json.JSONDecodeError:
            raise ContractViolation("Abstractor output is not valid JSON")
        
        if "ir" not in data:
            raise ContractViolation("Missing intermediate representation (ir)")
        
        if "interface" not in data:
            raise ContractViolation("Missing interface definition")
        
        return data
    
    @staticmethod
    def validate_generalizer_to_validator(generalizer_output: str) -> dict:
        """
        Contract: Generalizer → Validator
        
        Generalizer must produce:
        - rtl: Valid Verilog/SystemVerilog source
        - top_module: Name of top module
        - changes_summary: What was modified
        """
        try:
            data = json.loads(generalizer_output)
        except json.JSONDecodeError:
            # Might be raw RTL, not JSON
            if "module " in generalizer_output:
                return {"rtl": generalizer_output, "top_module": "unknown"}
            raise ContractViolation("Cannot parse generalizer output")
        
        if "rtl" not in data:
            raise ContractViolation("Missing RTL source")
        
        # Basic RTL sanity check
        if "module " not in data["rtl"]:
            raise ContractViolation("RTL does not contain module definition")
        
        return data


class IntegrationTest:
    """Run pipeline and check contracts at each handoff."""
    
    def __init__(self, flow):
        self.flow = flow
        self.contract_results: list[dict] = []
    
    def run_with_contract_checks(self, test_case: TestCase) -> dict:
        """Execute flow with contract validation at each step."""
        
        # We'll instrument the flow to capture intermediate outputs
        # This is a simplified version; real implementation would use callbacks
        
        results = {
            "test_case_id": test_case.id,
            "contracts_passed": [],
            "contracts_failed": [],
            "final_output": None
        }
        
        try:
            # Run the flow
            final = self.flow.kickoff(inputs={
                "original_design": test_case.rtl_path.read_text()
            })
            results["final_output"] = final
            
            # Check contracts from captured state
            state = self.flow.state
            
            # Contract 1: Analyzer → Abstractor
            if hasattr(state, "analyzer_output"):
                try:
                    ContractValidator.validate_analyzer_to_abstractor(
                        state.analyzer_output
                    )
                    results["contracts_passed"].append("analyzer_to_abstractor")
                except ContractViolation as e:
                    results["contracts_failed"].append({
                        "contract": "analyzer_to_abstractor",
                        "error": str(e)
                    })
            
            # Contract 2: Abstractor → Generalizer
            if hasattr(state, "abstractor_output"):
                try:
                    ContractValidator.validate_abstractor_to_generalizer(
                        state.abstractor_output
                    )
                    results["contracts_passed"].append("abstractor_to_generalizer")
                except ContractViolation as e:
                    results["contracts_failed"].append({
                        "contract": "abstractor_to_generalizer",
                        "error": str(e)
                    })
            
            # Contract 3: Generalizer → Validator
            if hasattr(state, "generalizer_output"):
                try:
                    ContractValidator.validate_generalizer_to_validator(
                        state.generalizer_output
                    )
                    results["contracts_passed"].append("generalizer_to_validator")
                except ContractViolation as e:
                    results["contracts_failed"].append({
                        "contract": "generalizer_to_validator",
                        "error": str(e)
                    })
                    
        except Exception as e:
            results["error"] = str(e)
        
        return results
```

---

### Part 6: Trajectory Evaluation — Did the System Take the Right Steps?

This checks whether agents used tools appropriately and followed a reasonable path:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryStep:
    """A single step in the agent's execution."""
    timestamp: float
    agent_role: str
    action_type: str  # "tool_call", "llm_call", "delegation"
    action_name: str  # Tool name or "generate_response"
    input_summary: str
    output_summary: str
    duration_ms: float
    tokens_used: int
    success: bool
    error: Optional[str] = None


class TrajectoryCollector:
    """Collects execution trajectory via callbacks."""
    
    def __init__(self):
        self.steps: list[TrajectoryStep] = []
        self._start_time: float = 0
    
    def step_callback(self, step_output):
        """Called after each agent step."""
        import time
        
        self.steps.append(TrajectoryStep(
            timestamp=time.time() - self._start_time,
            agent_role=getattr(step_output, 'agent', 'unknown'),
            action_type=self._classify_action(step_output),
            action_name=getattr(step_output, 'tool', 'llm_response'),
            input_summary=str(getattr(step_output, 'tool_input', ''))[:200],
            output_summary=str(getattr(step_output, 'result', ''))[:200],
            duration_ms=getattr(step_output, 'execution_time', 0) * 1000,
            tokens_used=getattr(step_output, 'tokens', 0),
            success=not getattr(step_output, 'error', False),
            error=getattr(step_output, 'error_message', None)
        ))
    
    def _classify_action(self, step_output) -> str:
        if hasattr(step_output, 'tool') and step_output.tool:
            return "tool_call"
        return "llm_call"
    
    def get_tool_sequence(self) -> list[str]:
        """Extract ordered list of tools called."""
        return [s.action_name for s in self.steps if s.action_type == "tool_call"]


class TrajectoryEvaluator:
    """Evaluates execution trajectories against expectations."""
    
    def __init__(self, test_case: TestCase):
        self.test_case = test_case
    
    def evaluate(self, trajectory: TrajectoryCollector) -> dict:
        """Compute trajectory metrics."""
        results = {
            "tool_coverage": self._compute_tool_coverage(trajectory),
            "tool_order_correct": self._check_tool_order(trajectory),
            "unnecessary_tool_calls": self._find_unnecessary_calls(trajectory),
            "failed_steps": self._count_failures(trajectory),
            "efficiency_score": self._compute_efficiency(trajectory),
        }
        return results
    
    def _compute_tool_coverage(self, traj: TrajectoryCollector) -> float:
        """What fraction of expected tools were called?"""
        expected = set(self.test_case.expected_tools_called)
        actual = set(traj.get_tool_sequence())
        
        if not expected:
            return 1.0
        
        covered = expected.intersection(actual)
        return len(covered) / len(expected)
    
    def _check_tool_order(self, traj: TrajectoryCollector) -> bool:
        """Were tools called in the expected order?"""
        expected = self.test_case.expected_tools_called
        actual = traj.get_tool_sequence()
        
        # Check if expected is a subsequence of actual
        it = iter(actual)
        return all(tool in it for tool in expected)
    
    def _find_unnecessary_calls(self, traj: TrajectoryCollector) -> list[str]:
        """Identify redundant or suspicious tool calls."""
        tool_calls = traj.get_tool_sequence()
        
        unnecessary = []
        
        # Repeated identical calls
        for i, tool in enumerate(tool_calls[1:], 1):
            if tool == tool_calls[i-1]:
                unnecessary.append(f"Repeated: {tool} at position {i}")
        
        # Tools that failed and were retried > 3 times
        from collections import Counter
        failures = Counter(
            s.action_name for s in traj.steps 
            if s.action_type == "tool_call" and not s.success
        )
        for tool, count in failures.items():
            if count > 3:
                unnecessary.append(f"Excessive retries: {tool} ({count} failures)")
        
        return unnecessary
    
    def _count_failures(self, traj: TrajectoryCollector) -> int:
        return sum(1 for s in traj.steps if not s.success)
    
    def _compute_efficiency(self, traj: TrajectoryCollector) -> float:
        """
        Efficiency = (expected steps) / (actual steps)
        
        Score > 1.0 means system was more efficient than expected.
        Score < 1.0 means system took more steps than necessary.
        """
        expected_steps = len(self.test_case.expected_tools_called) + 5  # Buffer for LLM calls
        actual_steps = len(traj.steps)
        
        if actual_steps == 0:
            return 0.0
        
        return min(expected_steps / actual_steps, 1.5)  # Cap at 1.5
```

---

### Part 7: End-to-End Evaluation — The Ground Truth

This is the ultimate test: did the system actually improve the design?

```python
import subprocess
from pathlib import Path


class PPAEvaluator:
    """
    Runs actual EDA tools to measure PPA of generated designs.
    
    This is the ground truth — no LLM involved in scoring.
    """
    
    def __init__(self, synth_tool, sta_tool, power_tool):
        self.synth = synth_tool
        self.sta = sta_tool
        self.power = power_tool
    
    def evaluate_design(self, rtl: str, top_module: str, constraints: str) -> PPAMetrics:
        """Run full EDA flow and extract metrics."""
        
        # Step 1: Synthesis
        synth_result = json.loads(self.synth._run(rtl, top_module))
        if not synth_result.get("success"):
            raise EvaluationError(f"Synthesis failed: {synth_result.get('errors')}")
        
        # Step 2: STA
        sta_result = json.loads(self.sta._run(
            synth_result["netlist_path"], 
            constraints
        ))
        if not sta_result.get("success"):
            raise EvaluationError(f"STA failed: {sta_result.get('errors')}")
        
        # Step 3: Power
        power_result = json.loads(self.power._run(
            synth_result["netlist_path"],
            sta_result.get("switching_activity", "default.saif")
        ))
        
        return PPAMetrics(
            area_um2=synth_result["area_um2"],
            wns_ns=sta_result["worst_negative_slack"],
            tns_ns=sta_result["total_negative_slack"],
            power_total_mw=power_result.get("total_mw", 0),
            power_dynamic_mw=power_result.get("dynamic_mw", 0),
            power_leakage_mw=power_result.get("leakage_mw", 0),
            critical_path_stages=sta_result.get("critical_path_stages", 0),
        )


class EndToEndEvaluator:
    """
    Complete end-to-end evaluation against ground truth.
    """
    
    def __init__(self, ppa_evaluator: PPAEvaluator):
        self.ppa = ppa_evaluator
    
    def evaluate(
        self, 
        test_case: TestCase,
        generated_rtl: str,
        generated_top: str
    ) -> dict:
        """Compare generated design against baseline."""
        
        results = {
            "test_case_id": test_case.id,
            "baseline_ppa": test_case.baseline_ppa.model_dump(),
            "generated_ppa": None,
            "improvement": None,
            "functional_correct": None,
            "meets_constraints": None,
            "verdict": None,
        }
        
        try:
            # Get PPA of generated design
            generated_ppa = self.ppa.evaluate_design(
                generated_rtl,
                generated_top,
                test_case.constraints_sdc.read_text()
            )
            results["generated_ppa"] = generated_ppa.model_dump()
            
            # Compute improvement
            improvement = generated_ppa.improvement_over(test_case.baseline_ppa)
            results["improvement"] = improvement
            
            # Check constraints
            results["meets_constraints"] = self._check_constraints(
                test_case, generated_ppa
            )
            
            # Functional verification (if testbench available)
            if hasattr(test_case, 'testbench_path'):
                results["functional_correct"] = self._run_simulation(
                    generated_rtl, test_case.testbench_path
                )
            
            # Final verdict
            results["verdict"] = self._compute_verdict(results)
            
        except EvaluationError as e:
            results["error"] = str(e)
            results["verdict"] = "FAIL_SYNTHESIS"
        
        return results
    
    def _check_constraints(self, test_case: TestCase, ppa: PPAMetrics) -> dict:
        """Verify design meets all constraints."""
        checks = {}
        
        # Timing constraint
        checks["timing_met"] = ppa.wns_ns >= test_case.min_timing_slack_ns
        
        # Area constraint
        area_increase_pct = (
            (ppa.area_um2 - test_case.baseline_ppa.area_um2) 
            / test_case.baseline_ppa.area_um2 * 100
        )
        checks["area_acceptable"] = area_increase_pct <= test_case.max_acceptable_area_increase_pct
        
        # All checks pass?
        checks["all_passed"] = all(checks.values())
        
        return checks
    
    def _run_simulation(self, rtl: str, testbench: Path) -> bool:
        """Run functional simulation."""
        # This would invoke Verilator, VCS, etc.
        # Simplified here
        try:
            result = subprocess.run(
                ["verilator", "--lint-only", "-Wall", "/dev/stdin"],
                input=rtl,
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _compute_verdict(self, results: dict) -> str:
        """Final pass/fail determination."""
        if results.get("error"):
            return "FAIL_ERROR"
        
        if not results.get("meets_constraints", {}).get("all_passed", False):
            return "FAIL_CONSTRAINTS"
        
        if results.get("functional_correct") is False:
            return "FAIL_FUNCTIONAL"
        
        improvement = results.get("improvement", {})
        if improvement.get("area_pct", 0) > 0 or improvement.get("power_pct", 0) > 0:
            return "PASS_IMPROVED"
        
        return "PASS_NO_REGRESSION"
```

---

### Part 8: LLM-as-Judge for Qualitative Aspects

Some things are hard to measure automatically. Use a separate LLM to judge:

```python
from crewai import Agent, Task, Crew


class QualityJudge:
    """
    Uses an LLM to evaluate qualitative aspects of agent outputs.
    
    This is NOT the same as asking "is this a good design?" —
    we use ground-truth EDA for that. This judges things like:
    - Is the analysis comprehensive?
    - Are the explanations clear?
    - Did the agent reason correctly?
    """
    
    def __init__(self, judge_model: str = "gpt-4o"):
        self.judge = Agent(
            role="Expert Reviewer",
            goal="Evaluate AI agent outputs for quality and correctness",
            backstory="""You are a senior hardware architect with 20 years 
            of experience. You review AI-generated design analyses and 
            optimizations with a critical eye.""",
            llm=judge_model,
            verbose=False,
        )
    
    def evaluate_analysis_quality(
        self, 
        rtl: str, 
        analysis_output: str
    ) -> dict:
        """Judge the quality of the analyzer's output."""
        
        task = Task(
            description=f"""
            Evaluate this design analysis for quality.
            
            Original RTL:
            ```verilog
            {rtl[:2000]}  # Truncate for context
            ```
            
            Analysis Output:
            ```
            {analysis_output}
            ```
            
            Score each dimension from 1-5:
            1. COMPLETENESS: Did it identify all major components?
            2. ACCURACY: Are the identified features correct?
            3. DEPTH: Does it go beyond surface-level observation?
            4. ACTIONABILITY: Does it suggest useful optimization directions?
            
            Respond with JSON only:
            {{
                "completeness": {{"score": N, "reason": "..."}},
                "accuracy": {{"score": N, "reason": "..."}},
                "depth": {{"score": N, "reason": "..."}},
                "actionability": {{"score": N, "reason": "..."}}
            }}
            """,
            expected_output="JSON scores",
            agent=self.judge,
        )
        
        crew = Crew(agents=[self.judge], tasks=[task], verbose=False)
        result = crew.kickoff()
        
        try:
            return json.loads(result.raw)
        except json.JSONDecodeError:
            return {"error": "Judge output not valid JSON", "raw": result.raw}
    
    def evaluate_optimization_reasoning(
        self,
        original_rtl: str,
        optimized_rtl: str,
        agent_explanation: str
    ) -> dict:
        """Judge whether the optimization reasoning makes sense."""
        
        task = Task(
            description=f"""
            Evaluate whether this optimization reasoning is sound.
            
            Original design (excerpt):
            ```verilog
            {original_rtl[:1500]}
            ```
            
            Optimized design (excerpt):
            ```verilog
            {optimized_rtl[:1500]}
            ```
            
            Agent's explanation:
            ```
            {agent_explanation}
            ```
            
            Evaluate:
            1. LOGICAL_SOUNDNESS: Does the reasoning follow logically? (1-5)
            2. TECHNICAL_ACCURACY: Are the technical claims correct? (1-5)
            3. TRADEOFF_AWARENESS: Does it acknowledge tradeoffs? (1-5)
            4. FALSIFIABLE: Could this be verified/disproven? (1-5)
            
            Also flag any RED_FLAGS (e.g., impossible claims, contradictions).
            
            Respond with JSON only.
            """,
            expected_output="JSON evaluation",
            agent=self.judge,
        )
        
        crew = Crew(agents=[self.judge], tasks=[task], verbose=False)
        result = crew.kickoff()
        
        try:
            return json.loads(result.raw)
        except json.JSONDecodeError:
            return {"error": "Judge output not valid JSON", "raw": result.raw}
```

**Important caveats about LLM-as-judge:**

1. LLM-as-a-judge is a general technique where you use LLM to approximate human labeling — it's not ground truth
2. Use it for things that are hard to measure automatically (explanation quality, reasoning soundness)
3. Don't use it for things you can measure (PPA metrics, functional correctness)
4. Calibrate the judge against human ratings periodically

---

### Part 9: Observability and Continuous Evaluation

CrewAI's built-in tracing provides comprehensive observability for your AI agents, including agent decisions, task execution timelines, tool usage, and LLM calls.

For production, integrate with observability platforms:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


def setup_tracing():
    """Configure OpenTelemetry tracing for production."""
    provider = TracerProvider()
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint="http://observability-server:4317")
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)


class ProductionEvaluationPipeline:
    """
    Continuous evaluation running alongside production.
    """
    
    def __init__(self, sample_rate: float = 0.1):
        self.sample_rate = sample_rate  # Evaluate 10% of production runs
        self.tracer = trace.get_tracer("soc_optimization")
    
    def should_evaluate(self) -> bool:
        import random
        return random.random() < self.sample_rate
    
    def wrap_flow_with_evaluation(self, flow, test_case: TestCase):
        """Wrap a production flow with evaluation hooks."""
        
        original_kickoff = flow.kickoff
        
        def instrumented_kickoff(*args, **kwargs):
            with self.tracer.start_as_current_span("soc_optimization_run") as span:
                span.set_attribute("test_case_id", test_case.id)
                span.set_attribute("complexity", test_case.complexity.value)
                
                # Collect trajectory
                collector = TrajectoryCollector()
                flow.step_callback = collector.step_callback
                
                # Run the flow
                result = original_kickoff(*args, **kwargs)
                
                # Evaluate if sampled
                if self.should_evaluate():
                    self._run_async_evaluation(
                        test_case, result, collector, span
                    )
                
                return result
        
        flow.kickoff = instrumented_kickoff
        return flow
    
    def _run_async_evaluation(
        self, 
        test_case: TestCase, 
        result, 
        trajectory: TrajectoryCollector,
        span
    ):
        """Queue evaluation for background processing."""
        import threading
        
        def evaluate():
            # Trajectory evaluation (fast)
            traj_eval = TrajectoryEvaluator(test_case)
            traj_results = traj_eval.evaluate(trajectory)
            
            # Record metrics
            span.set_attribute("trajectory.tool_coverage", traj_results["tool_coverage"])
            span.set_attribute("trajectory.efficiency", traj_results["efficiency_score"])
            
            # Store for later analysis
            self._store_evaluation_result({
                "test_case_id": test_case.id,
                "timestamp": time.time(),
                "trajectory": traj_results,
                "result_summary": str(result)[:1000]
            })
        
        # Don't block production on evaluation
        thread = threading.Thread(target=evaluate, daemon=True)
        thread.start()
    
    def _store_evaluation_result(self, result: dict):
        """Persist evaluation results for analysis."""
        # Could be PostgreSQL, ClickHouse, BigQuery, etc.
        pass
```

---

### Part 10: Putting It All Together — The Evaluation Runner

```python
import time
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class EvaluationConfig:
    """Configuration for evaluation runs."""
    corpus_path: Path
    results_db_path: Path
    parallelism: int = 4
    timeout_per_case_seconds: int = 600
    run_functional_sim: bool = True
    run_llm_judge: bool = True
    sample_for_judge: float = 0.2  # Only LLM-judge 20% of cases


class EvaluationRunner:
    """
    Orchestrates the complete evaluation pipeline.
    """
    
    def __init__(self, config: EvaluationConfig, flow_factory):
        self.config = config
        self.flow_factory = flow_factory  # Callable that returns a fresh Flow
        self.corpus = TestCorpus(config.corpus_path)
        self.ppa_evaluator = PPAEvaluator(
            CachedSynthesisTool(),
            StaticTimingAnalysisTool(),
            PowerAnalysisTool()
        )
        self.e2e_evaluator = EndToEndEvaluator(self.ppa_evaluator)
        self.quality_judge = QualityJudge() if config.run_llm_judge else None
    
    def run_full_suite(self) -> "EvaluationReport":
        """Run evaluation on the complete test corpus."""
        cases = self.corpus.get_full_suite()
        return self._run_cases(cases, "full_suite")
    
    def run_smoke_tests(self) -> "EvaluationReport":
        """Run fast evaluation on minimal test set."""
        cases = self.corpus.get_smoke_tests()
        return self._run_cases(cases, "smoke")
    
    def run_single(self, case_id: str) -> dict:
        """Run evaluation on a single test case."""
        case = self.corpus._cases.get(case_id)
        if not case:
            raise ValueError(f"Unknown test case: {case_id}")
        return self._evaluate_single_case(case)
    
    def _run_cases(self, cases: list[TestCase], suite_name: str) -> "EvaluationReport":
        """Run evaluation on a list of cases with parallelism."""
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.config.parallelism) as executor:
            future_to_case = {
                executor.submit(self._evaluate_single_case, case): case 
                for case in cases
            }
            
            for future in as_completed(future_to_case):
                case = future_to_case[future]
                try:
                    result = future.result(timeout=self.config.timeout_per_case_seconds)
                    results.append(result)
                    print(f"✓ {case.id}: {result.get('verdict', 'UNKNOWN')}")
                except Exception as e:
                    results.append({
                        "test_case_id": case.id,
                        "error": str(e),
                        "verdict": "FAIL_EXCEPTION"
                    })
                    print(f"✗ {case.id}: {e}")
        
        total_time = time.time() - start_time
        
        return EvaluationReport(
            suite_name=suite_name,
            timestamp=time.time(),
            total_cases=len(cases),
            results=results,
            total_time_seconds=total_time,
        )
    
    def _evaluate_single_case(self, case: TestCase) -> dict:
        """Complete evaluation of a single test case."""
        
        result = {
            "test_case_id": case.id,
            "complexity": case.complexity.value,
        }
        
        # 1. Create fresh flow
        flow = self.flow_factory()
        
        # 2. Instrument with trajectory collection
        trajectory = TrajectoryCollector()
        # Note: actual implementation would hook into flow callbacks
        
        # 3. Run the flow
        try:
            flow_result = flow.kickoff(inputs={
                "original_design": case.rtl_path.read_text()
            })
            result["flow_completed"] = True
        except Exception as e:
            result["flow_completed"] = False
            result["flow_error"] = str(e)
            result["verdict"] = "FAIL_FLOW"
            return result
        
        # 4. Trajectory evaluation
        traj_evaluator = TrajectoryEvaluator(case)
        result["trajectory"] = traj_evaluator.evaluate(trajectory)
        
        # 5. Contract validation
        try:
            # Would validate contracts from flow state
            result["contracts_passed"] = True
        except ContractViolation as e:
            result["contracts_passed"] = False
            result["contract_error"] = str(e)
        
        # 6. End-to-end PPA evaluation
        if hasattr(flow.state, 'synthesized_design'):
            e2e_result = self.e2e_evaluator.evaluate(
                case,
                flow.state.synthesized_design.get("rtl", ""),
                flow.state.synthesized_design.get("top_module", case.top_module)
            )
            result["e2e"] = e2e_result
            result["verdict"] = e2e_result.get("verdict", "UNKNOWN")
        else:
            result["verdict"] = "FAIL_NO_OUTPUT"
        
        # 7. LLM-as-judge (sampled)
        if self.quality_judge and self._should_run_judge():
            if hasattr(flow.state, 'analyzer_output'):
                result["judge_analysis"] = self.quality_judge.evaluate_analysis_quality(
                    case.rtl_path.read_text()[:2000],
                    flow.state.analyzer_output
                )
        
        return result
    
    def _should_run_judge(self) -> bool:
        import random
        return random.random() < self.config.sample_for_judge


@dataclass
class EvaluationReport:
    """Summary of an evaluation run."""
    suite_name: str
    timestamp: float
    total_cases: int
    results: list[dict]
    total_time_seconds: float
    
    @property
    def pass_rate(self) -> float:
        passing = sum(1 for r in self.results if r.get("verdict", "").startswith("PASS"))
        return passing / self.total_cases if self.total_cases > 0 else 0.0
    
    @property
    def improvement_rate(self) -> float:
        improved = sum(1 for r in self.results if r.get("verdict") == "PASS_IMPROVED")
        return improved / self.total_cases if self.total_cases > 0 else 0.0
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        verdict_counts = {}
        for r in self.results:
            v = r.get("verdict", "UNKNOWN")
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        
        lines = [
            f"=== Evaluation Report: {self.suite_name} ===",
            f"Total cases: {self.total_cases}",
            f"Pass rate: {self.pass_rate:.1%}",
            f"Improvement rate: {self.improvement_rate:.1%}",
            f"Total time: {self.total_time_seconds:.1f}s",
            "",
            "Verdict breakdown:",
        ]
        for verdict, count in sorted(verdict_counts.items()):
            lines.append(f"  {verdict}: {count}")
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        return json.dumps({
            "suite_name": self.suite_name,
            "timestamp": self.timestamp,
            "total_cases": self.total_cases,
            "pass_rate": self.pass_rate,
            "improvement_rate": self.improvement_rate,
            "total_time_seconds": self.total_time_seconds,
            "results": self.results,
        }, indent=2)
```

---

### Part 11: CI/CD Integration

```yaml
# .github/workflows/agent-evaluation.yml
name: Agent Evaluation

on:
  push:
    branches: [main]
    paths:
      - 'agents/**'
      - 'flows/**'
      - 'tools/**'
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 2 * * *'  # Nightly full suite

jobs:
  smoke-tests:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
      
      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -e ".[eval]"
      
      - name: Run smoke tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python -m evaluation.runner --suite smoke --output smoke-results.json
      
      - name: Check pass rate
        run: |
          python -c "
          import json
          with open('smoke-results.json') as f:
              report = json.load(f)
          if report['pass_rate'] < 0.9:
              print(f'FAIL: Pass rate {report[\"pass_rate\"]:.1%} < 90%')
              exit(1)
          print(f'PASS: Pass rate {report[\"pass_rate\"]:.1%}')
          "
      
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: smoke-results
          path: smoke-results.json

  full-evaluation:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'
    timeout-minutes: 240
    steps:
      - uses: actions/checkout@v4
      
      # ... similar setup ...
      
      - name: Run full evaluation
        run: |
          python -m evaluation.runner --suite full --output full-results.json
      
      - name: Generate regression report
        run: |
          python -m evaluation.regression_check \
            --current full-results.json \
            --baseline ${{ github.workspace }}/baselines/latest.json
      
      - name: Update baseline on main
        if: github.ref == 'refs/heads/main'
        run: |
          cp full-results.json baselines/$(date +%Y%m%d).json
          cp full-results.json baselines/latest.json
```

---

### Summary: Evaluation Strategy

| What | How | When | Threshold |
|------|-----|------|-----------|
| **Unit tests** | pytest + golden outputs | Every commit | 100% pass |
| **Contract validation** | Pydantic schemas | Every commit | 100% pass |
| **Trajectory coverage** | Expected tool sequences | Every commit | >90% coverage |
| **Smoke tests** | 3-5 trivial designs | Every PR | >90% pass |
| **Full PPA evaluation** | Actual EDA tools | Nightly | No regression |
| **LLM-as-judge** | Sampled quality assessment | Nightly | >3.5/5 average |
| **Functional simulation** | Verilator/VCS | Weekly | 100% pass on sampled |

**Key principles:**

1. **Ground truth over vibes** — PPA metrics from real EDA tools beat LLM opinions
2. **Fast feedback loop** — Smoke tests in <5 min, full suite overnight
3. **Regression detection** — Compare against baselines, not just pass/fail
4. **Trace everything** — You'll need the logs when debugging agent decisions
5. **Separate concerns** — Unit, integration, trajectory, e2e are different problems

Want me to dive deeper into any specific aspect of the evaluation harness?

## Trajectory Coverage

Trajectory coverage is one of the more nuanced aspects of agent evaluation. Let me break it down comprehensively.

## What Is Trajectory Coverage?

A **trajectory** is the sequence of actions an agent system takes to accomplish a task. For your SoC optimization pipeline, a trajectory might look like:

```
1. [Analyzer] LLM call: "Analyze this RTL..."
2. [Analyzer] Tool call: rtl_parser(design.v)
3. [Analyzer] LLM call: "Based on parsing results..."
4. [Abstractor] LLM call: "Abstract this analysis..."
5. [Generalizer] LLM call: "Generate optimized architecture..."
6. [Generalizer] Tool call: rtl_synthesis(new_design.v)  
7. [Generalizer] Tool call: rtl_synthesis(new_design.v)  ← retry after failure
8. [Validator] Tool call: simulation(netlist, testbench)
9. [Validator] LLM call: "Interpret simulation results..."
10. [Scorer] Tool call: ppa_analysis(netlist)
11. [Scorer] LLM call: "Compare metrics..."
```

**Trajectory coverage** answers: "Did the system take a reasonable path through the problem space?"

This is distinct from:
- **Output quality** — "Is the final answer good?" (end-to-end eval)
- **Contract compliance** — "Are intermediate outputs well-formed?" (integration eval)

---

## Why Trajectory Matters

Consider two runs that produce the same final output:

**Run A (Good trajectory):**
```
analyze → abstract → synthesize → validate(pass) → score
```

**Run B (Problematic trajectory):**
```
analyze → abstract → synthesize → validate(fail) → 
synthesize → validate(fail) → synthesize → validate(fail) →
synthesize → validate(pass) → score
```

Both might produce a valid optimized design, but Run B:
- Consumed 4x the compute
- Indicates the generalizer is producing unreliable outputs
- Might have gotten lucky on the 4th try

Trajectory analysis catches systemic issues that end-to-end metrics miss.

---

## Defining Expected Trajectories

### Level 1: Required Tools (Minimal)

At minimum, define which tools *must* be called:

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrajectoryExpectation:
    """Defines expected trajectory for a test case."""
    
    # Tools that MUST be called
    required_tools: list[str]
    
    # Tools that SHOULD be called (warning if missing)
    recommended_tools: list[str] = None
    
    # Tools that MUST NOT be called
    forbidden_tools: list[str] = None
    
    # Maximum allowed tool calls (efficiency bound)
    max_total_tool_calls: Optional[int] = None
    
    # Maximum retries per tool
    max_retries_per_tool: int = 3


# Example for your SoC pipeline
SOC_OPTIMIZATION_TRAJECTORY = TrajectoryExpectation(
    required_tools=[
        "rtl_parser",       # Must parse the input
        "rtl_synthesis",    # Must synthesize
        "static_timing",    # Must run STA
        "ppa_analysis",     # Must compute final metrics
    ],
    recommended_tools=[
        "power_analysis",   # Should run power analysis
        "lint_check",       # Should lint before synthesis
    ],
    forbidden_tools=[
        "web_search",       # Should not search the web for RTL
        "code_execution",   # Should not run arbitrary code
    ],
    max_total_tool_calls=20,
    max_retries_per_tool=3,
)
```

### Level 2: Ordered Sequences (Stricter)

Some tool orderings are semantically required:

```python
@dataclass
class OrderedSequence:
    """A sequence of tools that must appear in order."""
    tools: list[str]
    strict: bool = False  # If True, must be consecutive
    

@dataclass 
class TrajectoryExpectationV2(TrajectoryExpectation):
    """Extended expectations with ordering constraints."""
    
    # Sequences that must appear in order
    required_sequences: list[OrderedSequence] = None
    
    # Tools that must come before others
    precedence_rules: dict[str, list[str]] = None  # tool -> must_come_after


SOC_TRAJECTORY_V2 = TrajectoryExpectationV2(
    required_tools=["rtl_synthesis", "static_timing", "ppa_analysis"],
    
    required_sequences=[
        # Synthesis must come before STA
        OrderedSequence(["rtl_synthesis", "static_timing"], strict=False),
        # STA must come before power (needs timing info)
        OrderedSequence(["static_timing", "power_analysis"], strict=False),
    ],
    
    precedence_rules={
        # static_timing must come after rtl_synthesis
        "static_timing": ["rtl_synthesis"],
        # ppa_analysis must come after both
        "ppa_analysis": ["rtl_synthesis", "static_timing"],
    },
    
    max_total_tool_calls=25,
)
```

### Level 3: State-Dependent Expectations (Most Flexible)

Real trajectories depend on intermediate results:

```python
from typing import Callable


@dataclass
class ConditionalExpectation:
    """Trajectory expectations that depend on runtime state."""
    
    condition: Callable[[dict], bool]  # Function of trajectory state
    then_expect: TrajectoryExpectation
    else_expect: Optional[TrajectoryExpectation] = None


class DynamicTrajectorySpec:
    """
    Specification that adapts based on what happens during execution.
    """
    
    def __init__(self):
        self.base = TrajectoryExpectationV2(
            required_tools=["rtl_synthesis"],
            max_total_tool_calls=30,
        )
        self.conditionals: list[ConditionalExpectation] = []
    
    def add_conditional(
        self, 
        condition: Callable[[dict], bool],
        then_expect: TrajectoryExpectation,
        else_expect: Optional[TrajectoryExpectation] = None
    ):
        self.conditionals.append(ConditionalExpectation(
            condition=condition,
            then_expect=then_expect,
            else_expect=else_expect,
        ))
    
    def get_expectations(self, trajectory_state: dict) -> TrajectoryExpectation:
        """Resolve expectations based on current trajectory state."""
        result = self.base
        
        for cond in self.conditionals:
            if cond.condition(trajectory_state):
                result = self._merge(result, cond.then_expect)
            elif cond.else_expect:
                result = self._merge(result, cond.else_expect)
        
        return result
    
    def _merge(self, base: TrajectoryExpectation, overlay: TrajectoryExpectation):
        # Merge logic here
        pass


# Example: If synthesis fails, expect retry behavior
spec = DynamicTrajectorySpec()

spec.add_conditional(
    condition=lambda state: state.get("synthesis_failed", False),
    then_expect=TrajectoryExpectation(
        required_tools=["rtl_synthesis"],  # Must retry
        max_retries_per_tool=5,            # Allow more retries
    ),
)

spec.add_conditional(
    condition=lambda state: state.get("timing_violated", False),
    then_expect=TrajectoryExpectation(
        required_tools=["constraint_relaxation", "rtl_synthesis"],
        recommended_tools=["timing_optimization"],
    ),
)
```

---

## Trajectory Matching Algorithms

Given an actual trajectory and expected trajectory, how do you score the match?

### Algorithm 1: Set Coverage (Simplest)

Just check if required tools were called:

```python
def compute_set_coverage(
    actual_tools: list[str],
    expected: TrajectoryExpectation
) -> dict:
    """Simple set-based coverage metric."""
    
    actual_set = set(actual_tools)
    required_set = set(expected.required_tools)
    recommended_set = set(expected.recommended_tools or [])
    forbidden_set = set(expected.forbidden_tools or [])
    
    # Required coverage: what fraction of required tools were called?
    required_covered = required_set.intersection(actual_set)
    required_coverage = len(required_covered) / len(required_set) if required_set else 1.0
    
    # Required missing: which required tools were NOT called?
    required_missing = required_set - actual_set
    
    # Recommended coverage
    recommended_covered = recommended_set.intersection(actual_set)
    recommended_coverage = len(recommended_covered) / len(recommended_set) if recommended_set else 1.0
    
    # Forbidden violations: which forbidden tools WERE called?
    forbidden_violations = forbidden_set.intersection(actual_set)
    
    return {
        "required_coverage": required_coverage,
        "required_missing": list(required_missing),
        "recommended_coverage": recommended_coverage,
        "recommended_missing": list(recommended_set - actual_set),
        "forbidden_violations": list(forbidden_violations),
        "passed": required_coverage == 1.0 and len(forbidden_violations) == 0,
    }
```

### Algorithm 2: Subsequence Matching (Order-Aware)

Check if required sequences appear in order:

```python
def is_subsequence(needle: list[str], haystack: list[str]) -> bool:
    """Check if needle appears as a subsequence of haystack."""
    it = iter(haystack)
    return all(item in it for item in needle)


def find_subsequence_positions(needle: list[str], haystack: list[str]) -> list[int]:
    """Find positions where subsequence elements appear."""
    positions = []
    it = iter(enumerate(haystack))
    
    for target in needle:
        for idx, item in it:
            if item == target:
                positions.append(idx)
                break
        else:
            return []  # Not found
    
    return positions


def compute_sequence_coverage(
    actual_tools: list[str],
    expected: TrajectoryExpectationV2
) -> dict:
    """Check ordered sequence requirements."""
    
    results = {
        "sequences": [],
        "all_sequences_present": True,
        "precedence_violations": [],
    }
    
    # Check each required sequence
    for seq in (expected.required_sequences or []):
        positions = find_subsequence_positions(seq.tools, actual_tools)
        present = len(positions) == len(seq.tools)
        
        seq_result = {
            "sequence": seq.tools,
            "present": present,
            "positions": positions if present else None,
            "strict": seq.strict,
        }
        
        # If strict, check consecutiveness
        if present and seq.strict:
            is_consecutive = all(
                positions[i+1] == positions[i] + 1 
                for i in range(len(positions)-1)
            )
            seq_result["consecutive"] = is_consecutive
            if not is_consecutive:
                seq_result["present"] = False
        
        results["sequences"].append(seq_result)
        if not seq_result["present"]:
            results["all_sequences_present"] = False
    
    # Check precedence rules
    tool_positions = {tool: i for i, tool in enumerate(actual_tools)}
    
    for tool, must_come_after in (expected.precedence_rules or {}).items():
        if tool not in tool_positions:
            continue
            
        tool_pos = tool_positions[tool]
        for predecessor in must_come_after:
            if predecessor in tool_positions:
                pred_pos = tool_positions[predecessor]
                if pred_pos > tool_pos:
                    results["precedence_violations"].append({
                        "tool": tool,
                        "should_come_after": predecessor,
                        "actual_positions": {tool: tool_pos, predecessor: pred_pos}
                    })
    
    return results
```

### Algorithm 3: Edit Distance (Fuzzy Matching)

For comparing against a "golden" trajectory:

```python
def trajectory_edit_distance(
    actual: list[str],
    expected: list[str],
    costs: dict = None
) -> dict:
    """
    Compute edit distance between trajectories.
    
    Allows custom costs for different operations.
    """
    costs = costs or {
        "insert": 1.0,    # Extra tool call
        "delete": 2.0,    # Missing required tool (more serious)
        "substitute": 1.5, # Wrong tool
    }
    
    m, n = len(actual), len(expected)
    
    # DP table
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    
    # Base cases
    for i in range(m + 1):
        dp[i][0] = i * costs["insert"]
    for j in range(n + 1):
        dp[0][j] = j * costs["delete"]
    
    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if actual[i-1] == expected[j-1]:
                dp[i][j] = dp[i-1][j-1]  # Match, no cost
            else:
                dp[i][j] = min(
                    dp[i-1][j] + costs["insert"],      # Insert
                    dp[i][j-1] + costs["delete"],      # Delete
                    dp[i-1][j-1] + costs["substitute"] # Substitute
                )
    
    distance = dp[m][n]
    max_distance = max(m, n) * max(costs.values())
    similarity = 1.0 - (distance / max_distance) if max_distance > 0 else 1.0
    
    return {
        "edit_distance": distance,
        "similarity": similarity,
        "actual_length": m,
        "expected_length": n,
    }
```

### Algorithm 4: State Machine Matching (Most Expressive)

Model valid trajectories as a state machine:

```python
from enum import Enum, auto
from typing import Set


class TrajectoryState(Enum):
    """States in the SoC optimization trajectory."""
    START = auto()
    ANALYZED = auto()
    ABSTRACTED = auto()
    SYNTHESIZED = auto()
    SYNTHESIS_FAILED = auto()
    VALIDATED = auto()
    VALIDATION_FAILED = auto()
    SCORED = auto()
    TERMINAL = auto()


class TrajectoryStateMachine:
    """
    Defines valid trajectory transitions.
    
    This captures complex conditional logic like:
    - "After synthesis failure, you can retry or adjust constraints"
    - "Validation must come after synthesis"
    """
    
    def __init__(self):
        self.transitions: dict[TrajectoryState, dict[str, TrajectoryState]] = {
            TrajectoryState.START: {
                "rtl_parser": TrajectoryState.ANALYZED,
                "lint_check": TrajectoryState.START,  # Can lint at start
            },
            TrajectoryState.ANALYZED: {
                "abstraction": TrajectoryState.ABSTRACTED,
                "rtl_parser": TrajectoryState.ANALYZED,  # Can re-parse
            },
            TrajectoryState.ABSTRACTED: {
                "rtl_synthesis": TrajectoryState.SYNTHESIZED,
                "constraint_relaxation": TrajectoryState.ABSTRACTED,
            },
            TrajectoryState.SYNTHESIZED: {
                "static_timing": TrajectoryState.SYNTHESIZED,
                "power_analysis": TrajectoryState.SYNTHESIZED,
                "simulation": TrajectoryState.VALIDATED,
                "formal_verification": TrajectoryState.VALIDATED,
            },
            TrajectoryState.SYNTHESIS_FAILED: {
                "rtl_synthesis": TrajectoryState.SYNTHESIZED,  # Retry
                "constraint_relaxation": TrajectoryState.ABSTRACTED,  # Adjust
                "abort": TrajectoryState.TERMINAL,
            },
            TrajectoryState.VALIDATED: {
                "ppa_analysis": TrajectoryState.SCORED,
            },
            TrajectoryState.VALIDATION_FAILED: {
                "rtl_synthesis": TrajectoryState.SYNTHESIZED,  # Retry with new design
                "abort": TrajectoryState.TERMINAL,
            },
            TrajectoryState.SCORED: {
                "report": TrajectoryState.TERMINAL,
            },
        }
        
        # Special transitions based on tool results
        self.failure_transitions = {
            "rtl_synthesis": TrajectoryState.SYNTHESIS_FAILED,
            "simulation": TrajectoryState.VALIDATION_FAILED,
            "formal_verification": TrajectoryState.VALIDATION_FAILED,
        }
        
        self.terminal_states = {TrajectoryState.TERMINAL, TrajectoryState.SCORED}
    
    def validate_trajectory(
        self, 
        tool_calls: list[tuple[str, bool]]  # (tool_name, success)
    ) -> dict:
        """
        Validate a trajectory against the state machine.
        
        Returns detailed analysis of valid/invalid transitions.
        """
        current_state = TrajectoryState.START
        path = [(current_state, None, True)]
        violations = []
        
        for i, (tool, success) in enumerate(tool_calls):
            # Check if transition is valid
            valid_transitions = self.transitions.get(current_state, {})
            
            if tool not in valid_transitions:
                violations.append({
                    "step": i,
                    "tool": tool,
                    "from_state": current_state.name,
                    "valid_tools": list(valid_transitions.keys()),
                    "error": "invalid_transition"
                })
                # Try to recover by staying in current state
                next_state = current_state
            else:
                # Determine next state based on success/failure
                if success:
                    next_state = valid_transitions[tool]
                elif tool in self.failure_transitions:
                    next_state = self.failure_transitions[tool]
                else:
                    next_state = current_state  # Stay in current state
            
            path.append((next_state, tool, success))
            current_state = next_state
        
        # Check if we reached a valid terminal state
        reached_terminal = current_state in self.terminal_states
        
        return {
            "valid": len(violations) == 0 and reached_terminal,
            "violations": violations,
            "path": [(s.name, t, ok) for s, t, ok in path],
            "final_state": current_state.name,
            "reached_terminal": reached_terminal,
        }
    
    def visualize(self) -> str:
        """Generate Mermaid diagram of the state machine."""
        lines = ["stateDiagram-v2"]
        
        for from_state, transitions in self.transitions.items():
            for tool, to_state in transitions.items():
                lines.append(f"    {from_state.name} --> {to_state.name}: {tool}")
        
        return "\n".join(lines)
```

---

## Comprehensive Trajectory Evaluator

Putting it all together:

```python
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter
import json


@dataclass
class TrajectoryStep:
    """A single step in execution."""
    index: int
    timestamp_ms: float
    agent: str
    action_type: str  # "tool_call", "llm_call", "delegation"
    action_name: str
    input_hash: str  # For detecting repeated inputs
    success: bool
    duration_ms: float
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete execution trajectory."""
    steps: list[TrajectoryStep]
    
    def tool_calls(self) -> list[str]:
        """Extract just tool names."""
        return [s.action_name for s in self.steps if s.action_type == "tool_call"]
    
    def tool_calls_with_success(self) -> list[tuple[str, bool]]:
        """Extract tool names with success status."""
        return [
            (s.action_name, s.success) 
            for s in self.steps 
            if s.action_type == "tool_call"
        ]
    
    def llm_calls(self) -> list[TrajectoryStep]:
        """Extract LLM call steps."""
        return [s for s in self.steps if s.action_type == "llm_call"]
    
    def by_agent(self) -> dict[str, list[TrajectoryStep]]:
        """Group steps by agent."""
        result = {}
        for step in self.steps:
            if step.agent not in result:
                result[step.agent] = []
            result[step.agent].append(step)
        return result
    
    def total_duration_ms(self) -> float:
        return sum(s.duration_ms for s in self.steps)
    
    def total_tokens(self) -> int:
        return sum(s.tokens_in + s.tokens_out for s in self.steps)


class TrajectoryEvaluator:
    """
    Comprehensive trajectory evaluation.
    
    Combines multiple evaluation strategies.
    """
    
    def __init__(
        self,
        expectation: TrajectoryExpectationV2,
        state_machine: Optional[TrajectoryStateMachine] = None,
        golden_trajectory: Optional[list[str]] = None,
    ):
        self.expectation = expectation
        self.state_machine = state_machine
        self.golden = golden_trajectory
    
    def evaluate(self, trajectory: Trajectory) -> dict:
        """Run all trajectory evaluations."""
        
        tool_calls = trajectory.tool_calls()
        tool_calls_with_success = trajectory.tool_calls_with_success()
        
        results = {
            "summary": {},
            "set_coverage": self._evaluate_set_coverage(tool_calls),
            "sequence_coverage": self._evaluate_sequence_coverage(tool_calls),
            "efficiency": self._evaluate_efficiency(trajectory),
            "anti_patterns": self._detect_anti_patterns(trajectory),
        }
        
        # Optional: state machine validation
        if self.state_machine:
            results["state_machine"] = self.state_machine.validate_trajectory(
                tool_calls_with_success
            )
        
        # Optional: golden trajectory comparison
        if self.golden:
            results["golden_comparison"] = trajectory_edit_distance(
                tool_calls, self.golden
            )
        
        # Compute summary score
        results["summary"] = self._compute_summary_score(results)
        
        return results
    
    def _evaluate_set_coverage(self, tool_calls: list[str]) -> dict:
        return compute_set_coverage(tool_calls, self.expectation)
    
    def _evaluate_sequence_coverage(self, tool_calls: list[str]) -> dict:
        return compute_sequence_coverage(tool_calls, self.expectation)
    
    def _evaluate_efficiency(self, trajectory: Trajectory) -> dict:
        """Evaluate execution efficiency."""
        
        tool_calls = trajectory.tool_calls()
        
        # Total calls vs expected
        expected_min = len(self.expectation.required_tools)
        actual = len(tool_calls)
        
        # Tool call distribution
        call_counts = Counter(tool_calls)
        
        # Detect excessive retries
        max_calls_per_tool = max(call_counts.values()) if call_counts else 0
        
        # Time efficiency
        total_time = trajectory.total_duration_ms()
        tool_time = sum(
            s.duration_ms for s in trajectory.steps 
            if s.action_type == "tool_call"
        )
        llm_time = sum(
            s.duration_ms for s in trajectory.steps 
            if s.action_type == "llm_call"
        )
        
        # Token efficiency
        total_tokens = trajectory.total_tokens()
        
        return {
            "total_tool_calls": actual,
            "expected_minimum": expected_min,
            "efficiency_ratio": expected_min / actual if actual > 0 else 0,
            "call_distribution": dict(call_counts),
            "max_calls_per_tool": max_calls_per_tool,
            "exceeds_retry_limit": max_calls_per_tool > self.expectation.max_retries_per_tool,
            "exceeds_total_limit": (
                self.expectation.max_total_tool_calls and 
                actual > self.expectation.max_total_tool_calls
            ),
            "time_breakdown": {
                "total_ms": total_time,
                "tool_ms": tool_time,
                "llm_ms": llm_time,
                "overhead_ms": total_time - tool_time - llm_time,
            },
            "total_tokens": total_tokens,
        }
    
    def _detect_anti_patterns(self, trajectory: Trajectory) -> list[dict]:
        """Detect problematic patterns in the trajectory."""
        
        anti_patterns = []
        
        # Anti-pattern 1: Repeated identical calls
        seen_calls = {}  # (tool, input_hash) -> list of indices
        for step in trajectory.steps:
            if step.action_type != "tool_call":
                continue
            key = (step.action_name, step.input_hash)
            if key not in seen_calls:
                seen_calls[key] = []
            seen_calls[key].append(step.index)
        
        for (tool, _), indices in seen_calls.items():
            if len(indices) > 1:
                anti_patterns.append({
                    "type": "repeated_identical_call",
                    "tool": tool,
                    "occurrences": len(indices),
                    "indices": indices,
                    "severity": "warning" if len(indices) <= 2 else "error",
                })
        
        # Anti-pattern 2: Tool call immediately after failure without change
        prev_step = None
        for step in trajectory.steps:
            if step.action_type != "tool_call":
                prev_step = step
                continue
            
            if prev_step and prev_step.action_type == "tool_call":
                if (prev_step.action_name == step.action_name and
                    not prev_step.success and
                    prev_step.input_hash == step.input_hash):
                    anti_patterns.append({
                        "type": "retry_without_change",
                        "tool": step.action_name,
                        "indices": [prev_step.index, step.index],
                        "severity": "warning",
                        "suggestion": "Modify input or adjust parameters before retry",
                    })
            
            prev_step = step
        
        # Anti-pattern 3: Wrong tool order
        tool_calls = trajectory.tool_calls()
        critical_order = ["rtl_synthesis", "static_timing"]  # STA must come after synth
        
        synth_idx = None
        sta_idx = None
        for i, tool in enumerate(tool_calls):
            if tool == "rtl_synthesis" and synth_idx is None:
                synth_idx = i
            if tool == "static_timing" and sta_idx is None:
                sta_idx = i
        
        if synth_idx is not None and sta_idx is not None and sta_idx < synth_idx:
            anti_patterns.append({
                "type": "wrong_order",
                "expected": "rtl_synthesis before static_timing",
                "actual": f"static_timing at {sta_idx}, rtl_synthesis at {synth_idx}",
                "severity": "error",
            })
        
        # Anti-pattern 4: Excessive LLM calls between tool calls
        llm_streak = 0
        max_llm_streak = 0
        for step in trajectory.steps:
            if step.action_type == "llm_call":
                llm_streak += 1
                max_llm_streak = max(max_llm_streak, llm_streak)
            else:
                llm_streak = 0
        
        if max_llm_streak > 5:
            anti_patterns.append({
                "type": "excessive_llm_calls",
                "max_consecutive": max_llm_streak,
                "severity": "warning",
                "suggestion": "Agent may be stuck in reasoning loop",
            })
        
        # Anti-pattern 5: Forbidden tool usage
        forbidden = set(self.expectation.forbidden_tools or [])
        used_forbidden = forbidden.intersection(set(trajectory.tool_calls()))
        for tool in used_forbidden:
            anti_patterns.append({
                "type": "forbidden_tool",
                "tool": tool,
                "severity": "error",
            })
        
        return anti_patterns
    
    def _compute_summary_score(self, results: dict) -> dict:
        """Compute overall trajectory score."""
        
        score = 100.0
        issues = []
        
        # Penalize missing required tools
        set_cov = results["set_coverage"]
        if set_cov["required_coverage"] < 1.0:
            penalty = (1.0 - set_cov["required_coverage"]) * 30
            score -= penalty
            issues.append(f"Missing required tools: {set_cov['required_missing']}")
        
        # Penalize forbidden tool usage
        if set_cov["forbidden_violations"]:
            score -= 20 * len(set_cov["forbidden_violations"])
            issues.append(f"Used forbidden tools: {set_cov['forbidden_violations']}")
        
        # Penalize sequence violations
        seq_cov = results["sequence_coverage"]
        if not seq_cov["all_sequences_present"]:
            score -= 15
            issues.append("Required sequences not present")
        if seq_cov["precedence_violations"]:
            score -= 10 * len(seq_cov["precedence_violations"])
            issues.append(f"Precedence violations: {len(seq_cov['precedence_violations'])}")
        
        # Penalize efficiency issues
        eff = results["efficiency"]
        if eff["exceeds_total_limit"]:
            score -= 10
            issues.append(f"Exceeded tool call limit: {eff['total_tool_calls']}")
        if eff["exceeds_retry_limit"]:
            score -= 5
            issues.append(f"Excessive retries: {eff['max_calls_per_tool']}")
        
        # Penalize anti-patterns
        for ap in results["anti_patterns"]:
            if ap["severity"] == "error":
                score -= 10
            else:
                score -= 3
            issues.append(f"{ap['type']}: {ap.get('tool', ap.get('expected', ''))}")
        
        # Bonus for efficiency
        if eff["efficiency_ratio"] > 0.8:
            score += 5
        
        return {
            "score": max(0, min(100, score)),
            "grade": self._score_to_grade(score),
            "issues": issues,
            "passed": score >= 70,
        }
    
    def _score_to_grade(self, score: float) -> str:
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
```

---

## Trajectory Collection from CrewAI

To collect trajectories, hook into CrewAI's callback system:

```python
from crewai import Crew
import time
import hashlib


class TrajectoryCollector:
    """Collects trajectory from CrewAI execution."""
    
    def __init__(self):
        self.steps: list[TrajectoryStep] = []
        self._start_time: float = 0
        self._step_index: int = 0
    
    def start(self):
        self._start_time = time.time()
        self._step_index = 0
        self.steps = []
    
    def step_callback(self, step_output):
        """Called after each agent step."""
        
        now = time.time()
        
        # Determine action type
        if hasattr(step_output, 'tool') and step_output.tool:
            action_type = "tool_call"
            action_name = step_output.tool
        else:
            action_type = "llm_call"
            action_name = "generate"
        
        # Hash the input for duplicate detection
        input_str = str(getattr(step_output, 'tool_input', ''))
        input_hash = hashlib.md5(input_str.encode()).hexdigest()[:8]
        
        # Detect success/failure
        result = getattr(step_output, 'result', None)
        error = getattr(step_output, 'error', None)
        success = error is None and result is not None
        
        step = TrajectoryStep(
            index=self._step_index,
            timestamp_ms=(now - self._start_time) * 1000,
            agent=getattr(step_output, 'agent', 'unknown'),
            action_type=action_type,
            action_name=action_name,
            input_hash=input_hash,
            success=success,
            duration_ms=getattr(step_output, 'execution_time', 0) * 1000,
            tokens_in=getattr(step_output, 'prompt_tokens', 0),
            tokens_out=getattr(step_output, 'completion_tokens', 0),
            error=str(error) if error else None,
            metadata={
                "input_preview": input_str[:200],
                "output_preview": str(result)[:200] if result else None,
            }
        )
        
        self.steps.append(step)
        self._step_index += 1
    
    def task_callback(self, task_output):
        """Called after each task completion."""
        # Can add task-level markers to trajectory
        pass
    
    def get_trajectory(self) -> Trajectory:
        return Trajectory(steps=self.steps)


def run_with_trajectory_collection(
    crew: Crew,
    inputs: dict,
) -> tuple[any, Trajectory]:
    """Execute crew and collect trajectory."""
    
    collector = TrajectoryCollector()
    collector.start()
    
    # Inject callbacks
    original_step_callback = crew.step_callback
    crew.step_callback = collector.step_callback
    
    try:
        result = crew.kickoff(inputs=inputs)
    finally:
        crew.step_callback = original_step_callback
    
    return result, collector.get_trajectory()
```

---

## Trajectory Visualization

For debugging, visualize trajectories:

```python
def trajectory_to_mermaid(trajectory: Trajectory) -> str:
    """Generate Mermaid sequence diagram."""
    
    lines = ["sequenceDiagram"]
    
    current_agent = None
    for step in trajectory.steps:
        agent = step.agent.replace(" ", "_")
        
        if step.action_type == "tool_call":
            status = "✓" if step.success else "✗"
            lines.append(
                f"    {agent}->>Tools: {step.action_name} {status}"
            )
            if step.success:
                lines.append(f"    Tools-->>{ agent}: result")
            else:
                lines.append(f"    Tools--x{agent}: error")
        else:
            lines.append(f"    Note over {agent}: LLM reasoning")
    
    return "\n".join(lines)


def trajectory_to_gantt(trajectory: Trajectory) -> str:
    """Generate Mermaid Gantt chart for timing."""
    
    lines = [
        "gantt",
        "    title Trajectory Timeline",
        "    dateFormat X",
        "    axisFormat %s",
    ]
    
    for step in trajectory.steps:
        start = int(step.timestamp_ms - step.duration_ms)
        duration = int(step.duration_ms)
        
        status = "" if step.success else "crit,"
        lines.append(
            f"    {step.action_name}: {status}{start}, {duration}ms"
        )
    
    return "\n".join(lines)


def print_trajectory_summary(trajectory: Trajectory):
    """Print human-readable trajectory summary."""
    
    print(f"{'='*60}")
    print(f"TRAJECTORY SUMMARY")
    print(f"{'='*60}")
    print(f"Total steps: {len(trajectory.steps)}")
    print(f"Total time: {trajectory.total_duration_ms():.0f}ms")
    print(f"Total tokens: {trajectory.total_tokens()}")
    print()
    
    print("Steps:")
    for step in trajectory.steps:
        status = "✓" if step.success else "✗"
        print(f"  {step.index:3d}. [{step.agent:20s}] {step.action_type:10s} "
              f"{step.action_name:25s} {status} ({step.duration_ms:.0f}ms)")
    
    print()
    print("Tool call frequency:")
    for tool, count in Counter(trajectory.tool_calls()).most_common():
        print(f"  {tool}: {count}")
```

---

## Statistical Analysis Across Runs

Single-run trajectory metrics are noisy. Aggregate across runs:

```python
import statistics
from typing import List


class TrajectoryStatistics:
    """Aggregate trajectory metrics across multiple runs."""
    
    def __init__(self, evaluations: list[dict]):
        """
        Args:
            evaluations: List of results from TrajectoryEvaluator.evaluate()
        """
        self.evaluations = evaluations
    
    def compute_statistics(self) -> dict:
        """Compute aggregate statistics."""
        
        scores = [e["summary"]["score"] for e in self.evaluations]
        tool_counts = [e["efficiency"]["total_tool_calls"] for e in self.evaluations]
        
        # Pass rate
        passed = sum(1 for e in self.evaluations if e["summary"]["passed"])
        
        # Anti-pattern frequency
        all_anti_patterns = []
        for e in self.evaluations:
            all_anti_patterns.extend(e["anti_patterns"])
        
        ap_counts = Counter(ap["type"] for ap in all_anti_patterns)
        
        # Efficiency distribution
        efficiency_ratios = [e["efficiency"]["efficiency_ratio"] for e in self.evaluations]
        
        return {
            "n_runs": len(self.evaluations),
            "pass_rate": passed / len(self.evaluations),
            "score": {
                "mean": statistics.mean(scores),
                "median": statistics.median(scores),
                "stdev": statistics.stdev(scores) if len(scores) > 1 else 0,
                "min": min(scores),
                "max": max(scores),
            },
            "tool_calls": {
                "mean": statistics.mean(tool_counts),
                "median": statistics.median(tool_counts),
                "stdev": statistics.stdev(tool_counts) if len(tool_counts) > 1 else 0,
            },
            "efficiency": {
                "mean": statistics.mean(efficiency_ratios),
                "median": statistics.median(efficiency_ratios),
            },
            "anti_pattern_frequency": dict(ap_counts),
            "most_common_issues": Counter(
                issue 
                for e in self.evaluations 
                for issue in e["summary"]["issues"]
            ).most_common(5),
        }
    
    def detect_regressions(
        self, 
        baseline_stats: dict,
        thresholds: dict = None
    ) -> list[dict]:
        """Compare against baseline statistics to detect regressions."""
        
        thresholds = thresholds or {
            "score_drop": 5.0,           # Alert if mean score drops by 5+
            "pass_rate_drop": 0.05,      # Alert if pass rate drops by 5%+
            "efficiency_drop": 0.1,      # Alert if efficiency drops by 10%+
        }
        
        current = self.compute_statistics()
        regressions = []
        
        # Score regression
        score_diff = baseline_stats["score"]["mean"] - current["score"]["mean"]
        if score_diff > thresholds["score_drop"]:
            regressions.append({
                "metric": "score",
                "baseline": baseline_stats["score"]["mean"],
                "current": current["score"]["mean"],
                "diff": score_diff,
                "severity": "high" if score_diff > 10 else "medium",
            })
        
        # Pass rate regression
        pass_diff = baseline_stats["pass_rate"] - current["pass_rate"]
        if pass_diff > thresholds["pass_rate_drop"]:
            regressions.append({
                "metric": "pass_rate",
                "baseline": baseline_stats["pass_rate"],
                "current": current["pass_rate"],
                "diff": pass_diff,
                "severity": "high" if pass_diff > 0.1 else "medium",
            })
        
        # Efficiency regression
        eff_diff = baseline_stats["efficiency"]["mean"] - current["efficiency"]["mean"]
        if eff_diff > thresholds["efficiency_drop"]:
            regressions.append({
                "metric": "efficiency",
                "baseline": baseline_stats["efficiency"]["mean"],
                "current": current["efficiency"]["mean"],
                "diff": eff_diff,
                "severity": "medium",
            })
        
        # New anti-patterns
        baseline_aps = set(baseline_stats.get("anti_pattern_frequency", {}).keys())
        current_aps = set(current["anti_pattern_frequency"].keys())
        new_aps = current_aps - baseline_aps
        if new_aps:
            regressions.append({
                "metric": "new_anti_patterns",
                "patterns": list(new_aps),
                "severity": "medium",
            })
        
        return regressions
```

---

## Integration with CI

```python
# evaluation/trajectory_check.py

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to trajectory eval results JSON")
    parser.add_argument("--baseline", help="Path to baseline stats JSON")
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--min-score", type=float, default=70.0)
    args = parser.parse_args()
    
    with open(args.results) as f:
        evaluations = json.load(f)
    
    stats = TrajectoryStatistics(evaluations)
    current = stats.compute_statistics()
    
    print("Trajectory Evaluation Summary")
    print("=" * 40)
    print(f"Runs: {current['n_runs']}")
    print(f"Pass rate: {current['pass_rate']:.1%}")
    print(f"Mean score: {current['score']['mean']:.1f}")
    print(f"Mean efficiency: {current['efficiency']['mean']:.2f}")
    print()
    
    failures = []
    
    # Check thresholds
    if current["pass_rate"] < args.min_pass_rate:
        failures.append(
            f"Pass rate {current['pass_rate']:.1%} < {args.min_pass_rate:.1%}"
        )
    
    if current["score"]["mean"] < args.min_score:
        failures.append(
            f"Mean score {current['score']['mean']:.1f} < {args.min_score}"
        )
    
    # Check for regressions
    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        
        regressions = stats.detect_regressions(baseline)
        for reg in regressions:
            if reg["severity"] == "high":
                failures.append(f"Regression: {reg['metric']} ({reg.get('diff', '')})")
            else:
                print(f"Warning: {reg['metric']} regressed")
    
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    
    print("\n✓ All trajectory checks passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
```

---

## Summary

Trajectory coverage evaluates **how** the system solves problems, not just **what** it produces:

| Aspect | What It Catches | Method |
|--------|----------------|--------|
| **Required tools** | Missing critical steps | Set coverage |
| **Tool order** | Logical errors (STA before synth) | Sequence matching |
| **Efficiency** | Wasted compute, stuck loops | Call counts, ratios |
| **Anti-patterns** | Repeated failures, forbidden tools | Pattern detection |
| **State validity** | Impossible transitions | State machine |
| **Regression** | System getting worse over time | Baseline comparison |

For your SoC optimization system specifically, trajectory coverage would catch:
- Generalizer generating designs without first analyzing
- Validator being skipped
- Excessive synthesis retries indicating bad design generation
- Wrong tool order (power analysis before synthesis)
- Agents calling forbidden tools (web search for RTL)

This gives you visibility into agent behavior that end-to-end metrics miss entirely.
