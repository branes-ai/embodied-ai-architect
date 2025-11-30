# Phase 2

## Natural Language Constraints (Milestone 2.2)

Tthe **Natural Language Constraints** interface is a critical step for developing the Agent's MVP. This feature validates the Agent's core ability to perform **reasoning and translation**—converting high-level goals into low-level engineering parameters for the Design Space Exploration (DSE).

Here are examples of how a user interacts with the system versus the technical translation performed by the Agent.

### Constraint Translation Examples

| User Input (Natural Language Constraint) | Agent's Reasoning & Translation Process |
| :--- | :--- |
| **"Maximize flight time for a standard delivery mission, while keeping total BoM cost below $350."** | 1. **Identify KPI:** Maximize $\text{Range} / \text{Watt}$. 2. **Constraint Check:** Filter hardware options where $\text{Cost} < \$350$. 3. **Action:** Prioritize lower clock frequencies, highly quantized $int8$ models, and smaller L2 cache sizes. |
| **"The drone must reliably detect objects at a minimum distance of 10 meters, regardless of weather."** | 1. **Identify KPI:** Minimize False Negatives at 10m range. 2. **Action:** **Prohibit** high-loss $int8$ quantization or aggressive model pruning. **Mandate** running the perception model at full $\text{fp32}$ or $\text{fp16}$ precision. 3. **Sim Check:** Introduce **Domain Randomization** (virtual fog/rain) into the Flight Simulator. |
| **"We are using a proprietary flight control loop that requires a minimum loop rate of 250 Hz."** | 1. **Identify Constraint:** $\text{Max Latency}_{\text{Control Loop}} < 4 \text{ms}$. 2. **Action:** **Fix** the CPU core frequency needed for the control stack. **Allocate** dedicated compute resources (CPU core affinity) to the control loop. 3. **Search Space Reduction:** Prune any HW/SW design combination that violates the $4 \text{ms}$ deadline. |
| **"The compute needs to operate in environments up to $60^\circ\text{C}$ without active cooling."** | 1. **Identify Constraint:** $\text{Max Thermal Design Power (TDP)} < 5 \text{W}$ (estimated for passive cooling). 2. **Action:** Use the **Power Estimator** (McPAT) to enforce a search boundary. 3. **Prioritize:** Designs that rely on highly efficient custom accelerators (NPU) over general-purpose CPUs/GPUs to keep **Package Power (W)** low. |

***

### Example Interface Visualization

To showcase this feature, your MVP interface would need to make the constraint-to-design-space reduction clear to the user.


This visualization demonstrates that the Agent's value is not just in *finding* an optimal point, but in **intelligently constraining the massive design space** based on human context.

***

## User Story and Acceptance Criteria for Milestone 2.2
### **The Agentic Software Explorer (MVP)**

Milestone 2.2 focuses on proving the Agent's ability to automate the **software** design space exploration (DSE) to maximize energy efficiency on a **fixed, target hardware platform** (e.g., a specific drone flight controller or companion computer).

---

### User Story

> **As a Drone Systems Architect,**
>
> **I want** to input high-level mission goals and technical constraints (e.g., maximum flight time, minimum detection range) into the platform,
>
> **So that** the Agent can automatically analyze, re-architect, and optimize the software pipeline (models, framerates, schedules) to deliver the highest possible energy efficiency while guaranteeing mission success metrics.

---

### Acceptance Criteria (A/C)

These criteria are binary (pass/fail) tests that validate the Agent's core claim of autonomous, energy-aware DSE.

| ID | Criterion Description | Key Metrics (Must be Reported) | Engineering Focus |
| :--- | :--- | :--- | :--- |
| **A/C 2.2.1** | **Constraint Enforcement** | The Agent must deliver a final design where **zero** critical mission constraints (e.g., $250 \text{ Hz}$ control loop rate, $10 \text{ meter}$ detection range) are violated in the Co-Simulation Environment. | **Tool Use:** Successful integration of the Agent with the simulation metrics to validate hard constraints. |
| **A/C 2.2.2** | **Energy Maximization** | The Agent's final optimized design must achieve at least a **$20\%$ improvement** in the $\text{Energy Efficiency Ratio}$ ($\text{Performance} / \text{Watt}$) compared to the manually profiled baseline software stack. | **Optimization:** Demonstrating the Agent's ability to learn and exploit the non-linear trade-offs between accuracy, latency, and power. |
| **A/C 2.2.3** | **Software Architecture Autonomy** | The Agent must demonstrate the ability to autonomously explore **at least three distinct architectural changes** (e.g., model swap, scheduling change, or precision reduction) within a single optimization run. | **Reasoning:** Proving the Agent isn't just parameter-tuning but is making high-level architectural decisions based on power-aware reasoning. |
| **A/C 2.2.4** | **Failure Interpretation** | Given a simulation run that results in a drone crash (e.g., failure to detect an obstacle), the Agent must generate a **textual explanation** identifying the root cause (e.g., "Quantization of the perception model was too aggressive, reducing the $\text{IoU}$ below the safe threshold") and propose a correction. | **Observability:** Validating the Agent's ability to read and interpret complex simulation and compute logs to guide future design choices. |

---

### Milestone Success Definition

Milestone 2.2 is successful when the **Orchestrator Agent** can take a natural language request, execute a DSE campaign using the simulator tools, and output an optimized software configuration that meets all hard constraints while demonstrating significant energy efficiency improvements ($>20\%$) over the original, unoptimized stack.


