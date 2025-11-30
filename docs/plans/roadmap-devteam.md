# Devteam Size and Make-up

Reflecting on the transformative impact of AI assistants on engineering productivity, when leveraging **AI-powered productivity (10x)**, the focus shifts from hiring large teams for manual, repetitive tasks (like profiling and initial DSE) to hiring **senior, specialized experts** who can *guide, validate, and build the tools* the AI uses.

The goal is to hire **Leverage Engineers**—those who can enable 10x productivity by designing the right prompts, tools, and validation layers for the AI.

Here is the revised, AI-centric **Personnel Hiring Plan** for Phase 1 (Foundation) and Phase 2 (Software Agent MVP), focusing on high leverage and reduced headcount.

***

## Phase 1 & 2: AI-Enhanced Hiring Plan

We will hire a small, highly effective core team, leveraging AI Assistants (e.g., code generation, tool/API lookup, initial simulation setup) to dramatically reduce the need for junior or mid-level engineers.

| Role | Count | Specialization & Leverage | Estimated Start | Core Deliverables (Tied to A/C) |
| :--- | :---: | :--- | :--- | :--- |
| **1. Lead Systems Architect & Tech Lead** | 1 | **Performance Engineering Maturity.** Must bridge the gap between AI, HW, and SW. Responsible for defining the overall Co-Simulation architecture and validating the $\text{Sim-to-Real}$ strategy. | Month 1 | A/C 2.2.1: **Constraint Design**. Defining the rules the Agent uses to enforce constraints. |
| **2. Agentic AI / Reinforcement Learning Engineer** | 1 | **AI Tooling & Orchestration.** Responsible for building the **Orchestrator Agent** (the "brain"). Focus on the tool-use layer (calling simulators) and designing the **Reward Function** for energy efficiency (maximizing $\text{Perf}/\text{Watt}$). | Month 1 | A/C 2.2.2: **Energy Maximization**. Training the Agent to achieve the $>20\%$ efficiency goal. |
| **3. Simulation & Tooling Engineer** | 1 | **HW/SW Simulator Expert.** Focuses on the **Co-Simulation Pipeline** (integrating AirSim $\to$ gem5 $\to$ McPAT) and building the necessary wrappers/APIs for the Agent to access them as "tools." | Month 1 | A/C 2.2.4: **Failure Interpretation**. Structuring simulation logs into actionable feedback for the Agent. |
| **4. Embedded/FPGA Specialist** | 1 | **Software Optimization & Real-World Validation.** Seniority allows them to focus on advanced topics (e.g., writing highly optimized kernels for the Agent to choose from, defining low-level hardware constraints). **AI Assistant Leverage:** Use AI for routine driver coding, initial profiling, and boilerplate. | Month 3 | A/C 2.2.3: **Architecture Autonomy**. Defining the set of optimized software components (the search space) the Agent can swap. |
| **Total Headcount** | **4** | | | |

---

## The AI-Leverage Assessment

The headcount is intentionally lean, recognizing the 10x productivity boost in three key areas:

### 1. Code Generation and Debugging
* **Reduced Need:** Junior/Mid-level Software Engineers.
* **AI Function:** AI assistants write the majority of API wrappers, configuration file parsers (YAML/JSON), and simulator glue code. They also handle the initial debugging of complex integration errors between tools (e.g., fixing `gem5` configuration syntax).
* **Result:** The **Simulation & Tooling Engineer** (Role 3) focuses solely on architectural correctness and high-level integration, rather than line-by-line coding.

### 2. Design Space Exploration (DSE) Automation
* **Reduced Need:** Performance Engineers focused on manual profiling and tuning.
* **AI Function:** The **Orchestrator Agent** *is* the productivity tool. It runs thousands of simulation permutations autonomously, performs the initial analysis, and guides the design process.
* **Result:** The **Agentic AI Engineer** (Role 2) focuses on building the *system* that does the DSE, not executing the DSE itself, leading to a productivity gain far exceeding $10\text{x}$ on the DSE task.

### 3. Documentation and Knowledge Transfer
* **Reduced Need:** Technical Writers and Analysts for routine reports.
* **AI Function:** AI assistants can instantly generate reports, documentation, and executive summaries based on the raw data from the simulation pipeline (A/C 3.2.4).
* **Result:** The **Lead Systems Architect** (Role 1) spends less time on reporting and more time on strategic validation and architectural oversight.

This plan maintains the high performance engineering sensibility required for embodied AI solutions while keeping the budget optimized.

---

# Phase 3: Personnel Plan for Hardware Co-Design

Phase 3 introduces the highest technical complexity—the Agent must design *hardware* that can be physically built. This requires adding specialists who understand the constraints and synthesis processes of custom silicon (FPGAs/ASICs) and can bridge the final **Sim-to-Real gap** (Milestone 3.2).

Since the Phase 1 & 2 team established the Agent's brain and the simulation tools, the Phase 3 team can be small, focusing purely on deep technical validation and deployment. They will heavily leverage AI assistants for tasks like writing Hardware Description Language (HDL) boilerplate and validating synthesis results.

| Role | Count | Specialization & Leverage | Estimated Start | Core Deliverables (Tied to A/C) |
| :--- | :---: | :--- | :--- | :--- |
| **5. Custom Hardware Synthesis Engineer** | 1 | **RTL/FPGA Design & HLS (High-Level Synthesis) Expert.** Responsible for taking the Agent's optimized virtual hardware blueprint and translating it into deployable RTL (Verilog/VHDL) for the drone's target platform (likely an FPGA or SoC). **AI Assistant Leverage:** Use AI to generate initial HLS code, check timing constraints, and optimize synthesis directives. | Month 7 | A/C 3.1.1: **Architectural Generation**. Ensuring the Agent-designed parameters map correctly to synthesizable hardware. |
| **6. Verification & Validation (V&V) Engineer** | 1 | **Measurement and Calibration Expert.** Focuses entirely on bridging the **Sim-to-Real Gap** (A/C 3.2.2). Designs the physical testbench, selects power measurement tools, and develops the calibration methodology to ensure the simulation model is accurate to within $10\%$. | Month 8 | A/C 3.2.2: **Sim-to-Real Accuracy**. Designing the methodology for physical power measurements and flight time validation. |
| **Total Added Headcount** | **2** | | | |

---

## Final Team Structure & Function

By the start of Phase 3 (Month 7), your highly leveraged, six-person core team will be fully operational, covering all necessary technical domains:

| Role (Total: 6) | Core Function in Phase 3 | AI Leverage |
| :--- | :--- | :--- |
| **Lead Systems Architect** | **Guidance & Oversight.** Defines final system requirements and signs off on the **Blueprint Package** (A/C 3.2.4). | Uses AI for competitive analysis and generating executive summaries. |
| **Agentic AI / RL Engineer** | **Agent Refinement.** Fine-tunes the reward function to balance $W/\text{flight time}$ and handles the larger search space of HW/SW co-design. | Uses AI for generating complex reward function structures and debugging learning curves. |
| **Simulation & Tooling Engineer** | **Pipeline Scalability.** Ensures the coupled simulator can handle the increased complexity of the custom hardware models and run the massive DSE campaigns efficiently. | Uses AI for writing and debugging complex system scripts for parallel cloud execution. |
| **Embedded/FPGA Specialist** | **Deployment & Testing.** Handles the physical flashing, boot process, and real-time software stack on the drone. | Uses AI for generating device drivers and interfacing with low-level kernel code. |
| **Custom Hardware Synthesis Engineer** | **Silicon Translation.** Takes Agent's output $\to$ synthesizable code. This is the hand-off point to manufacturing. | Uses AI to quickly translate high-level synthesis (HLS) C++ code into optimized RTL (Verilog). |
| **V&V Engineer** | **Accuracy Assurance.** Designs the **physical validation tests** to prove the $50\%$ flight time increase (A/C 3.2.3) and verifies the $10\%$ Sim-to-Real accuracy (A/C 3.2.2). | Uses AI for drafting test plans, documenting measurement procedures, and analyzing large datasets from physical logs. |

This lean team structure maximizes the impact of each hire, reserving the bulk of the repetitive labor for the Agentic AI and standard AI assistants. 
