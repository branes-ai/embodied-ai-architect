# Agentic AI Efforts in SoC design

The field of **Agentic RTL and ASIC design** is moving from simple assistive "Copilots" to autonomous "Agentic Workflows" where multiple AI agents collaborate to design, verify, and debug hardware.

As of late 2025, the landscape is divided between major EDA (Electronic Design Automation) giants, specialized startups, and academic research labs.

### 1. Major EDA Vendors

The "Big Three" have pivoted from single-model AI to agentic orchestration within their stacks.

* **Synopsys:** Has integrated **Synopsys.ai Copilot** and **DSO.ai** into a multi-agent framework. Their latest focus (as of late 2025) is **Agentic EDA**, where agents handle multi-step reasoning for RTL-to-GDSII flows and autonomously optimize PPA (Power, Performance, Area).
* **Cadence:** Utilizes **JedAI** (Joint Enterprise Data and AI) as an agentic backbone that allows cross-tool collaboration between synthesis, verification, and implementation tools.
* **Siemens EDA (Mentor):** Focuses on agentic verification through their **Questa** platform, using AI agents to autonomously close coverage gaps and root-cause simulation failures.

### 2. Specialized Startups & New Entrants

Several specialized players have emerged to challenge traditional workflows with "AI-first" agent architectures.

* **ChipAgents:** A pioneer in the space, they developed a dedicated environment for multi-agent hardware design. They recently achieved a 97.4% pass rate on NVIDIA’s VerilogEval benchmark, outperforming many general-purpose models.
* **MosChip:** Recently released **Agentic RTL Coder**, which uses a multi-agent loop (Llama-3 and DeepSeek-Coder) for RTL generation, linting (via Verilator), and automated testbench iteration.
* **RapidSilicon:** Focuses on AI-driven FPGA and ASIC design tools, utilizing agents to handle complex timing closure and routing.

### 3. Corporate Research Labs

* **NVIDIA Research:** Highly active in this space with **ChipNeMo** and **VerilogCoder**. Their 2025 work focuses on GPU-accelerated RTL simulation and agentic reflection to fix syntax errors.
* **Intel & AMD:** Both are partnering heavily with Synopsys to deploy internal agentic agents that learn from proprietary design histories to suggest "organization-specific" RTL architectures.

### 4. Open-Source & Academic Projects

The research community has provided the "Blueprints" for these agentic tools:

* **Spec2RTL-Agent:** A research framework that uses iterative reflection and progressive coding to turn natural language specifications into working Verilog.
* **MAGE (Multi-Agent Generation & Evaluation):** Developed at UC San Diego, this is a benchmarking and generation framework for hardware-specific agents.
* **OpenROAD / ORFS-Agent:** A project aiming to provide an agentic interface for the OpenROAD open-source RTL-to-GDSII flow.

---

### Comparison of Agent Roles

In a typical 2025 agentic workflow, you will find these specialized roles:

| Agent Role | Primary Function | Tools They "Use" |
| --- | --- | --- |
| **Spec Enhancer** | Refines messy text into strict technical specs. | LLMs (GPT-4.1, Claude 4) |
| **RTL Generator** | Writes the Verilog/VHDL logic. | DeepSeek-Coder, CodeV |
| **Lint Agent** | Checks for syntax and CDC issues. | Verilator, SpyGlass |
| **Verification Agent** | Generates UVM/SystemVerilog testbenches. | VCS, Xcelium, Questa |
| **Trackback Agent** | Diagnoses failures and tells the Generator how to fix them. | Internal Reasoning Loops |

## Open-Source agentic tools

The field of agentic RTL (Register Transfer Level) design has matured significantly in 2025. We have moved from simple "code completion" to "multi-agent orchestration" where different AI agents act as the Architect, the RTL Coder, and the Verification Engineer.

Below are the key open-source repositories and research papers driving this shift.

### 1. Key GitHub Repositories (Active 2024–2025)

These projects provide the actual "engine" or framework to run agentic hardware design workflows.

* **[MAGE (Multi-Agent Engine)](https://github.com/stable-lab/MAGE-A-Multi-Agent-Engine-for-Automated-RTL-Code-Generation):**
* **What it is:** A specialized multi-agent framework specifically for RTL generation.
* **Core Feature:** It uses a "Plan-Execute-Verify" loop. It can interface with benchmarks like *VerilogEval* to autonomously iterate until the code passes functional tests.


* **[Spec2RTL (by Cirkitly)](https://www.google.com/search?q=https://github.com/Cirkitly/spec2rtl):**
* **What it is:** An autonomous AI assistant that transforms Markdown-based hardware specs into verified Verilog and SystemVerilog testbenches.
* **Core Feature:** It features an "AI Self-Critique" agent that acts as a peer reviewer, checking for logic flaws before running actual simulations.


* **[VerilogCoder](https://www.google.com/search?q=https://github.com/nv-tlabs/VerilogCoder) (NVIDIA Research):**
* **What it is:** A system of multiple agents that autonomously write Verilog and use collaborative tools (syntax checkers, simulators) to fix errors.
* **Impact:** It has achieved over 94% functional correctness on the *VerilogEval-Human* benchmark.


* **[AutoAgents](https://github.com/Link-AGI/AutoAgents):**
* **What it is:** While general-purpose, this framework is being used in the hardware community to "spawn" specialized agents (e.g., a "Constraint Agent" or a "Timing Agent") dynamically based on the design task.



---

### 2. High-Impact Research Papers (2025)

If you are looking for the technical "how-to" behind these agents, these three papers are the current gold standard:

| Paper Title | Venue / Date | Key Innovation |
| --- | --- | --- |
| **"Spec2RTL-Agent: Automated Hardware Code Generation from Complex Specifications"** | *ICLAD 2025* | First fully automated system that handles *unstructured* specs (PDFs, tables) by translating them into intermediate C++/HLS before generating RTL. |
| **"Revisiting VerilogEval: A Year of Improvements in LLMs for Hardware"** | *ACM TAES 2025* | Comprehensive look at how models like **DeepSeek-R1** and **GPT-5** (released 2025) perform in agentic loops vs. traditional zero-shot prompting. |
| **"VToT: Automatic Verilog Generation via Tree of Thoughts"** | *IEEE 2025* | Introduces a "Tree of Thoughts" reasoning path for agents, allowing them to explore multiple architectural branches for a single module. |

---

### 3. Benchmarks Used to Test Agents

To see who is currently "winning" in the agentic space, keep an eye on these leaderboards:

* **VerilogEval 2.0:** The industry standard for Verilog generation.
* **RTLLM 2.0:** Focuses on larger, more complex "real-world" design tasks (e.g., UART, SPI, and RISC-V components).
* **XYZ Benchmark (New for 2025):** The first benchmark specifically designed for **Dockerized agents** that have permission to use actual EDA tools (like Vivado or Verilator) during their reasoning process.

### Next Step for You

Would you like me to **break down a specific agentic workflow** (e.g., how a "Verification Agent" talks to a "Designer Agent" to fix a bug) or help you **set up one of the open-source repos** like MAGE?