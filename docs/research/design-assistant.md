# AI Assistant for SoC Design

Designing an AI assistant for chip design (Electronic Design Automation or EDA) requires a fundamentally different mindset than designing a standard coding assistant like Copilot. In software, a bug can be patched over the air. In hardware, a bug can cost \$10 million in mask sets and delay a product by six months.

Here are the governing principles that must form the "constitution" of our AI architect.

### **The Core Philosophy: "Trust but Verify"**
The overarching theme of your architecture must be **verifiability**. The AI should never be the final authority; it is a generator of *candidates* that must be validated by rigorous, deterministic physics and logic engines.

---

### **I. The 5 Governing Principles**

#### **1. The Principle of Formal Correctness (The "Zero-Hallucination" Mandate)**
In chip design, "mostly correct" is effectively "broken." An AI that hallucinates a non-existent pin on a logic gate or invents a timing constraint will destroy a chip.
* **The Rule:** The AI must operate within a **Correct-by-Construction** framework. It should not generate raw text/code directly for critical paths without passing it through a semantic validator.
* **Architectural Implication:** You must architect a **"Sandwich Model"**.
    * *Layer 1 (AI):* Generates the RTL (Register Transfer Level) code.
    * *Layer 2 (Symbolic check):* A rigid, non-AI parser immediately checks syntax, linting, and formal equivalence.
    * *Layer 3 (AI):* If the check fails, the error log is fed back to the AI to self-correct before the human ever sees it.

#### **2. The Principle of IP Sovereignty (The "Air-Gap" Mental Model)**
Chip designs (ISAs, Data path circuits, Netlists, Floorplans, GDSII layouts) are among the most valuable trade secrets in the world. Users will not use our assistant if there is a none 0 percent chance their proprietary architecture leaks into a public model training set.
* **The Rule:** Data egress is forbidden. The AI must "forget" the specific circuit details after the session or be a frozen model instance running locally/in a private cloud (VPC).
* **Architectural Implication:** Use **RAG (Retrieval-Augmented Generation)** with strict access controls. The AI model itself should be generic (trained on open-source RISC-V cores, etc.), but it "learns" the proprietary project specifics only temporarily via a vector database that the user controls.

#### **3. The Principle of Physical Reality (Physics-Aware Inference)**
Software has infinite canvas; chips have physical boundaries (Die Area, Thermal limits, Speed of light). An AI that optimizes logic but creates a hotspot that causes continuous thermal throttling does not create a valuable solution.
* **The Rule:** Optimization suggestions must be **PPA-Aware** (Power, Performance, Area). The AI cannot suggest a change without estimating its cost in these three currencies.
* **Architectural Implication:** The AI needs "tools usage" capabilities. It should be able to query an external estimator (like a synthesis tool) to ask, *"If we change this adder to a multiplier, what happens to the area?"* before suggesting it to the user.

#### **4. The Principle of Determinism**
Hardware engineering relies on regression testing. The same script run today and tomorrow, must generate the exact same GDSII layout. Stochastic (random) AI behavior is the enemy of regression.
* **The Rule:** The AI’s "creative" mode must be separable from its "implementation" mode. For implementation tasks (e.g., "Write a script to place these macros"), the output must be deterministic.
* **Architectural Implication:** You generally need to set the `temperature` (randomness parameter) to 0 for code generation tasks, or architect the AI to generate a *script* (Tcl/Python) that produces the result, rather than producing the result directly.

#### **5. The Principle of Explainability (White-Box Engineering)**
An engineer cannot sign off on a design they don't understand. If the AI optimizes a circuit block to be 20% faster, the engineer needs to know *why* and *how*, to ensure it didn't sacrifice accuracy (e.g. quantization might destabilize the inference), or reliability (e.g., removing safety redundancies).
* **The Rule:** Every generative action requires a **Chain of Thought** citation.
* **Architectural Implication:** The interface must support "Click-to-Source." If the AI suggests a specific Verilog module, it must cite the specific requirement document or the specific optimization pattern it referenced (e.g., *"I used a carry-lookahead adder here because you specified a timing constraint of < 500ps"*).

---

### **II. Architectural Blueprint: The "Agentic" Approach**
To satisfy these principles, you cannot simply use a Chatbot. You need an **Agentic System**.



* **The Orchestrator (The Brain):** An LLM that understands natural language requests (e.g., "Optimize this block for low power").
* **The Tool Belt (The Hands):** APIs that connect to industry-standard tools (Synopsys, Cadence, OpenROAD). The AI writes Tcl scripts to drive these tools.
* **The Critic (The Conscience):** A separate, smaller model or rule-set that critiques the output of the Orchestrator against the "Principles" before showing it to the user.
* **The Knowledge Base (The Memory):** A secure, local vector store containing the user's specs, previous designs, and process design kits (PDKs).

### **III. Summary of Responsibilities**

| Feature | Standard AI Assistant | Chip Design AI Assistant |
| :--- | :--- | :--- |
| **Output** | Text / Code Snippets | Architecture spec, ISAs, Verilog/VHDL / Tcl Scripts / Constraints |
| **Verification** | Unit Tests (Software) | Formal Verification / LVS / DRC |
| **Cost Function** | Latency / Compute | Latency, Throughput, Efficiency, PPA (Power, Performance, Area) |
| **Risk Tolerance** | High (Fix it later) | Zero (Respins are fatal) |
| **Context** | 4k - 32k tokens | Gigabytes of specs and netlists |

### **IV. Layers of design support**
There are three core **domains** of chip design. 
  - **Architecture** SW-HW interaction and co-design
  - **Front-End** RTL creation and verification
  - **Back-End** Physical layout and routing

An AI for **Architecture** exploration, creating hw/sw compositions, will look different from an AI for **Logic Design**, writing Verilog, which will look very different from an AI for **Physical Design**, placing components on a floor plan.

## AI Architect

The pivot to **HW/SW Co-Design** moves us from "Construction" (how to build it) to "Architecture" (what to build). This is indeed where the highest leverage lies—a 10% gain in RTL optimization is nice, but a 10x gain comes from moving the right compute kernel from the CPU to a dedicated accelerator.

In this phase, your AI is no longer a "Junior Engineer" checking syntax; it acts as a **Chief Architect**.

Here are the governing principles for an AI Architect focused on Design Space Exploration (DSE) and HW/SW partitioning.

### **The Core Philosophy: "Workload-Driven Plasticity"**
In co-design, hardware is not a fixed container; it is fluid. The software workload dictates the shape of the hardware. The AI must treat the hardware architecture as a variable to be solved for, based entirely on the demands of the target software.

---

### **I. The 5 Governing Principles of Co-Design AI**

#### **1. The Principle of Workload First Mandate**
Custom hardware acceleration cannot be designed in a vacuum. The request to design and optimize an embodied AI system must start with the embodied AI workload, and its characterization.
* **The Rule:** All architectural proposals must be justified by **Dynamic Profiling Data**. The AI must ingest execution traces (e.g., from QEMU, fsim/psim, or PyTorch profilers) to identify "Hotspots"—the 10% of code that runs 90% of the time.
* **Architectural Implication:** The AI needs an **Ingestion Agent** that parses software profiling logs (flame graphs, instruction counts, cache miss rates). It then maps these software bottlenecks to specific hardware candidates (e.g., *"The MatMul operation consumes 60% of cycles; I propose a systolic array accelerator"*).

#### **2. The Principle of Isomorphic Partitioning (The "Fluid Boundary")**
A function, such as, convolution, encryption, or BLAS L3 matmul, can live as a C++ library (Software), a fast DSP instruction (Firmware/ISA extension), or a dedicated block (Hardware accelerator).
* **The Rule:** The AI must view HW and SW as **interchangeable implementation details** of the same functional intent. It should present a menu of options: "Pure Soft," "Soft+Instruction," or "Hard IP," with the cost/benefit of each.
* **Architectural Implication:** The AI needs a **Cost Modeler**. It must instantaneously estimate the "Tax" of moving functionality to hardware.
    * *Software Cost:* Cycles, Energy.
    * *Hardware Cost:* Area (mm²), Design NRE (Non-Recurring Engineering) time, Data Transfer overhead (the "Tax" of moving data to the accelerator).

#### **3. The Principle of "Good Enough" Accuracy (Abstract Simulation)**
In the implementation phase, we will need nanosecond accuracy. In the architecture phase, we need **Directional Correctness**. Fast, high-level of abstraction modelers are most valuable to iterate through a large state space of possibilities.
* **The Rule:** The AI works with **Transaction-Level Models (TLM)**, not Register-Level models. It optimizes for simulation speed (millions of instructions per second) over timing precision.
* **Architectural Implication:** The AI leverages **Virtual Prototyping**. It configures high-level models to test an architecture in seconds, rather than generating Verilog that takes hours to synthesize.

#### **4. The Principle of Bottleneck Visibility (The "Roofline" Mental Model)**
Architectures rarely fail because of raw compute; they fail because of **data movement**. An AI that adds 1,000 cores but leaves the memory bandwidth unchanged is designing a traffic jam.
* **The Rule:** Every architectural proposal must come with a **Roofline Analysis**. The AI must calculate the "Arithmetic Intensity" (Operations per Byte) and verify if the design is Compute-Bound or Memory-Bound.
* **Architectural Implication:** The AI must explicitly model **Interconnects and Caches**. It shouldn't just suggest "Add an NPU"; it must suggest "Add an NPU *and* double the L2 cache bandwidth," identifying the NoC (Network on Chip) congestion that the new unit will create.

#### **5. The Principle of Pareto Frontier Navigation**
There is no "best" chip. There is only the "best trade-off" between Power, Performance, and Area (PPA).
* **The Rule:** The AI never outputs a single solution. It outputs a **Pareto Curve**. It shows the user the "Efficient Frontier"—a set of designs where you cannot improve one metric without sacrificing another.
* **Architectural Implication:** The AI runs **Multi-Objective Optimization**. It explores thousands of configuration permutations (cache sizes, bus widths, clock frequencies) to plot the curve, allowing the human architect to pick the "sweet spot."

---

### **II. The "Virtual Sandbox" Architecture**

To support this, the AI design assistant deploys a **"Digital Twin" Loop**:

1.  **Input (The Spec):** User provides the Target Workload (e.g., "Llama-3 Inference") and Constraints (e.g., "< 5 Watts").
2.  **The Mapper (AI Agent):** Analyzes the workload and maps functions to abstract resources (CPU vs. DSP vs. GPU vs. TPU vs. NPU vs. KPU).
3.  **The Simulator (The Oracle):** A high-speed **functional/performance simulator** runs the workload on the virtual hardware model.
4.  **The Feedback Loop:** The simulator reports specific friction points (e.g., "DRAM bandwidth saturated").
5.  **The Iterator:** The AI tweaks the parameters (e.g., "Switch from LPDDR4 to LPDDR5") and re-runs.

### **III. Summary of Responsibilities: Co-Design Phase**

| Feature | Implementation AI (Backend) | Co-Design AI (Architect) |
| :--- | :--- | :--- |
| **Core Unit** | Logic Gates / Transistors | **Transactions / Data Flows** |
| **Input Data** | Verilog / Netlists | **Software Traces / C++ Code** |
| **Key Risk** | Timing Violation / DRC Error | **Amdahl's Law / Data Starvation** |
| **"Truth"** | Physics (SPICE/RC Extraction) | **SystemC Simulation** |
| **Goal** | "Make it manufacturable" | **"Make it the right product"** |

---

### **IV. Next Step**

We are effectively building a "System-Level Synthesis" assistant. The most critical component to define next is the **"Cost Function"** the AI uses to judge success.

[Accelerating Early Stage Exploration with Virtual Prototyping](https://www.youtube.com/watch?v=UYlxEe5kbe8)
This video details the methodology of "Shift Left" and using Virtual Prototyping for early design space exploration, matching the architectural principles discussed above.



