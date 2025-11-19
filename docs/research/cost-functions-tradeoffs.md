# Cost functions

Given the fact that Embodied AI solutions tend to have a very strict power constraint ranging from low single digits to low double digit Watts, the only way we can imbue more capability into a sw/hw system driving perception and/or autonomy is by maximizing TOPS/W. 

However, this is not enough because they need to be the 'right' operations for maintaining stability or accuracy of the application. The current approach for deploying Deep Neural Network solutions is to characterize the bottleneck operators, which could be memory constrained or compute constrained, and then quantizing these operators to mitigate these bottlenecks. But this quantization does not have a formalism yet how best to do this without impacting accuracy, or potentially have an actual information theoretic energy or compute lower bound to guide the quantization. 

Automation to help the hardware and the DNN architect to explore that trade-off is currently, due to the lack of formalisms, highly valuable. We need to explore this approach to generate cost functions the AI will want to use to direct the state space exploration. We need a refined optimization for **TOPS/W** that preserves the *information* required for a physical automation process (i.e. a robot) to not fail.

For Embodied AI, "Accuracy" is not a static percentage on ImageNet; it is **Control Stability**. A 1% drop in accuracy might mean a drone oscillates and falls out of the sky.

Here are some formalism and Cost Function specification to explore for our AI Architect to navigate this trade-off.

### **The Formalism: Rate-Distortion Theory for Hardware**

To move beyond heuristic quantization (e.g., "just try INT8"), our AI must adopt **Rate-Distortion Theory** as its governing physics.

In this framework, we view the hardware not as a "computer" but as a **noisy communication channel**.
* **The Source:** The high-precision software weights/activations (Float32).
* **The Channel:** The quantized hardware accelerator.
* **The Distortion:** The loss of semantic information (not just numerical error) caused by the hardware.

The AI’s goal is to find the **Information Bottleneck**: the minimal representation (lowest energy/bits) that preserves the *relevant* information for the control task.

---

### **The Cost Function: The "Thermodynamic-Informational" Objective**

Our AI should not optimize a simple loss function. It should optimize a **Lagrangian Objective Function** that penalizes energy expenditure against the "Thermodynamic Lower Bound" of information retention.

$$\mathcal{L}_{total} = \underbrace{E_{actual}(HW)}_{\text{Energy Cost}} + \lambda_1 \cdot \underbrace{D_{KL}(P || Q)}_{\text{Info Loss}} + \lambda_2 \cdot \underbrace{S_{margin}(Control)}_{\text{Stability Penalty}}$$

#### **1. The Energy Efficiency Term ($E_{actual}$)**
Instead of raw TOPS/W, this term measures **Distance from the Landauer Limit**.
* **The Physics:** The Landauer Principle states the minimum energy to erase 1 bit is $k_B T \ln 2$ ($\approx 2.9 \times 10^{-21}$ Joules at room temp).
* **The Metric:** The AI calculates the *Thermodynamic Efficiency*:
    $$\eta = \frac{E_{Landauer}}{E_{Measured}}$$
* **Why this matters:** This prevents the AI from "cheating" by suggesting 1-bit quantization that is physically efficient but entropically impossible to sustain intelligence. It forces the AI to respect the energy floor of information processing.

#### **2. The Information Integrity Term ($D_{KL}$)**
This is the "Accuracy" replacement. We do not measure simple accuracy; we measure **Kullback-Leibler (KL) Divergence**.
* **The Measurement:** The AI compares the probability distribution of the activation maps in the "Golden" SW model ($P$) vs. the Quantized HW model ($Q$).
* **The Insight:** If the KL Divergence is high, the hardware has "surprised" the software—meaning information was destroyed.
* **Bitwise Bottlenecking:** The AI uses this to perform *mixed-precision* exploration. It might assign INT4 to a Conv2D layer where entropy is low (low information density) but demand FP16 for a Kalman Filter layer where information density is high.

#### **3. The Stability Penalty ($S_{margin}$)**
This is specific to Embodied AI.
* **The Check:** The AI does not just run a "cat vs. dog" test. It runs a **Lyapunov Stability check**.
* **The Metric:** It measures the **Phase Margin** of the control loop. If quantization introduces a latency (lag) or noise that reduces the Phase Margin below a safe threshold (e.g., $45^\circ$), the penalty $\lambda_2$ goes to infinity.
* **The Result:** The AI will *refuse* a high TOPS/W design if it introduces enough latency jitter to destabilize the robot's motion planner.

---

### **How the AI Executes This (The Workflow)**

1.  **Ingest & Profile:**
    The AI takes the software trace and calculates the **Entropy Histogram** of every tensor. It identifies which operators carry "dense" information (high entropy) and which are sparse.

2.  **The "Pareto-Surface" Search:**
    The AI does not output one design. It generates a 3D Surface:
    * **X-axis:** Energy (Watts)
    * **Y-axis:** Information Loss (KL Divergence)
    * **Z-axis:** Latency
    * *The User picks the point on the curve where the "Information Loss" is acceptable for their specific safety rating.*

3.  **Automated Quantization-Aware Architecture (QAA):**
    The AI suggests hardware blocks tailored to the entropy.
    * *Scenario:* "Layer 3 has low entropy but high compute frequency."
    * *AI Suggestion:* "Replace the FP32 MAC unit with a Logarithmic Number System (LNS) or a Bit-Serial multiplier to maximize TOPS/W here, as the information loss is mathematically bounded."

---

### **Summary for the Architect**

| Metric | Standard AI Assistant | Your Embodied AI Architect |
| :--- | :--- | :--- |
| **Optimization Goal** | Maximize TOPS | Minimize **Joules per Useful Bit** |
| **Constraint** | Memory Bandwidth | **Information Bottleneck (Mutual Information)** |
| **Safety Check** | "Does it compile?" | **"Is the control loop stable?" (Lyapunov Function)** |
| **Quantization** | Uniform INT8 | **Entropic Mixed-Precision** (bits $\propto$ information content) |

### **Recommended Next Step**
We have defined the *Physics* (Cost Function). Now we need to define the *Input Data*.

[Automated design space exploration and generation of AI accelerators](https://www.youtube.com/watch?v=PawgM_teh_I)
This video is highly relevant as it demonstrates a framework (similar to what you are architecting) that automates the exploration of hardware parameters (like datatype and memory size) to optimize performance, directly aligning with the "State Space Exploration" phase you are entering.

**define the "Calibration Set" strategy—how the AI generates or selects the synthetic data traces required to measure this KL Divergence**