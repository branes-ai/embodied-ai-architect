# Vision Language Action models

This is a structured research guide designed for a graduate student. It moves from theoretical foundations to a concrete implementation roadmap for building a VLA specifically for aerial robotics.

---

# Research Directive: Vision-Language-Action (VLA) Architectures for Autonomous Aerial Systems

**To:** Graduate Student Researcher
**From:** Principal Investigator / Advisor
**Subject:** Roadmap for Developing a Drone-Based VLA for Autonomy

## 1. Executive Summary & Objective

We are moving beyond standard "perception-planning-control" stacks. Your task is to develop a **Vision-Language-Action (VLA)** model for a drone. Unlike ground robots, drones require 6-DoF (Degree of Freedom) spatial awareness and operate under strict size, weight, and power (SWaP) constraints.

**Your Goal:** Build/Adapt a VLA that takes raw camera feeds and high-level text commands (e.g., *"Inspect the red structural beam on the second floor"*) and outputs direct flight control commands (velocities/waypoints).

---

## 2. Theoretical Foundation: How VLAs Work

A VLA is not just a Visual Question Answering (VQA) model; it is a VQA model where the "answer" is a physical movement.

### 2.1 The Core Architecture

Most State-of-the-Art (SOTA) VLAs (e.g., RT-2, OpenVLA) follow this pipeline:

1. **Vision Encoder (The Eyes):** Processes the incoming video stream.
* *Standard:* ViT (Vision Transformer) or CLIP.
* *SOTA:* **SigLIP** (Sigmoid Loss for Language Image Pre-training) or **DINOv2**. These are preferred because they capture semantic meaning (what an object is) *and* geometric structure (where it is), which is critical for flying.


2. **LLM Backbone (The Brain):** A pre-trained Large Language Model (e.g., Llama 3, Vicuna, or Phi-3). It acts as the reasoning engine.
3. **Action Head (The Body):** This is where the magic happens. The model projects the LLM's latent features into an **Action Space**.

### 2.2 Designing the Latent Embedding Space

The "Latent Space" is the mathematical representation where vision, text, and action meet.

* **The Problem:** Text is discrete ("Apple"), but drone flight is continuous ().
* **The Solution (Action Tokenization):** We discrete the continuous action space into "tokens."
* Example: A velocity of  to  m/s is divided into 256 bins.
* Bin 0 = Token `[ACTION_0]`, Bin 255 = Token `[ACTION_255]`.
* The LLM predicts `[ACTION_124]` just like it predicts the word "the."



### 2.3 How to Evaluate Embedding Quality

You cannot wait to fly the drone to know if your model is learning. Use these "probing" techniques:

* **Linear Probing:** Freeze the model. Train a single linear layer on top of the latent vectors to predict the drone's *height*. If it fails, your latent space has lost spatial awareness.
* **t-SNE Visualization:** Project your embeddings into 2D. You should see clusters formed not by *object type* (all cars together) but by *action type* (all "inspecting" maneuvers together, regardless of the object).

---

## 3. Implementation Roadmap

### Phase A: Data Collection & Simulation (Weeks 1-4)

You cannot train on a physical drone initially (you will crash too many times).

1. **Simulator:** Use **AirSim** or **Gazebo**.
2. **The "Cross-Embodiment" Trick:** Drones lack massive training datasets. You must use the **Open X-Embodiment** dataset (ground robot data) to teach the model general physics/logic, then fine-tune on drone data.
3. **Data Format:** Collect tuples of .
* *Action_t* should be velocity commands .



### Phase B: Training Strategy (Weeks 5-8)

Do not train from scratch. Use **OpenVLA** or **Prismatic** as your codebase.

1. **VLM Pre-training:** Start with a pre-trained VLM (e.g., LLaVA or Prism). It already knows what a "building" looks like.
2. **Action Fine-Tuning:**
* Freeze the Vision Encoder (mostly).
* Use **LoRA (Low-Rank Adaptation)** on the LLM. This allows you to train a 7B parameter model on a single consumer GPU (e.g., RTX 4090).
* *Loss Function:* Standard Cross-Entropy Loss on the action tokens.





### Phase C: Edge Deployment (The "Drone" Constraint)

A standard VLA (7B params) requires ~14GB VRAM. A drone (NVIDIA Jetson Orin) has ~8-16GB total.

1. **Quantization:** You must quantize the model to **4-bit (AWQ or GPTQ)**. This reduces memory usage by 70% with minimal accuracy loss.
2. **The Dual-System Setup (Critical for Drones):**
* **Slow Loop (1Hz):** The VLA runs once per second to decide *intent* ("Fly to the red car").
* **Fast Loop (100Hz):** A traditional PID or geometric controller handles stability and obstacle avoidance while following the VLA's waypoint. **Do not let the VLA control motor RPMs directly.**



---

## 4. Key Papers to Read (In Order)

1. **"RT-2: Vision-Language-Action Models with Web-Scale Knowledge" (Google DeepMind)**
* *Why:* The foundational paper. Explains how to treat actions as language tokens.


2. **"OpenVLA: An Open-Source Vision-Language-Action Model"**
* *Why:* Your likely starting codebase. Read their sections on "Action Quantization" carefully.


3. **"Do As I Can, Not As I Say" (SayCan)**
* *Why:* Teaches you how to ground language in "affordances" (what the robot is actually capable of doing).


4. **"LoRA: Low-Rank Adaptation of Large Language Models"**
* *Why:* Essential for training these huge models on university hardware.



## 5. Next Step for You

1. **Clone the [OpenVLA repository](https://github.com/openvla/openvla).**
2. **Download the "Bridge" dataset** (a subset of Open X-Embodiment) to inspect how they format the `(image, text, action)` tuples.
3. **Set up a simple AirSim environment** where a drone just needs to "find the red ball." Try to collect 50 manual flight trajectories.
