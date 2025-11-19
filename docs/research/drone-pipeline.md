# Industrial Embodied AI Drone Architecture

We need to build a **semantic-level world model** (not just a map) capable of onboard reasoning, running on hardware that balances high-compute density with strict SWaP (Size, Weight, and Power) constraints.

Here is a proposed architecture that specifically targets high-efficiency compute (>2 TOPS/W), lightweight industrial sensors, and the specific software stack that turns "sensed data" into a "reasoned digital twin."

### 1. The Compute Layer: High-Efficiency Inference
The NVIDIA Jetson Orin Nano, with an energy profile of 0.2TOPS/Watt is not energy-efficient enough for the perception task. For a drone effectively running a digital twin and reasoning engine, we will need a dedicated AI accelerator with a 10x higher performance-per-watt profile.

**Primary Recommendation: Qualcomm Flight RB5 5G Platform**
* **Why:** This is built specifically for this use case. It integrates the Flight Controller and Mission Computer functions but allows for the "split brain" architecture you want via a companion setup.
* **Specs:** ~15 TOPS at very low power (dedicated Hexagon DSPs). It is vastly more efficient than the Jetson architecture for this specific envelope (approx 10 TOPS/W efficiency on the DSP).
* **Form Factor:** It eliminates the bulk of carrier boards. It is designed to be the *core* of the drone, not a strap-on computer.

**Alternative (NVIDIA Path): Jetson Orin NX (Not Nano)**
* **Why:** If you must use CUDA for specific libraries, the Orin **NX** (100 TOPS) is the entry point.
* **Correction on Efficiency:** To beat the efficiency bottleneck, you must not rely solely on the GPU. You need to offload the vision pipeline to the **Deep Learning Accelerator (DLA)** cores on the Orin, or use an external accelerator like the **Hailo-8**.
* **Hailo-8 Integration:** If you stick with a lighter host (like a Raspberry Pi 5 or CM4), adding a **Hailo-8 M.2 module** gives you 26 TOPS at ~2.5W (~10 TOPS/W). This is the efficiency king for edge inference right now.

### 2. The Sensor Layer: De-coupling Sensing from Compute
The RealSense stereo camera is too large of a "brick" with a USB bottleneck. For a drone, we need sensors that communicate via **MIPI CSI-2** or **GMSL2** (direct to ISP) to reduce latency and weight, or smart image sensors that process depth on-chip.

* **Option A (Smart Sensor): Luxonis OAK-D Pro W (Wide)**
    * **Why:** It weighs ~30g (stripped). Crucially, it performs the stereo depth matching and object detection (YOLO) *on the camera itself* (using its own Myriad X / RVC3 chip).
    * **Benefit:** Your main flight computer receives only the metadata (Object Class: "Person", XYZ coordinates: [2.1, 0.5, 1.0]) and the depth map, saving your main CPU for the "reasoning" and digital twin construction.
* **Option B (Raw Industrial): Stereolabs ZED X Mini**
    * **Why:** Uses **GMSL2** (Gigabit Multimedia Serial Link). This is a rugged, industrial protocol that allows high-bandwidth data without the USB CPU overhead or connector instability. It is designed for high-vibration drone environments.

### 3. The "Digital Twin" Software Stack
For situational awareness, we need a capable Digital Twin of the perceived world. Kimera builds a mesh (a "sensed representation"). To get a "digital twin" we can *reason* about, we need a **Scene Graph**.

The perception pipeline needs to create a **Hierarchical 3D Scene Graph (H-3DSG)**.

* **The Project:** **Hydra** (by MIT-SPARK Lab)
    * **What it does:** It takes the mesh (from something like Kimera) and abstracts it into a graph.
        * *Layer 1 (Metric):* The mesh/occupancy grid (Obstacle avoidance).
        * *Layer 2 (Semantic):* Objects (Chairs, Tables, Trees).
        * *Layer 3 (Topological):* Places (Room A, Corridor B).
    * **Why this fits the requirements:** We can query Hydra in natural language: *"Is there a chair in Room A?"* or *"Plan a path to the nearest exit."* This is **reasoning**, not just collision checking.

* **The "Cutting Edge" Alternative:** **ConceptGraphs**
    * **What it does:** It builds an **Open-Vocabulary 3D Scene Graph**.
    * **How:** It projects features from VLM (Vision-Language Models) like CLIP or LLaVA onto the 3D map.
    * **Result:** You can reason about objects the drone has never been explicitly trained on. You can give the drone a text command: *"Find the red backpack that looks damaged,"* and it can query the digital twin for those semantic features.

### The SoTA Pipeline

1.  **Sensing:** **Luxonis OAK-D Pro** runs `YoloV8` onboard and outputs `Depth` + `Semantic Bounding Boxes` via SPI/Ethernet to the mission computer.
2.  **State Estimation:** **OpenVINS** or **VINS-Fusion** (running on the Mission Computer) fuses IMU and Visual Odometry to give perfect localization.
3.  **Twin Construction:** **Hydra** runs on the Mission Computer (e.g., Orin NX or RB5). It consumes the OAK-D data to build a **Dynamic Scene Graph**.
4.  **Reasoning:** A high-level planner (e.g., `BehaviorTree.CPP` in ROS 2) queries the **Hydra Scene Graph**.
    * *Example Logic:* `IF (Battery < 20%) AND (Distance(Home) > 500m) AND (SceneGraph.contains("LandingSpot")) -> THEN Land.`
5.  **Control:** The planner sends setpoints via **MAVROS/MAVSDK** to the Flight Controller (Pixhawk/PX4) which executes the motor mixing.

This architecture provides the >10 TOPS/W efficiency required (via RB5 or Hailo), solves the form-factor issues (MIPI/Smart sensors), and delivers the actual reasoning capability (Scene Graphs) needed.

[Structured Interfaces for Automated Reasoning with 3D Scene Graphs](https://www.youtube.com/watch?v=zY_YI9giZSA)
This video demonstrates the "reasoning" capability you are looking for, showing a robot using a 3D Scene Graph (similar to Hydra) to understand a query, correct a mislabeled object in its digital twin, and plan a task based on that semantic update.
