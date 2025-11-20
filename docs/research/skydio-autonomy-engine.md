# Skydio's hw/sw architecture

## "9 Concurrent DNNs" Claim

Skydio's drones run nine custom deep neural networks as part of their Autonomy Engine, but the term "concurrent" here is misleading. These aren't all running simultaneously in parallel threads. Instead, they're running in a **pipelined architecture** where different stages of perception execute sequentially or in partially overlapping stages.

### What These DNNs Actually Do

From Skydio's technical publications and the GC-Net paper they published, here's what their perception stack appears to include:

1. **Stereo depth estimation networks** - Multiple instances running on stereo pairs from their 6 navigation cameras
2. **Object detection/tracking networks** - For identifying and tracking people, vehicles
3. **Semantic segmentation** - Context-aware understanding of scene elements
4. **Feature extraction networks** - Unary feature generation for stereo matching
5. **Cost volume regularization** - 3D convolutional networks for depth refinement

The GC-Net paper shows their stereo architecture uses a sophisticated pipeline: 2D convolutions for feature extraction, cost volume formation, then 3D convolutions for regularization. They're not running 9 monolithic networks - they have a multi-stage pipeline with different specialized networks at each stage.

## Hardware Evolution and Power Efficiency

**1. Architectural Flexibility**

The Jetson Orin SoC includes GPU, DLAs (Deep Learning Accelerators), and ARM-based general-purpose compute. Skydio's autonomy engine isn't just running DNNs - it's doing:
- Visual odometry and SLAM
- Nonlinear optimization for path planning
- Classical geometric processing
- Real-time 3D reconstruction

Hailo is an **inference-only ASIC** optimized for small DNNs. It cannot do training, and it cannot easily run non-DNN algorithms. For a system that needs to blend classical computer vision with deep learning and optimization, it would need to be coupled with a general purpose compute facility.

**2. The Concurrent Processing Reality**

The basic Skydio pipeline consists of:
- 6 cameras at ~200° FOV each, building 360° depth maps updated at over 1 million points per second
- The pipeline has multiple stages that can overlap: while one stereo pair is in the cost volume stage, another might be in feature extraction
- The DLAs handle DNN inference while the CPU and GPU processes non-DNN tasks

This heterogeneous compute is why they leverage the NVIDIA Jetson platform: **the DLAs offload neural net inference (running at high TOPS/W efficiency similar to Hailo), while the GPU handles the geometric processing, SLAM, and path planning that Hailo cannot do.**

**3. Software Ecosystem**

NVIDIA's ecosystem (TensorRT, cuDNN, CUDA) allows Skydio to:
- Rapidly iterate on network architectures
- Deploy the same models from datacenter training to edge inference
- Use a mature toolchain with extensive optimization support

Hailo requires model conversion through their proprietary compiler and is limited to TensorFlow/PyTorch/ONNX models that fit their architecture constraints.

### The Real Performance Picture

Looking at NVIDIA's own DRIVE AV team perception pipeline, they report that DNNs account for 60%+ of compute, and they rely heavily on DLAs as dedicated inference engines to meet latency KPIs. The same principle applies to Skydio.

How do deliver stringent latency concerns are valid but solved through:
- **Pipelining**: Different network stages overlap
- **Heterogeneous compute**: DLAs for inference, GPU for everything else
- **Quantization**: INT8 inference on DLAs provides much higher throughput
- **Network optimization**: Their networks are designed to be small and fast (note how they integrate geometric priors to reduce network size)

### The Design Trade-off

Skydio chose **flexibility and capability** over pure inference efficiency. For an autonomous drone that must:
- Navigate GPS-denied environments
- Avoid ½-inch wires in real-time  
- Build 3D maps on the fly
- Run path planning and control

...they need the full SoC capabilities, not just an inference accelerator.

For a pure vision-only inference workload, such as a security camera running object detection, Hailo would be far superior in terms of energy and latency. But Skydio's system is much more than just running DNNs - it's a complete autonomous robotics stack, and that's where NVIDIA Jetson's generality wins despite the power cost.

## Concurrency afforded by the hardware

The "9 concurrent DNNs" claim might sound like marketing fluff, but it is technically grounded in how the NVIDIA Jetson Orin architecture facilitates **hardware-level parallelism**.

Here is the deep dive into the architecture, the specific models, and the design trade-offs.

### 1. How Skydio Runs "Concurrent" DNNs (The "Secret" is Hardware)

Skydio achieves concurrency not just through software scheduling, but by utilizing distinct hardware blocks on the Jetson Orin SoC.

The Jetson Orin is a heterogeneous SoC. Skydio likely splits the workload across three distinct compute engines running simultaneously:

  * **The Ampere GPU:** Runs the heaviest, most complex models (e.g., 3D reconstruction, semantic segmentation).
  * **The DLA (Deep Learning Accelerator):** Orin has **2 dedicated DLA cores**. These are fixed-function NPU blocks designed specifically for standard CNN operations (convolution/pooling). Skydio offloads standard "backbone" tasks (like obstacle detection or subject tracking) to the DLA.
      * *Benefit:* The DLA runs independently of the GPU. While the GPU is crunching 3D map data, the DLA is processing obstacle avoidance frames in parallel without context switching.
  * **The PVA (Programmable Vision Accelerator):** Handles low-level computer vision tasks (optical flow, feature tracking) to feed the VIO (Visual Inertial Odometry) system, freeing up the GPU/CPU.

**Result:** The "9 concurrent DNNs" are likely distributed: ~2-3 on the GPU, ~4-5 on the DLAs, and lighter tasks on the PVA or CPU.

-----

### 2. What are the 9 DNNs?

Based on Skydio’s technical disclosures and the standard autonomy stack requirements, the 9 models likely break down as follows:

| Category | Likely Model Function | Compute Engine |
| :--- | :--- | :--- |
| **Perception (Geometry)** | **1. Deep Stereo / Depth Estimation:** Converting 6x fisheye camera feeds into depth maps. | **GPU** (Heavy compute) |
| **Perception (Semantics)** | **2. Object Detection:** Identifying people, cars, and obstacles. | **DLA** (Standard CNN) |
| | **3. Semantic Segmentation:** Classifying "sky," "water," "ground," "tree" to aid path planning. | **GPU** |
| **State Estimation** | **4. Visual Inertial Odometry (VIO) Correction:** A learned model to correct drift in the IMU+Vision pose estimation. | **PVA / DLA** |
| **Motion** | **5. Subject Tracking (Shadow):** Predicting subject movement (Kalman filter + visual embeddings). | **DLA** |
| **Enhancement** | **6. NightSense (De-noising):** Low-light raw image enhancement (detecting photons in darkness). | **GPU** (Pixel-wise ops) |
| **Mapping** | **7. 3D Voxel/Surface Reconstruction:** Converting depth maps into the persistent 3D map. | **GPU** |
| **Control** | **8. Path Planning Policy:** A learned policy (reinforcement learning) for complex maneuvers (e.g., "squeeze through gap"). | **CPU / DLA** |
| **Safety** | **9. Collision Prediction:** A separate, lightweight safety net model verifying the planned trajectory is clear. | **DLA** (Safety critical) |

-----

### 3. The Design Space: Why Jetson Orin@0.2 TOPS/W?

For a fully autonomous flying robot, the Jetson Orin is often the superior architectural choice for three reasons:

#### A. The "Host Processor" Bottleneck

  * **Hailo is an accelerator (NPU), not a computer.** It needs a host CPU (like a Raspberry Pi, Intel Core, or Rockchip) to feed it data.
  * **The Bottleneck:** Skydio drones process **1 million 3D points per second**. Moving 6 streams of 4K video from the camera ISP -\> System RAM -> PCIe Bus -> Hailo NPU -> PCIe Bus -> System RAM creates massive latency and power overhead.
  * **Jetson Advantage:** It is a **Unified Memory Architecture**. The CPU, GPU, and DLA share the *same* physical RAM. The cameras write data to memory once, and the GPU/DLA read it instantly without copying. This saves significant power and latency that specs like "TOPS/W" ignore.

#### B. Non-Neural Compute (The "Robotics" Part)

Autonomy is only ~40% neural networks. The other ~60% is classical robotics math:

  * **VIO (Visual Inertial Odometry):** Heavy linear algebra (matrix multiplication) to fuse IMU and camera data at 500Hz.
  * **Motion Planning:** Rapidly exploring search trees (RRT*, A*) to find flight paths.
  * **Video Encoding:** Compressing 4K video for the user.
  * **Hailo cannot do any of this.** You would need a powerful (and power-hungry) Intel/AMD CPU alongside the Hailo to handle these tasks, negating the efficiency gains. The Jetson's 12-core ARM CPU + GPU handles both the AI and the math.

#### C. Flexibility vs. Efficiency

  * **Hailo-8** excels at standard CNNs (ResNet, YOLO).
  * **Skydio's Needs:** They use custom, weird architectures for **3D Geometry** (Transformers, NeRF-like structures, or graph networks). NPUs like Hailo often struggle to run these custom non-standard layers efficiently. The Jetson GPU (CUDA) is fully programmable and can run *any* mathematical operation, ensuring Skydio isn't "stuck" if they invent a new model architecture next month.

### Summary

Skydio picked the Jetson Orin because it offers the **best system-level efficiency** for a robot that needs to do heavy CPU math *and* heavy AI inference on the same shared memory. Hailo wins on pure AI inference, but loses when you have to build a complete flying computer around it.

### Relevant Technical Talk

This video from NVIDIA features a deep dive into how the **Deep Learning Accelerator (DLA)** works specifically on the Jetson Orin to offload the GPU, which perfectly explains Skydio's concurrency strategy.

[NVIDIA Jetson Orin DLA Deep Dive](https://www.google.com/search?q=https://www.youtube.com/watch%3Fv%3DL8dY3fU_aG4)

*This video is relevant because it technically explains the hardware block (DLA) that allows companies like Skydio to run concurrent models without choking the main GPU.*

