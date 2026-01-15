---
title: Operator Catalog
description: Browse operators for embodied AI pipelines.
---

Operators are the building blocks of embodied AI pipelines.

:::note
This page will be auto-generated from the `embodied-schemas` operator catalog.
:::

## Perception

### Detection

| Operator | Model | Latency* | Status |
|----------|-------|----------|--------|
| YOLO Detector (nano) | YOLOv8n | 28ms | Full support |
| YOLO Detector (small) | YOLOv8s | 45ms | Full support |
| RT-DETR Detector | RT-DETR-L | 65ms | Full support |

### Tracking

| Operator | Algorithm | Latency* | Status |
|----------|-----------|----------|--------|
| ByteTrack | ByteTrack | 2ms | Full support |
| DeepSORT | DeepSORT | 15ms | Partial support |

### Depth

| Operator | Model | Latency* | Status |
|----------|-------|----------|--------|
| Stereo Depth | SGM | 8ms | Full support |
| Mono Depth | MiDaS-Small | 35ms | Partial support |

## State Estimation

| Operator | Algorithm | Rate | Status |
|----------|-----------|------|--------|
| Kalman Filter | EKF | 1kHz | Full support |
| Particle Filter | PF | 100Hz | Full support |

## Planning

| Operator | Algorithm | Latency* | Status |
|----------|-----------|----------|--------|
| RRT* | RRT* | 50ms | Full support |
| A* | A* | 10ms | Full support |

## Control

| Operator | Algorithm | Rate | Status |
|----------|-----------|------|--------|
| PID Controller | PID | 1kHz | Full support |
| MPC Controller | MPC | 100Hz | Partial support |

*Latency measured on Jetson Orin Nano
