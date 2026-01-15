---
title: Sensor Catalog
description: Browse supported sensors for embodied AI systems.
---

This catalog lists sensors commonly used in embodied AI systems.

:::note
This page will be auto-generated from the `embodied-schemas` sensor catalog.
:::

## RGB Cameras

| Sensor | Resolution | FPS | Interface | Status |
|--------|------------|-----|-----------|--------|
| IMX477 | 4056x3040 | 30 | CSI-2 | Full support |
| IMX219 | 3280x2464 | 30 | CSI-2 | Full support |
| OV5647 | 2592x1944 | 30 | CSI-2 | Full support |

## Depth Sensors

| Sensor | Range | Resolution | Interface | Status |
|--------|-------|------------|-----------|--------|
| Intel RealSense D435 | 0.3-3m | 1280x720 | USB 3.0 | Full support |
| Intel RealSense D455 | 0.4-6m | 1280x720 | USB 3.0 | Full support |
| OAK-D | 0.2-35m | 1280x800 | USB 3.0 | Full support |

## LiDAR

| Sensor | Range | Points/sec | Interface | Status |
|--------|-------|------------|-----------|--------|
| Livox Mid-360 | 70m | 200K | Ethernet | Partial support |
| Ouster OS1-32 | 120m | 1.3M | Ethernet | Partial support |

## IMUs

| Sensor | Axes | Rate | Interface | Status |
|--------|------|------|-----------|--------|
| BMI088 | 6-axis | 2kHz | I2C/SPI | Full support |
| ICM-42688 | 6-axis | 32kHz | SPI | Full support |
