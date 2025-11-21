# Utility and management scripts

## Quick Start:

  ** 1. Record webcam video**
  python scripts/record_webcam.py

  ** 2. Generate depth map using MiDaS**
  python scripts/generate_depth_maps.py --input test_data/videos/webcam/recording_YYYYMMDD_HHMMSS.mp4

## Recording Tips:

Good scenarios to record:
  - Walk toward/away from camera (1-5m range)
  - Move objects at different depths
  - Indoor scene with furniture at 1-3m
  - Hallway with objects near and far
  - Desktop setup with monitor (1m), wall (3m), doorway (5m)

What to avoid:
  - Overhead/downward angles (like drone footage)
  - Objects all at same distance
  - Very fast movement
  - Poor lighting

The script shows:
  - ⚫ Red dot when recording
  - Frame count and elapsed time
  - On-screen tips for good footage
  - Controls: SPACE to record, Q to quit, I to toggle tips

