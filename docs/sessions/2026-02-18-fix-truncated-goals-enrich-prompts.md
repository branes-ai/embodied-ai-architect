# Session: Fix Truncated Goals + Enrich Demo Prompts

**Date:** 2026-02-18
**Commit:** `b5412da`

## Summary

Demo scripts truncated goal text to 80-90 chars with `...`, hiding the most important parts of each prompt. Since goals functionally drive workload estimation via keyword matching in `_estimate_workload_from_goal()` (specialists.py:121-242), richer and more distinct prompts produce visibly different pipeline outputs — different workloads, hardware scores, and architectures.

## What Was Changed

### Truncation Fix (6 demo scripts)

Replaced every `print(f"  Goal: {GOAL[:N]}...")` pattern with `textwrap.fill()` that wraps the full goal text:

```python
import textwrap
print(f"\n  Goal:")
print(textwrap.fill(goal, width=W - 4, initial_indent="    ", subsequent_indent="    "))
```

### Enriched Goal Strings (7 demo scripts)

Each goal was expanded to 3-5 sentences with specific workload keywords that trigger different code paths in the workload analyzer:

| Demo | Script | Workloads Triggered | Key Enrichments |
|------|--------|-------------------|-----------------|
| 1 | `demo_soc_designer.py` | detection + tracking + perception | YOLOv8-nano, ByteTrack, 720p, thermal cycling |
| 2 | `demo_dse_pareto.py` | detection + slam | Localization, mapping, GPS-denied aisles, deterministic scheduling |
| 3 | `demo_soc_optimizer.py` | detection + tracking + perception | Camera ISP, buffering latency, thermal throttling headroom |
| 4 | `demo_kpu_rtl.py` | detection + tracking + perception | Camera vision/ISP, systolic array, 12nm process |
| 5 | `demo_hitl_safety.py` | detection + tracking + voice | **Voice/speech** for surgeon commands, dual-redundant lockstep, watchdog |
| 6a | `demo_experience_cache.py` | detection + tracking | Urban YOLO + ByteTrack for pedestrian avoidance |
| 6b | `demo_experience_cache.py` | detection + tracking + slam + perception | Multispectral camera, precision agriculture mapping |
| 7 | `demo_full_campaign.py` | detection + slam + perception + voice + lidar | All 5 workload types, concurrent deterministic scheduling |

### Demo 3 Dynamic Goal Fix

`demo_soc_optimizer.py` previously had a hardcoded one-liner for display and a separate dynamic goal for `create_initial_soc_state()`. Now both share the same enriched `goal_text` variable.

## Recognized Keywords

From `_estimate_workload_from_goal()` (specialists.py:130-206):

| Workload | Keywords |
|----------|----------|
| object_detection | detection, yolo, object |
| object_tracking | tracking, track, bytetrack |
| visual_slam | slam, localization, mapping |
| visual_perception | perception, vision, camera |
| voice_recognition | voice, speech, audio |
| lidar_processing | lidar, point cloud |

## Files Modified

| File | Change |
|------|--------|
| `examples/demo_soc_designer.py` | `import textwrap`, enriched DEMO_1_GOAL, `textwrap.fill()` display |
| `examples/demo_dse_pareto.py` | `import textwrap`, enriched DEMO_2_GOAL, `textwrap.fill()` display |
| `examples/demo_soc_optimizer.py` | `import textwrap`, enriched dynamic goal, reused for state + display |
| `examples/demo_kpu_rtl.py` | `import textwrap`, enriched DEMO_4_GOAL, `textwrap.fill()` display |
| `examples/demo_hitl_safety.py` | `import textwrap`, enriched DEMO_5_GOAL with voice/speech, `textwrap.fill()` |
| `examples/demo_experience_cache.py` | Enriched DRONE_1_GOAL and DRONE_2_GOAL (no truncation to fix) |
| `examples/demo_full_campaign.py` | `import textwrap`, enriched DEMO_7_GOAL, `textwrap.fill()` display |

## Verification

All 7 files parse without syntax errors. Keyword matching was verified to produce the intended workload combinations per demo, ensuring each demo generates visibly different workload tables and hardware rankings.
