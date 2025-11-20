#!/usr/bin/env python3
"""
Compare monocular vs stereo pipeline results.

Analyzes log files from both pipelines and compares:
- Detection counts
- Tracking stability
- Processing speed
- Scene graph statistics

Usage:
    python scripts/compare_mono_vs_stereo.py --video-name 247589_tiny
"""

import sys
from pathlib import Path
import argparse
import re
from typing import Dict, List
import yaml


def parse_log_file(log_path: Path) -> Dict:
    """
    Parse pipeline log file to extract statistics.

    Returns:
        dict with stats
    """
    if not log_path.exists():
        return {"error": "Log file not found"}

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    stats = {
        "frames_processed": 0,
        "detections_per_frame": [],
        "tracks_per_frame": [],
        "scene_objects": [],
        "fps_values": [],
        "errors": []
    }

    # Extract frame processing info
    # Format: "Frame 0030 | FPS: 15.3 | Detections: 5 | Tracks: 4 | 3D Objects: 3"
    pattern = r"Frame (\d+).*?FPS: ([\d.]+).*?Detections: (\d+).*?Tracks: (\d+).*?Scene objects: (\d+)"

    for match in re.finditer(pattern, content, re.IGNORECASE):
        frame_num = int(match.group(1))
        fps = float(match.group(2))
        detections = int(match.group(3))
        tracks = int(match.group(4))
        objects_3d = int(match.group(5))

        stats["frames_processed"] = max(stats["frames_processed"], frame_num + 1)
        stats["fps_values"].append(fps)
        stats["detections_per_frame"].append(detections)
        stats["tracks_per_frame"].append(tracks)
        stats["scene_objects"].append(objects_3d)

    # Check for errors
    if "error" in content.lower() or "exception" in content.lower():
        error_lines = [line for line in content.split('\n')
                      if 'error' in line.lower() or 'exception' in line.lower()]
        stats["errors"] = error_lines[:5]  # First 5 errors

    return stats


def compute_summary_stats(values: List[float]) -> Dict:
    """Compute mean, min, max, std from list of values."""
    if not values:
        return {"mean": 0, "min": 0, "max": 0, "std": 0}

    import numpy as np
    arr = np.array(values)

    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr))
    }


def compare_results(mono_log: Path, stereo_log: Path, video_name: str):
    """
    Compare monocular vs stereo results.
    """
    print("\n" + "="*70)
    print(f"COMPARISON: Monocular vs Stereo - {video_name}")
    print("="*70 + "\n")

    # Parse logs
    print("Parsing logs...")
    mono_stats = parse_log_file(mono_log)
    stereo_stats = parse_log_file(stereo_log)

    # Check for errors
    if "error" in mono_stats:
        print(f"ERROR: Could not parse monocular log: {mono_stats['error']}")
        return

    if "error" in stereo_stats:
        print(f"ERROR: Could not parse stereo log: {stereo_stats['error']}")
        return

    # Basic stats
    print(f"Monocular processed: {mono_stats['frames_processed']} frames")
    print(f"Stereo processed:    {stereo_stats['frames_processed']} frames")
    print()

    # Performance comparison
    print("="*70)
    print("PERFORMANCE")
    print("="*70)

    mono_fps = compute_summary_stats(mono_stats['fps_values'])
    stereo_fps = compute_summary_stats(stereo_stats['fps_values'])

    print(f"FPS:")
    print(f"  Monocular: {mono_fps['mean']:.1f} avg (min: {mono_fps['min']:.1f}, max: {mono_fps['max']:.1f})")
    print(f"  Stereo:    {stereo_fps['mean']:.1f} avg (min: {stereo_fps['min']:.1f}, max: {stereo_fps['max']:.1f})")

    if mono_fps['mean'] > 0:
        fps_diff_pct = ((stereo_fps['mean'] - mono_fps['mean']) / mono_fps['mean']) * 100
        print(f"  Difference: {fps_diff_pct:+.1f}%")
    print()

    # Detection comparison
    print("="*70)
    print("DETECTIONS")
    print("="*70)

    mono_det = compute_summary_stats(mono_stats['detections_per_frame'])
    stereo_det = compute_summary_stats(stereo_stats['detections_per_frame'])

    print(f"Detections per frame:")
    print(f"  Monocular: {mono_det['mean']:.1f} avg (std: {mono_det['std']:.2f})")
    print(f"  Stereo:    {stereo_det['mean']:.1f} avg (std: {stereo_det['std']:.2f})")
    print()

    # Tracking comparison
    print("="*70)
    print("TRACKING")
    print("="*70)

    mono_track = compute_summary_stats(mono_stats['tracks_per_frame'])
    stereo_track = compute_summary_stats(stereo_stats['tracks_per_frame'])

    print(f"Tracks per frame:")
    print(f"  Monocular: {mono_track['mean']:.1f} avg (std: {mono_track['std']:.2f})")
    print(f"  Stereo:    {stereo_track['mean']:.1f} avg (std: {stereo_track['std']:.2f})")

    # Lower std = more stable tracking
    if stereo_track['std'] < mono_track['std']:
        improvement = ((mono_track['std'] - stereo_track['std']) / mono_track['std']) * 100
        print(f"  ✓ Stereo tracking {improvement:.1f}% more stable")
    print()

    # 3D scene graph
    print("="*70)
    print("3D SCENE GRAPH")
    print("="*70)

    mono_obj = compute_summary_stats(mono_stats['scene_objects'])
    stereo_obj = compute_summary_stats(stereo_stats['scene_objects'])

    print(f"3D objects:")
    print(f"  Monocular: {mono_obj['mean']:.1f} avg (std: {mono_obj['std']:.2f})")
    print(f"  Stereo:    {stereo_obj['mean']:.1f} avg (std: {stereo_obj['std']:.2f})")
    print()

    # Errors
    if mono_stats['errors']:
        print("="*70)
        print("MONOCULAR ERRORS")
        print("="*70)
        for err in mono_stats['errors']:
            print(f"  {err}")
        print()

    if stereo_stats['errors']:
        print("="*70)
        print("STEREO ERRORS")
        print("="*70)
        for err in stereo_stats['errors']:
            print(f"  {err}")
        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)

    print("\nExpected improvements with stereo:")
    print("  ✓ More accurate depth (real vs estimated)")
    print("  ✓ More stable tracking (consistent 3D positions)")
    print("  ✓ Better velocity estimates (metric accuracy)")

    print("\nExpected trade-offs:")
    print("  - Slightly slower (depth processing overhead)")
    print("  - Requires depth camera or pre-generated depth maps")

    print()


def batch_compare():
    """Compare all toll booth videos."""
    test_dir = Path(__file__).parent.parent / "test_data" / "videos" / "traffic"

    videos = ["247589_tiny", "247589_small", "247589_medium"]

    for video_name in videos:
        mono_log = test_dir / f"{video_name}.log"
        stereo_log = test_dir / f"{video_name}_stereo.log"

        if mono_log.exists() and stereo_log.exists():
            compare_results(mono_log, stereo_log, video_name)
        else:
            print(f"\nSkipping {video_name} (missing logs)")
            if not mono_log.exists():
                print(f"  Missing: {mono_log}")
            if not stereo_log.exists():
                print(f"  Missing: {stereo_log}")


def main():
    parser = argparse.ArgumentParser(description='Compare monocular vs stereo results')
    parser.add_argument('--video-name', type=str, default=None,
                        help='Video name (e.g., 247589_tiny)')
    parser.add_argument('--batch', action='store_true',
                        help='Compare all toll booth videos')

    args = parser.parse_args()

    test_dir = Path(__file__).parent.parent / "test_data" / "videos" / "traffic"

    if args.batch:
        batch_compare()
    elif args.video_name:
        mono_log = test_dir / f"{args.video_name}.log"
        stereo_log = test_dir / f"{args.video_name}_stereo.log"
        compare_results(mono_log, stereo_log, args.video_name)
    else:
        print("ERROR: Specify --video-name or --batch")
        return


if __name__ == "__main__":
    main()
