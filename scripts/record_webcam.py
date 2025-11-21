#!/usr/bin/env python3
"""
Webcam recording utility for creating stereo pipeline test videos.

Records forward-facing webcam footage suitable for depth estimation testing.
Provides visual feedback and instructions for capturing good test data.

Usage:
    python scripts/record_webcam.py --output test_data/videos/webcam/test1.mp4
    python scripts/record_webcam.py --camera 0 --resolution 1280x720 --fps 30
"""

import sys
from pathlib import Path
import argparse
import cv2
import time
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_resolution(res_str):
    """Parse resolution string like '1280x720' to (width, height)."""
    try:
        width, height = map(int, res_str.lower().split('x'))
        return width, height
    except:
        raise ValueError(f"Invalid resolution format: {res_str}. Use WIDTHxHEIGHT (e.g., 1280x720)")


def record_webcam(
    camera_id=0,
    output_path=None,
    resolution=(1280, 720),
    fps=30,
    show_instructions=True
):
    """
    Record webcam footage with live preview.

    Args:
        camera_id: Camera device ID (0 for default webcam)
        output_path: Path to save video
        resolution: (width, height) tuple
        fps: Frames per second
        show_instructions: Show recording tips overlay
    """
    # Open camera
    print(f"Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {camera_id}")

    # Set resolution
    width, height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Get actual resolution (might differ from requested)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} fps")

    # Default output path if not provided
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"test_data/videos/webcam/recording_{timestamp}.mp4")
    else:
        output_path = Path(output_path)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (actual_width, actual_height)
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    print(f"\nOutput: {output_path}")
    print("\n" + "="*60)
    print("WEBCAM RECORDING")
    print("="*60)
    print("\nControls:")
    print("  SPACE - Start/Stop recording")
    print("  'q'   - Quit")
    print("  'i'   - Toggle instructions")
    print("\nTips for good depth test footage:")
    print("  • Keep camera steady or move slowly")
    print("  • Include objects at multiple distances (1-10m)")
    print("  • Use forward-facing perspective (not overhead)")
    print("  • Record for 10-30 seconds")
    print("  • Good lighting helps detection")
    print("="*60)

    recording = False
    frame_count = 0
    start_time = None

    instructions = [
        "TIPS FOR DEPTH TESTING:",
        "- Objects at 1-3m (near)",
        "- Objects at 3-7m (mid)",
        "- Objects at 7-15m (far)",
        "- Walk toward/away slowly",
        "- Move objects in scene"
    ]

    show_tips = show_instructions

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            display_frame = frame.copy()

            # Add overlay
            overlay_y = 30

            # Recording indicator
            if recording:
                # Red recording dot
                cv2.circle(display_frame, (30, 30), 15, (0, 0, 255), -1)
                cv2.putText(display_frame, "RECORDING", (60, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Frame counter and time
                elapsed = time.time() - start_time
                cv2.putText(display_frame,
                           f"Frames: {frame_count} | Time: {elapsed:.1f}s",
                           (60, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                # Ready indicator
                cv2.putText(display_frame, "READY - Press SPACE to record",
                           (30, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Instructions overlay
            if show_tips:
                tips_y = actual_height - 180
                for i, tip in enumerate(instructions):
                    cv2.putText(display_frame, tip,
                               (10, tips_y + i*25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show preview
            cv2.imshow("Webcam Recording", display_frame)

            # Write frame if recording
            if recording:
                writer.write(frame)
                frame_count += 1

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar
                if not recording:
                    # Start recording
                    recording = True
                    start_time = time.time()
                    frame_count = 0
                    print("\n▶ Recording started...")
                else:
                    # Stop recording
                    recording = False
                    elapsed = time.time() - start_time
                    print(f"■ Recording stopped. {frame_count} frames, {elapsed:.1f}s")
            elif key == ord('i'):
                # Toggle instructions
                show_tips = not show_tips

    finally:
        # Cleanup
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

        if frame_count > 0:
            print(f"\n✓ Video saved: {output_path}")
            print(f"  {frame_count} frames")
            print(f"\nNext steps:")
            print(f"  1. Generate depth map:")
            print(f"     python scripts/generate_depth_maps.py --input {output_path}")
            print(f"  2. Run stereo pipeline:")
            print(f"     python examples/stereo_pipeline.py --backend recorded \\")
            print(f"       --rgb-video {output_path} \\")
            print(f"       --depth-video {output_path.with_name(output_path.stem + '_depth.mp4')}")
        else:
            print("\nNo frames recorded. Video not saved.")
            output_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description='Record webcam footage for stereo pipeline testing'
    )
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path (default: auto-generated in test_data/videos/webcam/)')
    parser.add_argument('--resolution', type=str, default='1280x720',
                       help='Resolution as WIDTHxHEIGHT (default: 1280x720)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second (default: 30)')
    parser.add_argument('--no-instructions', action='store_true',
                       help='Hide instruction overlay')

    args = parser.parse_args()

    try:
        resolution = parse_resolution(args.resolution)

        record_webcam(
            camera_id=args.camera,
            output_path=args.output,
            resolution=resolution,
            fps=args.fps,
            show_instructions=not args.no_instructions
        )

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
