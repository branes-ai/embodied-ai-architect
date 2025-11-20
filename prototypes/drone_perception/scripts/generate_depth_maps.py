#!/usr/bin/env python3
"""
Generate depth maps from RGB videos using MiDaS.

Creates synthetic depth videos for testing stereo pipeline without hardware.
Uses MiDaS v3.1 for monocular depth estimation.

Usage:
    # Generate depth for a single video
    python scripts/generate_depth_maps.py --input test_data/videos/traffic/247589_tiny.mp4

    # Generate depth for all toll booth videos
    python scripts/generate_depth_maps.py --batch toll_booth

    # Use specific MiDaS model
    python scripts/generate_depth_maps.py --input video.mp4 --model dpt_large
"""

import sys
from pathlib import Path
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_midas_model(model_type="dpt_hybrid"):
    """
    Load MiDaS depth estimation model.

    Args:
        model_type: "dpt_large", "dpt_hybrid", or "midas_small"

    Returns:
        model, transform, device
    """
    print(f"Loading MiDaS model: {model_type}")

    # Check if MPS (Apple Silicon) is available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load model
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.to(device)
    midas.eval()

    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "dpt_large" or model_type == "dpt_hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    print(f"Model loaded on {device}")

    return midas, transform, device


def estimate_depth_frame(model, transform, device, image_rgb):
    """
    Estimate depth for a single frame.

    Args:
        model: MiDaS model
        transform: MiDaS transform
        device: Torch device
        image_rgb: RGB image (H, W, 3)

    Returns:
        Depth map (H, W) in arbitrary units (relative depth)
    """
    # Prepare input
    input_batch = transform(image_rgb).to(device)

    # Predict
    with torch.no_grad():
        prediction = model(input_batch)

        # Resize to original resolution
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth = prediction.cpu().numpy()

    return depth


def normalize_depth_for_video(depth, min_depth=0.0, max_depth=10.0):
    """
    Normalize depth map to 16-bit range for video encoding.

    Args:
        depth: Depth map (inverse depth from MiDaS)
        min_depth: Minimum depth in meters (far = small values)
        max_depth: Maximum depth in meters (close = large values)

    Returns:
        16-bit depth map (uint16)
    """
    # MiDaS outputs inverse depth (larger = closer)
    # Invert to get depth (larger = farther)
    depth_inv = 1.0 / (depth + 1e-6)

    # Normalize to 0-1 range
    depth_min = depth_inv.min()
    depth_max = depth_inv.max()

    if depth_max > depth_min:
        depth_normalized = (depth_inv - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth_inv)

    # Scale to physical depth range (0 to max_depth meters)
    depth_meters = depth_normalized * max_depth

    # Convert to 16-bit (millimeter precision)
    # Store as mm in uint16: 0-65535 mm (0-65.535 m)
    depth_mm = (depth_meters * 1000.0).astype(np.uint16)

    return depth_mm


def generate_depth_video(
    input_video_path,
    output_depth_path,
    model,
    transform,
    device,
    max_depth=10.0,
    show_preview=False
):
    """
    Generate depth video from RGB video.

    Args:
        input_video_path: Path to input RGB video
        output_depth_path: Path to output depth video
        model: MiDaS model
        transform: MiDaS transform
        device: Torch device
        max_depth: Maximum depth in meters for normalization
        show_preview: Show preview window during processing
    """
    print(f"\nProcessing: {input_video_path}")
    print(f"Output: {output_depth_path}")

    # Open input video
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {input_video_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")

    # Create output video writer
    # Use grayscale 16-bit format
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')  # Lossless codec for 16-bit
    depth_writer = cv2.VideoWriter(
        str(output_depth_path),
        fourcc,
        fps,
        (width, height),
        isColor=False
    )

    if not depth_writer.isOpened():
        # Fallback to visualization (8-bit colormap)
        print("Warning: Failed to create 16-bit video, using 8-bit colormap instead")
        output_depth_path = output_depth_path.with_suffix('.vis.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        depth_writer = cv2.VideoWriter(
            str(output_depth_path),
            fourcc,
            fps,
            (width, height),
            isColor=True
        )

    # Process frames
    frame_count = 0
    with tqdm(total=total_frames, desc="Generating depth") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Estimate depth
            depth = estimate_depth_frame(model, transform, device, frame_rgb)

            # Normalize for storage
            depth_16bit = normalize_depth_for_video(depth, max_depth=max_depth)

            # Write frame
            # Convert 16-bit to 8-bit for standard video (scale down)
            depth_8bit = (depth_16bit / 256).astype(np.uint8)

            # For visualization, create colormap
            depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_MAGMA)
            depth_writer.write(depth_colormap)

            # Show preview
            if show_preview and frame_count % 10 == 0:
                preview = np.hstack([
                    cv2.resize(frame_bgr, (width//2, height//2)),
                    cv2.resize(depth_colormap, (width//2, height//2))
                ])
                cv2.imshow("RGB | Depth", preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1
            pbar.update(1)

    # Cleanup
    cap.release()
    depth_writer.release()
    if show_preview:
        cv2.destroyAllWindows()

    print(f"✓ Generated {frame_count} depth frames")
    print(f"✓ Saved to: {output_depth_path}")

    # Also save metadata
    metadata_path = output_depth_path.with_suffix('.yaml')
    with open(metadata_path, 'w') as f:
        f.write(f"# Depth map metadata\n")
        f.write(f"source_video: {input_video_path.name}\n")
        f.write(f"depth_video: {output_depth_path.name}\n")
        f.write(f"model: MiDaS\n")
        f.write(f"frames: {frame_count}\n")
        f.write(f"resolution: {width}x{height}\n")
        f.write(f"fps: {fps}\n")
        f.write(f"max_depth_meters: {max_depth}\n")
        f.write(f"depth_scale: 0.001  # meters per unit (1mm)\n")

    print(f"✓ Saved metadata: {metadata_path}")


def batch_process_toll_booth():
    """Generate depth maps for all toll booth test videos."""
    test_dir = Path(__file__).parent.parent / "test_data" / "videos" / "traffic"

    videos = [
        "247589_tiny.mp4",
        "247589_small.mp4",
        "247589_medium.mp4"
    ]

    # Load model once
    model, transform, device = load_midas_model(model_type="dpt_hybrid")

    # Process each video
    for video_name in videos:
        input_path = test_dir / video_name
        if not input_path.exists():
            print(f"Skipping {video_name} (not found)")
            continue

        output_path = test_dir / video_name.replace(".mp4", "_depth.mp4")

        try:
            generate_depth_video(
                input_path,
                output_path,
                model,
                transform,
                device,
                max_depth=15.0,  # Traffic scenes can be farther
                show_preview=False
            )
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            continue

    print("\n" + "="*60)
    print("Batch processing complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Generate depth maps from RGB videos')
    parser.add_argument('--input', type=str, default=None,
                        help='Input RGB video path')
    parser.add_argument('--output', type=str, default=None,
                        help='Output depth video path (default: input_depth.mp4)')
    parser.add_argument('--model', type=str, default='dpt_hybrid',
                        choices=['dpt_large', 'dpt_hybrid', 'midas_small'],
                        help='MiDaS model type')
    parser.add_argument('--max-depth', type=float, default=10.0,
                        help='Maximum depth in meters for normalization')
    parser.add_argument('--batch', type=str, default=None,
                        choices=['toll_booth'],
                        help='Batch process predefined video sets')
    parser.add_argument('--preview', action='store_true',
                        help='Show preview during processing')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("DEPTH MAP GENERATION (MiDaS)")
    print("="*60)

    # Check dependencies
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed!")
        print("Install with: pip install torch torchvision")
        return

    # Batch processing
    if args.batch == 'toll_booth':
        batch_process_toll_booth()
        return

    # Single video processing
    if args.input is None:
        print("ERROR: --input required (or use --batch)")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input video not found: {input_path}")
        return

    output_path = Path(args.output) if args.output else input_path.with_name(
        input_path.stem + "_depth.mp4"
    )

    # Load model
    model, transform, device = load_midas_model(model_type=args.model)

    # Generate depth video
    generate_depth_video(
        input_path,
        output_path,
        model,
        transform,
        device,
        max_depth=args.max_depth,
        show_preview=args.preview
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
