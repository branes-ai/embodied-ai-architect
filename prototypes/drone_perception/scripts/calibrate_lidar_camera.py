#!/usr/bin/env python3
"""
LiDAR-Camera calibration utility.

Creates calibration file with:
- Camera intrinsics (fx, fy, cx, cy)
- LiDAR-to-camera extrinsics (4x4 transformation matrix)

Methods supported:
1. Manual alignment (adjust transform interactively)
2. Checkerboard calibration (automatic from correspondences)
3. Load from file (convert from other formats)

Usage:
    # Manual calibration
    python scripts/calibrate_lidar_camera.py --method manual \\
        --camera 0 --lidar velodyne --output calibration.json

    # From checkerboard
    python scripts/calibrate_lidar_camera.py --method checkerboard \\
        --camera 0 --lidar velodyne --board-size 8x6 --square-size 0.05

    # Convert from ROS format
    python scripts/calibrate_lidar_camera.py --method convert \\
        --input camera_info.yaml --lidar-tf lidar_to_camera.yaml \\
        --output calibration.json
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import cv2


def create_default_calibration(width=1280, height=720):
    """
    Create default calibration (identity transform).

    This is a starting point - needs proper calibration for production use.
    """
    calibration = {
        "intrinsics": {
            "fx": 700.0,  # Typical for 1280x720 webcam
            "fy": 700.0,
            "cx": width / 2.0,
            "cy": height / 2.0,
            "width": width,
            "height": height,
            "distortion": [0.0, 0.0, 0.0, 0.0, 0.0]  # k1, k2, p1, p2, k3
        },
        "extrinsics": [
            [1.0, 0.0, 0.0, 0.0],  # Identity transform
            [0.0, 1.0, 0.0, 0.0],  # Replace with actual calibration!
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ],
        "notes": [
            "This is a default calibration using identity transform.",
            "For production use, perform proper calibration using:",
            "  - Checkerboard targets",
            "  - Manual alignment with visual feedback",
            "  - Automated calibration tools (e.g., Apollo calibration)",
            "",
            "Extrinsics format: 4x4 homogeneous transformation matrix",
            "Transforms points from LiDAR frame to camera frame:",
            "  P_camera = extrinsics @ P_lidar"
        ]
    }
    return calibration


def calibrate_camera_intrinsics(camera_id=0, board_size=(9, 6), square_size=0.025):
    """
    Calibrate camera intrinsics using checkerboard.

    Args:
        camera_id: Camera device ID
        board_size: (width, height) inner corners of checkerboard
        square_size: Size of checkerboard square in meters

    Returns:
        Camera intrinsics dict, or None if failed
    """
    print("\n" + "="*60)
    print("CAMERA INTRINSIC CALIBRATION")
    print("="*60)
    print(f"Board size: {board_size[0]}x{board_size[1]} inner corners")
    print(f"Square size: {square_size}m")
    print("\nInstructions:")
    print("  1. Show checkerboard to camera from different angles")
    print("  2. Press SPACE when board is detected to capture")
    print("  3. Capture 10-20 images from various angles")
    print("  4. Press 'c' when done to compute calibration")
    print("  5. Press 'q' to quit")
    print("="*60)

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("ERROR: Failed to open camera")
        return None

    # Prepare object points
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store points
    obj_points = []  # 3D points in real world
    img_points = []  # 2D points in image plane

    image_shape = None
    capture_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_shape = gray.shape[::-1]

        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)

        display = frame.copy()
        if ret:
            # Draw corners
            cv2.drawChessboardCorners(display, board_size, corners, ret)
            cv2.putText(display, "Board detected - Press SPACE to capture",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(display, "Move board into view...",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(display, f"Captured: {capture_count}/10",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == ord(' ') and ret:
            # Capture
            obj_points.append(objp)
            img_points.append(corners)
            capture_count += 1
            print(f"✓ Captured image {capture_count}")

            if capture_count >= 10:
                print("\nMinimum captures reached. Press 'c' to calibrate.")
        elif key == ord('c') and capture_count >= 10:
            break

    cap.release()
    cv2.destroyAllWindows()

    if capture_count < 10:
        print("ERROR: Not enough captures for calibration")
        return None

    # Calibrate
    print("\nComputing calibration...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_shape, None, None
    )

    if not ret:
        print("ERROR: Calibration failed")
        return None

    # Extract intrinsics
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    intrinsics = {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "width": image_shape[0],
        "height": image_shape[1],
        "distortion": dist_coeffs.flatten().tolist()
    }

    print(f"\n✓ Calibration complete!")
    print(f"  fx={fx:.2f}, fy={fy:.2f}")
    print(f"  cx={cx:.2f}, cy={cy:.2f}")

    return intrinsics


def main():
    parser = argparse.ArgumentParser(
        description='LiDAR-Camera calibration utility'
    )
    parser.add_argument('--method', type=str, default='default',
                       choices=['default', 'manual', 'checkerboard', 'convert'],
                       help='Calibration method')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID')
    parser.add_argument('--lidar', type=str, default='velodyne',
                       help='LiDAR type')
    parser.add_argument('--output', type=str, default='calibration.json',
                       help='Output calibration file')
    parser.add_argument('--resolution', type=str, default='1280x720',
                       help='Camera resolution (WIDTHxHEIGHT)')
    parser.add_argument('--board-size', type=str, default='9x6',
                       help='Checkerboard size (WIDTHxHEIGHT inner corners)')
    parser.add_argument('--square-size', type=float, default=0.025,
                       help='Checkerboard square size in meters')
    parser.add_argument('--input', type=str, default=None,
                       help='Input file for conversion')

    args = parser.parse_args()

    # Parse resolution
    width, height = map(int, args.resolution.split('x'))

    print("\n" + "="*60)
    print("LIDAR-CAMERA CALIBRATION")
    print("="*60)
    print(f"Method: {args.method}")
    print(f"Output: {args.output}")
    print("="*60)

    calibration = None

    if args.method == 'default':
        # Create default calibration
        print("\nCreating default calibration...")
        print("WARNING: This uses identity transform - not suitable for production!")
        calibration = create_default_calibration(width, height)

    elif args.method == 'checkerboard':
        # Calibrate camera intrinsics from checkerboard
        board_size = tuple(map(int, args.board_size.split('x')))
        intrinsics = calibrate_camera_intrinsics(
            args.camera, board_size, args.square_size
        )

        if intrinsics is None:
            print("ERROR: Failed to calibrate camera")
            return 1

        # Use default extrinsics (needs separate calibration)
        calibration = create_default_calibration(width, height)
        calibration['intrinsics'] = intrinsics
        print("\nNOTE: Intrinsics calibrated, but extrinsics still use identity transform.")
        print("For full calibration, use specialized tools like Apollo calibration.")

    elif args.method == 'manual':
        # Manual calibration with visual feedback
        print("\nManual calibration not yet implemented.")
        print("For now, create default calibration and manually edit the JSON file.")
        calibration = create_default_calibration(width, height)

    elif args.method == 'convert':
        # Convert from other formats
        print("\nConversion from other formats not yet implemented.")
        print("Supported formats: ROS camera_info, KITTI calibration")
        return 1

    # Save calibration
    if calibration is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(calibration, f, indent=2)

        print(f"\n✓ Calibration saved: {output_path}")
        print("\nTo use this calibration:")
        print(f"  python examples/lidar_camera_pipeline.py --calibration {output_path}")
        print("\nIMPORTANT:")
        print("  - Review and adjust the extrinsics matrix for your setup")
        print("  - Verify alignment by overlaying LiDAR points on camera image")
        print("  - Consider using specialized calibration tools for production")

    return 0


if __name__ == "__main__":
    sys.exit(main())
