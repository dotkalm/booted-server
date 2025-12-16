"""
Camera calibration using homography and solvePnP for AR placement.

This module implements camera parameter estimation from detected wheel positions
using OpenCV's Perspective-n-Point (PnP) solver. The estimated camera parameters
can be used to match Three.js camera perspective with the original photo.

Reference: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def estimate_camera_from_wheels(
    front_wheel_bbox: Optional[Dict],
    rear_wheel_bbox: Optional[Dict],
    image_width: int,
    image_height: int,
    wheelbase_m: float = 2.7,  # Average car wheelbase in meters
    wheel_diameter_m: float = 0.65,  # Average wheel diameter in meters
) -> Dict:
    """
    Estimate camera parameters from detected wheel positions using solvePnP.

    This function assumes wheels lie on a common ground plane (Y=0) and uses
    the known physical dimensions of a car to solve for the camera pose.

    Args:
        front_wheel_bbox: Bounding box of front wheel {"x1", "y1", "x2", "y2", "width", "height"}
        rear_wheel_bbox: Bounding box of rear wheel
        image_width: Image width in pixels
        image_height: Image height in pixels
        wheelbase_m: Distance between front and rear wheel centers (default: 2.7m)
        wheel_diameter_m: Diameter of wheels (default: 0.65m)

    Returns:
        Dictionary containing:
        - intrinsic_matrix: 3x3 camera intrinsic matrix
        - rotation_vector: 3x1 Rodrigues rotation vector
        - translation_vector: 3x1 camera translation
        - fov: Field of view in degrees
        - camera_height: Estimated camera height in meters
        - method: Calibration method used
    """
    # Validate inputs
    if not front_wheel_bbox or not rear_wheel_bbox:
        logger.warning("Missing wheel bounding boxes, using default camera parameters")
        return _default_camera_params(image_width, image_height)

    # Extract wheel centers and radii
    rear_center_x = (rear_wheel_bbox["x1"] + rear_wheel_bbox["x2"]) / 2
    rear_center_y = (rear_wheel_bbox["y1"] + rear_wheel_bbox["y2"]) / 2
    rear_radius = (rear_wheel_bbox["width"] + rear_wheel_bbox["height"]) / 4

    front_center_x = (front_wheel_bbox["x1"] + front_wheel_bbox["x2"]) / 2
    front_center_y = (front_wheel_bbox["y1"] + front_wheel_bbox["y2"]) / 2
    front_radius = (front_wheel_bbox["width"] + front_wheel_bbox["height"]) / 4

    # Define 3D world points on ground plane (Y=0)
    # Coordinate system: X=forward, Y=up, Z=right (from car's perspective)
    # Rear wheel at origin, front wheel at (-wheelbase, 0, 0)
    world_points = np.array([
        [0, 0, 0],              # Rear wheel ground contact (center)
        [-wheelbase_m, 0, 0],   # Front wheel ground contact (center)
        [0, 0, wheel_diameter_m / 2],     # Point to side of rear wheel
        [-wheelbase_m, 0, wheel_diameter_m / 2],  # Point to side of front wheel
    ], dtype=np.float32)

    # Corresponding 2D image points
    # Use bottom of wheels for ground contact points
    # Use top of wheels for side reference points
    image_points = np.array([
        [rear_center_x, rear_center_y + rear_radius],    # Rear wheel bottom
        [front_center_x, front_center_y + front_radius],  # Front wheel bottom
        [rear_center_x, rear_center_y - rear_radius],     # Rear wheel top
        [front_center_x, front_center_y - front_radius],  # Front wheel top
    ], dtype=np.float32)

    # Initial camera intrinsic matrix guess
    # Focal length estimate: approximate as image width
    focal_guess = max(image_width, image_height)
    K = np.array([
        [focal_guess, 0, image_width / 2],
        [0, focal_guess, image_height / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    # Solve PnP to get camera pose
    success, rvec, tvec = cv2.solvePnP(
        world_points,
        image_points,
        K,
        None,  # No distortion coefficients
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        logger.warning("solvePnP failed, using default camera parameters")
        return _default_camera_params(image_width, image_height)

    # Refine focal length using wheel size constraint
    refined_focal = _refine_focal_length(
        front_radius, rear_radius,
        wheelbase_m, wheel_diameter_m,
        image_width, image_height
    )

    if refined_focal:
        K[0, 0] = refined_focal
        K[1, 1] = refined_focal

        # Re-solve with refined intrinsics
        success, rvec, tvec = cv2.solvePnP(
            world_points,
            image_points,
            K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

    # Calculate FOV from focal length
    focal_length = K[0, 0]
    fov = 2 * np.arctan(image_height / (2 * focal_length)) * (180 / np.pi)

    # Estimate camera height from translation vector
    # tvec[1] represents the Y-component (height) in world coordinates
    camera_height = abs(tvec[1][0])

    logger.info(f"Camera calibration: FOV={fov:.1f}°, height={camera_height:.2f}m, "
                f"focal={focal_length:.0f}px")

    return {
        'intrinsic_matrix': K.tolist(),
        'rotation_vector': rvec.flatten().tolist(),
        'translation_vector': tvec.flatten().tolist(),
        'fov': float(fov),
        'camera_height': float(camera_height),
        'focal_length': float(focal_length),
        'method': 'solvePnP'
    }


def _refine_focal_length(
    front_radius_px: float,
    rear_radius_px: float,
    wheelbase_m: float,
    wheel_diameter_m: float,
    image_width: int,
    image_height: int
) -> Optional[float]:
    """
    Refine focal length estimate using known wheel diameter.

    A wheel of known real-world diameter appears as a certain pixel size.
    This relationship constrains the focal length.

    Args:
        front_radius_px: Front wheel radius in pixels
        rear_radius_px: Rear wheel radius in pixels
        wheelbase_m: Distance between wheels in meters
        wheel_diameter_m: Real wheel diameter in meters
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Refined focal length in pixels, or None if estimation failed
    """
    # Average wheel radius in pixels
    avg_radius_px = (front_radius_px + rear_radius_px) / 2

    if avg_radius_px < 10:  # Too small, unreliable
        return None

    # Estimate focal length from wheel size
    # focal ≈ (pixel_size * depth) / real_size
    # We use a heuristic depth estimate based on typical car photography
    estimated_depth_m = 3.0  # Assume camera is ~3m away from car

    focal_estimate = (avg_radius_px * estimated_depth_m) / (wheel_diameter_m / 2)

    # Sanity check: focal length should be in reasonable range
    min_focal = image_width * 0.5
    max_focal = image_width * 2.0

    if min_focal <= focal_estimate <= max_focal:
        logger.info(f"Refined focal length: {focal_estimate:.0f}px (from wheel size)")
        return float(focal_estimate)
    else:
        logger.warning(f"Focal refinement out of range: {focal_estimate:.0f}px")
        return None


def estimate_camera_from_homography(
    front_wheel_bbox: Optional[Dict],
    rear_wheel_bbox: Optional[Dict],
    image_width: int,
    image_height: int,
    wheelbase_m: float = 2.7,
) -> Dict:
    """
    Alternative approach: Estimate camera from homography matrix.

    Computes homography from ground plane to image, then decomposes it
    into camera parameters.

    Args:
        front_wheel_bbox: Front wheel bounding box
        rear_wheel_bbox: Rear wheel bounding box
        image_width: Image width in pixels
        image_height: Image height in pixels
        wheelbase_m: Wheelbase in meters

    Returns:
        Camera calibration dictionary
    """
    if not front_wheel_bbox or not rear_wheel_bbox:
        return _default_camera_params(image_width, image_height)

    # Extract wheel centers
    rear_center_x = (rear_wheel_bbox["x1"] + rear_wheel_bbox["x2"]) / 2
    rear_center_y = (rear_wheel_bbox["y1"] + rear_wheel_bbox["y2"]) / 2
    rear_radius = (rear_wheel_bbox["width"] + rear_wheel_bbox["height"]) / 4

    front_center_x = (front_wheel_bbox["x1"] + front_wheel_bbox["x2"]) / 2
    front_center_y = (front_wheel_bbox["y1"] + front_wheel_bbox["y2"]) / 2
    front_radius = (front_wheel_bbox["width"] + front_wheel_bbox["height"]) / 4

    # Define 2D ground plane points (X, Z) - ignoring Y since ground is Y=0
    world_pts_2d = np.array([
        [0, 0],               # Rear wheel
        [-wheelbase_m, 0],    # Front wheel
        [0, 0.5],             # Rear + 0.5m sideways
        [-wheelbase_m, 0.5],  # Front + 0.5m sideways
    ], dtype=np.float32)

    # Corresponding image points
    image_pts = np.array([
        [rear_center_x, rear_center_y + rear_radius],
        [front_center_x, front_center_y + front_radius],
        [rear_center_x, rear_center_y - rear_radius],
        [front_center_x, front_center_y - front_radius],
    ], dtype=np.float32)

    # Compute homography
    H, status = cv2.findHomography(world_pts_2d, image_pts, cv2.RANSAC, 5.0)

    if H is None:
        logger.warning("Homography computation failed")
        return _default_camera_params(image_width, image_height)

    # Camera intrinsic matrix guess
    focal_guess = max(image_width, image_height)
    K = np.array([
        [focal_guess, 0, image_width / 2],
        [0, focal_guess, image_height / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    # Decompose homography
    num_solutions, rvecs, tvecs, normals = cv2.decomposeHomographyMat(H, K)

    # Choose best solution (camera should be in front of plane)
    best_idx = 0
    for i in range(num_solutions):
        if tvecs[i][2] > 0:  # Positive Z = camera in front
            best_idx = i
            break

    rvec = rvecs[best_idx]
    tvec = tvecs[best_idx]

    # Calculate FOV
    focal_length = K[0, 0]
    fov = 2 * np.arctan(image_height / (2 * focal_length)) * (180 / np.pi)
    camera_height = abs(tvec[1][0])

    return {
        'intrinsic_matrix': K.tolist(),
        'rotation_vector': rvec.flatten().tolist(),
        'translation_vector': tvec.flatten().tolist(),
        'fov': float(fov),
        'camera_height': float(camera_height),
        'focal_length': float(focal_length),
        'method': 'homography'
    }


def _default_camera_params(width: int, height: int) -> Dict:
    """
    Fallback camera parameters when calibration fails.

    Provides reasonable defaults based on typical mobile phone camera specs.
    """
    focal_length = width  # Typical for smartphone cameras
    fov = 2 * np.arctan(height / (2 * focal_length)) * (180 / np.pi)

    return {
        'intrinsic_matrix': [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ],
        'rotation_vector': [0.0, 0.0, 0.0],
        'translation_vector': [0.0, 1.5, 3.0],  # 1.5m high, 3m back
        'fov': float(fov),
        'camera_height': 1.5,
        'focal_length': float(focal_length),
        'method': 'default'
    }


def opencv_to_threejs_camera(
    camera_params: Dict,
    image_width: int,
    image_height: int
) -> Dict:
    """
    Convert OpenCV camera parameters to Three.js camera configuration.

    Handles coordinate system conversion:
    - OpenCV: X=right, Y=down, Z=forward
    - Three.js: X=right, Y=up, Z=toward camera

    Args:
        camera_params: Output from estimate_camera_from_wheels()
        image_width: Image width in pixels
        image_height: Image height in pixels

    Returns:
        Dictionary with Three.js camera parameters
    """
    fov = camera_params['fov']
    rvec = np.array(camera_params['rotation_vector'])
    tvec = np.array(camera_params['translation_vector'])

    # Convert rotation vector to quaternion
    R, _ = cv2.Rodrigues(rvec)
    quaternion = _rotation_matrix_to_quaternion(R)

    # Convert translation (OpenCV to Three.js coordinate system)
    # OpenCV: Y-down, Z-forward
    # Three.js: Y-up, Z-toward-camera
    position = [
        tvec[0],   # X: same
        -tvec[1],  # Y: flip
        -tvec[2]   # Z: flip
    ]

    return {
        'fov': fov,
        'aspect': image_width / image_height,
        'near': 0.01,
        'far': 100.0,
        'position': position,
        'quaternion': {
            'x': quaternion[0],
            'y': quaternion[1],
            'z': quaternion[2],
            'w': quaternion[3]
        }
    }


def _rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return (x, y, z, w)
