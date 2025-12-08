"""
Geometry utilities for estimating 3D wheel orientation from 2D detections.

This module calculates rotation matrices to transform standard basis vectors
[1,0,0], [0,1,0], [0,0,1] into the local coordinate frame of a detected wheel,
suitable for Three.js asset placement.

Coordinate System (Three.js convention):
- X: Right (positive = right side of screen/car)
- Y: Up (positive = up)
- Z: Forward (positive = towards camera, out of screen)

For a wheel viewed from the side:
- Local X-axis: Points into the wheel (axle direction)
- Local Y-axis: Points up (world up)
- Local Z-axis: Tangent to wheel, along car's forward direction
"""

import math
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def calculate_wheel_center(bbox: Dict[str, int]) -> Tuple[float, float]:
    """Calculate the center point of a wheel from its bounding box."""
    center_x = (bbox["x1"] + bbox["x2"]) / 2
    center_y = (bbox["y1"] + bbox["y2"]) / 2
    return center_x, center_y


def calculate_wheel_radius(bbox: Dict[str, int]) -> float:
    """
    Estimate wheel radius from bounding box.
    Uses average of width and height to account for elliptical appearance.
    """
    return (bbox["width"] + bbox["height"]) / 4


def estimate_viewing_angle(bbox: Dict[str, int]) -> float:
    """
    Estimate the viewing angle of the wheel based on its ellipse ratio.
    
    A wheel viewed straight-on appears circular (ratio = 1.0).
    A wheel viewed from the side appears as a narrow ellipse (ratio < 1.0).
    
    Returns:
        angle in radians (0 = viewing wheel face straight-on, π/2 = viewing from pure side)
    """
    width = bbox["width"]
    height = bbox["height"]
    
    if height == 0:
        return math.pi / 2
    
    # The ratio of width to height indicates the viewing angle
    # For a side view of a car, wheels appear roughly circular but slightly elliptical
    ratio = width / height
    
    # Clamp ratio to valid range [0, 1] for arccos
    # If ratio > 1, wheel is wider than tall (unusual, might be measurement noise)
    ratio = min(max(ratio, 0.1), 1.0)
    
    # angle = arccos(ratio) gives us the angle from straight-on view
    angle = math.acos(ratio)
    
    return angle


def identify_front_rear_wheels(wheels: List[Dict[str, Any]], car_bbox: Dict[str, int]) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Identify which wheel is front and which is rear based on position.
    
    Assumptions:
    - Side view of car
    - Maximum 2 visible wheels (front and rear on visible side)
    - Rear wheel is typically closer to the back of the car bounding box
    
    Returns:
        Dictionary with 'front' and 'rear' wheel detections (or None if not found)
    """
    if not wheels:
        return {"front": None, "rear": None}
    
    if len(wheels) == 1:
        # Single wheel - determine if front or rear based on position
        wheel = wheels[0]
        center_x, _ = calculate_wheel_center(wheel["bbox"])
        car_center_x = (car_bbox["x1"] + car_bbox["x2"]) / 2
        
        # If wheel is in front half of car, it's likely front wheel
        if center_x < car_center_x:
            return {"front": wheel, "rear": None}
        else:
            return {"front": None, "rear": wheel}
    
    # Multiple wheels - sort by x position
    sorted_wheels = sorted(wheels, key=lambda w: calculate_wheel_center(w["bbox"])[0])
    
    # The leftmost and rightmost wheels are our candidates
    # We need to determine car orientation to know which is front/rear
    
    # Use car bounding box to infer direction
    # Typically, if we're viewing the left side of the car, 
    # the left wheel is front and right wheel is rear
    # For right side view, it's reversed
    
    # Heuristic: Use the car's aspect ratio and wheel positions
    car_width = car_bbox["width"]
    car_center_x = (car_bbox["x1"] + car_bbox["x2"]) / 2
    
    left_wheel = sorted_wheels[0]
    right_wheel = sorted_wheels[-1]
    
    left_center_x, _ = calculate_wheel_center(left_wheel["bbox"])
    right_center_x, _ = calculate_wheel_center(right_wheel["bbox"])
    
    # Determine which side of the car we're viewing
    # If left wheel is closer to left edge of car bbox, we're viewing left side
    left_dist_to_car_left = left_center_x - car_bbox["x1"]
    right_dist_to_car_right = car_bbox["x2"] - right_center_x
    
    # For a typical side view, front wheel is at the "front" of the car
    # We'll use a simple heuristic: smaller x = front for left-side view
    # This can be refined with more sophisticated pose estimation
    
    return {
        "front": left_wheel,
        "rear": right_wheel
    }


def estimate_car_direction(wheels: List[Dict[str, Any]], car_bbox: Dict[str, int]) -> Tuple[float, float]:
    """
    Estimate the direction the car is facing in 2D image space.
    
    Returns:
        Normalized 2D direction vector (dx, dy) pointing in car's forward direction
    """
    wheel_positions = identify_front_rear_wheels(wheels, car_bbox)
    
    front = wheel_positions["front"]
    rear = wheel_positions["rear"]
    
    if front and rear:
        front_center = calculate_wheel_center(front["bbox"])
        rear_center = calculate_wheel_center(rear["bbox"])
        
        # Direction from rear to front
        dx = front_center[0] - rear_center[0]
        dy = front_center[1] - rear_center[1]
        
        # Normalize
        length = math.sqrt(dx*dx + dy*dy)
        if length > 0:
            return (dx / length, dy / length)
    
    # Default: assume car faces left (common in side-view photos)
    return (-1.0, 0.0)


def estimate_ground_plane_angle(wheels: List[Dict[str, Any]]) -> float:
    """
    Estimate the angle of the ground plane from wheel bottom positions.
    
    Returns:
        Angle in radians (0 = horizontal ground, positive = tilted)
    """
    if len(wheels) < 2:
        return 0.0
    
    # Get bottom center of each wheel
    wheel_bottoms = []
    for wheel in wheels:
        center_x = (wheel["bbox"]["x1"] + wheel["bbox"]["x2"]) / 2
        bottom_y = wheel["bbox"]["y2"]
        wheel_bottoms.append((center_x, bottom_y))
    
    # Sort by x position
    wheel_bottoms.sort(key=lambda p: p[0])
    
    # Calculate angle between leftmost and rightmost wheel bottoms
    left = wheel_bottoms[0]
    right = wheel_bottoms[-1]
    
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    
    if dx == 0:
        return 0.0
    
    return math.atan2(dy, dx)


def calculate_rotation_matrix(
    wheels: List[Dict[str, Any]], 
    car_bbox: Dict[str, int],
    image_width: int,
    image_height: int,
    target_wheel: str = "rear"
) -> Dict[str, Any]:
    """
    Calculate the rotation matrix to transform standard basis vectors to wheel local frame.
    
    This computes the columns of a 3x3 rotation matrix:
    - Column 1 (X-axis/Right): Points along the wheel's axle (into the car)
    - Column 2 (Y-axis/Up): Points upward in world space
    - Column 3 (Z-axis/Forward): Tangent to wheel, along car's direction
    
    Args:
        wheels: List of detected wheels
        car_bbox: Bounding box of the car
        image_width: Width of the image in pixels
        image_height: Height of the image in pixels
        target_wheel: Which wheel to calculate for ("front" or "rear")
    
    Returns:
        Dictionary containing:
        - rotation_matrix: 3x3 rotation matrix as nested lists
        - basis_vectors: The three basis vectors (x, y, z axes)
        - euler_angles: Rotation as Euler angles (radians) for Three.js
        - quaternion: Rotation as quaternion for Three.js
        - metadata: Additional geometric information
    """
    # Identify wheels
    wheel_positions = identify_front_rear_wheels(wheels, car_bbox)
    target = wheel_positions.get(target_wheel)
    
    if not target:
        logger.warning(f"No {target_wheel} wheel found, using first available wheel")
        target = wheels[0] if wheels else None
    
    if not target:
        # Return identity matrix if no wheel found
        return _create_identity_result()
    
    # Get geometric measurements
    viewing_angle = estimate_viewing_angle(target["bbox"])
    car_direction = estimate_car_direction(wheels, car_bbox)
    ground_angle = estimate_ground_plane_angle(wheels)
    
    # Determine which side of the car we're viewing
    # This affects the direction of the X-axis (into car vs out of car)
    wheel_center_x, _ = calculate_wheel_center(target["bbox"])
    car_center_x = (car_bbox["x1"] + car_bbox["x2"]) / 2
    
    # Heuristic: if wheel is right of car center in image, we're likely viewing left side
    viewing_left_side = wheel_center_x > car_center_x * 0.8  # Some tolerance
    
    # Build rotation matrix
    # 
    # Y-axis (Up): Always points up, adjusted for ground plane tilt
    y_axis = np.array([
        -math.sin(ground_angle),
        math.cos(ground_angle),
        0.0
    ])
    
    # Z-axis (Forward): Along car's direction of travel
    # Project car_direction into 3D, accounting for perspective
    # For a side view, forward is roughly along the image X-axis
    z_axis = np.array([
        car_direction[0],
        car_direction[1] * 0.1,  # Minimal Y component for side view
        -0.1  # Slight depth component (car going slightly away)
    ])
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # X-axis (Right/Axle): Perpendicular to Y and Z
    # For left side view: points into screen (negative Z in camera space)
    # For right side view: points out of screen (positive Z in camera space)
    x_axis = np.cross(y_axis, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Flip X-axis based on which side we're viewing
    if viewing_left_side:
        x_axis = -x_axis
    
    # Re-orthogonalize Z to ensure perfect orthonormality
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Construct rotation matrix (columns are the basis vectors)
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # Convert to Euler angles (XYZ order, which Three.js uses by default)
    euler_angles = rotation_matrix_to_euler(rotation_matrix)
    
    # Convert to quaternion
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    
    return {
        "rotation_matrix": rotation_matrix.tolist(),
        "basis_vectors": {
            "x_axis": x_axis.tolist(),  # Right / Axle direction
            "y_axis": y_axis.tolist(),  # Up
            "z_axis": z_axis.tolist()   # Forward
        },
        "euler_angles": {
            "x": euler_angles[0],
            "y": euler_angles[1],
            "z": euler_angles[2],
            "order": "XYZ"
        },
        "quaternion": {
            "x": quaternion[0],
            "y": quaternion[1],
            "z": quaternion[2],
            "w": quaternion[3]
        },
        "metadata": {
            "viewing_angle_rad": viewing_angle,
            "viewing_angle_deg": math.degrees(viewing_angle),
            "ground_angle_rad": ground_angle,
            "ground_angle_deg": math.degrees(ground_angle),
            "car_direction_2d": list(car_direction),
            "viewing_side": "left" if viewing_left_side else "right",
            "target_wheel": target_wheel
        }
    }


def calculate_wheel_transform(
    wheel: Dict[str, Any],
    car_bbox: Dict[str, int],
    image_width: int,
    image_height: int,
    all_wheels: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Calculate complete transform (position + rotation) for a wheel in normalized coordinates.
    
    Position is normalized to [-1, 1] range for easy Three.js integration.
    Origin is at image center.
    
    Args:
        wheel: The target wheel detection
        car_bbox: Bounding box of the car
        image_width: Image width in pixels
        image_height: Image height in pixels
        all_wheels: All detected wheels (for context)
    
    Returns:
        Complete transform information for Three.js
    """
    # Calculate center position in normalized coordinates
    center_x, center_y = calculate_wheel_center(wheel["bbox"])
    
    # Normalize to [-1, 1] range (image center = origin)
    normalized_x = (center_x / image_width) * 2 - 1
    normalized_y = -((center_y / image_height) * 2 - 1)  # Flip Y for Three.js
    
    # Estimate depth (Z) based on wheel size
    # Larger wheels = closer to camera = larger Z
    wheel_radius = calculate_wheel_radius(wheel["bbox"])
    avg_dimension = (image_width + image_height) / 2
    # Normalize radius relative to image size, scale to reasonable depth range
    normalized_z = (wheel_radius / avg_dimension) * 2 - 0.5
    
    # Calculate rotation
    rotation_data = calculate_rotation_matrix(
        all_wheels, car_bbox, image_width, image_height, 
        target_wheel="rear"  # We'll determine actual position separately
    )
    
    return {
        "position": {
            "x": normalized_x,
            "y": normalized_y,
            "z": normalized_z,
            "pixel_x": center_x,
            "pixel_y": center_y
        },
        "rotation": rotation_data,
        "scale": {
            "uniform": wheel_radius / avg_dimension * 4,  # Scale factor for 3D asset
            "radius_pixels": wheel_radius
        },
        "bounding_box": wheel["bbox"],
        "confidence": wheel.get("confidence", 0)
    }


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (XYZ order).
    
    Returns:
        Tuple of (x, y, z) rotation angles in radians
    """
    # Handle gimbal lock cases
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    
    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    
    return (x, y, z)


def rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert rotation matrix to quaternion.
    
    Returns:
        Tuple of (x, y, z, w) quaternion components
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return (x, y, z, w)


def _create_identity_result() -> Dict[str, Any]:
    """Create a default identity transform result."""
    return {
        "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "basis_vectors": {
            "x_axis": [1, 0, 0],
            "y_axis": [0, 1, 0],
            "z_axis": [0, 0, 1]
        },
        "euler_angles": {
            "x": 0,
            "y": 0,
            "z": 0,
            "order": "XYZ"
        },
        "quaternion": {
            "x": 0,
            "y": 0,
            "z": 0,
            "w": 1
        },
        "metadata": {
            "viewing_angle_rad": 0,
            "viewing_angle_deg": 0,
            "ground_angle_rad": 0,
            "ground_angle_deg": 0,
            "car_direction_2d": [-1, 0],
            "viewing_side": "unknown",
            "target_wheel": "unknown",
            "warning": "No wheel detected, returning identity transform"
        }
    }


def enrich_detection_with_geometry(
    detection_results: Dict[str, Any],
    image_width: int,
    image_height: int
) -> Dict[str, Any]:
    """
    Enrich detection results with 3D geometry information for each car.
    
    This is the main function to call after running detection.
    It adds wheel transforms and rotation matrices to each detected car.
    
    Args:
        detection_results: Results from CarWheelDetector.detect_cars_and_wheels()
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        Enriched detection results with geometry data
    """
    enriched = detection_results.copy()
    enriched["image_dimensions"] = {
        "width": image_width,
        "height": image_height
    }
    
    for car_result in enriched.get("results", []):
        car_bbox = car_result["car"]["bbox"]
        wheels = car_result.get("wheels", [])
        
        # Identify front and rear wheels
        wheel_positions = identify_front_rear_wheels(wheels, car_bbox)
        car_result["wheel_positions"] = {
            "front": wheel_positions["front"],
            "rear": wheel_positions["rear"]
        }
        
        # Calculate transform for rear wheel (primary target for 3D asset)
        rear_wheel = wheel_positions.get("rear")
        if rear_wheel:
            car_result["rear_wheel_transform"] = calculate_wheel_transform(
                rear_wheel, car_bbox, image_width, image_height, wheels
            )
        else:
            car_result["rear_wheel_transform"] = None
        
        # Calculate transform for front wheel as well (optional use)
        front_wheel = wheel_positions.get("front")
        if front_wheel:
            car_result["front_wheel_transform"] = calculate_wheel_transform(
                front_wheel, car_bbox, image_width, image_height, wheels
            )
        else:
            car_result["front_wheel_transform"] = None
        
        # Add overall car geometry info
        car_result["car_geometry"] = {
            "direction_2d": list(estimate_car_direction(wheels, car_bbox)),
            "ground_angle_deg": math.degrees(estimate_ground_plane_angle(wheels)),
            "viewing_side": "left" if wheel_positions.get("rear") and 
                calculate_wheel_center(wheel_positions["rear"]["bbox"])[0] > 
                (car_bbox["x1"] + car_bbox["x2"]) / 2 * 0.8 else "right"
        }
    
    return enriched
