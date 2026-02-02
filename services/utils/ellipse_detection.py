"""
Ellipse detection for tire geometry analysis.

This module uses OpenCV to detect ellipses within tire bounding boxes.
The ellipse parameters (especially the rotation angle) help determine
the 3D viewing angle of the wheel.

Key insight: A circular wheel viewed at an angle appears as an ellipse.
- The ratio of minor/major axis tells us the viewing angle
- The rotation angle of the ellipse tells us the tilt of the axle

Tire validation: Real tires have a dark rubber ring around them.
We validate detected ellipses by checking for dark pixels along the ellipse perimeter.
"""

import math
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Threshold for "dark" pixels (tire rubber is typically very dark)
DARK_THRESHOLD = 80  # Pixels with brightness below this are considered dark
DARK_RING_RATIO_THRESHOLD = 0.4  # At least 40% of perimeter should be dark


def sample_ellipse_perimeter(
    img_array: np.ndarray,
    center: Tuple[float, float],
    axes: Tuple[float, float],
    angle_deg: float,
    num_samples: int = 36,
    offset_x: int = 0,
    offset_y: int = 0
) -> List[int]:
    """
    Sample pixel brightness values along an ellipse perimeter.
    
    Args:
        img_array: Grayscale image array
        center: Ellipse center (in ROI coordinates)
        axes: (major_axis, minor_axis) semi-axes
        angle_deg: Rotation angle of ellipse in degrees
        num_samples: Number of points to sample around the ellipse
        offset_x, offset_y: Offset to convert ROI coords to image coords
    
    Returns:
        List of brightness values (0-255) at sampled points
    """
    major, minor = axes
    angle_rad = math.radians(angle_deg)
    h, w = img_array.shape[:2]
    
    brightness_values = []
    
    for i in range(num_samples):
        t = 2 * math.pi * i / num_samples
        
        # Parametric ellipse point
        x = major * math.cos(t)
        y = minor * math.sin(t)
        
        # Rotate by ellipse angle
        rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        
        # Translate to center and add offset
        px = int(center[0] + rx)
        py = int(center[1] + ry)
        
        # Check bounds
        if 0 <= px < w and 0 <= py < h:
            brightness_values.append(int(img_array[py, px]))
    
    return brightness_values


def has_dark_ring(
    img_array: np.ndarray,
    ellipse: Tuple,
    offset_x: int = 0,
    offset_y: int = 0,
    dark_threshold: int = DARK_THRESHOLD,
    min_dark_ratio: float = DARK_RING_RATIO_THRESHOLD
) -> Tuple[bool, float, float]:
    """
    Check if an ellipse has a dark ring around it (indicating a tire).
    
    Args:
        img_array: Grayscale image array (ROI)
        ellipse: OpenCV ellipse tuple ((cx, cy), (w, h), angle)
        offset_x, offset_y: Offset to convert ROI coords to full image coords
        dark_threshold: Brightness threshold for "dark" pixels
        min_dark_ratio: Minimum ratio of dark pixels required
    
    Returns:
        (is_valid, dark_ratio, avg_brightness) tuple
    """
    center, axes, angle = ellipse
    major = max(axes[0], axes[1]) / 2
    minor = min(axes[0], axes[1]) / 2
    
    # Sample points along the ellipse perimeter
    brightness_values = sample_ellipse_perimeter(
        img_array, center, (major, minor), angle,
        num_samples=36, offset_x=offset_x, offset_y=offset_y
    )
    
    if not brightness_values:
        return False, 0.0, 255.0
    
    # Count dark pixels
    dark_count = sum(1 for b in brightness_values if b < dark_threshold)
    dark_ratio = dark_count / len(brightness_values)
    avg_brightness = sum(brightness_values) / len(brightness_values)
    
    is_valid = dark_ratio >= min_dark_ratio
    
    return is_valid, dark_ratio, avg_brightness


def detect_tire_ellipse(
    image: Image.Image,
    wheel_bbox: Dict[str, int],
    padding: int = 10
) -> Optional[Dict[str, Any]]:
    """
    Detect the ellipse of a tire within its bounding box using OpenCV.
    
    Validates that the detected ellipse has a dark ring around it (tire rubber).
    This helps distinguish the actual tire from the wheel well.
    
    Args:
        image: PIL Image
        wheel_bbox: Bounding box of the wheel (x1, y1, x2, y2)
        padding: Extra pixels around bbox to include
    
    Returns:
        Dictionary with ellipse parameters:
        - center: (cx, cy) in image coordinates
        - axes: (major_axis, minor_axis) - semi-axes lengths
        - angle: rotation angle in degrees (0° = horizontal major axis)
        - axis_ratio: minor/major ratio (1.0 = circle, <1.0 = ellipse)
        - viewing_angle_deg: estimated angle from head-on view
        - confidence: 0.0-1.0 confidence in the detection
        - dark_ring_ratio: ratio of dark pixels on perimeter
        
        Returns None if detection fails.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed, cannot detect tire ellipse")
        return None
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        if img_array.shape[2] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_array
    
    # Crop to wheel region with padding
    x1 = max(0, wheel_bbox["x1"] - padding)
    y1 = max(0, wheel_bbox["y1"] - padding)
    x2 = min(image.width, wheel_bbox["x2"] + padding)
    y2 = min(image.height, wheel_bbox["y2"] + padding)
    
    roi_bgr = img_bgr[y1:y2, x1:x2]
    
    if roi_bgr.size == 0:
        logger.warning("Empty ROI for ellipse detection")
        return None
    
    # Convert to grayscale for edge detection and dark ring validation
    if len(roi_bgr.shape) == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection with Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Dilate edges to connect broken contours
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.debug("No contours found for ellipse detection")
        return _fallback_ellipse_from_bbox(wheel_bbox, x1, y1)
    
    # Collect all valid ellipse candidates with their scores
    candidates = []
    
    roi_center = ((x2 - x1) / 2, (y2 - y1) / 2)
    roi_size = max(x2 - x1, y2 - y1)
    
    for contour in contours:
        # Need at least 5 points to fit an ellipse
        if len(contour) < 5:
            continue
        
        try:
            # Fit ellipse to contour
            ellipse = cv2.fitEllipse(contour)
            center, axes, angle = ellipse
            
            # Validate ellipse parameters
            major_axis = max(axes[0], axes[1]) / 2
            minor_axis = min(axes[0], axes[1]) / 2
            
            if major_axis < 10 or minor_axis < 5:
                continue  # Too small
            
            # Check if ellipse is within ROI bounds
            if (center[0] < 0 or center[0] >= (x2 - x1) or 
                center[1] < 0 or center[1] >= (y2 - y1)):
                continue
            
            # === DARK RING VALIDATION ===
            # Check if the ellipse perimeter has dark pixels (tire rubber)
            is_dark, dark_ratio, avg_brightness = has_dark_ring(gray, ellipse)
            
            # Score based on multiple factors
            # 1. Size (larger is better, likely the tire)
            size_score = min((major_axis * minor_axis) / (roi_size * roi_size / 4), 1.0)
            
            # 2. Proximity to ROI center
            dist_to_center = math.sqrt(
                (center[0] - roi_center[0])**2 + 
                (center[1] - roi_center[1])**2
            )
            center_score = 1.0 - min(dist_to_center / roi_size, 1.0)
            
            # 3. Reasonable aspect ratio for a tire (not too elongated)
            axis_ratio = minor_axis / major_axis if major_axis > 0 else 0
            ratio_score = 1.0 if 0.3 < axis_ratio < 1.0 else 0.3
            
            # 4. Dark ring score (KEY: tires are dark!)
            dark_score = dark_ratio  # 0.0 to 1.0
            
            # Combined score - dark ring is important
            score = (size_score * 0.25 + 
                    center_score * 0.2 + 
                    ratio_score * 0.15 + 
                    dark_score * 0.4)  # 40% weight on dark ring!
            
            candidates.append({
                'ellipse': ellipse,
                'score': score,
                'dark_ratio': dark_ratio,
                'avg_brightness': avg_brightness,
                'is_dark': is_dark,
                'major_axis': major_axis,
                'minor_axis': minor_axis
            })
            
        except cv2.error:
            continue
    
    if not candidates:
        logger.debug("No valid ellipse candidates found, using bbox fallback")
        return _fallback_ellipse_from_bbox(wheel_bbox, x1, y1)
    
    # Sort by score (best first)
    candidates.sort(key=lambda c: c['score'], reverse=True)
    
    # Try to find a candidate with a dark ring
    best_candidate = None
    for cand in candidates:
        if cand['is_dark']:
            best_candidate = cand
            logger.debug(f"Selected ellipse with dark ring: dark_ratio={cand['dark_ratio']:.2f}, "
                        f"brightness={cand['avg_brightness']:.1f}")
            break
    
    # If no dark ring candidate, take the best overall but log a warning
    if best_candidate is None:
        best_candidate = candidates[0]
        logger.warning(f"No ellipse with dark ring found, using best candidate: "
                      f"dark_ratio={best_candidate['dark_ratio']:.2f}, "
                      f"brightness={best_candidate['avg_brightness']:.1f}")
    
    # Extract ellipse parameters from best candidate
    best_ellipse = best_candidate['ellipse']
    center, axes, angle = best_ellipse
    
    # Convert to image coordinates (add ROI offset)
    center_x = center[0] + x1
    center_y = center[1] + y1
    
    # OpenCV returns full axes (diameter), we want semi-axes (radius)
    major_axis = max(axes[0], axes[1]) / 2
    minor_axis = min(axes[0], axes[1]) / 2
    
    # Normalize angle: OpenCV angle is for the first axis
    # We want angle of major axis from horizontal
    if axes[0] < axes[1]:
        # First axis is minor, so add 90 to get major axis angle
        angle = (angle + 90) % 180
    
    # Calculate axis ratio and viewing angle
    axis_ratio = minor_axis / major_axis if major_axis > 0 else 1.0
    
    # Viewing angle: cos(θ) = minor/major for a circle viewed at angle θ
    # θ = 0° means head-on (circle), θ = 90° means edge-on (line)
    viewing_angle_rad = math.acos(min(axis_ratio, 1.0))
    viewing_angle_deg = math.degrees(viewing_angle_rad)
    
    return {
        "center": (center_x, center_y),
        "axes": (major_axis, minor_axis),
        "angle": angle,  # Angle of major axis from horizontal (degrees)
        "axis_ratio": axis_ratio,
        "viewing_angle_deg": viewing_angle_deg,
        "confidence": best_candidate['score'],
        "dark_ring_ratio": best_candidate['dark_ratio'],
        "avg_brightness": best_candidate['avg_brightness'],
        "has_dark_ring": best_candidate['is_dark'],
        "method": "contour_fit_dark_validated"
    }


def _fallback_ellipse_from_bbox(
    wheel_bbox: Dict[str, int],
    roi_x1: int = 0,
    roi_y1: int = 0
) -> Dict[str, Any]:
    """
    Create ellipse parameters from bounding box when detection fails.
    Assumes the bbox roughly matches the tire.
    """
    center_x = (wheel_bbox["x1"] + wheel_bbox["x2"]) / 2
    center_y = (wheel_bbox["y1"] + wheel_bbox["y2"]) / 2
    
    width = wheel_bbox["x2"] - wheel_bbox["x1"]
    height = wheel_bbox["y2"] - wheel_bbox["y1"]
    
    major_axis = max(width, height) / 2
    minor_axis = min(width, height) / 2
    
    # Angle: if wider than tall, major axis is horizontal (0°)
    # if taller than wide, major axis is vertical (90°)
    angle = 0 if width >= height else 90
    
    axis_ratio = minor_axis / major_axis if major_axis > 0 else 1.0
    viewing_angle_rad = math.acos(min(axis_ratio, 1.0))
    viewing_angle_deg = math.degrees(viewing_angle_rad)
    
    return {
        "center": (center_x, center_y),
        "axes": (major_axis, minor_axis),
        "angle": angle,
        "axis_ratio": axis_ratio,
        "viewing_angle_deg": viewing_angle_deg,
        "confidence": 0.3,  # Low confidence for bbox fallback
        "method": "bbox_fallback"
    }


def calculate_axle_angle_from_ellipse(
    ellipse: Dict[str, Any],
    wheel_to_wheel_angle_deg: float
) -> float:
    """
    Calculate the 3D axle angle from ellipse parameters.
    
    The ellipse's rotation angle, combined with the wheel-to-wheel line angle,
    gives us information about the viewing angle of the wheel/axle.
    
    Args:
        ellipse: Ellipse detection result
        wheel_to_wheel_angle_deg: Angle of wheel-to-wheel line from horizontal
    
    Returns:
        Estimated angle of the axle (X-axis) from horizontal in degrees.
        Positive = tilted up toward viewer, Negative = tilted away.
    """
    ellipse_angle = ellipse["angle"]
    axis_ratio = ellipse["axis_ratio"]
    
    # The ellipse major axis should be roughly perpendicular to the viewing direction
    # For a side-view of a car:
    # - If ellipse is nearly circular (ratio ~1), we're viewing nearly head-on
    # - If ellipse is elongated (ratio < 1), we're viewing at an angle
    
    # The ellipse angle tells us the tilt of the tire
    # A perfectly level tire has ellipse major axis horizontal (angle = 0)
    # Camber/road tilt causes the ellipse to rotate
    
    # Compute the difference between ellipse angle and wheel-to-wheel angle
    # This difference indicates how much the tire is rotated relative to the ground plane
    angle_diff = ellipse_angle - wheel_to_wheel_angle_deg
    
    # Normalize to [-90, 90] range
    while angle_diff > 90:
        angle_diff -= 180
    while angle_diff < -90:
        angle_diff += 180
    
    return angle_diff


def draw_detected_ellipse(
    draw,  # ImageDraw object
    ellipse: Dict[str, Any],
    color: Tuple[int, int, int] = (255, 0, 255),  # Magenta
    width: int = 2
):
    """
    Draw the detected ellipse on an image.
    
    Args:
        draw: PIL ImageDraw object
        ellipse: Ellipse detection result
        color: RGB color tuple
        width: Line width
    """
    if not ellipse:
        return
    
    center = ellipse["center"]
    major, minor = ellipse["axes"]
    angle_deg = ellipse["angle"]
    
    # PIL doesn't directly support rotated ellipses, so we draw it as a polygon
    # Generate points on the ellipse
    points = []
    angle_rad = math.radians(angle_deg)
    
    for i in range(36):  # 36 points = 10° increments
        t = 2 * math.pi * i / 36
        # Parametric ellipse
        x = major * math.cos(t)
        y = minor * math.sin(t)
        # Rotate by ellipse angle
        rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        # Translate to center
        points.append((center[0] + rx, center[1] + ry))
    
    # Draw as polygon outline
    points.append(points[0])  # Close the loop
    draw.line(points, fill=color, width=width)
    
    # Draw center point
    draw.ellipse(
        [center[0] - 3, center[1] - 3, center[0] + 3, center[1] + 3],
        fill=color
    )
    
    # Draw major axis line
    end1 = (
        center[0] + major * math.cos(angle_rad),
        center[1] + major * math.sin(angle_rad)
    )
    end2 = (
        center[0] - major * math.cos(angle_rad),
        center[1] - major * math.sin(angle_rad)
    )
    draw.line([end1, end2], fill=color, width=1)
