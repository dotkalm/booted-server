"""
Visualization utilities for drawing detected geometry on images.

This module provides functions to visualize:
- Bounding boxes for cars and wheels
- 3D coordinate axes projected onto 2D image
- Wheel centers and orientations
- Ground plane indicators
- Hough transform line detection
"""

import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


# Color scheme (RGB)
COLORS = {
    "car_bbox": (0, 255, 0),        # Green
    "wheel_bbox": (255, 165, 0),    # Orange
    "front_wheel": (0, 191, 255),   # Deep sky blue
    "rear_wheel": (255, 0, 128),    # Hot pink
    "x_axis": (255, 0, 0),          # Red (right/axle)
    "y_axis": (0, 255, 0),          # Green (up)
    "z_axis": (0, 0, 255),          # Blue (forward)
    "hough_lines": (255, 0, 255),   # Magenta - detected Hough lines (car body color match)
    "wheel_to_wheel": (255, 165, 0), # Orange - wheel-to-wheel line
    "center_point": (255, 255, 0),  # Yellow
    "ground_line": (128, 128, 128), # Gray
    "text": (255, 255, 255),        # White
    "text_bg": (0, 0, 0),           # Black
}


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    color: Tuple[int, int, int],
    width: int = 3,
    arrow_size: int = 10
):
    """Draw an arrow from start to end point."""
    # Draw main line
    draw.line([start, end], fill=color, width=width)
    
    # Calculate arrow head
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx*dx + dy*dy)
    
    if length < 1:
        return
    
    # Normalize direction
    dx /= length
    dy /= length
    
    # Arrow head points
    angle = math.pi / 6  # 30 degrees
    arrow_len = arrow_size
    
    # Left arrow point
    left_x = end[0] - arrow_len * (dx * math.cos(angle) - dy * math.sin(angle))
    left_y = end[1] - arrow_len * (dy * math.cos(angle) + dx * math.sin(angle))
    
    # Right arrow point
    right_x = end[0] - arrow_len * (dx * math.cos(angle) + dy * math.sin(angle))
    right_y = end[1] - arrow_len * (dy * math.cos(angle) - dx * math.sin(angle))
    
    # Draw arrow head
    draw.polygon([end, (left_x, left_y), (right_x, right_y)], fill=color)


def draw_3d_axes(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    basis_vectors: Dict[str, List[float]],
    axis_length: float = 60,
    line_width: int = 3,
    wheel_to_wheel_2d: Optional[Tuple[float, float]] = None
):
    """
    Draw 3D coordinate axes projected onto 2D image.
    
    Args:
        draw: PIL ImageDraw object
        center: Center point (pixel coordinates)
        basis_vectors: Dictionary with x_axis, y_axis, z_axis vectors
        axis_length: Length of each axis in pixels
        line_width: Width of axis lines
        wheel_to_wheel_2d: If provided, use this 2D direction for Z-axis (more accurate)
    """
    # Project 3D basis vectors to 2D
    x_axis = basis_vectors["x_axis"]
    y_axis = basis_vectors["y_axis"]
    z_axis = basis_vectors["z_axis"]
    
    def project_vector(vec: List[float]) -> Tuple[float, float]:
        """Project 3D vector to 2D screen coordinates."""
        # For X and Y axes: use simple projection with Z influence for depth
        proj_x = vec[0] + vec[2] * 0.3
        proj_y = -vec[1] + vec[2] * 0.2  # Y is inverted in image coords
        return (proj_x, proj_y)
    
    def scale_vector(v: Tuple[float, float], length: float) -> Tuple[float, float]:
        mag = math.sqrt(v[0]**2 + v[1]**2)
        if mag < 0.001:
            return (0, 0)
        return (v[0] / mag * length, v[1] / mag * length)
    
    # Calculate end points for X and Y axes (projected from 3D)
    x_proj = project_vector(x_axis)
    y_proj = project_vector(y_axis)
    
    x_end = scale_vector(x_proj, axis_length)
    y_end = scale_vector(y_proj, axis_length)
    
    # For Z-axis: use the actual 2D wheel-to-wheel direction if provided
    # This ensures Z lies exactly on the wheel-to-wheel line
    if wheel_to_wheel_2d:
        # wheel_to_wheel_2d is in image coordinates (Y not inverted)
        z_end = scale_vector(wheel_to_wheel_2d, axis_length)
    else:
        z_proj = project_vector(z_axis)
        z_end = scale_vector(z_proj, axis_length)
    
    # Draw axes as arrows
    # X-axis (Red) - Axle direction
    draw_arrow(
        draw,
        center,
        (center[0] + x_end[0], center[1] + x_end[1]),
        COLORS["x_axis"],
        width=line_width
    )
    
    # Y-axis (Green) - Up direction
    draw_arrow(
        draw,
        center,
        (center[0] + y_end[0], center[1] + y_end[1]),
        COLORS["y_axis"],
        width=line_width
    )
    
    # Z-axis (Blue) - Forward direction
    draw_arrow(
        draw,
        center,
        (center[0] + z_end[0], center[1] + z_end[1]),
        COLORS["z_axis"],
        width=line_width
    )
    
    # Draw axis labels
    label_offset = 15
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()
    
    draw.text(
        (center[0] + x_end[0] + label_offset, center[1] + x_end[1]),
        "X",
        fill=COLORS["x_axis"],
        font=font
    )
    draw.text(
        (center[0] + y_end[0], center[1] + y_end[1] - label_offset),
        "Y",
        fill=COLORS["y_axis"],
        font=font
    )
    draw.text(
        (center[0] + z_end[0] + label_offset, center[1] + z_end[1]),
        "Z",
        fill=COLORS["z_axis"],
        font=font
    )


def draw_wheel_to_wheel_line(
    draw: ImageDraw.ImageDraw,
    front_center: Tuple[float, float],
    rear_center: Tuple[float, float],
    color: Tuple[int, int, int] = COLORS["wheel_to_wheel"],
    line_width: int = 3
):
    """Draw a line connecting the centers of front and rear wheels."""
    draw.line([front_center, rear_center], fill=color, width=line_width)
    
    # Draw small circles at each wheel center
    for center in [front_center, rear_center]:
        radius = 5
        draw.ellipse(
            [center[0] - radius, center[1] - radius, 
             center[0] + radius, center[1] + radius],
            fill=color,
            outline=color
        )


def get_dominant_color(
    image: Image.Image,
    bbox: Dict[str, int],
    sample_ratio: float = 0.1
) -> Tuple[int, int, int]:
    """
    Get the most common color in a bounding box region.
    Uses histogram binning to find the dominant color.
    
    Args:
        image: PIL Image
        bbox: Bounding box dict with x1, y1, x2, y2
        sample_ratio: Ratio of pixels to sample (for performance)
    
    Returns:
        (R, G, B) tuple of the dominant color
    """
    # Crop to bbox
    region = image.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
    
    # Resize for faster processing
    max_dim = 100
    w, h = region.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        region = region.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    
    # Quantize colors to reduce noise
    quantized = region.quantize(colors=16, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    color_counts = quantized.getcolors()
    
    if not color_counts or not palette:
        return (128, 128, 128)  # Default gray
    
    # Find the most common color
    most_common_idx = max(color_counts, key=lambda x: x[0])[1]
    r = palette[most_common_idx * 3]
    g = palette[most_common_idx * 3 + 1]
    b = palette[most_common_idx * 3 + 2]
    
    return (r, g, b)


def color_distance(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    """
    Calculate Euclidean distance between two colors in RGB space.
    """
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2)


def get_line_average_color(
    image: Image.Image,
    start: Tuple[int, int],
    end: Tuple[int, int],
    num_samples: int = 10
) -> Tuple[int, int, int]:
    """
    Get the average color along a line by sampling points.
    
    Args:
        image: PIL Image
        start: Start point (x, y)
        end: End point (x, y)
        num_samples: Number of points to sample along the line
    
    Returns:
        (R, G, B) tuple of the average color
    """
    pixels = []
    img_rgb = image.convert("RGB")
    w, h = image.size
    
    for i in range(num_samples):
        t = i / (num_samples - 1) if num_samples > 1 else 0.5
        x = int(start[0] + t * (end[0] - start[0]))
        y = int(start[1] + t * (end[1] - start[1]))
        
        # Clamp to image bounds
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        
        pixels.append(img_rgb.getpixel((x, y)))
    
    if not pixels:
        return (128, 128, 128)
    
    avg_r = sum(p[0] for p in pixels) // len(pixels)
    avg_g = sum(p[1] for p in pixels) // len(pixels)
    avg_b = sum(p[2] for p in pixels) // len(pixels)
    
    return (avg_r, avg_g, avg_b)


def detect_hough_lines(
    image: Image.Image,
    region_bbox: Dict[str, int],
    color_tolerance: float = 60.0,
    vertical_threshold_deg: float = 70.0
) -> List[Tuple[Tuple[int, int], Tuple[int, int], float]]:
    """
    Detect lines using Hough transform that match the car body color.
    Filters out near-vertical lines and lines whose color doesn't match
    the dominant color of the car bounding box.
    
    Args:
        image: PIL Image
        region_bbox: Bounding box to search within (car bbox)
        color_tolerance: Max color distance (Euclidean in RGB space) to accept
        vertical_threshold_deg: Reject lines steeper than this angle from horizontal
    
    Returns:
        List of (start_point, end_point, angle_deg) tuples for detected lines
    """
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV not installed, skipping Hough transform")
        return []
    
    # Get the dominant color of the car body
    car_color = get_dominant_color(image, region_bbox)
    logger.info(f"Car dominant color: RGB{car_color}")
    
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Crop to region of interest (car bounding box with some padding)
    x1 = max(0, region_bbox["x1"] - 20)
    y1 = max(0, region_bbox["y1"] - 20)
    x2 = min(image.width, region_bbox["x2"] + 20)
    y2 = min(image.height, region_bbox["y2"] + 20)
    
    roi = gray[y1:y2, x1:x2]
    
    # Edge detection
    edges = cv2.Canny(roi, 50, 150, apertureSize=3)
    
    # Hough Line Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=50,
        minLineLength=30,
        maxLineGap=10
    )
    
    if lines is None:
        return []
    
    matching_lines = []
    
    for line in lines:
        lx1, ly1, lx2, ly2 = line[0]
        
        # Calculate line angle from horizontal
        dx = lx2 - lx1
        dy = ly2 - ly1
        line_angle_rad = math.atan2(abs(dy), abs(dx))  # Always positive angle from horizontal
        line_angle_deg = math.degrees(line_angle_rad)
        
        # Skip near-vertical lines (more than vertical_threshold_deg from horizontal)
        if line_angle_deg > vertical_threshold_deg:
            continue
        
        # Convert to full image coordinates
        start = (lx1 + x1, ly1 + y1)
        end = (lx2 + x1, ly2 + y1)
        
        # Get the average color along this line
        line_color = get_line_average_color(image, start, end, num_samples=15)
        
        # Check if line color is within tolerance of car color
        dist = color_distance(line_color, car_color)
        
        if dist <= color_tolerance:
            # Return full angle for display (not just from horizontal)
            full_angle = math.degrees(math.atan2(dy, dx))
            matching_lines.append((start, end, full_angle))
            logger.debug(f"Line accepted: color RGB{line_color}, dist={dist:.1f}, angle={full_angle:.1f}°")
    
    logger.info(f"Hough: {len(lines)} total lines, {len(matching_lines)} matching car color")
    return matching_lines


def draw_hough_lines(
    draw: ImageDraw.ImageDraw,
    lines: List[Tuple[Tuple[int, int], Tuple[int, int], float]],
    color: Tuple[int, int, int] = COLORS["hough_lines"],
    line_width: int = 2
):
    """Draw detected Hough lines on the image."""
    for start, end, angle in lines:
        draw.line([start, end], fill=color, width=line_width)


def draw_wheel_center(
    draw: ImageDraw.ImageDraw,
    center: Tuple[float, float],
    radius: float,
    color: Tuple[int, int, int] = COLORS["center_point"],
    crosshair_size: int = 10
):
    """Draw a crosshair at the wheel center."""
    cx, cy = center
    
    # Draw crosshair
    draw.line([(cx - crosshair_size, cy), (cx + crosshair_size, cy)], fill=color, width=2)
    draw.line([(cx, cy - crosshair_size), (cx, cy + crosshair_size)], fill=color, width=2)
    
    # Draw center dot
    dot_radius = 4
    draw.ellipse(
        [cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius],
        fill=color
    )


def draw_bbox(
    draw: ImageDraw.ImageDraw,
    bbox: Dict[str, int],
    color: Tuple[int, int, int],
    label: Optional[str] = None,
    line_width: int = 2
):
    """Draw a bounding box with optional label."""
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    
    # Draw rectangle
    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
    
    # Draw label if provided
    if label:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        except:
            font = ImageFont.load_default()
        
        # Get text size
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw background for text
        padding = 2
        draw.rectangle(
            [x1, y1 - text_height - padding * 2, x1 + text_width + padding * 2, y1],
            fill=COLORS["text_bg"]
        )
        draw.text((x1 + padding, y1 - text_height - padding), label, fill=color, font=font)


def draw_ground_line(
    draw: ImageDraw.ImageDraw,
    wheels: List[Dict[str, Any]],
    image_width: int,
    color: Tuple[int, int, int] = COLORS["ground_line"],
    line_width: int = 1
):
    """Draw estimated ground line based on wheel positions."""
    if len(wheels) < 2:
        return
    
    # Get bottom center of each wheel
    points = []
    for wheel in wheels:
        cx = (wheel["bbox"]["x1"] + wheel["bbox"]["x2"]) / 2
        cy = wheel["bbox"]["y2"]  # Bottom of wheel
        points.append((cx, cy))
    
    points.sort(key=lambda p: p[0])
    
    if len(points) >= 2:
        # Extend line across image
        x1, y1 = points[0]
        x2, y2 = points[-1]
        
        if x2 != x1:
            slope = (y2 - y1) / (x2 - x1)
            
            # Extend to image edges
            start_y = y1 - slope * x1
            end_y = y1 + slope * (image_width - x1)
            
            draw.line(
                [(0, start_y), (image_width, end_y)],
                fill=color,
                width=line_width
            )


def draw_info_panel(
    draw: ImageDraw.ImageDraw,
    image_width: int,
    image_height: int,
    metadata: Dict[str, Any],
    position: str = "top-left"
):
    """Draw an info panel with metadata."""
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except:
        font = ImageFont.load_default()
    
    lines = [
        f"Viewing: {metadata.get('viewing_side', 'unknown')} side",
        f"View angle: {metadata.get('viewing_angle_deg', 0):.1f}°",
        f"Ground tilt: {metadata.get('ground_angle_deg', 0):.1f}°",
    ]
    
    # Calculate panel size
    line_height = 16
    padding = 8
    max_width = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
    panel_width = max_width + padding * 2
    panel_height = len(lines) * line_height + padding * 2
    
    # Position panel
    if position == "top-left":
        panel_x, panel_y = 10, 10
    elif position == "top-right":
        panel_x, panel_y = image_width - panel_width - 10, 10
    elif position == "bottom-left":
        panel_x, panel_y = 10, image_height - panel_height - 10
    else:
        panel_x, panel_y = image_width - panel_width - 10, image_height - panel_height - 10
    
    # Draw panel background
    draw.rectangle(
        [panel_x, panel_y, panel_x + panel_width, panel_y + panel_height],
        fill=(0, 0, 0, 180)
    )
    
    # Draw text
    for i, line in enumerate(lines):
        draw.text(
            (panel_x + padding, panel_y + padding + i * line_height),
            line,
            fill=COLORS["text"],
            font=font
        )


def draw_legend(
    draw: ImageDraw.ImageDraw,
    image_width: int,
    image_height: int
):
    """Draw a legend explaining the axes."""
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except:
        font = ImageFont.load_default()
    
    legend_items = [
        ("X (Red)", COLORS["x_axis"], "Axle direction"),
        ("Y (Green)", COLORS["y_axis"], "Up"),
        ("Z (Blue)", COLORS["z_axis"], "Forward"),
    ]
    
    line_height = 16
    padding = 8
    panel_width = 140
    panel_height = len(legend_items) * line_height + padding * 2
    
    # Position in bottom-right
    panel_x = image_width - panel_width - 10
    panel_y = image_height - panel_height - 10
    
    # Draw background
    draw.rectangle(
        [panel_x, panel_y, panel_x + panel_width, panel_y + panel_height],
        fill=(0, 0, 0, 200)
    )
    
    # Draw legend items
    for i, (label, color, desc) in enumerate(legend_items):
        y = panel_y + padding + i * line_height
        
        # Color swatch
        draw.rectangle(
            [panel_x + padding, y + 2, panel_x + padding + 12, y + 12],
            fill=color
        )
        
        # Text
        draw.text(
            (panel_x + padding + 18, y),
            f"{label}: {desc}",
            fill=COLORS["text"],
            font=font
        )


def visualize_detection(
    image: Image.Image,
    detection_data: Dict[str, Any],
    draw_car_bbox: bool = True,
    draw_wheel_bboxes: bool = True,
    draw_axes: bool = True,
    draw_centers: bool = True,
    draw_ground: bool = True,
    draw_legend_panel: bool = True,
    draw_info: bool = True,
    draw_hough: bool = True,
    hough_color_tolerance: float = 60.0,
    hough_vertical_threshold: float = 70.0,
    target_wheel: str = "rear",
    axis_length: int = 60
) -> Image.Image:
    """
    Draw detection visualization on an image.
    
    Args:
        image: PIL Image to draw on (will be copied)
        detection_data: Detection results from the API
        draw_car_bbox: Whether to draw car bounding box
        draw_wheel_bboxes: Whether to draw wheel bounding boxes
        draw_axes: Whether to draw 3D coordinate axes
        draw_centers: Whether to draw wheel center crosshairs
        draw_ground: Whether to draw ground line
        draw_legend_panel: Whether to draw axis legend
        draw_info: Whether to draw info panel
        draw_hough: Whether to detect and draw Hough lines matching car color
        hough_color_tolerance: Max color distance to car body color for Hough lines
        hough_vertical_threshold: Exclude lines steeper than this angle from horizontal
        target_wheel: Which wheel to draw axes on ("rear", "front", or "both")
        axis_length: Length of axis arrows in pixels
    
    Returns:
        New PIL Image with visualizations drawn
    """
    # Copy image to avoid modifying original
    viz_image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(viz_image)
    
    for car_result in detection_data.get("detections", []):
        # Draw car bounding box
        if draw_car_bbox:
            car = car_result.get("car", {})
            draw_bbox(
                draw,
                car["bbox"],
                COLORS["car_bbox"],
                label=f"Car {car_result['car_id']} ({car['confidence']:.0%})"
            )
        
        # Draw wheel bounding boxes
        if draw_wheel_bboxes:
            for wheel in car_result.get("wheels", []):
                draw_bbox(
                    draw,
                    wheel["bbox"],
                    COLORS["wheel_bbox"],
                    label=f"Wheel ({wheel['confidence']:.0%})"
                )
        
        # Draw ground line
        if draw_ground:
            draw_ground_line(
                draw,
                car_result.get("wheels", []),
                image.width
            )
        
        # Get wheel positions for wheel-to-wheel line
        wheel_positions = car_result.get("wheel_positions", {})
        front_wheel_data = wheel_positions.get("front")
        rear_wheel_data = wheel_positions.get("rear")
        
        front_center = None
        rear_center = None
        wheel_to_wheel_2d = None
        
        if front_wheel_data and rear_wheel_data:
            front_bbox = front_wheel_data["bbox"]
            rear_bbox = rear_wheel_data["bbox"]
            front_center = ((front_bbox["x1"] + front_bbox["x2"]) / 2,
                           (front_bbox["y1"] + front_bbox["y2"]) / 2)
            rear_center = ((rear_bbox["x1"] + rear_bbox["x2"]) / 2,
                          (rear_bbox["y1"] + rear_bbox["y2"]) / 2)
            
            # Draw the wheel-to-wheel line (orange)
            draw_wheel_to_wheel_line(draw, front_center, rear_center)
            
            # Calculate 2D direction for Z-axis (from rear to front = forward)
            dx = front_center[0] - rear_center[0]
            dy = front_center[1] - rear_center[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                wheel_to_wheel_2d = (dx / length, dy / length)
            
            # Detect and draw Hough lines that match car body color (not near-vertical)
            if draw_hough:
                car_bbox = car_result.get("car", {}).get("bbox", {})
                if car_bbox:
                    hough_lines = detect_hough_lines(
                        image,
                        car_bbox,
                        color_tolerance=hough_color_tolerance,
                        vertical_threshold_deg=hough_vertical_threshold
                    )
                    if hough_lines:
                        draw_hough_lines(draw, hough_lines, color=COLORS["hough_lines"], line_width=1)
                        logger.info(f"Found {len(hough_lines)} Hough lines matching car color")
        
        # Draw rear wheel visualization
        rear_transform = car_result.get("rear_wheel_transform")
        if rear_transform and target_wheel in ("rear", "both"):
            center = (rear_transform["position"]["pixel_x"], 
                     rear_transform["position"]["pixel_y"])
            
            if draw_centers:
                draw_wheel_center(
                    draw, center,
                    rear_transform["scale"]["radius_pixels"],
                    color=COLORS["rear_wheel"]
                )
            
            if draw_axes:
                draw_3d_axes(
                    draw, center,
                    rear_transform["rotation"]["basis_vectors"],
                    axis_length=axis_length,
                    wheel_to_wheel_2d=wheel_to_wheel_2d
                )
            
            if draw_info:
                draw_info_panel(
                    draw,
                    image.width,
                    image.height,
                    rear_transform["rotation"]["metadata"],
                    position="top-left"
                )
        
        # Draw front wheel visualization
        front_transform = car_result.get("front_wheel_transform")
        if front_transform and target_wheel in ("front", "both"):
            center = (front_transform["position"]["pixel_x"],
                     front_transform["position"]["pixel_y"])
            
            if draw_centers:
                draw_wheel_center(
                    draw, center,
                    front_transform["scale"]["radius_pixels"],
                    color=COLORS["front_wheel"]
                )
            
            if draw_axes and target_wheel == "front":
                draw_3d_axes(
                    draw, center,
                    front_transform["rotation"]["basis_vectors"],
                    axis_length=axis_length,
                    wheel_to_wheel_2d=wheel_to_wheel_2d
                )
    
    # Draw legend
    if draw_legend_panel:
        draw_legend(draw, image.width, image.height)
    
    return viz_image


def visualize_from_json(
    image_path: str,
    json_path: str,
    output_path: Optional[str] = None,
    **kwargs
) -> Image.Image:
    """
    Convenience function to visualize detection from JSON file.
    
    Args:
        image_path: Path to the source image
        json_path: Path to the detection JSON file
        output_path: Optional path to save the visualization
        **kwargs: Additional arguments passed to visualize_detection
    
    Returns:
        PIL Image with visualizations
    """
    import json
    
    image = Image.open(image_path)
    
    with open(json_path, 'r') as f:
        detection_data = json.load(f)
    
    viz = visualize_detection(image, detection_data, **kwargs)
    
    if output_path:
        viz.save(output_path)
        logger.info(f"Saved visualization to: {output_path}")
    
    return viz


# CLI entry point
if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) < 3:
        print("Usage: python visualization.py <image_path> <json_path> [output_path]")
        print("Example: python visualization.py spec/one.jpg spec/fixtures/one_detection.json spec/fixtures/one_viz.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    json_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    
    if output_path is None:
        # Default output path
        from pathlib import Path
        p = Path(image_path)
        output_path = str(p.parent / f"{p.stem}_visualization{p.suffix}")
    
    viz = visualize_from_json(image_path, json_path, output_path)
    print(f"✓ Visualization saved to: {output_path}")
