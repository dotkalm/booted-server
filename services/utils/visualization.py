"""
Visualization utilities for drawing detected geometry on images.

This module provides functions to visualize:
- Bounding boxes for cars and wheels
- 3D coordinate axes projected onto 2D image
- Wheel centers and orientations
- Ground plane indicators
"""

import math
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
    line_width: int = 3
):
    """
    Draw 3D coordinate axes projected onto 2D image.
    
    Args:
        draw: PIL ImageDraw object
        center: Center point (pixel coordinates)
        basis_vectors: Dictionary with x_axis, y_axis, z_axis vectors
        axis_length: Length of each axis in pixels
        line_width: Width of axis lines
    """
    # Project 3D basis vectors to 2D
    # Simple projection: use X and Y components, ignore Z for visualization
    # But we'll use a pseudo-3D projection for better visualization
    
    x_axis = basis_vectors["x_axis"]
    y_axis = basis_vectors["y_axis"]
    z_axis = basis_vectors["z_axis"]
    
    # Project to 2D with simple perspective
    # X component maps to image X
    # Y component maps to image Y (inverted)
    # Z component adds slight offset for depth perception
    
    def project_vector(vec: List[float]) -> Tuple[float, float]:
        """Project 3D vector to 2D screen coordinates."""
        # Simple orthographic-ish projection with Z influence
        proj_x = vec[0] + vec[2] * 0.3  # Z adds some X offset
        proj_y = -vec[1] + vec[2] * 0.2  # Z adds some Y offset, Y is inverted
        return (proj_x, proj_y)
    
    # Calculate end points for each axis
    x_proj = project_vector(x_axis)
    y_proj = project_vector(y_axis)
    z_proj = project_vector(z_axis)
    
    # Scale to axis_length
    def scale_vector(v: Tuple[float, float], length: float) -> Tuple[float, float]:
        mag = math.sqrt(v[0]**2 + v[1]**2)
        if mag < 0.001:
            return (0, 0)
        return (v[0] / mag * length, v[1] / mag * length)
    
    x_end = scale_vector(x_proj, axis_length)
    y_end = scale_vector(y_proj, axis_length)
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
                    axis_length=axis_length
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
                    axis_length=axis_length
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
