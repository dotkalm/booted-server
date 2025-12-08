"""
Blender-based rendering service for tire boot compositing.

This service orchestrates Blender to render 3D tire boots onto car images
based on wheel detection and geometry calculations.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)

# Path to Blender executable (configurable via environment)
BLENDER_PATH = os.environ.get('BLENDER_PATH', 'blender')

# Path to our Blender rendering script
SCRIPT_DIR = Path(__file__).parent / 'blender_scripts'
RENDER_SCRIPT = SCRIPT_DIR / 'render_tire_boot.py'

# Default tire boot asset (full model with tire for now)
ASSETS_DIR = Path(__file__).parent.parent.parent / 'assets' / 'tire-boot'
DEFAULT_BLEND_FILE = ASSETS_DIR / 'Security_Tire_Claw_Boot_blender_base.blend'


class BlenderRenderError(Exception):
    """Raised when Blender rendering fails."""
    pass


def check_blender_available() -> bool:
    """Check if Blender is available on the system."""
    try:
        result = subprocess.run(
            [BLENDER_PATH, '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            logger.info(f"Blender available: {version}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Blender not available: {e}")
    return False


def render_tire_boot(
    wheel_center: Tuple[float, float],
    wheel_radius: float,
    image_size: Tuple[int, int],
    rotation_matrix: Optional[list] = None,
    blend_file: Optional[Path] = None,
    output_path: Optional[Path] = None,
    hide_objects: Optional[list] = None
) -> Path:
    """
    Render a tire boot at the specified wheel position.
    
    Args:
        wheel_center: (x, y) pixel coordinates of wheel center
        wheel_radius: Radius of wheel in pixels
        image_size: (width, height) of the target image
        rotation_matrix: 3x3 rotation matrix for orientation (optional)
        blend_file: Path to .blend file (uses default if not specified)
        output_path: Path for rendered output (creates temp file if not specified)
        hide_objects: List of object names to hide from render (optional)
    
    Returns:
        Path to the rendered PNG with transparency
    
    Raises:
        BlenderRenderError: If rendering fails
    """
    if not check_blender_available():
        raise BlenderRenderError("Blender is not available. Install Blender or set BLENDER_PATH.")
    
    blend_file = blend_file or DEFAULT_BLEND_FILE
    if not blend_file.exists():
        raise BlenderRenderError(f"Blend file not found: {blend_file}")
    
    # Create output path if not specified
    if output_path is None:
        output_fd, output_path = tempfile.mkstemp(suffix='.png', prefix='tire_boot_')
        os.close(output_fd)
        output_path = Path(output_path)
    
    # Create config file for the Blender script
    config = {
        'blend_file': str(blend_file.absolute()),
        'wheel_center': list(wheel_center),
        'wheel_radius': wheel_radius,
        'rotation_matrix': rotation_matrix,
        'image_size': {
            'width': image_size[0],
            'height': image_size[1]
        },
        'output_path': str(output_path.absolute()),
        'hide_objects': hide_objects or []
    }
    
    # Write config to temp file
    config_fd, config_path = tempfile.mkstemp(suffix='.json', prefix='blender_config_')
    with os.fdopen(config_fd, 'w') as f:
        json.dump(config, f)
    
    try:
        # Run Blender in background mode
        cmd = [
            BLENDER_PATH,
            '--background',
            '--python', str(RENDER_SCRIPT),
            '--',
            config_path
        ]
        
        print(f"Running Blender: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        # Always print output for debugging
        if result.stdout:
            print(f"Blender stdout:\n{result.stdout}")
        if result.stderr:
            print(f"Blender stderr:\n{result.stderr}")
        
        if result.returncode != 0:
            raise BlenderRenderError(f"Blender rendering failed (exit code {result.returncode})")
        
        if not output_path.exists():
            raise BlenderRenderError("Blender did not create output file")
        
        return output_path
        
    finally:
        # Clean up config file
        os.unlink(config_path)


def composite_tire_boot(
    background_image: Image.Image,
    tire_boot_render: Path,
    output_path: Optional[Path] = None
) -> Image.Image:
    """
    Composite the rendered tire boot onto the background image.
    
    Args:
        background_image: PIL Image of the car photo
        tire_boot_render: Path to the rendered tire boot PNG with alpha
        output_path: Optional path to save the composite
    
    Returns:
        PIL Image with tire boot composited
    """
    # Load the tire boot render (with transparency)
    boot_image = Image.open(tire_boot_render).convert('RGBA')
    
    # Ensure background is RGBA
    if background_image.mode != 'RGBA':
        background_image = background_image.convert('RGBA')
    
    # Resize boot render to match background if needed
    if boot_image.size != background_image.size:
        boot_image = boot_image.resize(background_image.size, Image.Resampling.LANCZOS)
    
    # Composite using alpha blending
    composite = Image.alpha_composite(background_image, boot_image)
    
    # Save if output path specified
    if output_path:
        # Convert to RGB if saving as JPEG (no alpha support)
        output_str = str(output_path).lower()
        if output_str.endswith(('.jpg', '.jpeg')):
            composite = composite.convert('RGB')
        composite.save(output_path)
        logger.info(f"Saved composite to: {output_path}")
    
    return composite


def render_and_composite(
    image: Image.Image,
    wheel_data: Dict[str, Any],
    output_path: Optional[Path] = None
) -> Image.Image:
    """
    Full pipeline: render tire boot and composite onto image.
    
    Args:
        image: PIL Image of the car
        wheel_data: Detection data with wheel_center, wheel_radius, rotation_matrix
        output_path: Optional path to save result
    
    Returns:
        Composited PIL Image
    """
    # Extract wheel info
    wheel_center = (
        wheel_data['center']['x'],
        wheel_data['center']['y']
    )
    wheel_radius = wheel_data.get('radius', wheel_data['bbox']['width'] / 2)
    rotation_matrix = wheel_data.get('rotation_matrix')
    
    # Render tire boot
    boot_render = render_tire_boot(
        wheel_center=wheel_center,
        wheel_radius=wheel_radius,
        image_size=(image.width, image.height),
        rotation_matrix=rotation_matrix
    )
    
    try:
        # Composite
        result = composite_tire_boot(image, boot_render, output_path)
        return result
    finally:
        # Clean up temp render
        if boot_render.exists():
            os.unlink(boot_render)
