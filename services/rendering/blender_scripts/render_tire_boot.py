"""
Blender rendering script for tire boot compositing.

This script is executed by Blender in background mode:
    blender --background --python render_tire_boot.py -- [args]

It loads the tire boot model, positions it based on wheel detection data,
renders with transparency, and outputs a PNG for compositing.
"""

import bpy
import sys
import json
import math
import os
from mathutils import Vector, Euler, Matrix


def clear_scene():
    """Remove default objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def setup_render_settings(width: int, height: int, transparent: bool = True):
    """Configure render settings for compositing output."""
    scene = bpy.context.scene
    
    # Output dimensions match input image (temporarily 2x for higher quality)
    scene.render.resolution_x = width * 2
    scene.render.resolution_y = height * 2
    scene.render.resolution_percentage = 100
    
    # PNG with alpha for compositing
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    
    # Transparent background for compositing
    if transparent:
        scene.render.film_transparent = True
    
    # Use Eevee for speed (Blender 5.0 uses BLENDER_EEVEE_NEXT)
    # Fall back to BLENDER_EEVEE for older versions
    if hasattr(bpy.types, 'SceneEEVEE') and 'BLENDER_EEVEE_NEXT' in dir(bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items.keys):
        scene.render.engine = 'BLENDER_EEVEE_NEXT'
    else:
        try:
            scene.render.engine = 'BLENDER_EEVEE_NEXT'
        except:
            scene.render.engine = 'BLENDER_EEVEE'
    
    # Eevee settings - use try/except for API compatibility across versions
    # Higher samples for better quality
    try:
        scene.eevee.taa_render_samples = 128
    except AttributeError:
        pass
    
    try:
        scene.eevee.use_ssr = True
    except AttributeError:
        pass  # Renamed or removed in Blender 5.0
    
    try:
        scene.eevee.use_soft_shadows = True
    except AttributeError:
        pass


def setup_camera(wheel_center: tuple, wheel_radius: float, image_width: int, image_height: int):
    """
    Create and position an orthographic camera to match the 2D image perspective.
    
    For compositing, we use an orthographic camera positioned to look at the model
    from the front (along the -Y axis), matching how the model appears in the preview.
    
    Note: We render at 2x resolution for quality, so camera position and ortho_scale
    are calculated for the 2x coordinate system.
    """
    # Scale factor for 2x render resolution
    scale = 2
    
    # Create camera - position it in front of the scene, looking toward +Y
    # Camera is placed at negative Y, looking toward positive Y
    # Position is centered on the scaled image
    bpy.ops.object.camera_add(location=(image_width * scale / 2, -100, image_height * scale / 2))
    camera = bpy.context.object
    camera.name = "CompositeCamera"
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    # Use orthographic projection for 2D compositing
    camera.data.type = 'ORTHO'
    
    # Scale orthographic view to match the 2x render dimensions
    camera.data.ortho_scale = max(image_width, image_height) * scale
    
    # Rotate camera to look along +Y axis (toward the scene)
    # 90 degrees around X axis makes camera look from -Y toward +Y
    camera.rotation_euler = (math.radians(90), 0, 0)
    
    return camera


def setup_lighting():
    """Add basic lighting for the render."""
    # Key light
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
    sun = bpy.context.object
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(45), math.radians(30), 0)
    
    # Fill light (softer)
    bpy.ops.object.light_add(type='SUN', location=(-5, -5, 8))
    fill = bpy.context.object
    fill.data.energy = 0.5
    fill.rotation_euler = (math.radians(60), math.radians(-30), 0)
    
    # Add environment lighting
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes.get('Background')
    if bg_node:
        bg_node.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)
        bg_node.inputs['Strength'].default_value = 0.3


def load_tire_boot(blend_file_path: str) -> bpy.types.Object:
    """
    Load the tire boot by opening the blend file directly.
    
    This preserves all object relationships, transforms, and groupings
    exactly as they appear in the original file.
    
    Returns an empty object that parents all objects for easy manipulation.
    """
    # Open the blend file directly - this preserves everything
    bpy.ops.wm.open_mainfile(filepath=blend_file_path)
    
    # Get all mesh objects from the scene
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == 'MESH']
    
    if not mesh_objects:
        raise ValueError("No mesh objects found in blend file")
    
    print(f"Loaded {len(mesh_objects)} mesh objects from blend file")
    
    # Create an empty at the origin to parent everything
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    parent_empty = bpy.context.object
    parent_empty.name = "TireBootParent"
    
    # Parent all top-level objects to the empty
    for obj in mesh_objects:
        if obj.parent is None:
            obj.parent = parent_empty
            obj.matrix_parent_inverse = parent_empty.matrix_world.inverted()
    
    return parent_empty


def position_tire_boot(
    tire_boot: bpy.types.Object,
    wheel_center: tuple,
    wheel_radius: float,
    rotation_matrix: list,
    image_height: int
):
    """
    Position and orient the tire boot to match the detected wheel.
    
    Args:
        tire_boot: The tire boot parent object (empty)
        wheel_center: (x, y) pixel coordinates of wheel center
        wheel_radius: Radius of wheel in pixels
        rotation_matrix: 3x3 rotation matrix from our geometry calculations
        image_height: Image height for Y-coordinate flip
    """
    # Calculate bounding box across all child mesh objects
    all_coords = []
    for child in tire_boot.children_recursive:
        if child.type == 'MESH':
            for corner in child.bound_box:
                world_coord = child.matrix_world @ Vector(corner)
                all_coords.append(world_coord)
    
    if all_coords:
        min_x = min(v.x for v in all_coords)
        max_x = max(v.x for v in all_coords)
        min_y = min(v.y for v in all_coords)
        max_y = max(v.y for v in all_coords)
        min_z = min(v.z for v in all_coords)
        max_z = max(v.z for v in all_coords)
        boot_size = max(max_x - min_x, max_y - min_y, max_z - min_z)
    else:
        boot_size = 1.0
    
    # Scale to match wheel size (boot should be roughly wheel diameter)
    target_size = wheel_radius * 2 * 0.8 * 1.5  # 80% of wheel diameter, scaled up 1.5x for inspection
    # Scale by 2 since we're rendering at 2x resolution
    target_size = target_size * 2
    scale_factor = target_size / boot_size if boot_size > 0 else 1.0
    tire_boot.scale = (scale_factor, scale_factor, scale_factor)
    
    # Convert image coordinates to Blender 3D coordinates
    # Camera is looking along +Y axis, so:
    # - Image X -> Blender X (horizontal)
    # - Image Y -> Blender Z (vertical, but inverted: image Y=0 is top, Blender Z=high is top)
    # - Blender Y is depth (0 for our flat compositing)
    # Scale by 2 since we're rendering at 2x resolution
    scale = 2
    blender_x = wheel_center[0] * scale
    blender_z = (image_height - wheel_center[1]) * scale  # Flip Y to Z
    blender_y = 0  # Depth = 0
    
    # Position at wheel center
    tire_boot.location = (blender_x, blender_y, blender_z)
    
    # Apply base rotation to orient the claw correctly
    # Try no rotation first to see if both handles are visible
    base_rotation = Euler((0, 0, 0), 'XYZ')
    tire_boot.rotation_euler = base_rotation
    
    # Apply additional rotation from our geometry calculations if provided
    if rotation_matrix:
        # Convert our 3x3 rotation matrix to Blender's Matrix
        rot_mat = Matrix([
            [rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2]],
            [rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2]],
            [rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2]]
        ])
        # Combine base rotation with geometry rotation
        combined = rot_mat @ base_rotation.to_matrix()
        tire_boot.rotation_euler = combined.to_euler()


def render_to_file(output_path: str):
    """Render the scene to a file."""
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    print(f"Rendered to: {output_path}")


def main():
    """Main entry point when called from command line."""
    # Parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    if len(argv) < 1:
        print("Usage: blender --background --python render_tire_boot.py -- <config.json>")
        print("Config JSON should contain: blend_file, wheel_center, wheel_radius, rotation_matrix, image_size, output_path")
        sys.exit(1)
    
    # Load config
    config_path = argv[0]
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    blend_file = config['blend_file']
    wheel_center = tuple(config['wheel_center'])
    wheel_radius = config['wheel_radius']
    rotation_matrix = config.get('rotation_matrix')
    image_width = config['image_size']['width']
    image_height = config['image_size']['height']
    output_path = config['output_path']
    hide_objects = config.get('hide_objects', [])  # Optional list of object names to hide
    
    print(f"Rendering tire boot:")
    print(f"  Blend file: {blend_file}")
    print(f"  Wheel center: {wheel_center}")
    print(f"  Wheel radius: {wheel_radius}")
    print(f"  Image size: {image_width}x{image_height}")
    print(f"  Output: {output_path}")
    if hide_objects:
        print(f"  Hiding objects: {hide_objects}")
    
    # Load the blend file first (this replaces the current scene)
    tire_boot = load_tire_boot(blend_file)
    
    # Now set up our render settings, camera and lighting
    # (must be done AFTER loading because open_mainfile replaces everything)
    setup_render_settings(image_width, image_height, transparent=True)
    setup_camera(wheel_center, wheel_radius, image_width, image_height)
    setup_lighting()
    
    # Position the tire boot
    position_tire_boot(tire_boot, wheel_center, wheel_radius, rotation_matrix, image_height)
    
    # Hide specified objects (for testing which objects to remove)
    for obj_name in hide_objects:
        if obj_name in bpy.data.objects:
            bpy.data.objects[obj_name].hide_render = True
            print(f"  Hidden: {obj_name}")
    
    # Render
    render_to_file(output_path)
    
    print("Done!")


if __name__ == "__main__":
    main()
