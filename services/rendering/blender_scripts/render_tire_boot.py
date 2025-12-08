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
    
    # Output dimensions match input image
    scene.render.resolution_x = width
    scene.render.resolution_y = height
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
    try:
        scene.eevee.taa_render_samples = 64
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
    
    For compositing, we use an orthographic camera positioned to match
    the input image's coordinate system.
    """
    # Create camera
    bpy.ops.object.camera_add(location=(0, 0, 10))
    camera = bpy.context.object
    camera.name = "CompositeCamera"
    
    # Set as active camera
    bpy.context.scene.camera = camera
    
    # Use orthographic projection for 2D compositing
    camera.data.type = 'ORTHO'
    
    # Scale orthographic view to match image dimensions
    # We'll work in a coordinate system where 1 unit = 1 pixel
    camera.data.ortho_scale = max(image_width, image_height)
    
    # Position camera looking down -Z axis
    camera.location = (image_width / 2, image_height / 2, 100)
    camera.rotation_euler = (0, 0, 0)
    
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
    Load the tire boot from a .blend file.
    
    Returns the main tire boot object.
    """
    # Get list of objects before appending
    existing_objects = set(bpy.data.objects.keys())
    
    # Load all objects from the blend file
    with bpy.data.libraries.load(blend_file_path, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects
        data_to.materials = data_from.materials
        data_to.textures = data_from.textures
        data_to.images = data_from.images
    
    # Link loaded objects to current scene
    tire_boot_objects = []
    for obj in data_to.objects:
        if obj is not None:
            bpy.context.collection.objects.link(obj)
            tire_boot_objects.append(obj)
    
    # Find the main mesh object (usually the largest one)
    mesh_objects = [obj for obj in tire_boot_objects if obj.type == 'MESH']
    
    if not mesh_objects:
        raise ValueError("No mesh objects found in blend file")
    
    # Return the largest mesh by vertex count
    main_object = max(mesh_objects, key=lambda o: len(o.data.vertices))
    
    # Parent all objects to the main one for easier manipulation
    for obj in tire_boot_objects:
        if obj != main_object and obj.parent is None:
            obj.parent = main_object
    
    return main_object


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
        tire_boot: The tire boot Blender object
        wheel_center: (x, y) pixel coordinates of wheel center
        wheel_radius: Radius of wheel in pixels
        rotation_matrix: 3x3 rotation matrix from our geometry calculations
        image_height: Image height for Y-coordinate flip
    """
    # Get the tire boot's bounding box to determine its size
    bbox = [tire_boot.matrix_world @ Vector(corner) for corner in tire_boot.bound_box]
    boot_size = max(
        max(v.x for v in bbox) - min(v.x for v in bbox),
        max(v.y for v in bbox) - min(v.y for v in bbox),
        max(v.z for v in bbox) - min(v.z for v in bbox)
    )
    
    # Scale to match wheel size (boot should be roughly wheel diameter)
    target_size = wheel_radius * 2 * 0.8  # 80% of wheel diameter
    scale_factor = target_size / boot_size if boot_size > 0 else 1.0
    tire_boot.scale = (scale_factor, scale_factor, scale_factor)
    
    # Convert image coordinates to Blender coordinates
    # Image: origin top-left, Y down
    # Blender: origin bottom-left, Y up (for 2D view)
    blender_x = wheel_center[0]
    blender_y = image_height - wheel_center[1]
    
    # Position at wheel center
    tire_boot.location = (blender_x, blender_y, 0)
    
    # Apply base rotation to orient the claw correctly
    # The model needs to be rotated to face the camera properly
    # Rotate 90 degrees clockwise around Z axis (negative rotation)
    base_rotation = Euler((0, 0, math.radians(-90)), 'XYZ')
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
    
    print(f"Rendering tire boot:")
    print(f"  Blend file: {blend_file}")
    print(f"  Wheel center: {wheel_center}")
    print(f"  Wheel radius: {wheel_radius}")
    print(f"  Image size: {image_width}x{image_height}")
    print(f"  Output: {output_path}")
    
    # Clear and setup scene
    clear_scene()
    setup_render_settings(image_width, image_height, transparent=True)
    setup_camera(wheel_center, wheel_radius, image_width, image_height)
    setup_lighting()
    
    # Load and position tire boot
    tire_boot = load_tire_boot(blend_file)
    position_tire_boot(tire_boot, wheel_center, wheel_radius, rotation_matrix, image_height)
    
    # Render
    render_to_file(output_path)
    
    print("Done!")


if __name__ == "__main__":
    main()
