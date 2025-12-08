#!/usr/bin/env python
"""Test render with fixed object grouping."""

import os
import sys
import traceback

# Write all output to a log file
log_file = open('/tmp/render_test.log', 'w')

def log(msg):
    log_file.write(str(msg) + '\n')
    log_file.flush()
    print(msg)
    sys.stdout.flush()

try:
    # Set the Blender path
    os.environ['BLENDER_PATH'] = '/Applications/Blender.app/Contents/MacOS/Blender'
    log('Blender path set')

    from services.rendering.blender_renderer import render_tire_boot, composite_tire_boot
    from PIL import Image
    log('Imports successful')

    # Load test image
    test_image = Image.open('spec/two.jpg')
    log(f'Loaded test image: {test_image.size}')

    # Wheel data for first wheel
    wheel_data = {
        'center': (450, 350),
        'radius': 80,
        'rotation_matrix': None
    }

    # Render the tire boot
    log('Rendering tire boot...')
    render_path = render_tire_boot(
        wheel_center=wheel_data['center'],
        wheel_radius=wheel_data['radius'],
        rotation_matrix=wheel_data['rotation_matrix'],
        image_size=test_image.size
    )

    log(f'Rendered to: {render_path}')

    # Composite onto image
    log('Compositing...')
    result = composite_tire_boot(test_image, render_path)

    result.convert('RGB').save('spec/fixtures/grouped_test.jpg')
    log('Saved to spec/fixtures/grouped_test.jpg')
except Exception as e:
    log(f'ERROR: {e}')
    log(traceback.format_exc())
finally:
    log_file.close()
