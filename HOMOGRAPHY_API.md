# Homography Detection API

## Overview

The `/detect/homography` endpoint provides advanced car and wheel detection with camera calibration using OpenCV's solvePnP method. This enables accurate AR (Augmented Reality) placement of 3D assets by matching the Three.js camera perspective with the original photo.

## Endpoint

```
POST /detect/homography
```

## Features

✅ Two-stage car and wheel detection (YOLO-based)
✅ Camera calibration using homography/solvePnP
✅ 3D geometry transforms for detected wheels
✅ Three.js-ready camera configuration
✅ Typed responses with Pydantic schemas
✅ TypeScript type definitions for frontend

## Request

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | Image file to process |
| `car_confidence` | float | 0.3 | Car detection confidence threshold (0-1) |
| `wheel_confidence` | float | 0.3 | Wheel detection confidence threshold (0-1) |
| `wheelbase_m` | float | 2.7 | Expected wheelbase distance in meters |
| `wheel_diameter_m` | float | 0.65 | Expected wheel diameter in meters |

### Example (cURL)

```bash
curl -X POST "http://localhost:8000/detect/homography" \
  -F "file=@car_image.jpg" \
  -F "car_confidence=0.3" \
  -F "wheel_confidence=0.3"
```

### Example (Python)

```python
import requests

with open("car_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/detect/homography",
        files={"file": f},
        data={
            "car_confidence": 0.3,
            "wheel_confidence": 0.3,
            "wheelbase_m": 2.7,
            "wheel_diameter_m": 0.65
        }
    )

data = response.json()
print(f"Detected {data['total_cars']} cars")
```

### Example (JavaScript/TypeScript)

```typescript
import type { HomographyDetectionResponse } from './types/detection';

async function detectWithHomography(imageFile: File): Promise<HomographyDetectionResponse> {
  const formData = new FormData();
  formData.append('file', imageFile);
  formData.append('car_confidence', '0.3');
  formData.append('wheel_confidence', '0.3');

  const response = await fetch('/detect/homography', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    throw new Error(`Detection failed: ${response.statusText}`);
  }

  return response.json();
}
```

## Response

### Structure

```typescript
{
  "filename": string,
  "image_dimensions": {
    "width": number,
    "height": number
  },
  "total_cars": number,
  "detections": [
    {
      "car_id": number,
      "car": { /* car detection */ },
      "wheels": [ /* wheel detections */ ],
      "wheel_count": number,
      "camera_calibration": { /* camera parameters */ },
      "threejs_camera": { /* Three.js config */ },
      "rear_wheel_transform": { /* 3D transform */ },
      "front_wheel_transform": { /* 3D transform */ },
      "car_geometry": { /* geometry info */ }
    }
  ],
  "filtered_to_largest": boolean
}
```

### Camera Calibration

When both front and rear wheels are detected, the response includes camera calibration:

```json
{
  "camera_calibration": {
    "intrinsic_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "rotation_vector": [rx, ry, rz],
    "translation_vector": [tx, ty, tz],
    "fov": 55.0,
    "camera_height": 1.5,
    "focal_length": 1920.0,
    "method": "solvePnP"
  }
}
```

**Calibration Methods:**
- `solvePnP`: OpenCV's Perspective-n-Point solver (most accurate)
- `homography`: Homography matrix decomposition
- `default`: Fallback when calibration fails

### Three.js Camera Configuration

Ready-to-use camera parameters for Three.js:

```json
{
  "threejs_camera": {
    "fov": 55.0,
    "aspect": 1.777,
    "near": 0.01,
    "far": 100.0,
    "position": [0.0, 1.5, 3.0],
    "quaternion": {
      "x": 0.0,
      "y": 0.0,
      "z": 0.0,
      "w": 1.0
    }
  }
}
```

### Wheel Transform

Complete 3D transform data for positioning assets:

```json
{
  "rear_wheel_transform": {
    "position": {
      "x": 0.5,          // Normalized [-1, 1]
      "y": -0.2,
      "z": -0.4,
      "pixel_x": 650.0,  // Original pixels
      "pixel_y": 420.0
    },
    "rotation": {
      "quaternion": { "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0 },
      "euler_angles": { "x": 0.0, "y": 1.57, "z": 0.0, "order": "XYZ" },
      "rotation_matrix": [[ /* 3x3 matrix */ ]],
      "basis_vectors": {
        "x_axis": [1, 0, 0],
        "y_axis": [0, 1, 0],
        "z_axis": [0, 0, 1]
      }
    },
    "scale": {
      "uniform": 0.25,
      "radius_pixels": 65.0
    }
  }
}
```

## Usage with Three.js

### 1. Configure Camera

```typescript
const detection = response.detections[0];

if (detection.threejs_camera) {
  const cam = detection.threejs_camera;

  const camera = new THREE.PerspectiveCamera(
    cam.fov,
    cam.aspect,
    cam.near,
    cam.far
  );

  camera.position.set(...cam.position);
  camera.quaternion.set(
    cam.quaternion.x,
    cam.quaternion.y,
    cam.quaternion.z,
    cam.quaternion.w
  );
}
```

### 2. Position 3D Asset

```typescript
if (detection.rear_wheel_transform) {
  const transform = detection.rear_wheel_transform;

  // Position asset at wheel location
  mesh.position.set(
    transform.position.x,
    transform.position.y,
    transform.position.z
  );

  // Apply rotation
  mesh.quaternion.set(
    transform.rotation.quaternion.x,
    transform.rotation.quaternion.y,
    transform.rotation.quaternion.z,
    transform.rotation.quaternion.w
  );

  // Apply scale
  mesh.scale.setScalar(transform.scale.uniform);
}
```

### 3. Complete Example with React Three Fiber

```tsx
import { Canvas } from '@react-three/fiber';
import { HomographyDetectionResponse } from './types/detection';

function ARScene({ detection }: { detection: HomographyDetectionResponse['detections'][0] }) {
  if (!detection.threejs_camera || !detection.rear_wheel_transform) {
    return null;
  }

  const { threejs_camera: cam, rear_wheel_transform: transform } = detection;

  return (
    <Canvas
      camera={{
        fov: cam.fov,
        aspect: cam.aspect,
        near: cam.near,
        far: cam.far,
        position: cam.position,
      }}
      onCreated={({ camera }) => {
        camera.quaternion.set(
          cam.quaternion.x,
          cam.quaternion.y,
          cam.quaternion.z,
          cam.quaternion.w
        );
      }}
    >
      <mesh
        position={[
          transform.position.x,
          transform.position.y,
          transform.position.z
        ]}
        quaternion={[
          transform.rotation.quaternion.x,
          transform.rotation.quaternion.y,
          transform.rotation.quaternion.z,
          transform.rotation.quaternion.w
        ]}
        scale={transform.scale.uniform}
      >
        <boxGeometry />
        <meshStandardMaterial color="red" />
      </mesh>
    </Canvas>
  );
}
```

## TypeScript Types

Full TypeScript type definitions are available in `/types/detection.ts`:

```typescript
import type {
  HomographyDetectionResponse,
  CameraCalibration,
  ThreeJSCamera,
  WheelTransform
} from './types/detection';
```

Helper functions:

```typescript
import {
  hasCameraCalibration,
  hasThreeJSCamera,
  hasRearWheelTransform
} from './types/detection';

if (hasCameraCalibration(detection)) {
  // TypeScript knows camera_calibration is non-null here
  const fov = detection.camera_calibration.fov;
}
```

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 422 | Validation error (missing/invalid parameters) |
| 500 | Internal server error (detection failed) |
| 503 | Models still loading |

### Example Error Response

```json
{
  "detail": "Detection failed"
}
```

## Testing

Run the test suite:

```bash
# All tests
pytest tests/test_homography.py -v

# With output
pytest tests/test_homography.py -v -s

# Specific test
pytest tests/test_homography.py::test_homography_endpoint_summary -v
```

## Performance

Typical processing times (on M1 Mac):
- Model inference: 30-50ms per image
- Camera calibration: <5ms
- Total response time: 50-100ms

## Limitations

1. **Two visible wheels required**: Camera calibration requires both front and rear wheels to be detected
2. **Side-view assumption**: Algorithm works best with side-angle car photos
3. **Flat ground assumption**: Ground plane estimation assumes relatively flat terrain
4. **Depth is relative**: Z-coordinates are estimated from wheel size, not true depth
5. **Single car focus**: When multiple cars detected, filters to largest by default

## Future Improvements

Based on the homography_approach_findings.md document:

### Phase 1: Side Mirror Detection (HIGH PRIORITY)
- Train YOLO to detect side mirrors
- Use mirrors for front/rear wheel identification
- Expected improvement: 80% → 85% success rate

### Phase 2: Multi-Object Geometry
- Add window and door detection
- Refine X/Y/Z axis heuristics
- Expected improvement: 85% → 90% success rate

### Phase 3: Multi-View Stereo (OPTIONAL)
- Support multiple photos from different angles
- True 3D reconstruction via triangulation
- Expected improvement: 90% → 95%+ success rate

## References

- [OpenCV solvePnP Documentation](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga549c2075fac14829ff4a58bc931c033d)
- [Three.js PerspectiveCamera](https://threejs.org/docs/#api/en/cameras/PerspectiveCamera)
- [Homography Approach Findings](./homography_approach_findings.md)
- [Detection Schema Documentation](./spec/DETECTION_SCHEMA.md)

## Support

For issues or questions:
1. Check the test suite for examples
2. Review TypeScript type definitions
3. Consult the homography approach documentation
4. Open an issue with sample images and error logs
