/**
 * TypeScript type definitions for /detect/homography API response.
 *
 * These types are auto-generated to match the Pydantic schemas in schemas/detection.py
 * Use these types in your React/Three.js frontend for type-safe API consumption.
 *
 * @packageDocumentation
 */

/**
 * Bounding box coordinates in pixels.
 */
export interface BoundingBox {
  /** Left edge X coordinate */
  x1: number;
  /** Top edge Y coordinate */
  y1: number;
  /** Right edge X coordinate */
  x2: number;
  /** Bottom edge Y coordinate */
  y2: number;
  /** Box width in pixels */
  width: number;
  /** Box height in pixels */
  height: number;
}

/**
 * 3D position in normalized and pixel coordinates.
 */
export interface Position3D {
  /** Normalized X position [-1, 1] */
  x: number;
  /** Normalized Y position [-1, 1] */
  y: number;
  /** Estimated depth (relative) */
  z: number;
  /** Center X in pixels */
  pixel_x: number;
  /** Center Y in pixels */
  pixel_y: number;
}

/**
 * Quaternion rotation representation (x, y, z, w).
 */
export interface Quaternion {
  x: number;
  y: number;
  z: number;
  w: number;
}

/**
 * Euler angles rotation representation.
 */
export interface EulerAngles {
  /** Rotation around X-axis (radians) */
  x: number;
  /** Rotation around Y-axis (radians) */
  y: number;
  /** Rotation around Z-axis (radians) */
  z: number;
  /** Rotation order */
  order: string;
}

/**
 * Orthonormal basis vectors defining local coordinate frame.
 */
export interface BasisVectors {
  /** X-axis direction (axle) */
  x_axis: [number, number, number];
  /** Y-axis direction (up) */
  y_axis: [number, number, number];
  /** Z-axis direction (forward) */
  z_axis: [number, number, number];
}

/**
 * Additional rotation and viewing metadata.
 */
export interface RotationMetadata {
  viewing_angle_rad: number;
  viewing_angle_deg: number;
  ground_angle_rad: number;
  ground_angle_deg: number;
  wheel_to_wheel_2d: [number, number];
  viewing_side: string;
  target_wheel: string;
  ellipse_angle_deg?: number | null;
  ellipse_viewing_angle_deg?: number | null;
  used_ellipse: boolean;
}

/**
 * Complete rotation information in multiple representations.
 */
export interface Rotation {
  quaternion: Quaternion;
  euler_angles: EulerAngles;
  /** 3x3 rotation matrix */
  rotation_matrix: number[][];
  basis_vectors: BasisVectors;
  metadata: RotationMetadata;
}

/**
 * Scale information for 3D asset.
 */
export interface Scale {
  /** Uniform scale factor */
  uniform: number;
  /** Wheel radius in pixels */
  radius_pixels: number;
}

/**
 * Complete transform data for a wheel.
 */
export interface WheelTransform {
  position: Position3D;
  rotation: Rotation;
  scale: Scale;
  bounding_box: BoundingBox;
  confidence: number;
}

/**
 * Camera calibration parameters from homography/solvePnP.
 *
 * These parameters can be used to configure a Three.js PerspectiveCamera
 * to match the original photo's viewpoint.
 */
export interface CameraCalibration {
  /** 3x3 camera intrinsic matrix */
  intrinsic_matrix: number[][];
  /** 3x1 Rodrigues rotation vector */
  rotation_vector: [number, number, number];
  /** 3x1 camera position */
  translation_vector: [number, number, number];
  /** Field of view in degrees */
  fov: number;
  /** Estimated camera height in meters */
  camera_height: number;
  /** Focal length in pixels */
  focal_length: number;
  /** Calibration method: 'solvePnP', 'homography', or 'default' */
  method: 'solvePnP' | 'homography' | 'default';
}

/**
 * Three.js compatible camera parameters.
 * Can be directly used to configure a PerspectiveCamera.
 */
export interface ThreeJSCamera {
  /** Field of view in degrees */
  fov: number;
  /** Aspect ratio */
  aspect: number;
  /** Near clipping plane */
  near: number;
  /** Far clipping plane */
  far: number;
  /** Camera position [x, y, z] */
  position: [number, number, number];
  /** Camera rotation */
  quaternion: Quaternion;
}

/**
 * Detected tire ellipse parameters.
 */
export interface WheelEllipse {
  /** Ellipse center [x, y] */
  center: [number, number];
  /** Major and minor axes */
  axes: [number, number];
  /** Rotation angle in degrees */
  angle: number;
  /** Minor/major axis ratio */
  axis_ratio: number;
  /** Viewing angle in degrees */
  viewing_angle_deg: number;
}

/**
 * Overall car geometry information.
 */
export interface CarGeometry {
  /** Normalized wheel-to-wheel direction vector */
  wheel_to_wheel_2d: [number, number];
  /** Ground plane tilt in degrees */
  ground_angle_deg: number;
  /** Which side of car is visible: 'left' or 'right' */
  viewing_side: 'left' | 'right';
}

/**
 * Individual wheel detection.
 */
export interface WheelDetection {
  /** Detection class name */
  class: string;
  /** Detection confidence score */
  confidence: number;
  bbox: BoundingBox;
}

/**
 * Individual car detection.
 */
export interface CarDetection {
  /** Detection class name */
  class: string;
  /** Detection confidence score */
  confidence: number;
  bbox: BoundingBox;
}

/**
 * Identified front and rear wheel positions.
 */
export interface WheelPositions {
  front?: WheelDetection | null;
  rear?: WheelDetection | null;
}

/**
 * Complete detection result for a single car.
 *
 * This includes the car bounding box, all detected wheels, wheel transforms,
 * camera calibration, and geometry information.
 */
export interface CarDetectionResult {
  /** Unique ID for this car */
  car_id: number;
  car: CarDetection;
  wheels: WheelDetection[];
  /** Number of wheels detected */
  wheel_count: number;
  wheel_positions?: WheelPositions | null;
  rear_wheel_transform?: WheelTransform | null;
  front_wheel_transform?: WheelTransform | null;
  rear_wheel_ellipse?: WheelEllipse | null;
  front_wheel_ellipse?: WheelEllipse | null;
  car_geometry?: CarGeometry | null;
  camera_calibration?: CameraCalibration | null;
  threejs_camera?: ThreeJSCamera | null;
}

/**
 * Image dimensions in pixels.
 */
export interface ImageDimensions {
  width: number;
  height: number;
}

/**
 * Complete response from /detect/homography endpoint.
 *
 * This response includes all detection data, geometry transforms,
 * and camera calibration suitable for AR overlay in Three.js.
 */
export interface HomographyDetectionResponse {
  /** Source image filename */
  filename: string;
  image_dimensions: ImageDimensions;
  /** Number of cars detected */
  total_cars: number;
  /** Car detection results */
  detections: CarDetectionResult[];
  /** Whether results were filtered to largest car only */
  filtered_to_largest?: boolean;
}

// ============================================================================
// Helper Types and Utility Functions
// ============================================================================

/**
 * Parameters for the /detect/homography POST request.
 */
export interface HomographyDetectionParams {
  /** Image file to process */
  file: File | Blob;
  /** Confidence threshold for car detection (0-1) */
  car_confidence?: number;
  /** Confidence threshold for wheel detection (0-1) */
  wheel_confidence?: number;
  /** Expected wheelbase distance in meters */
  wheelbase_m?: number;
  /** Expected wheel diameter in meters */
  wheel_diameter_m?: number;
}

/**
 * Type guard to check if camera calibration is available.
 */
export function hasCameraCalibration(
  detection: CarDetectionResult
): detection is CarDetectionResult & { camera_calibration: CameraCalibration } {
  return detection.camera_calibration !== null && detection.camera_calibration !== undefined;
}

/**
 * Type guard to check if Three.js camera is available.
 */
export function hasThreeJSCamera(
  detection: CarDetectionResult
): detection is CarDetectionResult & { threejs_camera: ThreeJSCamera } {
  return detection.threejs_camera !== null && detection.threejs_camera !== undefined;
}

/**
 * Type guard to check if rear wheel transform is available.
 */
export function hasRearWheelTransform(
  detection: CarDetectionResult
): detection is CarDetectionResult & { rear_wheel_transform: WheelTransform } {
  return detection.rear_wheel_transform !== null && detection.rear_wheel_transform !== undefined;
}

/**
 * Extract camera parameters ready for Three.js PerspectiveCamera constructor.
 *
 * @example
 * ```typescript
 * const cameraParams = extractThreeJSCameraParams(detection);
 * const camera = new THREE.PerspectiveCamera(
 *   cameraParams.fov,
 *   cameraParams.aspect,
 *   cameraParams.near,
 *   cameraParams.far
 * );
 * camera.position.set(...cameraParams.position);
 * camera.quaternion.set(
 *   cameraParams.quaternion.x,
 *   cameraParams.quaternion.y,
 *   cameraParams.quaternion.z,
 *   cameraParams.quaternion.w
 * );
 * ```
 */
export function extractThreeJSCameraParams(
  detection: CarDetectionResult
): ThreeJSCamera | null {
  return detection.threejs_camera || null;
}

/**
 * Convert API response quaternion to Three.js Quaternion.
 *
 * @param q - Quaternion from API
 * @returns Array in format [x, y, z, w] for Three.js
 */
export function toThreeQuaternion(q: Quaternion): [number, number, number, number] {
  return [q.x, q.y, q.z, q.w];
}

/**
 * Convert API response position to Three.js Vector3.
 *
 * @param pos - Position from API
 * @returns Array in format [x, y, z] for Three.js
 */
export function toThreeVector3(pos: Position3D | [number, number, number]): [number, number, number] {
  if (Array.isArray(pos)) {
    return pos;
  }
  return [pos.x, pos.y, pos.z];
}

// ============================================================================
// Example Usage
// ============================================================================

/**
 * Example: Fetch detection with homography calibration
 *
 * ```typescript
 * async function detectWithHomography(imageFile: File): Promise<HomographyDetectionResponse> {
 *   const formData = new FormData();
 *   formData.append('file', imageFile);
 *   formData.append('car_confidence', '0.3');
 *   formData.append('wheel_confidence', '0.3');
 *
 *   const response = await fetch('/detect/homography', {
 *     method: 'POST',
 *     body: formData
 *   });
 *
 *   if (!response.ok) {
 *     throw new Error(`Detection failed: ${response.statusText}`);
 *   }
 *
 *   return response.json();
 * }
 *
 * // Use the response
 * const result = await detectWithHomography(myImage);
 *
 * if (result.total_cars > 0) {
 *   const detection = result.detections[0];
 *
 *   if (hasThreeJSCamera(detection)) {
 *     // Configure Three.js camera
 *     const cam = detection.threejs_camera;
 *     camera.fov = cam.fov;
 *     camera.aspect = cam.aspect;
 *     camera.position.set(...cam.position);
 *     camera.quaternion.set(cam.quaternion.x, cam.quaternion.y, cam.quaternion.z, cam.quaternion.w);
 *   }
 *
 *   if (hasRearWheelTransform(detection)) {
 *     // Position 3D asset
 *     const transform = detection.rear_wheel_transform;
 *     mesh.position.set(transform.position.x, transform.position.y, transform.position.z);
 *     mesh.scale.setScalar(transform.scale.uniform);
 *   }
 * }
 * ```
 */
