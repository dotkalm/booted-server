"""
Pydantic schemas for detection API responses.

These schemas provide type safety and automatic OpenAPI documentation.
They are designed to be easily consumable by TypeScript frontends.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class BoundingBox(BaseModel):
    """Bounding box coordinates in pixels."""
    x1: int = Field(..., description="Left edge X coordinate")
    y1: int = Field(..., description="Top edge Y coordinate")
    x2: int = Field(..., description="Right edge X coordinate")
    y2: int = Field(..., description="Bottom edge Y coordinate")
    width: int = Field(..., description="Box width in pixels")
    height: int = Field(..., description="Box height in pixels")


class Position3D(BaseModel):
    """3D position in normalized and pixel coordinates."""
    x: float = Field(..., description="Normalized X position [-1, 1]")
    y: float = Field(..., description="Normalized Y position [-1, 1]")
    z: float = Field(..., description="Estimated depth (relative)")
    pixel_x: float = Field(..., description="Center X in pixels")
    pixel_y: float = Field(..., description="Center Y in pixels")


class Quaternion(BaseModel):
    """Quaternion rotation representation."""
    x: float
    y: float
    z: float
    w: float


class EulerAngles(BaseModel):
    """Euler angles rotation representation."""
    x: float = Field(..., description="Rotation around X-axis (radians)")
    y: float = Field(..., description="Rotation around Y-axis (radians)")
    z: float = Field(..., description="Rotation around Z-axis (radians)")
    order: str = Field(default="XYZ", description="Rotation order")


class BasisVectors(BaseModel):
    """Orthonormal basis vectors defining local coordinate frame."""
    x_axis: List[float] = Field(..., description="X-axis direction (axle)")
    y_axis: List[float] = Field(..., description="Y-axis direction (up)")
    z_axis: List[float] = Field(..., description="Z-axis direction (forward)")


class RotationMetadata(BaseModel):
    """Additional rotation and viewing metadata."""
    viewing_angle_rad: float
    viewing_angle_deg: float
    ground_angle_rad: float
    ground_angle_deg: float
    wheel_to_wheel_2d: List[float]
    viewing_side: str
    target_wheel: str
    ellipse_angle_deg: Optional[float] = None
    ellipse_viewing_angle_deg: Optional[float] = None
    used_ellipse: bool = False


class Rotation(BaseModel):
    """Complete rotation information in multiple representations."""
    quaternion: Quaternion
    euler_angles: EulerAngles
    rotation_matrix: List[List[float]] = Field(..., description="3x3 rotation matrix")
    basis_vectors: BasisVectors
    metadata: RotationMetadata


class Scale(BaseModel):
    """Scale information for 3D asset."""
    uniform: float = Field(..., description="Uniform scale factor")
    radius_pixels: float = Field(..., description="Wheel radius in pixels")


class WheelTransform(BaseModel):
    """Complete transform data for a wheel."""
    position: Position3D
    rotation: Rotation
    scale: Scale
    bounding_box: BoundingBox
    confidence: float


class CameraIntrinsics(BaseModel):
    """Camera intrinsic matrix."""
    intrinsic_matrix: List[List[float]] = Field(..., description="3x3 intrinsic matrix")
    focal_length: float = Field(..., description="Focal length in pixels")
    fov: float = Field(..., description="Field of view in degrees")


class CameraCalibration(BaseModel):
    """
    Camera calibration parameters from homography/solvePnP.

    These parameters can be used to configure a Three.js PerspectiveCamera
    to match the original photo's viewpoint.
    """
    intrinsic_matrix: List[List[float]] = Field(..., description="3x3 camera intrinsic matrix")
    rotation_vector: List[float] = Field(..., description="3x1 Rodrigues rotation vector")
    translation_vector: List[float] = Field(..., description="3x1 camera position")
    fov: float = Field(..., description="Field of view in degrees")
    camera_height: float = Field(..., description="Estimated camera height in meters")
    focal_length: float = Field(..., description="Focal length in pixels")
    method: str = Field(..., description="Calibration method: 'solvePnP', 'homography', or 'default'")


class ThreeJSCamera(BaseModel):
    """Three.js compatible camera parameters."""
    fov: float = Field(..., description="Field of view in degrees")
    aspect: float = Field(..., description="Aspect ratio")
    near: float = Field(..., description="Near clipping plane")
    far: float = Field(..., description="Far clipping plane")
    position: List[float] = Field(..., description="Camera position [x, y, z]")
    quaternion: Quaternion = Field(..., description="Camera rotation")


class WheelEllipse(BaseModel):
    """Detected tire ellipse parameters."""
    center: List[float] = Field(..., description="Ellipse center [x, y]")
    axes: List[float] = Field(..., description="Major and minor axes")
    angle: float = Field(..., description="Rotation angle in degrees")
    axis_ratio: float = Field(..., description="Minor/major axis ratio")
    viewing_angle_deg: float = Field(..., description="Viewing angle in degrees")


class CarGeometry(BaseModel):
    """Overall car geometry information."""
    wheel_to_wheel_2d: List[float] = Field(..., description="Normalized wheel-to-wheel direction vector")
    ground_angle_deg: float = Field(..., description="Ground plane tilt in degrees")
    viewing_side: str = Field(..., description="Which side of car is visible: 'left' or 'right'")


class WheelDetection(BaseModel):
    """Individual wheel detection."""
    class_name: str = Field(alias="class", description="Detection class name")
    confidence: float = Field(..., description="Detection confidence score")
    bbox: BoundingBox

    class Config:
        populate_by_name = True


class CarDetection(BaseModel):
    """Individual car detection."""
    class_name: str = Field(alias="class", description="Detection class name")
    confidence: float = Field(..., description="Detection confidence score")
    bbox: BoundingBox

    class Config:
        populate_by_name = True


class WheelPositions(BaseModel):
    """Identified front and rear wheel positions."""
    front: Optional[WheelDetection] = None
    rear: Optional[WheelDetection] = None


class CarDetectionResult(BaseModel):
    """
    Complete detection result for a single car.

    This includes the car bounding box, all detected wheels, wheel transforms,
    camera calibration, and geometry information.
    """
    car_id: int = Field(..., description="Unique ID for this car")
    car: CarDetection
    wheels: List[WheelDetection] = Field(default_factory=list)
    wheel_count: int = Field(..., description="Number of wheels detected")
    wheel_positions: Optional[WheelPositions] = None
    rear_wheel_transform: Optional[WheelTransform] = None
    front_wheel_transform: Optional[WheelTransform] = None
    rear_wheel_ellipse: Optional[WheelEllipse] = None
    front_wheel_ellipse: Optional[WheelEllipse] = None
    car_geometry: Optional[CarGeometry] = None
    camera_calibration: Optional[CameraCalibration] = None
    threejs_camera: Optional[ThreeJSCamera] = None


class ImageDimensions(BaseModel):
    """Image dimensions in pixels."""
    width: int
    height: int


class HomographyDetectionResponse(BaseModel):
    """
    Complete response from /detect/homography endpoint.

    This response includes all detection data, geometry transforms,
    and camera calibration suitable for AR overlay in Three.js.
    """
    filename: str = Field(..., description="Source image filename")
    image_dimensions: ImageDimensions
    total_cars: int = Field(..., description="Number of cars detected")
    detections: List[CarDetectionResult] = Field(default_factory=list, description="Car detection results")
    filtered_to_largest: Optional[bool] = Field(default=False, description="Whether results were filtered to largest car only")

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "car_image.jpg",
                "image_dimensions": {"width": 1920, "height": 1080},
                "total_cars": 1,
                "detections": [
                    {
                        "car_id": 0,
                        "car": {
                            "class": "car",
                            "confidence": 0.95,
                            "bbox": {"x1": 100, "y1": 200, "x2": 800, "y2": 600, "width": 700, "height": 400}
                        },
                        "wheels": [],
                        "wheel_count": 2,
                        "camera_calibration": {
                            "fov": 55.0,
                            "camera_height": 1.5,
                            "focal_length": 1920.0,
                            "method": "solvePnP"
                        }
                    }
                ]
            }
        }
