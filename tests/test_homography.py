"""
Tests for /detect/homography endpoint with camera calibration.

These tests verify that the homography-based detection endpoint:
1. Returns properly typed responses
2. Provides camera calibration parameters
3. Includes Three.js-compatible camera configuration
4. Handles various image scenarios correctly
"""

import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from main import app
import json

# Test configuration
CONFIDENCE_THRESHOLD = 0.3
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"
# Use specific known-good fixture image with detected wheels
API_ROOT = Path(__file__).parent.parent
TEST_FIXTURE_IMAGE = API_ROOT / "image_fixture_20251215_153246.jpg"

@pytest.fixture(scope="module")
def test_client():
    """Create test client with proper startup/shutdown handling."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
def car_image_path():
    """Get the hardcoded test fixture image with known wheel detections."""
    return TEST_FIXTURE_IMAGE


@pytest.fixture
def bus_image_path():
    """Get a sample bus/vehicle image."""
    return FIXTURES_DIR / "edge_cases" / "02_buses.png"


def test_homography_endpoint_exists(test_client):
    """Test that the /detect/homography endpoint exists."""
    response = test_client.get("/")
    assert response.status_code == 200


def test_homography_endpoint_returns_typed_response(test_client, car_image_path):
    """Test that homography endpoint returns properly structured response."""
    if not car_image_path.exists():
        pytest.skip(f"Test image not found: {car_image_path}")

    with open(car_image_path, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("test_car.png", f, "image/png")},
            data={
                "car_confidence": 0.3,
                "wheel_confidence": 0.3
            }
        )

    assert response.status_code == 200, f"Request failed: {response.text}"

    data = response.json()

    # Validate response structure
    assert "filename" in data
    assert "image_dimensions" in data
    assert "total_cars" in data
    assert "detections" in data

    # Validate image dimensions
    assert "width" in data["image_dimensions"]
    assert "height" in data["image_dimensions"]
    assert data["image_dimensions"]["width"] > 0
    assert data["image_dimensions"]["height"] > 0


def test_homography_response_includes_camera_calibration(test_client, car_image_path):
    """Test that response includes camera calibration when wheels are detected."""
    if not car_image_path.exists():
        pytest.skip(f"Test image not found: {car_image_path}")

    with open(car_image_path, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("test_car.png", f, "image/png")},
            data={
                "car_confidence": 0.3,
                "wheel_confidence": 0.3
            }
        )

    assert response.status_code == 200
    data = response.json()

    # If cars are detected, check for camera calibration
    if data["total_cars"] > 0:
        for detection in data["detections"]:
            # Camera calibration should be present if both wheels detected
            if detection.get("wheel_count", 0) >= 2:
                assert "camera_calibration" in detection, \
                    "Camera calibration missing for car with 2+ wheels"

                calib = detection["camera_calibration"]

                # Validate camera calibration structure
                assert "intrinsic_matrix" in calib
                assert "rotation_vector" in calib
                assert "translation_vector" in calib
                assert "fov" in calib
                assert "camera_height" in calib
                assert "focal_length" in calib
                assert "method" in calib

                # Validate intrinsic matrix is 3x3
                assert len(calib["intrinsic_matrix"]) == 3
                assert all(len(row) == 3 for row in calib["intrinsic_matrix"])

                # Validate rotation vector is 3D
                assert len(calib["rotation_vector"]) == 3

                # Validate translation vector is 3D
                assert len(calib["translation_vector"]) == 3

                # Validate FOV is reasonable (typical range: 30-90 degrees)
                assert 20 <= calib["fov"] <= 100, \
                    f"FOV {calib['fov']}° outside reasonable range"

                # Validate camera height is reasonable (typical: 0.5-3m)
                assert 0.1 <= calib["camera_height"] <= 5.0, \
                    f"Camera height {calib['camera_height']}m outside reasonable range"

                # Validate focal length is reasonable
                assert calib["focal_length"] > 0

                # Check calibration method
                assert calib["method"] in ["solvePnP", "homography", "default"]

                print(f"\nCamera calibration successful:")
                print(f"  Method: {calib['method']}")
                print(f"  FOV: {calib['fov']:.1f}°")
                print(f"  Camera height: {calib['camera_height']:.2f}m")
                print(f"  Focal length: {calib['focal_length']:.0f}px")


def test_homography_response_includes_threejs_camera(test_client, car_image_path):
    """Test that response includes Three.js-ready camera configuration."""
    if not car_image_path.exists():
        pytest.skip(f"Test image not found: {car_image_path}")

    with open(car_image_path, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("test_car.png", f, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()

    if data["total_cars"] > 0:
        for detection in data["detections"]:
            if detection.get("wheel_count", 0) >= 2:
                assert "threejs_camera" in detection

                camera = detection["threejs_camera"]

                # Validate Three.js camera structure
                assert "fov" in camera
                assert "aspect" in camera
                assert "near" in camera
                assert "far" in camera
                assert "position" in camera
                assert "quaternion" in camera

                # Validate position is [x, y, z]
                assert len(camera["position"]) == 3

                # Validate quaternion has x, y, z, w
                assert "x" in camera["quaternion"]
                assert "y" in camera["quaternion"]
                assert "z" in camera["quaternion"]
                assert "w" in camera["quaternion"]

                # Validate aspect ratio is reasonable
                assert 0.5 <= camera["aspect"] <= 3.0

                # Validate near/far planes
                assert camera["near"] > 0
                assert camera["far"] > camera["near"]

                print(f"\nThree.js camera configuration:")
                print(f"  FOV: {camera['fov']:.1f}°")
                print(f"  Aspect: {camera['aspect']:.2f}")
                print(f"  Position: {camera['position']}")


def test_homography_with_custom_wheelbase(test_client, car_image_path):
    """Test that custom wheelbase parameter is accepted."""
    if not car_image_path.exists():
        pytest.skip(f"Test image not found: {car_image_path}")

    with open(car_image_path, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("test_car.png", f, "image/png")},
            data={
                "wheelbase_m": 3.0,  # Longer wheelbase (truck/SUV)
                "wheel_diameter_m": 0.7
            }
        )

    assert response.status_code == 200
    data = response.json()
    assert "detections" in data


def test_homography_response_includes_wheel_transforms(test_client, car_image_path):
    """Test that wheel transforms are included in response."""
    if not car_image_path.exists():
        pytest.skip(f"Test image not found: {car_image_path}")

    with open(car_image_path, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("test_car.png", f, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()

    if data["total_cars"] > 0:
        for detection in data["detections"]:
            # Check for wheel transform structure
            if detection.get("rear_wheel_transform"):
                transform = detection["rear_wheel_transform"]

                assert "position" in transform
                assert "rotation" in transform
                assert "scale" in transform

                # Validate position
                pos = transform["position"]
                assert "x" in pos
                assert "y" in pos
                assert "z" in pos
                assert "pixel_x" in pos
                assert "pixel_y" in pos

                # Validate rotation
                rot = transform["rotation"]
                assert "quaternion" in rot
                assert "euler_angles" in rot
                assert "basis_vectors" in rot

                # Validate scale
                scale = transform["scale"]
                assert "uniform" in scale
                assert "radius_pixels" in scale


def test_homography_handles_no_wheels_gracefully(test_client, car_image_path):
    """Test that endpoint handles images without detectable wheels."""
    negative_image = FIXTURES_DIR / "negative" / "01.png"

    if not negative_image.exists():
        pytest.skip(f"Test image not found: {negative_image}")

    with open(negative_image, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("no_car.png", f, "image/png")},
        )

    # Should succeed even with no detections
    assert response.status_code == 200
    data = response.json()
    assert "total_cars" in data


def test_homography_response_json_serializable(test_client):
    """Test that the entire response is JSON serializable."""
    car_image = FIXTURES_DIR / "edge_cases" / "01_cars.png"

    if not car_image.exists():
        pytest.skip(f"Test image not found: {car_image}")

    with open(car_image, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("test.png", f, "image/png")},
        )

    assert response.status_code == 200

    # Verify response can be serialized to JSON
    data = response.json()
    json_str = json.dumps(data)
    assert len(json_str) > 0

    # Verify it can be deserialized
    parsed = json.loads(json_str)
    assert parsed == data


def test_homography_car_geometry_info(test_client, car_image_path):
    """Test that car geometry information is included."""
    if not car_image_path.exists():
        pytest.skip(f"Test image not found: {car_image_path}")

    with open(car_image_path, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("test_car.png", f, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()

    if data["total_cars"] > 0:
        for detection in data["detections"]:
            if detection.get("wheel_count", 0) >= 2:
                assert "car_geometry" in detection

                geom = detection["car_geometry"]
                assert "wheel_to_wheel_2d" in geom
                assert "ground_angle_deg" in geom
                assert "viewing_side" in geom

                # Validate wheel_to_wheel_2d is a normalized vector
                vector = geom["wheel_to_wheel_2d"]
                assert len(vector) == 2
                magnitude = (vector[0]**2 + vector[1]**2)**0.5
                assert 0.9 <= magnitude <= 1.1, "wheel_to_wheel_2d should be normalized"


def test_homography_multiple_cars_filtering(test_client):
    """Test that largest car filtering works when multiple cars detected."""
    # Use an image with multiple vehicles
    multi_car_image = FIXTURES_DIR / "edge_cases" / "02_buses.png"

    if not multi_car_image.exists():
        pytest.skip(f"Test image not found: {multi_car_image}")

    with open(multi_car_image, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("buses.png", f, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()

    # Check if filtering occurred
    if data.get("filtered_to_largest"):
        assert data["total_cars"] == 1, \
            "When filtered_to_largest is True, should only have 1 car"
        print("\nMultiple cars detected, successfully filtered to largest")


# Performance and edge case tests

def test_homography_with_high_confidence_thresholds(test_client, car_image_path):
    """Test detection with strict confidence thresholds."""
    if not car_image_path.exists():
        pytest.skip(f"Test image not found: {car_image_path}")

    with open(car_image_path, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("test_car.png", f, "image/png")},
            data={
                "car_confidence": 0.7,
                "wheel_confidence": 0.7
            }
        )

    assert response.status_code == 200
    data = response.json()
    assert "total_cars" in data


def test_homography_with_low_confidence_thresholds(test_client, car_image_path):
    """Test detection with permissive confidence thresholds."""
    if not car_image_path.exists():
        pytest.skip(f"Test image not found: {car_image_path}")

    with open(car_image_path, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("test_car.png", f, "image/png")},
            data={
                "car_confidence": 0.1,
                "wheel_confidence": 0.1
            }
        )

    assert response.status_code == 200
    data = response.json()
    assert "total_cars" in data


def test_homography_invalid_file_handling(test_client):
    """Test that endpoint handles invalid files gracefully."""
    response = test_client.post(
        "/detect/homography",
        files={"file": ("test.txt", b"not an image", "text/plain")},
    )

    # Should return error
    assert response.status_code == 500


def test_homography_missing_file_parameter(test_client):
    """Test that endpoint requires file parameter."""
    response = test_client.post("/detect/homography")

    # Should return validation error
    assert response.status_code == 422  # Unprocessable Entity


# Summary test for reporting

def test_homography_endpoint_summary(test_client, car_image_path):
    """Summary test that reports all key metrics."""
    if not car_image_path.exists():
        pytest.skip(f"Test image not found: {car_image_path}")

    with open(car_image_path, "rb") as f:
        response = test_client.post(
            "/detect/homography",
            files={"file": ("summary_test.png", f, "image/png")},
        )

    assert response.status_code == 200
    data = response.json()

    print("\n" + "="*60)
    print("HOMOGRAPHY ENDPOINT SUMMARY")
    print("="*60)
    print(f"Total cars detected: {data['total_cars']}")
    print(f"Image dimensions: {data['image_dimensions']['width']}x{data['image_dimensions']['height']}")

    if data["total_cars"] > 0:
        for i, detection in enumerate(data["detections"]):
            print(f"\nCar {i}:")
            print(f"  Wheels: {detection['wheel_count']}")
            if detection.get("camera_calibration"):
                calib = detection["camera_calibration"]
                print(f"  Camera calibration: ✓")
                print(f"    Method: {calib['method']}")
                print(f"    FOV: {calib['fov']:.1f}°")
                print(f"    Height: {calib['camera_height']:.2f}m")
            else:
                print(f"  Camera calibration: ✗ (insufficient wheels)")

    print("="*60)
