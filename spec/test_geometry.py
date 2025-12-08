"""
Unit tests for geometry calculations and detection pipeline.
These tests generate JSON fixtures for frontend Storybook integration.
"""

import pytest
import json
import os
from pathlib import Path
from PIL import Image
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.models.detector import CarWheelDetector
from services.utils.geometry import (
    enrich_detection_with_geometry,
    calculate_rotation_matrix,
    calculate_wheel_transform,
    identify_front_rear_wheels,
    calculate_wheel_center,
    calculate_wheel_radius,
    estimate_viewing_angle,
    estimate_car_direction,
    estimate_ground_plane_angle
)


# Paths
SPEC_DIR = Path(__file__).parent
FIXTURES_OUTPUT_DIR = SPEC_DIR / "fixtures"


@pytest.fixture(scope="module")
def car_wheel_detector():
    """Load the car wheel detector model once for all tests."""
    return CarWheelDetector()


@pytest.fixture
def image_one():
    """Load the one.jpg test image."""
    image_path = SPEC_DIR / "one.jpg"
    assert image_path.exists(), f"Test image not found: {image_path}"
    return Image.open(image_path)


@pytest.fixture
def image_two():
    """Load the two.jpg test image."""
    image_path = SPEC_DIR / "two.jpg"
    if image_path.exists():
        return Image.open(image_path)
    return None


@pytest.fixture
def image_three():
    """Load the three.jpg test image."""
    image_path = SPEC_DIR / "three.jpg"
    if image_path.exists():
        return Image.open(image_path)
    return None


@pytest.fixture
def all_test_images():
    """Load all test images from the spec directory."""
    images = {}
    for i, name in enumerate(["one", "two", "three", "four", "five", "six", "seven"], 1):
        image_path = SPEC_DIR / f"{name}.jpg"
        if image_path.exists():
            images[name] = Image.open(image_path)
    return images


def save_fixture(data: dict, filename: str):
    """Save detection results as a JSON fixture for Storybook."""
    FIXTURES_OUTPUT_DIR.mkdir(exist_ok=True)
    filepath = FIXTURES_OUTPUT_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n✓ Fixture saved: {filepath}")
    return filepath


class TestGeometryCalculations:
    """Test the geometry calculation functions."""
    
    def test_calculate_wheel_center(self):
        """Test wheel center calculation."""
        bbox = {"x1": 100, "y1": 200, "x2": 200, "y2": 300, "width": 100, "height": 100}
        center_x, center_y = calculate_wheel_center(bbox)
        
        assert center_x == 150.0
        assert center_y == 250.0
    
    def test_calculate_wheel_radius(self):
        """Test wheel radius calculation."""
        bbox = {"x1": 100, "y1": 200, "x2": 200, "y2": 300, "width": 100, "height": 100}
        radius = calculate_wheel_radius(bbox)
        
        assert radius == 50.0  # (100 + 100) / 4
    
    def test_estimate_viewing_angle_circular(self):
        """Test viewing angle estimation for circular wheel (straight-on view)."""
        bbox = {"x1": 0, "y1": 0, "x2": 100, "y2": 100, "width": 100, "height": 100}
        angle = estimate_viewing_angle(bbox)
        
        # Circular wheel = viewing straight on = 0 radians
        assert abs(angle) < 0.01
    
    def test_estimate_viewing_angle_elliptical(self):
        """Test viewing angle estimation for elliptical wheel (angled view)."""
        bbox = {"x1": 0, "y1": 0, "x2": 50, "y2": 100, "width": 50, "height": 100}
        angle = estimate_viewing_angle(bbox)
        
        # Width/Height = 0.5, so viewing at an angle
        assert angle > 0.5  # Should be around 1.05 radians (60 degrees)
    
    def test_identify_front_rear_wheels_two_wheels(self):
        """Test front/rear wheel identification with two wheels."""
        car_bbox = {"x1": 100, "y1": 200, "x2": 800, "y2": 500, "width": 700, "height": 300}
        wheels = [
            {"class": "wheel", "confidence": 0.9, "bbox": {"x1": 150, "y1": 380, "x2": 250, "y2": 480, "width": 100, "height": 100}},
            {"class": "wheel", "confidence": 0.85, "bbox": {"x1": 600, "y1": 385, "x2": 700, "y2": 485, "width": 100, "height": 100}}
        ]
        
        positions = identify_front_rear_wheels(wheels, car_bbox)
        
        assert positions["front"] is not None
        assert positions["rear"] is not None
        # Left wheel should be front, right wheel should be rear
        assert positions["front"]["bbox"]["x1"] == 150
        assert positions["rear"]["bbox"]["x1"] == 600
    
    def test_estimate_car_direction(self):
        """Test car direction estimation."""
        car_bbox = {"x1": 100, "y1": 200, "x2": 800, "y2": 500, "width": 700, "height": 300}
        wheels = [
            {"class": "wheel", "confidence": 0.9, "bbox": {"x1": 150, "y1": 380, "x2": 250, "y2": 480, "width": 100, "height": 100}},
            {"class": "wheel", "confidence": 0.85, "bbox": {"x1": 600, "y1": 385, "x2": 700, "y2": 485, "width": 100, "height": 100}}
        ]
        
        direction = estimate_car_direction(wheels, car_bbox)
        
        # Car should be facing left (negative x direction) based on wheel positions
        assert direction[0] < 0  # Negative x component
        assert abs(direction[1]) < 0.1  # Near-zero y component (level ground)
    
    def test_calculate_rotation_matrix_structure(self):
        """Test that rotation matrix has correct structure."""
        car_bbox = {"x1": 100, "y1": 200, "x2": 800, "y2": 500, "width": 700, "height": 300}
        wheels = [
            {"class": "wheel", "confidence": 0.9, "bbox": {"x1": 150, "y1": 380, "x2": 250, "y2": 480, "width": 100, "height": 100}},
            {"class": "wheel", "confidence": 0.85, "bbox": {"x1": 600, "y1": 385, "x2": 700, "y2": 485, "width": 100, "height": 100}}
        ]
        
        result = calculate_rotation_matrix(wheels, car_bbox, 1920, 1080)
        
        # Check all required fields are present
        assert "rotation_matrix" in result
        assert "basis_vectors" in result
        assert "euler_angles" in result
        assert "quaternion" in result
        assert "metadata" in result
        
        # Check rotation matrix is 3x3
        assert len(result["rotation_matrix"]) == 3
        assert all(len(row) == 3 for row in result["rotation_matrix"])
        
        # Check quaternion has 4 components
        assert all(k in result["quaternion"] for k in ["x", "y", "z", "w"])
        
        # Check euler angles
        assert all(k in result["euler_angles"] for k in ["x", "y", "z", "order"])


class TestDetectionPipeline:
    """Test the full detection pipeline with real images."""
    
    def test_detect_one_image(self, car_wheel_detector, image_one):
        """Test detection on one.jpg and generate fixture."""
        # Run detection
        results = car_wheel_detector.detect_cars_and_wheels(
            image_one,
            car_conf=0.3,
            wheel_conf=0.3
        )
        
        # Enrich with geometry
        enriched = enrich_detection_with_geometry(
            results, 
            image_one.width, 
            image_one.height
        )
        
        # Build fixture output
        fixture = {
            "source": "one.jpg",
            "timestamp": datetime.now().isoformat(),
            "image_dimensions": {
                "width": image_one.width,
                "height": image_one.height
            },
            "total_cars": enriched["total_cars"],
            "detections": enriched["results"]
        }
        
        # Save fixture
        save_fixture(fixture, "one_detection.json")
        
        # Basic assertions
        assert "total_cars" in enriched
        assert "results" in enriched
        
        # If cars were detected, verify structure
        if enriched["total_cars"] > 0:
            car_result = enriched["results"][0]
            assert "car" in car_result
            assert "wheels" in car_result
            assert "rear_wheel_transform" in car_result
            
            # If rear wheel was detected, verify transform
            if car_result["rear_wheel_transform"]:
                transform = car_result["rear_wheel_transform"]
                assert "position" in transform
                assert "rotation" in transform
                assert "scale" in transform
    
    def test_detect_all_images(self, car_wheel_detector, all_test_images):
        """Test detection on all images and generate combined fixture."""
        all_fixtures = []
        
        for name, image in all_test_images.items():
            # Run detection
            results = car_wheel_detector.detect_cars_and_wheels(
                image,
                car_conf=0.3,
                wheel_conf=0.3
            )
            
            # Enrich with geometry
            enriched = enrich_detection_with_geometry(
                results,
                image.width,
                image.height
            )
            
            fixture = {
                "source": f"{name}.jpg",
                "image_dimensions": {
                    "width": image.width,
                    "height": image.height
                },
                "total_cars": enriched["total_cars"],
                "detections": enriched["results"]
            }
            
            all_fixtures.append(fixture)
            
            # Also save individual fixture
            save_fixture(fixture, f"{name}_detection.json")
        
        # Save combined fixture
        combined = {
            "generated_at": datetime.now().isoformat(),
            "total_images": len(all_fixtures),
            "images": all_fixtures
        }
        save_fixture(combined, "all_detections.json")
        
        assert len(all_fixtures) > 0


class TestStorybookFixtures:
    """Generate specific fixtures optimized for Storybook."""
    
    def test_generate_storybook_fixture(self, car_wheel_detector, image_one):
        """Generate a Storybook-ready fixture with all transform data."""
        # Run detection
        results = car_wheel_detector.detect_cars_and_wheels(
            image_one,
            car_conf=0.3,
            wheel_conf=0.3
        )
        
        # Enrich with geometry
        enriched = enrich_detection_with_geometry(
            results,
            image_one.width,
            image_one.height
        )
        
        # Build Storybook fixture - flat structure for easy consumption
        storybook_fixture = {
            "_meta": {
                "generated_at": datetime.now().isoformat(),
                "source_image": "one.jpg",
                "image_width": image_one.width,
                "image_height": image_one.height,
                "description": "Detection results for Three.js wheel overlay"
            },
            "cars": []
        }
        
        for car_result in enriched.get("results", []):
            car_data = {
                "car_id": car_result["car_id"],
                "car_bbox": car_result["car"]["bbox"],
                "car_confidence": car_result["car"]["confidence"],
                "wheel_count": car_result["wheel_count"],
                "wheels": car_result["wheels"],
                "geometry": car_result.get("car_geometry", {})
            }
            
            # Add rear wheel transform (primary target for 3D asset)
            rear_transform = car_result.get("rear_wheel_transform")
            if rear_transform:
                car_data["rear_wheel"] = {
                    "detected": True,
                    "center": {
                        "normalized": {
                            "x": rear_transform["position"]["x"],
                            "y": rear_transform["position"]["y"],
                            "z": rear_transform["position"]["z"]
                        },
                        "pixels": {
                            "x": rear_transform["position"]["pixel_x"],
                            "y": rear_transform["position"]["pixel_y"]
                        }
                    },
                    "rotation": {
                        "quaternion": rear_transform["rotation"]["quaternion"],
                        "euler": rear_transform["rotation"]["euler_angles"],
                        "matrix": rear_transform["rotation"]["rotation_matrix"]
                    },
                    "scale": rear_transform["scale"]["uniform"],
                    "radius_pixels": rear_transform["scale"]["radius_pixels"],
                    "bbox": rear_transform["bounding_box"],
                    "confidence": rear_transform["confidence"],
                    "viewing_metadata": rear_transform["rotation"]["metadata"]
                }
            else:
                car_data["rear_wheel"] = {"detected": False}
            
            # Add front wheel transform
            front_transform = car_result.get("front_wheel_transform")
            if front_transform:
                car_data["front_wheel"] = {
                    "detected": True,
                    "center": {
                        "normalized": {
                            "x": front_transform["position"]["x"],
                            "y": front_transform["position"]["y"],
                            "z": front_transform["position"]["z"]
                        },
                        "pixels": {
                            "x": front_transform["position"]["pixel_x"],
                            "y": front_transform["position"]["pixel_y"]
                        }
                    },
                    "rotation": {
                        "quaternion": front_transform["rotation"]["quaternion"],
                        "euler": front_transform["rotation"]["euler_angles"],
                        "matrix": front_transform["rotation"]["rotation_matrix"]
                    },
                    "scale": front_transform["scale"]["uniform"],
                    "radius_pixels": front_transform["scale"]["radius_pixels"],
                    "bbox": front_transform["bounding_box"],
                    "confidence": front_transform["confidence"]
                }
            else:
                car_data["front_wheel"] = {"detected": False}
            
            storybook_fixture["cars"].append(car_data)
        
        # Save Storybook fixture
        save_fixture(storybook_fixture, "storybook_fixture.json")
        
        # Verify structure
        assert "_meta" in storybook_fixture
        assert "cars" in storybook_fixture
    
    def test_generate_threejs_ready_fixture(self, car_wheel_detector, image_one):
        """Generate a minimal fixture specifically for Three.js consumption."""
        results = car_wheel_detector.detect_cars_and_wheels(
            image_one, car_conf=0.3, wheel_conf=0.3
        )
        enriched = enrich_detection_with_geometry(
            results, image_one.width, image_one.height
        )
        
        # Minimal Three.js fixture
        threejs_fixture = {
            "imageSize": [image_one.width, image_one.height],
            "wheels": []
        }
        
        for car_result in enriched.get("results", []):
            rear = car_result.get("rear_wheel_transform")
            if rear:
                threejs_fixture["wheels"].append({
                    "type": "rear",
                    "carId": car_result["car_id"],
                    "position": [
                        rear["position"]["x"],
                        rear["position"]["y"],
                        rear["position"]["z"]
                    ],
                    "quaternion": [
                        rear["rotation"]["quaternion"]["x"],
                        rear["rotation"]["quaternion"]["y"],
                        rear["rotation"]["quaternion"]["z"],
                        rear["rotation"]["quaternion"]["w"]
                    ],
                    "scale": rear["scale"]["uniform"],
                    "pixelCenter": [
                        rear["position"]["pixel_x"],
                        rear["position"]["pixel_y"]
                    ],
                    "pixelRadius": rear["scale"]["radius_pixels"]
                })
            
            front = car_result.get("front_wheel_transform")
            if front:
                threejs_fixture["wheels"].append({
                    "type": "front",
                    "carId": car_result["car_id"],
                    "position": [
                        front["position"]["x"],
                        front["position"]["y"],
                        front["position"]["z"]
                    ],
                    "quaternion": [
                        front["rotation"]["quaternion"]["x"],
                        front["rotation"]["quaternion"]["y"],
                        front["rotation"]["quaternion"]["z"],
                        front["rotation"]["quaternion"]["w"]
                    ],
                    "scale": front["scale"]["uniform"],
                    "pixelCenter": [
                        front["position"]["pixel_x"],
                        front["position"]["pixel_y"]
                    ],
                    "pixelRadius": front["scale"]["radius_pixels"]
                })
        
        save_fixture(threejs_fixture, "threejs_fixture.json")
        
        assert "imageSize" in threejs_fixture
        assert "wheels" in threejs_fixture


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_no_wheels_detected(self):
        """Test handling when no wheels are detected."""
        mock_results = {
            "total_cars": 1,
            "results": [{
                "car_id": 0,
                "car": {
                    "class": "car",
                    "confidence": 0.9,
                    "bbox": {"x1": 100, "y1": 100, "x2": 500, "y2": 300, "width": 400, "height": 200}
                },
                "wheels": [],
                "wheel_count": 0
            }]
        }
        
        enriched = enrich_detection_with_geometry(mock_results, 1920, 1080)
        
        assert enriched["results"][0]["rear_wheel_transform"] is None
        assert enriched["results"][0]["front_wheel_transform"] is None
    
    def test_single_wheel_detected(self):
        """Test handling when only one wheel is detected."""
        mock_results = {
            "total_cars": 1,
            "results": [{
                "car_id": 0,
                "car": {
                    "class": "car",
                    "confidence": 0.9,
                    "bbox": {"x1": 100, "y1": 100, "x2": 500, "y2": 300, "width": 400, "height": 200}
                },
                "wheels": [
                    {"class": "wheel", "confidence": 0.85, "bbox": {"x1": 350, "y1": 220, "x2": 420, "y2": 290, "width": 70, "height": 70}}
                ],
                "wheel_count": 1
            }]
        }
        
        enriched = enrich_detection_with_geometry(mock_results, 1920, 1080)
        
        # Should still produce some result
        car_result = enriched["results"][0]
        # Either front or rear should have a value
        has_wheel = (car_result["front_wheel_transform"] is not None or 
                     car_result["rear_wheel_transform"] is not None)
        assert has_wheel
    
    def test_no_cars_detected(self):
        """Test handling when no cars are detected."""
        mock_results = {
            "total_cars": 0,
            "results": []
        }
        
        enriched = enrich_detection_with_geometry(mock_results, 1920, 1080)
        
        assert enriched["total_cars"] == 0
        assert len(enriched["results"]) == 0
