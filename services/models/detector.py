from ultralytics import YOLO
from PIL import Image
from typing import List, Dict, Any
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class TireDetector:
    def __init__(self, model_path: str = None):
        """Initialize the tire detector with the model."""
        if model_path is None:
            # Look for model 2 levels up from this file (in booted_server directory)
            base_path = Path(__file__).parent.parent.parent
            model_name = os.getenv("MODEL_PATH", "results/rb600_wheel_detection3/weights/best.pt")
            model_path = base_path / model_name

        try:
            self.model = YOLO(str(model_path))
            logger.info(f"Model loaded successfully from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def detect_tires(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect tires in the given image."""
        results = self.model(image)

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    confidence = box.conf[0].item()
                    class_id = int(box.cls[0].item())

                    # Filter for tire class (adjust class_id as needed)
                    if class_id == 1:  # Class 1 = Rim (tire/wheel)
                        detections.append({
                            "class": "tire",
                            "confidence": round(confidence, 3),
                            "bbox": {
                                "x1": int(x1),
                                "y1": int(y1),
                                "x2": int(x2),
                                "y2": int(y2),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1)
                            }
                        })

        return detections


class CarWheelDetector:
    def __init__(self, car_model_path: str = None, wheel_model_path: str = None):
        """Initialize with separate car and wheel detection models."""

        # Car detector - use pre-trained COCO model
        if car_model_path is None:
            car_model_path = "yolov8n.pt"  # Pre-trained, has 'car' class

        # Wheel detector - your custom trained model
        if wheel_model_path is None:
            base_path = Path(__file__).parent.parent.parent
            wheel_model_path = base_path / "results/street_wheel_detection_v1/weights/best.pt"

        try:
            self.car_model = YOLO(str(car_model_path))
            self.wheel_model = YOLO(str(wheel_model_path))
            logger.info(f"Car model loaded from {car_model_path}")
            logger.info(f"Wheel model loaded from {wheel_model_path}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise

    def detect_cars(self, image: Image.Image, confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Detect cars in the image using pre-trained model."""
        results = self.car_model(image)

        cars = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].item())
                    # COCO dataset: class 2 = car
                    if class_id == 2:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()

                        if confidence >= confidence_threshold:
                            cars.append({
                                "class": "car",
                                "confidence": round(confidence, 3),
                                "bbox": {
                                    "x1": int(x1),
                                    "y1": int(y1),
                                    "x2": int(x2),
                                    "y2": int(y2),
                                    "width": int(x2 - x1),
                                    "height": int(y2 - y1)
                                }
                            })

        return cars

    def detect_wheels_in_region(self, image: Image.Image, bbox: Dict[str, int],
                                confidence_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Detect wheels within a specific bounding box region."""
        # Crop the image to the car region
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

        # Add padding to capture wheels that might be slightly outside
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(image.width, x2 + padding)
        y2 = min(image.height, y2 + padding)

        cropped = image.crop((x1, y1, x2, y2))

        # Run wheel detection on cropped region
        results = self.wheel_model(cropped)

        wheels = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = box.conf[0].item()

                    if confidence >= confidence_threshold:
                        # Get coordinates relative to crop
                        crop_x1, crop_y1, crop_x2, crop_y2 = box.xyxy[0].tolist()

                        # Map back to original image coordinates
                        orig_x1 = int(crop_x1 + x1)
                        orig_y1 = int(crop_y1 + y1)
                        orig_x2 = int(crop_x2 + x1)
                        orig_y2 = int(crop_y2 + y1)

                        wheels.append({
                            "class": "wheel",
                            "confidence": round(confidence, 3),
                            "bbox": {
                                "x1": orig_x1,
                                "y1": orig_y1,
                                "x2": orig_x2,
                                "y2": orig_y2,
                                "width": orig_x2 - orig_x1,
                                "height": orig_y2 - orig_y1
                            }
                        })

        return wheels

    def detect_cars_and_wheels(self, image: Image.Image,
                               car_conf: float = 0.3,
                               wheel_conf: float = 0.3) -> Dict[str, Any]:
        """Full pipeline: detect cars, then detect wheels within each car."""
        # Step 1: Detect cars
        cars = self.detect_cars(image, car_conf)

        # Step 2: For each car, detect wheels
        results = []
        for i, car in enumerate(cars):
            wheels = self.detect_wheels_in_region(image, car["bbox"], wheel_conf)

            results.append({
                "car_id": i,
                "car": car,
                "wheels": wheels,
                "wheel_count": len(wheels)
            })

        return {
            "total_cars": len(cars),
            "results": results
        }