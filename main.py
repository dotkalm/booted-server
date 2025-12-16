from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from datetime import datetime
import json
import io
import logging
from services.models.detector import TireDetector, CarWheelDetector
from services.utils.geometry import enrich_detection_with_geometry
from services.utils.camera_calibration import (
    estimate_camera_from_wheels,
    opencv_to_threejs_camera
)
from schemas.detection import HomographyDetectionResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("=" * 50)
logger.info("Starting Booted API application")
logger.info("=" * 50)

app = FastAPI()

# Global detector instances
detector = None
car_wheel_detector = None
models_loaded = False

@app.on_event("startup")
async def load_models():
    """Load models after server starts listening"""
    global detector, car_wheel_detector, models_loaded
    try:
        logger.info("Server started successfully, beginning model load...")
        logger.info("Loading TireDetector model...")
        detector = TireDetector()
        logger.info("TireDetector loaded successfully")

        logger.info("Loading CarWheelDetector models...")
        car_wheel_detector = CarWheelDetector()
        logger.info("CarWheelDetector loaded successfully")

        models_loaded = True
        logger.info("✓ All models loaded and ready for inference")
    except Exception as e:
        logger.error(f"FATAL: Failed to load models: {e}", exc_info=True)
        models_loaded = False

@app.get("/")
def read_root():
    return {
        "status": "working",
        "models_loaded": models_loaded
    }

@app.get("/health")
def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy" if models_loaded else "starting",
        "tire_detector": detector is not None,
        "car_wheel_detector": car_wheel_detector is not None,
        "models_ready": models_loaded
    }

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    if detector is None:
        raise HTTPException(status_code=503, detail="Models still loading, please try again")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run detection
        detections = detector.detect_tires(image)

        return {
            "filename": file.filename,
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "tire_count": len(detections),
            "detections": detections
        }

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail="Detection failed")

@app.post("/detect/cars-and-wheels")
async def detect_cars_and_wheels(
    file: UploadFile = File(...),
    car_confidence: float = 0.3,
    wheel_confidence: float = 0.3
):
    """
        Main endpoint for two-stage car and wheel detection. 
        Detects cars first, then wheels within detected car regions followed by enrichment with 3D geometry for Three.js integration.
    """
    if car_wheel_detector is None:
        raise HTTPException(status_code=503, detail="Models still loading, please try again")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # export detection results to JSON and save image for fixures
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_output_path = f"image_fixture_{timestamp}.jpg"
        image.save(image_output_path, format="JPEG")
        logger.info(f"Saved image to: {image_output_path}")

        json_filename = f"detection_fixture_{timestamp}.json"
        print(f"JSON data successfully written to {json_filename}")
        """

        # Run two-stage detection
        results = car_wheel_detector.detect_cars_and_wheels(
            image,
            car_conf=car_confidence,
            wheel_conf=wheel_confidence
        )

        # Enrich with 3D geometry for Three.js integration
        results = enrich_detection_with_geometry(results, image.width, image.height, image)
        """
        with open(json_filename, 'w') as json_file:
            json.dump(results, json_file, indent=4)
        """
        print('Final detection results:', results["results"][0])
        return {
            "filename": file.filename,
            "width": image.width,
            "height": image.height,
            "total_cars": results["total_cars"],
            "detections": results["results"]
        }

    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/homography", response_model=HomographyDetectionResponse)
async def detect_with_homography(
    file: UploadFile = File(...),
    car_confidence: float = 0.3,
    wheel_confidence: float = 0.3,
    wheelbase_m: float = 2.7,
    wheel_diameter_m: float = 0.65
):
    """
    Advanced detection endpoint with homography-based camera calibration.

    This endpoint performs two-stage car and wheel detection, enriches results
    with 3D geometry transforms, and computes camera calibration parameters
    using OpenCV's solvePnP method for accurate AR placement.

    Args:
        file: Uploaded image file
        car_confidence: Confidence threshold for car detection (0-1)
        wheel_confidence: Confidence threshold for wheel detection (0-1)
        wheelbase_m: Expected wheelbase distance in meters (default: 2.7m)
        wheel_diameter_m: Expected wheel diameter in meters (default: 0.65m)

    Returns:
        Typed response with detections, geometry, and camera calibration
    """
    if car_wheel_detector is None:
        raise HTTPException(status_code=503, detail="Models still loading, please try again")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run two-stage detection
        results = car_wheel_detector.detect_cars_and_wheels(
            image,
            car_conf=car_confidence,
            wheel_conf=wheel_confidence
        )

        # Enrich with 3D geometry for Three.js integration
        results = enrich_detection_with_geometry(results, image.width, image.height, image)

        # Add camera calibration for each detected car
        for car_result in results.get("results", []):
            wheel_positions = car_result.get("wheel_positions", {})
            front_wheel = wheel_positions.get("front")
            rear_wheel = wheel_positions.get("rear")

            if front_wheel and rear_wheel:
                # Estimate camera parameters using homography/solvePnP
                camera_params = estimate_camera_from_wheels(
                    front_wheel_bbox=front_wheel.get("bbox"),
                    rear_wheel_bbox=rear_wheel.get("bbox"),
                    image_width=image.width,
                    image_height=image.height,
                    wheelbase_m=wheelbase_m,
                    wheel_diameter_m=wheel_diameter_m
                )

                car_result["camera_calibration"] = camera_params

                # Also provide Three.js-ready camera config
                car_result["threejs_camera"] = opencv_to_threejs_camera(
                    camera_params,
                    image.width,
                    image.height
                )

                logger.info(f"Camera calibration: method={camera_params['method']}, "
                           f"FOV={camera_params['fov']:.1f}°, "
                           f"height={camera_params['camera_height']:.2f}m")
            else:
                logger.warning(f"Car {car_result.get('car_id')} missing wheels, "
                             "cannot calibrate camera")
                car_result["camera_calibration"] = None
                car_result["threejs_camera"] = None

        # Format response
        response_data = {
            "filename": file.filename,
            "image_dimensions": {
                "width": image.width,
                "height": image.height
            },
            "total_cars": results["total_cars"],
            "detections": results["results"],
            "filtered_to_largest": results.get("filtered_to_largest", False)
        }

        return response_data

    except Exception as e:
        logger.error(f"Homography detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))