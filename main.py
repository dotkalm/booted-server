from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import logging
from services.models.detector import TireDetector, CarWheelDetector
from services.utils.geometry import enrich_detection_with_geometry

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

# Global detector instance
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