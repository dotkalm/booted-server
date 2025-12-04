from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import logging
from services.models.detector import TireDetector, CarWheelDetector

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

        output_path = "output_image_no_ext"
        image.save(output_path, format="JPEG")
        logger.info(f"Saved image to: {output_path}")

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