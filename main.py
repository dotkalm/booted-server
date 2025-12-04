from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import logging
from services.models.detector import TireDetector, CarWheelDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Global detector instances
detector = None
car_wheel_detector = None

@app.on_event("startup")
async def load_models():
    """Load models after server starts listening"""
    global detector, car_wheel_detector
    logger.info("Loading detection models...")
    detector = TireDetector()
    car_wheel_detector = CarWheelDetector()
    logger.info("Models loaded successfully")

@app.get("/")
def read_root():
    return {"status": "working"}

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