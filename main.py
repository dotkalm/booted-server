from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import logging
from services.models.detector import TireDetector, CarWheelDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize detectors once at startup
detector = TireDetector()
car_wheel_detector = CarWheelDetector()

@app.get("/")
def read_root():
    return {"status": "working"}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
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