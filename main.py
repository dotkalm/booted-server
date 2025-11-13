from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import logging
from services.models.detector import TireDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize detector once at startup
detector = TireDetector()

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