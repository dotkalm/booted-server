from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "working"}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    contents = await file.read()

    # Open with Pillow
    image = Image.open(io.BytesIO(contents))

    return {
        "filename": file.filename,
        "width": image.width,
        "height": image.height,
        "format": image.format,
        "detections": []
    }
