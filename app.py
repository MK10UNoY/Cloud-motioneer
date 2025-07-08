# app.py - FastAPI wrapper for Predictor

from fastapi import FastAPI, UploadFile, File
from predictor import Predictor
import shutil
import os
from typing import List
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Create app instance
app = FastAPI()

# Enable CORS (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
predictor = Predictor("checkpoints/cloud_motion_v1.pth", interval_minutes=30)

# Ensure temp dir exists
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    paths = []
    for f in files:
        dest_path = os.path.join(TEMP_DIR, f.filename)
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        paths.append(dest_path)

    try:
        result = predictor.predict_from_files(paths)
        return {
            "predicted_timestamps": result["predicted_timestamps"],
            "predicted_shape": list(result["predicted_frames"].shape)
        }
    finally:
        # Clean up temp files
        for p in paths:
            if os.path.exists(p):
                os.remove(p)

# Optional root check
@app.get("/")
def root():
    return {"status": "API running. POST 6 .h5 files to /predict."}

# For direct run
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
