from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
import uuid
from predictor import Predictor
from checkpoint import download_checkpoint

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure the Checkpoint directory exists
download_checkpoint("cloud_checkpoint", "cloud_motion_diffusion_with_time_embed_new_loss_final.pth", "checkpoints/cloud_motion_v1.pth")
# Load the model once
predictor = Predictor("checkpoints/cloud_motion_v1.pth", interval_minutes=30) 

@app.post("/predict")
async def predict(files: List[UploadFile] = File(...)):
    # Create a temporary directory to save uploaded files
    temp_dir = f"temp_uploads/{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)

    file_paths = []
    for file in files:
        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)

    try:
        result = predictor.predict_from_files(file_paths)
        return {
            "predicted_timestamps": result["predicted_timestamps"],
            "predicted_frames": result["predicted_frames"].tolist(),
            "predicted_shape": list(result["predicted_frames"].shape)
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        shutil.rmtree(temp_dir)
