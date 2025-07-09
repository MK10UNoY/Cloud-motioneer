from fastapi import FastAPI, File, UploadFile, Form, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
import uuid
from predictor import Predictor
from checkpoint import download_checkpoint
from cloud_uploader import download_and_unzip_from_gcs
from google.cloud import storage
from fastapi.responses import HTMLResponse, JSONResponse

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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Cloud Motion Predictor API."}

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

@app.post("/predict-gcs")
async def predict_gcs(payload: dict = Body(...)):
    """
    Expects JSON: { "gcs_uri": "gs://bucket/path/to/file.zip" }
    """
    gcs_uri = payload.get("gcs_uri")
    if not gcs_uri:
        return {"error": "Missing gcs_uri in request body."}
    temp_dir = f"temp_uploads/{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    try:
        file_paths = download_and_unzip_from_gcs(gcs_uri, temp_dir)
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

@app.post("/get-upload-url")
async def get_upload_url(payload: dict = Body(...)):
    """
    Expects JSON: { "filename": "yourfile.zip", "bucket": "optional-bucket-name" }
    Returns: { "url": signed_url, "gcs_uri": "gs://bucket/filename" }
    """
    filename = payload.get("filename")
    bucket_name = payload.get("bucket") or "cloud_checkpoint"  # Default bucket
    if not filename:
        return {"error": "Missing filename in request body."}
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(filename)
    url = blob.generate_signed_url(
        version="v4",
        expiration=600,  # 10 minutes
        method="PUT",
        content_type="application/zip"
    )
    gcs_uri = f"gs://{bucket_name}/{filename}"
    return {"url": url, "gcs_uri": gcs_uri}

@app.post("/sample")
async def sample_predict():
    """
    Predict using 6 .h5 files already uploaded to 'upload/' folder in the default bucket.
    """
    bucket_name = "cloud_checkpoint"  # Change if your bucket is different
    folder = "upload/"
    # List .h5 files in the folder
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=folder))
    h5_files = [b for b in blobs if b.name.endswith(".h5") and not b.name.endswith("/")]
    if len(h5_files) != 6:
        return {"error": f"Expected 6 .h5 files in gs://{bucket_name}/{folder}, found {len(h5_files)}."}
    # Download files to temp dir
    temp_dir = f"temp_uploads/{uuid.uuid4()}"
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    try:
        for blob in h5_files:
            fname = os.path.basename(blob.name)
            fpath = os.path.join(temp_dir, fname)
            blob.download_to_filename(fpath)
            file_paths.append(fpath)
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

@app.get("/isronaut/upload-ui", response_class=HTMLResponse)
def upload_ui():
    return """
    <html>
      <head>
        <title>Cloud Motion Predictor</title>
        <style>
          body { font-family: Arial; margin: 40px; background-color: #f7f7f7; }
          form { padding: 20px; background: white; border-radius: 8px; box-shadow: 0 0 10px #ccc; }
          input[type=file] { margin-bottom: 10px; }
          input[type=submit] { padding: 8px 16px; border: none; background-color: #4CAF50; color: white; border-radius: 4px; cursor: pointer; }
        </style>
      </head>
      <body>
        <h2>Upload 6 .h5 Files for Prediction</h2>
        <form action="/predict" enctype="multipart/form-data" method="post">
          <input type="file" name="files" multiple required><br>
          <input type="submit" value="Run Prediction">
        </form>
      </body>
    </html>
    """

@app.get("/sample-ui", response_class=HTMLResponse)
def sample_ui():
    return """
    <html>
      <head>
        <title>Sample Prediction</title>
        <style>
          body { font-family: Arial; margin: 40px; background-color: #f7f7f7; }
          .container { padding: 20px; background: white; border-radius: 8px; box-shadow: 0 0 10px #ccc; max-width: 500px; margin: auto; }
          button { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
          #result { margin-top: 20px; white-space: pre-wrap; }
        </style>
        <script>
        async function runSample() {
          document.getElementById('result').innerText = 'Running prediction...';
          const resp = await fetch('/sample', { method: 'POST' });
          const data = await resp.json();
          document.getElementById('result').innerText = JSON.stringify(data, null, 2);
        }
        </script>
      </head>
      <body>
        <div class="container">
          <h2>Run Sample Prediction</h2>
          <p>This will use the 6 .h5 files in the <b>upload/</b> folder of your bucket.</p>
          <button onclick="runSample()">Start Prediction</button>
          <div id="result"></div>
        </div>
      </body>
    </html>
    """
