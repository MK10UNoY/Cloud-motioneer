import os
import zipfile
from google.cloud import storage

def download_and_unzip_from_gcs(gcs_uri: str, dest_dir: str):
    """
    Downloads a zip file from GCS and extracts it to dest_dir.
    Returns a list of extracted file paths.
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs_uri must start with gs://")
    
    # Parse bucket and blob
    parts = gcs_uri[5:].split("/", 1)
    bucket_name, blob_path = parts[0], parts[1]
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "input.zip")

    # Download from GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(zip_path)

    # Unzip
    extracted_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
        for name in zip_ref.namelist():
            fpath = os.path.join(dest_dir, name)
            if os.path.isfile(fpath):
                extracted_files.append(fpath)
    os.remove(zip_path)
    return extracted_files
