from google.cloud import storage
import os

def download_checkpoint(bucket_name: str, blob_name: str, destination_path: str):
    if os.path.exists(destination_path):
        print(f"Checkpoint already exists at {destination_path}")
        return

    print(f"Downloading checkpoint {blob_name} from bucket {bucket_name}...")
    client = storage.Client()  # Uses default GCP credentials
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(destination_path)
    print(f"Downloaded checkpoint to {destination_path}")
