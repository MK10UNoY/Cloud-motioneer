import os
import re
import h5py
import torch
import numpy as np
from datetime import datetime, timedelta
from model_inference import normalize_sequence, load_model, predict_future_frames

class Predictor:
    def __init__(self, checkpoint_path, interval_minutes=30):
        self.diffusion, self.device = load_model(checkpoint_path)
        self.interval = timedelta(minutes=interval_minutes)

    def _parse_timestamp_from_filename(self, filename):
        match = re.search(r"(\d{2}[A-Z]{3}\d{4})_(\d{4})", filename)
        if not match:
            raise ValueError(f"Filename {filename} does not match expected format.")
        date_str, time_str = match.groups()
        datetime_str = date_str + time_str  # e.g. "01JAN2025" + "0000"
        dt = datetime.strptime(datetime_str, "%d%b%Y%H%M")
        return dt

    def _load_h5_sequence(self, file_paths):
        band_list = ['IMG_VIS', 'IMG_WV', 'IMG_TIR1', 'IMG_TIR2', 'IMG_MIR', 'IMG_SWIR']
        sequence = []
        timestamps = []

        sorted_paths = sorted(file_paths)  # optional: enforce chronological order
        for path in sorted_paths:
            with h5py.File(path, 'r') as f:
                frame = [f[band][0] for band in band_list]  # list of 6 arrays (128x128)
                stacked = np.stack(frame, axis=0)  # (6, 128, 128)
                sequence.append(stacked)
            timestamps.append(self._parse_timestamp_from_filename(os.path.basename(path)))

        x_np = np.stack(sequence, axis=0)  # (6, 6, 128, 128)
        x = torch.tensor(x_np, dtype=torch.float32)
        return x, timestamps

    def _load_tensor_file(self, tensor_path):
        if tensor_path.endswith(".npy"):
            x_np = np.load(tensor_path)
        elif tensor_path.endswith(".pt"):
            x_np = torch.load(tensor_path).numpy()
        else:
            raise ValueError("Unsupported tensor file format. Use .npy or .pt")
        x = torch.tensor(x_np, dtype=torch.float32)
        return x

    def predict_from_files(self, file_paths):
        if len(file_paths) == 6 and all(f.endswith(".h5") for f in file_paths):
            x, timestamps = self._load_h5_sequence(file_paths)
        elif len(file_paths) == 1 and file_paths[0].endswith(('.npy', '.pt')):
            x = self._load_tensor_file(file_paths[0])
            timestamps = []
        else:
            raise ValueError("Expected 6 .h5 files or 1 tensor file (.npy/.pt)")

        x = normalize_sequence(x)  # (6, 6, 128, 128)
        x = x.unsqueeze(0)         # (1, 6, 6, 128, 128)
        output = predict_future_frames(self.diffusion, x, self.device)  # (1, 8, 6, 128, 128)

        predicted = output[:, 6:]  # (1, 2, 6, 128, 128)

        if timestamps:
            t6 = timestamps[-1]
            t7 = t6 + self.interval
            t8 = t6 + 2 * self.interval
            predicted_times = [t7.strftime("%Y-%m-%d %H:%M"), t8.strftime("%Y-%m-%d %H:%M")]
        else:
            predicted_times = ["T7", "T8"]

        return {
            "predicted_timestamps": result["predicted_timestamps"],
            "predicted_frames": result["predicted_frames"].tolist(),
            "predicted_shape": list(result["predicted_frames"].shape)
        }   
