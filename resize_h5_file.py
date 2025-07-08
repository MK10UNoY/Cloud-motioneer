import h5py
import numpy as np
import cv2
import os

def resize_h5_file(input_path, output_path, size=(128, 128), image_keys=None):
    """
        Resizes selected image datasets in an HDF5 (.h5) file and saves the output.
        Parameters:
        - input_path (str): Path to the original .h5 file.
        - output_path (str): Path to save the resized .h5 file.
        - size (tuple): Target dimensions (width, height), default is (128, 128).
        - image_keys (list or None): Keys to resize. If None, default set is used.
    """
    if image_keys is None:
        image_keys = ['IMG_VIS', 'IMG_TIR1', 'IMG_TIR2', 'IMG_SWIR', 'IMG_MIR', 'IMG_WV']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with h5py.File(input_path, 'r') as fin, h5py.File(output_path, 'w') as fout:
        for key in fin:
            data = fin[key][()]
            if key in image_keys:
                img = np.squeeze(data)
                resized = cv2.resize(img.astype(np.float32), size, interpolation=cv2.INTER_AREA)
                resized = np.clip(resized, 0, 65535).astype(np.uint16)
                fout.create_dataset(key, data=resized[np.newaxis, :, :])
            else:
                fin.copy(key, fout)

# 🔽 Example usage (remove or comment this part out if integrating into backend API)
# if __name__ == "__main__":
#     input_file = "/path/to/input.h5"
#     output_file = "/path/to/output_resized.h5"
#     resize_h5_file(input_file, output_file)
