import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler
from utility import slice_good_cycle, load_data


def build_training_data(good_cycles_list):
    all_training_windows = []

    for cycle in good_cycles_list:
        windows = slice_good_cycle(cycle)
        all_training_windows.extend(windows)

    print(f"Data sliced into {len(all_training_windows)} windows (Batch).")

    huge_2d_matrix = np.vstack(all_training_windows)

    scaler = RobustScaler()
    scaler.fit(huge_2d_matrix)

    global_median = scaler.center_
    global_iqr = scaler.scale_

    normalized_2d_matrix = scaler.transform(huge_2d_matrix)
    final_3d_tensor = normalized_2d_matrix.reshape(-1, 500, 3)

    print(f"\nThe Tensor going into Pytorch has shape: {final_3d_tensor.shape}")

    return final_3d_tensor, scaler, global_median, global_iqr

if __name__ == "__main__":

    RAW_DATA_FOLDER = '../data/raw/M01/OP07/good'
    PROCESSED_DATA_FOLDER = '../data/processed'

    os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

    print("Start loading raw data...")
    cycle_feb = load_data(month='Feb', year='2019', end=4, folder_path=RAW_DATA_FOLDER, machineId='M01', operation='OP07')
    cycle_aug = load_data(month='Aug', year='2019', end=15, folder_path=RAW_DATA_FOLDER, machineId='M01', operation='OP07')

    cycle_data_list = cycle_feb + cycle_aug

    if len(cycle_data_list) > 0:
        training_tensor, scaler, global_median, global_iqr = build_training_data(cycle_data_list)
        edge_params = {
            "version": "1.0",
            "machineId": "M01",
            "operation": "OP07",
            "global_median": global_median.tolist(), 
            "global_iqr": global_iqr.tolist()
        }

        print("\n--- Edge Device Global Params ---")
        print(f"X, Y, Z global median: {global_median}")
        print(f"X, Y, Z global IQR: {global_iqr}")

        params_save_path = os.path.join(PROCESSED_DATA_FOLDER, 'edge_params.json')

        with open(params_save_path, 'w') as f:
            json.dump(edge_params, f, indent=4)
        print(f"[*] Edge params saved to JSON: {params_save_path}")

        tensor_save_path = os.path.join(PROCESSED_DATA_FOLDER, 'training_tensor.npy')
        np.save(tensor_save_path, training_tensor)
        print(f"[*] Training 3D Tensor saved to NumPy archive: {tensor_save_path}")

else:
    print("No cycles found. Please check the raw data files and paths.")
