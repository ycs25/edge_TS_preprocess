import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler

def apply_symlog(data):
    return np.sign(data) * np.log1p(np.abs(data))

def slice_good_cycle(cycle_data, window_size=500, step=250):
    """
    Slicing of good cycles: discard tail shorter than 500 
    cycle_data: shape (N, 3) numpy array
    """
    windows = []
    num_steps = len(cycle_data)
    
    # Discard Tail
    for start_idx in range(0, num_steps - window_size + 1, step):
        end_idx = start_idx + window_size
        window = cycle_data[start_idx:end_idx, :] # shape (500, 3)
        windows.append(window)
        
    return windows

def load_data(month, year, end, folder_path, machineId='M01', operation='OP07', start=0):
    cycle_data_list = []

    for i in range(start, end + 1):
        filename = f"{machineId}_{month}_{year}_{operation}_{i:03d}.csv"
        filepath = os.path.join(folder_path, filename)

        if not os.path.exists(filepath):
            print(f"Warning: Cannot find {filename}, file skiped.")
            continue

        df = pd.read_csv(filepath)
        raw_sensor_data = df[['x','y','z']].values

        symlog_data = apply_symlog(raw_sensor_data)
        cycle_data_list.append(symlog_data)

    return cycle_data_list

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

    print("\n--- Edge Device Global Params ---")
    print(f"X, Y, Z global median: {global_median}")
    print(f"X, Y, Z global IQR: {global_iqr}")

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
        params_save_path = os.path.join(PROCESSED_DATA_FOLDER, 'edge_params.json')

        with open(params_save_path, 'w') as f:
            json.dump(edge_params, f, indent=4)
        print(f"[*] Edge params saved to JSON: {params_save_path}")

        tensor_save_path = os.path.join(PROCESSED_DATA_FOLDER, 'training_tensor.npy')
        np.save(tensor_save_path, training_tensor)
        print(f"[*] Training 3D Tensor saved to NumPy archive: {tensor_save_path}")

else:
    print("No cycles found. Please check the raw data files and paths.")
