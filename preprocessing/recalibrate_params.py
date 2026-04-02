import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler
from utility import slice_bad_cycle, slice_good_cycle, load_cycle

def recalculate_scaler(validation_cycles_dict):
    """
    validation_cycles_dict: {'M01_Feb_2020_OP07_000_good': data, 'M01_Aug_2020_OP07_001_bad': data, ...}
    global_median, global_iqr: RobustScaler parameters from training data
    """
    processed_val_dict = {}
    good_cycle_windows = []

    for cycle_name, cycle_data in validation_cycles_dict.items():
        if 'bad' in cycle_name:
            windows = slice_bad_cycle(cycle_data)
        else:
            windows = slice_good_cycle(cycle_data)
            good_cycle_windows.extend(window)
        
        if not windows:
            print(f"Warning: No valid windows extracted from {cycle_name}. Skipping.")
            continue
          
        processed_val_dict[cycle_name] = cycle_3d
        # print(f"Processed {cycle_name}: {cycle_3d.shape} windows extracted.")

    good_cycle_matrix = np.vstack(good_cycle_windows)
    scaler = RobustScaler()
    scaler.fit(good_cycle_matrix)

    global_median = scaler.center_
    global_iqr = scaler.scale_

    for 
    cycle_normalized = scaler.transform(cycle_2d)
    cycle_3d = cycle_normalized.reshape(-1, 500, 3)

    return processed_val_dict, scaler, global_median, global_iqr

if __name__ == "__main__":
    GOOD_CYCLE_FOLDER = '../data/raw/M01_OP07/good'
    BAD_CYCLE_FOLDER = '../data/raw/M01_OP07/bad'
    PROCESSED_DATA_FOLDER = '/data/processed'

    os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

    print("Start loading raw validation data...")

    val_cycles = {}
    years = ['2020', '2021']
    month = 'Feb'
    for year in years:
        for i in range(4):
            file_id = f'M01_{month}_{year}_OP07_{i:03d}'
            good_cycle_path = os.path.join(GOOD_CYCLE_FOLDER, f'{file_id}.csv')
            good_cycle_name = f'{file_id}_good'
  
            if os.path.exists(good_cycle_path):
                val_cycles[good_cycle_name] = load_cycle(filepath=good_cycle_path)
            else:
                print(f"Skipping: File does not exist {good_cycle_path}")

    val_cycles['M01_Feb_2019_OP07_000_bad'] = load_cycle(filepath=os.path.join(BAD_CYCLE_FOLDER, 'M01_Feb_2019_OP07_000.csv'))
    val_cycles['M01_Aug_2019_OP07_000_bad'] = load_cycle(filepath=os.path.join(BAD_CYCLE_FOLDER, 'M01_Aug_2019_OP07_000.csv'))

    val_dict, _, global_median, global_iqr = recalculate_scaler(val_cycles)

    edge_params_new = {
        "version": "1.0",
        "machineId": "M01",
        "operation": "OP07",
        "global_median": global_median.tolist(), 
        "global_iqr": global_iqr.tolist()
    }
    params_path = os.path.join(PROCESSED_DATA_FOLDER, 'edge_params_new.json')
    with open(params_path, 'w') as f:
        json.dump(edge_params_new, f, indent=4)

    print("\n--- Edge Device Global Params ---")
    print(f"X, Y, Z global median: {global_median}")
    print(f"X, Y, Z global IQR: {global_iqr}")
    print(f"[*] Edge params saved to JSON: {params_path}")
    
    dict_path = os.path.join(PROCESSED_DATA_FOLDER, 'validation_new_scaler.npz')
    np.savez_compressed(dict_path, **processed_val_dict)
    print(f"\nValidation data archive saved to {save_path} with {len(processed_val_dict)} cycles.")
