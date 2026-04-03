import os
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import RobustScaler
from utility import slice_bad_cycle, slice_good_cycle, load_cycle

def build_testing_data(testing_cycles_dict, global_median, global_iqr, save_path='data/processed/testing_cycles.npz'):
    """
    testing_cycles_dict: {'M01_Feb_2020_OP07_000_good': data, 'M01_Aug_2020_OP07_001_bad': data, ...}
    global_median, global_iqr: RobustScaler parameters from training data
    """
    processed_val_dict = {}

    for cycle_name, cycle_data in testing_cycles_dict.items():
        if 'bad' in cycle_name:
            windows = slice_bad_cycle(cycle_data)
        else:
            windows = slice_good_cycle(cycle_data)
        
        if not windows:
            print(f"Warning: No valid windows extracted from {cycle_name}. Skipping.")
            continue

        cycle_2d = np.vstack(windows)
        cycle_normalized = (cycle_2d - global_median) / global_iqr
        cycle_3d = cycle_normalized.reshape(-1, 500, 3)

        processed_val_dict[cycle_name] = cycle_3d
        print(f"Processed {cycle_name}: {cycle_3d.shape} windows extracted.")
    
    np.savez_compressed(save_path, **processed_val_dict)
    print(f"\nTesting data archive saved to {save_path} with {len(processed_val_dict)} cycles.")


if __name__ == "__main__":
    GOOD_CYCLE_FOLDER = '../data/raw/M01_OP07/good'
    BAD_CYCLE_FOLDER = '../data/raw/M01_OP07/bad'
    PROCESSED_DATA_FOLDER = 'data/processed'

    os.makedirs(PROCESSED_DATA_FOLDER, exist_ok=True)

    print("Start loading raw testing data...")

    test_cycles = {}
    years = '2021'
    month = 'Feb'
    for i in range(4,13):
        file_id = f'M01_{month}_{years}_OP07_{i:03d}'
        good_cycle_path = os.path.join(GOOD_CYCLE_FOLDER, f'{file_id}.csv')
        good_cycle_name = f'{file_id}_good'

        if os.path.exists(good_cycle_path):
            test_cycles[good_cycle_name] = load_cycle(filepath=good_cycle_path)
        else:
            print(f"Skipping: File does not exist {good_cycle_path}")

    test_cycles['M01_Feb_2021_OP07_000_bad'] = load_cycle(filepath=os.path.join(BAD_CYCLE_FOLDER, 'M01_Aug_2021_OP07_000.csv'))

    with open(os.path.join(PROCESSED_DATA_FOLDER, 'edge_params.json'), 'r') as f:
        edge_params = json.load(f)
        global_median = np.array(edge_params['global_median'])
        global_iqr = np.array(edge_params['global_iqr'])

    build_testing_data(testing_cycles_dict=test_cycles, global_median=global_median, global_iqr=global_iqr, save_path=os.path.join(PROCESSED_DATA_FOLDER, 'testing_cycles.npz'))