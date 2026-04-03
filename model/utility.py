import pandas as pd
import os
import numpy as np

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

def slice_bad_cycle(cycle_data, window_size=500, step=250):
    """
    Slicing of bad cycles: keep features before machine stops (Snap-to-tail)。
    """
    windows = []
    num_steps = len(cycle_data)
    
    if num_steps < window_size:
        return []

    start_idx = 0
    # Normal sliding
    while start_idx + window_size <= num_steps:
        window = cycle_data[start_idx : start_idx + window_size, :]
        windows.append(window)
        start_idx += step
    

    # Checking the remaining tail, force a window backwards of determined size
    if (start_idx - step) + window_size < num_steps:
        last_window = cycle_data[-window_size:, :]
        windows.append(last_window)
        
    return windows

def window_enchancement(window_raw):
    """
    Input: raw data of shape (500, 3)
    Output: enhanced matrix (x, y, z, CF_x, CF_y, CF_z) of shape (500, 6)
    """
    # 1. Calculate RMS and Max
    # Along columns
    rms = np.sqrt(np.mean(np.square(window_raw), axis=0)) 
    max_abs = np.max(np.abs(window_raw), axis=0)
    
    # 2. Calculate Crest Factor (adding 1e-8 to prevent divison by 0)
    cf = max_abs / (rms + 1e-8) # shape (3,)
    
    # 3. Broadcasting：duplicate three scalers 500 times
    cf_broadcasted = np.tile(cf, (window_raw.shape[0], 1)) # shape (500, 3)
    
    # 4. Concatenate new features to raw data
    window_enhanced = np.concatenate((window_raw, cf_broadcasted), axis=1) # shape (500, 6)
    
    return window_enhanced

def apply_symlog(data):
    return np.sign(data) * np.log1p(np.abs(data))


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

def load_cycle(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: Cannot find {filepath}, file skiped.")
        return None

    df = pd.read_csv(filepath)
    raw_sensor_data = df[['x','y','z']].values
    symlog_data = apply_symlog(raw_sensor_data)

    return symlog_data