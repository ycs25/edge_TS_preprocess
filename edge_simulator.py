import os
import time

import numpy as np
import pandas as pd
import requests

from model.utility import apply_symlog

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")

CLOUD_URL = "http://127.0.0.1:5000"
WINDOW_SIZE = 500
STEP_SIZE = 250
SAMPLING_INTERVAL_SEC = 0.0005  # 2kHz (1 / 2000)

# Change these if your raw CSV uses different sensor columns
FEATURE_COLUMNS = ["x", "y", "z"]

MACHINE_ID = "M01"
OPERATION_ID = "OP07"

def fetch_init_data():
    """Waiting for Cloud signal"""
    print("Waiting for Cloud 'Start' signal and parameters...")
    while True:
        try:
            response = requests.get(f"{CLOUD_URL}/init", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "START":
                    edge_params = data.get("edge_params", {})
                    global_median = np.array(edge_params["global_median"], dtype=np.float32)
                    global_iqr = np.array(edge_params["global_iqr"], dtype=np.float32)
                    
                    global_iqr = np.where(global_iqr == 0, 1e-8, global_iqr)
                    
                    print("Received START signal and parameters from Cloud.")
                    return global_median, global_iqr
        except requests.exceptions.RequestException:
            pass
        
        print("Cloud not ready or unreachable, retrying in 2 seconds...")
        time.sleep(2)

def send_window_to_cloud(window_buffer, sent_windows, row_num, csv_name):
    """Send a window of 500 to the cloud"""
    payload = {
        "machine_id": MACHINE_ID,
        "operation_id": OPERATION_ID,
        "source_file": csv_name,
        "timestamp": time.time(),
        "data": window_buffer
    }

    # print(payload)

    try:
        response = requests.post(f"{CLOUD_URL}/predict", json=payload, timeout=5)
        response.raise_for_status()

        print(f"[{MACHINE_ID} - {OPERATION_ID} | {csv_name}] "
              f"Successfully sent Window {sent_windows} (reached Row {row_num})")
        
    except requests.exceptions.RequestException as e:
        print(f"Failed to send data to cloud: {e}")

def stream_csv_file(csv_path, global_median, global_iqr):
    csv_name = os.path.basename(csv_path)
    print(f"\nStreaming raw CSV: {csv_name}")
    df = pd.read_csv(csv_path)

    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path}: {missing_cols}")

    window_buffer = []
    sent_windows = 0

    for row_num, (idx, row) in enumerate(df[FEATURE_COLUMNS].iterrows(), start=1):
        # 1. sensor generate one record
        raw_x = row.to_numpy(dtype=np.float32)
        
        # 2. Immediate preprocessing：Symlog -> Scaler
        sym_x = apply_symlog(raw_x)
        norm_x = (sym_x - global_median) / global_iqr
        
        # 3. Save to buffer
        window_buffer.append(norm_x.tolist())

        # 4. Check if buffer is full
        if len(window_buffer) == WINDOW_SIZE:
            # Send current window
            sent_windows += 1
            send_window_to_cloud(window_buffer, sent_windows, row_num, csv_name)
            
            # 5. Sliding window that keep the step size
            window_buffer = window_buffer[STEP_SIZE:]
            # ===============================================

        # Simulate 2kHz sampling rate
        time.sleep(SAMPLING_INTERVAL_SEC)


def main():
    global_median, global_iqr = fetch_init_data()

    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DATA_DIR}")

    csv_files = sorted([os.path.join(RAW_DATA_DIR, f) for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(".csv")])

    for csv_path in csv_files:
        stream_csv_file(csv_path, global_median, global_iqr)


if __name__ == "__main__":
    main()