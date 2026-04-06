import os
import json
<<<<<<< HEAD
import numpy as np
import requests

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(ROOT_DIR, "data", "processed", "testing_cycles.npz")
EDGE_PARAMS_PATH = os.path.join(ROOT_DIR, "data", "processed", "edge_params.json")

API_URL = "http://127.0.0.1:5000/predict"
=======
import time
from collections import deque

import numpy as np
import pandas as pd
import requests

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
EDGE_PARAMS_PATH = os.path.join(ROOT_DIR, "data", "processed", "edge_params.json")

API_URL = "http://127.0.0.1:5000/predict"
WINDOW_SIZE = 500
SAMPLING_INTERVAL_SEC = 0.005  # as requested by teammate

# Change these if your raw CSV uses different sensor columns
FEATURE_COLUMNS = ["x", "y", "z"]
>>>>>>> bf4a32c (Added edge-cloud pipeline: Flask API, real-time edge simulator with CSV streaming + normalization, and Streamlit dashboard)


def load_edge_params():
    if not os.path.exists(EDGE_PARAMS_PATH):
<<<<<<< HEAD
        print("edge_params.json not found. Skipping edge parameter load.")
        return None

    with open(EDGE_PARAMS_PATH, "r") as f:
        return json.load(f)


def simulate_streaming(cycle_name, cycle_windows, batch_size=5):
    """
    cycle_windows shape: (num_windows, 500, 3)
    Sends small batches to backend to simulate edge streaming.
    """
    n = cycle_windows.shape[0]
    print(f"\nStreaming cycle: {cycle_name} | total windows: {n}")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = cycle_windows[start:end]

        payload = {
            "cycle_name": cycle_name,
            "data": batch.tolist()
        }

        response = requests.post(API_URL, json=payload, timeout=60)

        if response.status_code == 200:
            result = response.json()
            print(
                f"[{cycle_name}] windows {start}:{end} | "
=======
        raise FileNotFoundError(f"edge_params.json not found: {EDGE_PARAMS_PATH}")

    with open(EDGE_PARAMS_PATH, "r") as f:
        params = json.load(f)

    global_median = np.array(params["global_median"], dtype=np.float32)
    global_iqr = np.array(params["global_iqr"], dtype=np.float32)

    # protect against divide-by-zero
    global_iqr = np.where(global_iqr == 0, 1e-8, global_iqr)

    return global_median, global_iqr


def normalize_row(x, global_median, global_iqr):
    return (x - global_median) / global_iqr


def send_window(cycle_name, window_array):
    """
    window_array shape: (500, num_features)
    Backend expects: (num_windows, 500, num_features)
    so wrap as 1 window.
    """
    payload = {
        "cycle_name": cycle_name,
        "data": np.expand_dims(window_array, axis=0).tolist()
    }

    response = requests.post(API_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def stream_csv_file(csv_path, global_median, global_iqr, feature_columns, sleep_time=0.0):
    cycle_name = os.path.splitext(os.path.basename(csv_path))[0]
    print(f"\nStreaming raw CSV: {cycle_name}")

    df = pd.read_csv(csv_path)

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required feature columns in {csv_path}: {missing_cols}\n"
            f"Available columns: {list(df.columns)}"
        )

    buffer = deque(maxlen=WINDOW_SIZE)
    sent_windows = 0

    for idx, row in df[feature_columns].iterrows():
        raw_x = row.to_numpy(dtype=np.float32)
        norm_x = normalize_row(raw_x, global_median, global_iqr)
        buffer.append(norm_x)

        if len(buffer) == WINDOW_SIZE:
            window_array = np.array(buffer, dtype=np.float32)

            result = send_window(cycle_name, window_array)
            sent_windows += 1

            print(
                f"[{cycle_name}] sample={idx + 1} | "
                f"window={sent_windows} | "
>>>>>>> bf4a32c (Added edge-cloud pipeline: Flask API, real-time edge simulator with CSV streaming + normalization, and Streamlit dashboard)
                f"mean_mae={result['mean_mae']:.6f} | "
                f"alarm_3sigma={result['alarm_3sigma']} | "
                f"alarm_4sigma={result['alarm_4sigma']}"
            )
<<<<<<< HEAD
        else:
            print(f"Request failed for {cycle_name} windows {start}:{end}")
            print(response.text)


def main():
    edge_params = load_edge_params()
    if edge_params is not None:
        print("Loaded edge params successfully.")
        if "global_median" in edge_params:
            print("Found global_median in edge params.")
        if "global_iqr" in edge_params:
            print("Found global_iqr in edge params.")

    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Testing archive not found: {TEST_DATA_PATH}")

    archive = np.load(TEST_DATA_PATH)

    print("Available cycles in testing_cycles.npz:")
    for name in archive.files:
        print(" -", name)

    for cycle_name in archive.files:
        cycle_windows = archive[cycle_name]

        if cycle_windows.ndim != 3 or cycle_windows.shape[1:] != (500, 3):
            print(f"Skipping {cycle_name}: unexpected shape {cycle_windows.shape}")
            continue

        simulate_streaming(cycle_name, cycle_windows, batch_size=5)
=======

        if sleep_time > 0:
            time.sleep(sleep_time)

    print(f"Finished {cycle_name}: sent {sent_windows} windows")


def main():
    global_median, global_iqr = load_edge_params()
    print("Loaded edge normalization parameters.")
    print("global_median shape:", global_median.shape)
    print("global_iqr shape:", global_iqr.shape)

    if not os.path.exists(RAW_DATA_DIR):
        raise FileNotFoundError(f"Raw data folder not found: {RAW_DATA_DIR}")

    csv_files = sorted(
        [
            os.path.join(RAW_DATA_DIR, f)
            for f in os.listdir(RAW_DATA_DIR)
            if f.lower().endswith(".csv")
        ]
    )

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {RAW_DATA_DIR}")

    print("\nCSV files found:")
    for f in csv_files:
        print(" -", os.path.basename(f))

    for csv_path in csv_files:
        stream_csv_file(
            csv_path=csv_path,
            global_median=global_median,
            global_iqr=global_iqr,
            feature_columns=FEATURE_COLUMNS,
            sleep_time=0.0  # set to 0.005 for true live-like simulation
        )
>>>>>>> bf4a32c (Added edge-cloud pipeline: Flask API, real-time edge simulator with CSV streaming + normalization, and Streamlit dashboard)


if __name__ == "__main__":
    main()