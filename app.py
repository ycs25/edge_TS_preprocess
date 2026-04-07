from flask import Flask, request, jsonify
import json
import os
import sys
import numpy as np
import torch
import sqlite3
from datetime import datetime
import traceback

# Make sure Python can import files from model/
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from model.cnnlstm_autoencoder import CNNLSTMAutoencoder

# SQLite Path
DB_PATH = os.path.join(ROOT_DIR, "data", "processed", "inference_history.db")

app = Flask(__name__)

MODEL_PATH = os.path.join(ROOT_DIR, "data", "models", "cnnlstm_autoencoder_op07.pth")
EDGE_PARAMS_PATH = os.path.join(ROOT_DIR, "data", "processed", "edge_params.json")
ALARM_PARAMS_PATH = os.path.join(ROOT_DIR, "data", "processed", "cloud_alarm_params.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model object
model = None
alarm_params = {}
edge_params = {}

def init_db():
    """Initialize database"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            machine_id TEXT,
            operation_id TEXT,
            source_file TEXT,
            timestamp REAL,
            window_error REAL,
            threshold_3sigma REAL,
            threshold_4sigma REAL
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(record):
    """Write one record into db"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        INSERT INTO history 
        (machine_id, operation_id, source_file, timestamp, window_error, threshold_3sigma, threshold_4sigma)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        record["machine_id"], record["operation_id"],
        record["source_file"], record["timestamp"],
        record["window_error"], record["threshold_3sigma"], record["threshold_4sigma"]
    ))
    conn.commit()
    conn.close()

def load_params():
    global alarm_params, edge_params
    # Load Cloud Alarm Params
    if os.path.exists(ALARM_PARAMS_PATH):
        with open(ALARM_PARAMS_PATH, "r") as f:
            alarm_params = json.load(f)
    else:
        alarm_params = {"threshold_3sigma": 0.4, "threshold_4sigma": 0.5}
        
    # Load Edge Normalization Params
    if os.path.exists(EDGE_PARAMS_PATH):
        with open(EDGE_PARAMS_PATH, "r") as f:
            edge_params = json.load(f)
    else:
        edge_params = {"global_median": [0.0, 0.0, 0.0], "global_iqr": [1.0, 1.0, 1.0]}


def load_model():
    global model
    model = CNNLSTMAutoencoder(seq_len=500, n_features=3, lstm_hidden_dim=64).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully.")


def compute_window_errors(cycle_3d: np.ndarray):
    """
    cycle_3d shape: (num_windows, 500, 3)
    returns:
        error_array: per-window MAE, shape (num_windows,)
    """
    if model is None:
        raise RuntimeError("Model has not been initialized. Call load_model() first.")

    cycle_tensor = torch.tensor(cycle_3d, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        reconstructed_tensor = model(cycle_tensor)

    error_array = torch.mean(
        torch.abs(cycle_tensor - reconstructed_tensor), dim=(1, 2)
    ).cpu().numpy()

    return error_array


def get_thresholds(operation_id=None):
    threshold_3sigma = alarm_params.get("threshold_3sigma", 0.4)
    threshold_4sigma = alarm_params.get("threshold_4sigma", 0.5)

    # If operation-specific thresholds exist, use them
    if operation_id and operation_id in alarm_params:
        value = alarm_params[operation_id]
        if isinstance(value, dict):
            threshold_3sigma = value.get("threshold_3sigma", threshold_3sigma)
            threshold_4sigma = value.get("threshold_4sigma", threshold_4sigma)

    return threshold_3sigma, threshold_4sigma

@app.route("/init", methods=["GET"])
def init_edge():
    """
    Call this port when edge device start
    Return start command (status: "START") and parameters required for preprocess
    """
    return jsonify({
        "status": "START",
        "command": "Begin simulating sensor data",
        "edge_params": edge_params
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Edge-Cloud CNC Anomaly Detection API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:
        payload = request.get_json(force=True)

        machine_id = payload.get("machine_id", "unknown_machine")
        operation_id = payload.get("operation_id", "unknown_op")
        source_file = payload.get("source_file", "unknown_file")
        timestamp = payload.get("timestamp", 0.0)
        window_data = payload.get("data", None)

        if window_data is None:
            return jsonify({"error": "Missing 'data' in request body"}), 400

        cycle_3d = np.expand_dims(np.array(window_data, dtype=np.float32), axis=0)

        if cycle_3d.shape[1] != 500 or cycle_3d.shape[2] != 3:
            return jsonify({
                "error": f"Expected window shape (500, 3), got {np.array(window_data).shape}"
            }), 400

        error_array = compute_window_errors(cycle_3d)
        threshold_3sigma, threshold_4sigma = get_thresholds(operation_id)

        window_error = float(np.mean(error_array))

        record = {
            "machine_id": machine_id,
            "operation_id": operation_id,
            "source_file": source_file,
            "timestamp": timestamp,
            "window_error": window_error,
            "threshold_3sigma": float(threshold_3sigma),
            "threshold_4sigma": float(threshold_4sigma)
        }

        save_to_db(record)

        return jsonify({"status": "received"}), 202

    except Exception as e:
        print("\n" + "="*50)
        print("🚨 EXCEPTION IN /predict:")
        traceback.print_exc()
        print("="*50 + "\n")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_db()
    load_params()
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)