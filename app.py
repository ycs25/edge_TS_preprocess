from flask import Flask, request, jsonify
import json
import os
import sys
import numpy as np
import torch

# Make sure Python can import files from model/
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from cnnlstm_autoencoder import CNNLSTMAutoencoder

app = Flask(__name__)

MODEL_PATH = os.path.join(ROOT_DIR, "data", "models", "cnnlstm_autoencoder_op07.pth")
ALARM_PARAMS_PATH = os.path.join(ROOT_DIR, "data", "processed", "cloud_alarm_params.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model object
model = None
alarm_params = {}


def load_alarm_params():
    global alarm_params
    if os.path.exists(ALARM_PARAMS_PATH):
        with open(ALARM_PARAMS_PATH, "r") as f:
            alarm_params = json.load(f)
    else:
        alarm_params = {
            "threshold_3sigma": 0.4,
            "threshold_4sigma": 0.5
        }


def load_model():
    global model
    model = CNNLSTMAutoencoder(seq_len=500, n_features=3, lstm_hidden_dim=64).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()


def compute_window_errors(cycle_3d: np.ndarray):
    """
    cycle_3d shape: (num_windows, 500, 3)
    returns:
        error_array: per-window MAE, shape (num_windows,)
    """
    cycle_tensor = torch.tensor(cycle_3d, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        reconstructed_tensor = model(cycle_tensor)

    error_array = torch.mean(
        torch.abs(cycle_tensor - reconstructed_tensor), dim=(1, 2)
    ).cpu().numpy()

    return error_array


def get_thresholds(cycle_name=None):
    threshold_3sigma = alarm_params.get("threshold_3sigma", 0.4)
    threshold_4sigma = alarm_params.get("threshold_4sigma", 0.5)

    # If cycle-specific thresholds exist, use them
    if cycle_name and cycle_name in alarm_params:
        value = alarm_params[cycle_name]
        if isinstance(value, dict):
            threshold_3sigma = value.get("threshold_3sigma", threshold_3sigma)
            threshold_4sigma = value.get("threshold_4sigma", threshold_4sigma)

    return threshold_3sigma, threshold_4sigma


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Edge-Cloud CNC Anomaly Detection API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expected JSON:
    {
        "cycle_name": "M01_Feb_2021_OP07_004_good",
        "data": [[[... 3 features ...] x 500] x num_windows]
    }
    """
    try:
        payload = request.get_json(force=True)

        cycle_name = payload.get("cycle_name", "unknown_cycle")
        cycle_data = payload.get("data", None)

        if cycle_data is None:
            return jsonify({"error": "Missing 'data' in request body"}), 400

        cycle_3d = np.array(cycle_data, dtype=np.float32)

        if cycle_3d.ndim != 3:
            return jsonify({
                "error": f"Input data must be 3D, got shape {cycle_3d.shape}"
            }), 400

        if cycle_3d.shape[1] != 500 or cycle_3d.shape[2] != 3:
            return jsonify({
                "error": f"Expected shape (num_windows, 500, 3), got {cycle_3d.shape}"
            }), 400

        error_array = compute_window_errors(cycle_3d)

        threshold_3sigma, threshold_4sigma = get_thresholds(cycle_name)

        pct_above_3sigma = float(np.mean(error_array > threshold_3sigma))
        pct_above_4sigma = float(np.mean(error_array > threshold_4sigma))

        response = {
            "cycle_name": cycle_name,
            "num_windows": int(cycle_3d.shape[0]),
            "mean_mae": float(np.mean(error_array)),
            "max_mae": float(np.max(error_array)),
            "min_mae": float(np.min(error_array)),
            "threshold_3sigma": float(threshold_3sigma),
            "threshold_4sigma": float(threshold_4sigma),
            "pct_windows_above_3sigma": pct_above_3sigma,
            "pct_windows_above_4sigma": pct_above_4sigma,
            "alarm_3sigma": pct_above_3sigma > 0,
            "alarm_4sigma": pct_above_4sigma > 0,
            "window_errors": error_array.tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    load_alarm_params()
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)