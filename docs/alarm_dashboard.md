# Cloud-Edge Alarm System & Dashboard Logic Design

1. Overview
This document outlines the real-time anomaly detection logic for the CNC machining Edge-Cloud collaborative framework. The predictive model (CNN-LSTM Autoencoder) evaluates data in 500-timestep windows (approx. 0.25 seconds). To prevent false positives caused by transient noise and machine state changes, we must implement state masking, dynamic thresholding, and signal debouncing in the Cloud inference pipeline.

2. Edge Device Responsibilities (Protocol Extensions)
Beyond sending normalized payload windows, the Edge device MUST append state metadata to each request to provide context for the Cloud model.
- `operation_id`: Identifies the current machining operation (e.g., OP07, OP03). The Cloud will use this to dynamically load the corresponding model weights.
- `machine_state`: A status flag (STARTING, RUNNING, STOPPING, IDLE). The edge infers this based on overall spindle load or explicit G-code triggers.
- Payload Format Example:
    ```
    {
    "machine_id": "M01",
    "operation_id": "OP07",
    "state": "RUNNING",
    "window_data": [[...], [...], ...]
    }
    ```

3. Cloud Inference & Alarm Logic (The Filter Pipeline)
When the Cloud receives a payload window, it calculates the Mean Absolute Error (MAE) of the model's reconstruction. The system must pass this MAE through two filters before triggering an actual alarm:
- Filter 1: State Masking (Transient Avoidance)
  - Logic: Mechanical vibrations are highly unstable during machine start-up and shutdown.
  - Action: If `machine_state` is `STARTING` or `STOPPING` (or within $X$ seconds of these state transitions), the Cloud infers the data but mutates the Alarm Trigger to False, regardless of the MAE value.
- Filter 2: Debounce Mechanism (Noise Rejection)
  - Logic: A single 0.25s window experiencing high MAE could be a random electromagnetic spike. A true mechanical failure (e.g., tool breakage) persists over time.
  - Parameters: 
    - Threshold (T): The MAE threshold (e.g., $\mu + 3\sigma$ derived from historical good cycles).
    - Debounce_N: An integer (e.g., 3).
  - Action: The Cloud maintains a rolling counter per machine. If $MAE > T$, `counter += 1`. If $MAE \le T$, `counter = 0`. An anomaly is strictly defined as `counter >= Debounce_N`.

4. Streamlit Dashboard Integration
To demonstrate the system's robustness and Continuous Learning capabilities, the GUI application should include the following interactive modules:
- Live MAE Plot: A real-time line chart plotting the incoming reconstruction errors.
- Interactive Parameters Panel:
  - Slider for Threshold (T) (e.g., 0.02 to 0.10).
  - Slider for Debounce N (e.g., 1 to 10 windows).
  - Visual Effect: Changing these sliders should instantly recalculate historical anomalies on the chart to demonstrate how debouncing removes False Positives.
- Model Version Switcher (A/B Testing Concept Drift):
  - A dropdown/toggle to switch the underlying inference model between Base Model (v1.0) and Fine-tuned Model (v1.1).
  - Purpose: When testing 2021 data (aged machine), selecting Base Model should show rising MAE (drifting baseline), while switching to Fine-tuned Model will visually drop the MAE back to normal levels, showcasing successful Continuous Learning adaptation.

5. Data Type/variables
- A series of MAE or Reconstruction Error
- Threshold: $\mu$, $\sigma$ and $\mu + 3*\sigma$