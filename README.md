# Edge-Cloud Collaborative Anomaly Detection for CNC Machining

## Overview
This project implements an unsupervised deep learning system within a simulated IoT Edge-Cloud collaborative architecture to perform real-time anomaly detection on CNC machining cycles. By utilizing a hybrid Edge-Cloud framework, the system transitions manufacturing from schedule-based maintenance to predictive maintenance, instantly identifying defective machining cycles while handling the extreme variance of physical vibrations.

## Core Architecture
The system is distributed across three primary components to balance computational load and minimize transmission latency:
* **Edge Device (IoT Simulation):** Simulates a high-frequency (2 kHz) data stream from CNC sensors. It handles lightweight preprocessing by slicing data into 500-step windows with a 250-step overlap. It applies a Symmetric Logarithm (`symlog`) and decentralized normalization (`RobustScaler`) using pre-calculated global parameters from an edge_params.json configuration file.
* **Cloud Backend (Flask REST API):** Receives the compact, normalized payload and performs heavy deep learning inference. It runs a hybrid CNN-LSTM Autoencoder to calculate the Mean Absolute Error (MAE) of the reconstruction, storing the telemetry and metadata into an SQLite database.
* **Operational Dashboard (Streamlit):** Serves as a real-time playback engine querying the SQLite database. It features a dynamic scoring chart and a three-tier smart alarm system (Green, Yellow, Red) governed by a customizable "Debounce" mechanism to prevent false positives from isolated vibration spikes.

## The Model: CNN-LSTM Autoencoder
A pure LSTM model fails on this dataset due to "mean collapse," where the network outputs a flat moving average to avoid the penalty of high-frequency physical shocks. To solve this, our architecture uses:
1. **1D-CNN (Feature Extractor):** Captures sudden physical shocks and transient spikes.
2. **Bidirectional LSTM (Temporal Learner):** Learns the long-term temporal dependencies of the machining sequence from the CNN's feature maps.
To handle physical "Concept Drift" as machines age, the framework includes a Continuous Learning pipeline. By running `recalibrate_params.py`, the system calculates new scaling parameters for the validation dataset, fine-tunes the model on recent data, and executes an Over-The-Air (OTA) update to push the new `edge_params.json` back to the edge nodes.

## Technologies Used
* **Language:** Python 3.x 
* **Machine Learning:** PyTorch (CNN-LSTM Autoencoder) 
* **Data Processing:** Pandas, NumPy 
* **Backend & Simulation:** Flask, SQLite 
* **Frontend GUI:** Streamlit 

## Setup & Installation
1. Clone the repository and install dependencies:
```Bash
git clone https://github.com/ycs25/edge_TS_preprocess
cd <repository-directory>
pip install -r requirements.txt
```
2. Start the Cloud Backend:
```
python app.py
```
3. Launch the Edge Simulator:
In a separate terminal, start the edge device to begin streaming data to the cloud API:
```
python edge_simulator.py
```
4. Run the Streamlit Dashboard:
In a third terminal, start the visual frontend.
```
streamlit run app.py
```

Alternatively, run the [app](https://diffusionexample-5y4bhysmu8qdbvkbyvv4aw.streamlit.app/) directly from Streamlit community. It takes a while to load.

**Caution:** When using GitHub codespace, Streamlit app might not be working.
