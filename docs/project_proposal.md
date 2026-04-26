# Project Proposal: Edge-Cloud Collaborative Anomaly Detection for CNC Machining with Continuous Learning

## Project Objective
This project involves building a supervised machine learning system within an IoT Edge-Cloud collaborative architecture to classify [CNC machining cycles](https://www.kaggle.com/datasets/maximilianfellhuber/cnc-machining-data) as "Good" or "Bad". By leveraging the labeled Bosch CNC dataset, the system will compare a baseline Logistic Regression model against a deep learning Long Short-Term Memory (LSTM) network. A key focus is the implementation of an MLOps pipeline that handles Data Drift through continuous learning, ensuring the classifier remains accurate as mechanical components age or operational conditions shift.

## Dataset & Preprocessing Strategy
The dataset consists of multi-sensor time-series data with binary labels. To address the inherent class imbalance (where "Bad" cycles are typically rare), the following strategies will be applied:
- Task-Specific Segmentation: Separate baselines and models will be developed for different machining stages (e.g., Roughing vs. Finishing), as the physical signatures of failure differ across tasks.
- Windowing with Temporal Alignment: * Feature Engineering for Baseline: For Logistic Regression, windows will be flattened into statistical aggregates (mean, variance, kurtosis) and lagged features.
    - 3D Tensor Shaping for LSTM: For the LSTM model, data will be shaped into $(samples, time\_steps, features)$ to preserve sequential dependencies.
    - Boundary & Tail Handling: Strict adherence to Machine_ID boundaries and a hybrid "Force Tail Alignment" strategy to ensure the critical moments leading up to a labeled "Bad" state are captured.

## System Architecture (Edge-Cloud Workflow)
Phase 1: Cloud-Side Comparative Training
- The cloud environment will perform the heavy lifting of training two competing models:
    1.	Baseline (Logistic Regression): A high-interpretability model focused on engineered features.
    2.	Champion (LSTM): A recurrent architecture designed to capture complex temporal patterns without manual feature extraction.
- Global normalization parameters ($\mu$ and $\sigma$) are computed and stored for deployment.

Phase 2: Edge-Side Preprocessing & OTA Updates

To minimize cloud transmission payloads and ensure high data quality, simulated edge devices perform a strictly ordered, $O(N)$ real-time preprocessing pipeline before data leaves the factory floor:
- **Real-Time Data Ingestion:** Edge devices continuously ingest multi-sensor streams (e.g., vibration, spindle load) that already possess fixed-frequency sampling and absolute timestamps, eliminating the need for complex temporal alignment.
- **Threshold Clipping (Outlier Removal):** Incoming raw signals are immediately subjected to physical threshold clipping. This step instantly discards extreme, non-physical sensor glitches or electrical short-circuit spikes, preventing artificial anomalies from skewing the operational baselines.
- **Lightweight Denoising:** With a stable and fixed sampling frequency inherently guaranteed, a low-overhead Exponential Moving Average (EMA) filter is applied. This smooths out the high-frequency mechanical background noise typical of a factory environment, preserving only the core machine health signatures.
- **Decentralized Normalization:** Finally, devices receive global $\mu$ and $\sigma$ parameters via Over-The-Air (OTA) updates. The edge applies these parameters to zero-center and standardize the smoothed local data windows, ensuring the local distribution perfectly matches the exact specifications expected by the cloud-based LSTM or Logistic Regression models.

Phase 3: Cloud-Side Real-Time Inference
- The standardized windows are streamed to the cloud, where the active model calculates the Probability of Failure $P(Y=1 | X)$.
- If $P > \text{Threshold}$, an anomaly alert is triggered.

## Continuous Learning & Performance Adaptation
Using a Walk-Forward Validation approach, the system monitors model performance (F1-Score) over time to detect Concept Drift:
- Level 1: Feature Re-calibration: If the feature distribution shifts (detected via Kolmogorov-Smirnov tests), the cloud re-calculates $\mu$ and $\sigma$ and pushes them to the Edge.
- Level 2: Model Evolution: If classification accuracy drops below a pre-defined threshold due to mechanical wear-and-tear, the system triggers:
  - Incremental Learning: Updating the existing weights using recent "Bad" and "Good" samples.
  - Full Retraining: Re-optimizing the model architecture or hyperparameters if the drift is severe.

## Demonstration & Interactive Dashboard (Streamlit)
The project will conclude with a live dashboard demonstrating the MLOps lifecycle:
  1. Real-Time Classification Feed: Visualizing sensor data alongside the model's live "Failure Probability" score.
  2. Performance Decay Simulation: Demonstrating how the model’s F1-score degrades when "drifted" data is introduced.
  3. Active Learning Trigger: A button to "Retrain & Deploy," showing the immediate recovery of classification precision after the model adapts to the new data distribution.

## Reference
This project idea is inspired by *Tawakuli, A., Kaiser, D., and Engel, T. Modern data preprocessing is holistic, normalized and distributed.2022*.

The Bosch CNC machine dataset originates from *M.-A. Tnani, M. Feil, and K. Diepold, "Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring," Procedia CIRP, vol. 107, pp. 131–136, 2022, doi: 10.1016/j.procir.2022.04.022*.