import numpy as np
import json
from validation import model_validation
from validation import fast_validation
import matplotlib.pyplot as plt

""" summary = model_validation(dataset_path='data/processed/testing_cycles.npz',
                           model_path='data/models/cnnlstm_autoencoder_op07_v1_1.pth',
                           validation_plots_folder='data/plots/tuned_validation',
                           alarm_params_path='data/processed/cloud_alarm_params_tuned.json',
                           version='1.2') """

with open('data/processed/edge_params_mixed.json','r') as f:
    edge_params = json.load(f)

error_array = fast_validation(cycle_path='../data/raw/M01_OP07/good/202108/M01_Aug_2021_OP07_001.csv',
                               model_path='data/models/cnnlstm_autoencoder_op07_v1_1.pth',
                               global_median=np.array(edge_params['global_median']),
                               global_iqr=np.array(edge_params['global_iqr']))

with open('data/processed/cloud_alarm_params_tuned.json', 'r') as f:
    alarm_params = json.load(f)

threshold_3sigma = alarm_params["threshold_3sigma"]
threshold_4sigma = alarm_params["threshold_4sigma"]

error_array = np.array(error_array)
n3sigma_alarms = np.sum(error_array > threshold_3sigma)
n3sigma_ratio = n3sigma_alarms / len(error_array)
n4sigma_alarms = np.sum(error_array > threshold_4sigma)
n4sigma_ratio = n4sigma_alarms / len(error_array)
print(f"3Sigma Alarm Ratio: {n3sigma_ratio:.4f} ({n3sigma_alarms} out of {len(error_array)} windows)")
print(f"4Sigma Alarm Ratio: {n4sigma_ratio:.4f} ({n4sigma_alarms} out of {len(error_array)} windows)")

plt.figure(figsize=(10, 5))
plt.plot(error_array, color='blue')
plt.title('Subhealthy Cycle')
plt.xlabel('Window')
plt.ylabel('Reconstruction Error')
plt.show()