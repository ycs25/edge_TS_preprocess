import numpy as np

from validation import model_validation
from validation import fast_validation

#model_validation(dataset_path='data/processed/validation_new_scaler.npz', validation_plots_folder='data/plots/validation_new_scaler', alarm_params_path='data/processed/cloud_alarm_params_new.json')

error_array = fast_validation(cycle_path='../data/raw/M01_OP07/good/202108/M01_Aug_2021_OP07_002.csv',
                               model_path='data/models/cnnlstm_autoencoder_op07.pth',
                               global_median=[-2.302585092994046, 3.4011973816621555, -6.921658184151129],
                               global_iqr=[8.188411308079031, 8.720134035412928, 0.10952285257251315])

error_array = np.array(error_array)
n3sigma_alarms = np.sum(error_array > 0.04365578293800354)
n3sigma_ratio = n3sigma_alarms / len(error_array)
n4sigma_alarms = np.sum(error_array > 0.050473280251026154)
n4sigma_ratio = n4sigma_alarms / len(error_array)
print(f"3Sigma Alarm Ratio: {n3sigma_ratio:.4f} ({n3sigma_alarms} out of {len(error_array)} windows)")
print(f"4Sigma Alarm Ratio: {n4sigma_ratio:.4f} ({n4sigma_alarms} out of {len(error_array)} windows)")