import os
import json
import torch
import numpy as np
from matplotlib import pyplot as plt

from cnnlstm_autoencoder import CNNLSTMAutoencoder

def model_prediction(data_path, model_path='data/models/cnnlstm_autoencoder_op07.pth'):
    """Run the trained model on new data and generate predictions.

    Args:
        data_path (str): Path to the .npz archive containing the processed testing dataset.
    """
    test_archive = np.load(data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNNLSTMAutoencoder(seq_len=500, n_features=3, lstm_hidden_dim=64).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    all_predictions = {}

    for cycle_name in test_archive.files:
        cycle_numpy = test_archive[cycle_name]
        cycle_tensor = torch.tensor(cycle_numpy, dtype=torch.float32).to(device)

        with torch.no_grad():
            reconstructed_tensor = model(cycle_tensor)

        error_array = torch.mean(
            torch.abs(cycle_tensor - reconstructed_tensor), dim=(1, 2)
        ).cpu().numpy()

        all_predictions[cycle_name] = error_array
    
    return all_predictions


def alarm_triggering(predictions, alarm_params_path='data/processed/edge_alarm_params.json'):
    """Determine if an alarm should be triggered based on the predictions and predefined thresholds.

    Args:
        predictions (dict): Dictionary of cycle names to their corresponding error arrays.
        alarm_params_path (str): Path to the JSON file containing alarm threshold parameters.
    """
    with open(alarm_params_path, 'r') as f:
        alarm_params = json.load(f)

    alarms_triggered = {}

    for cycle_name, error_array in predictions.items():
        threshold_3sigma = alarm_params.get(cycle_name, alarm_params.get('threshold_3sigma', 0.4))
        threshold_4sigma = alarm_params.get(cycle_name, alarm_params.get('threshold_4sigma', 0.5))
        alarms_triggered[cycle_name] = [sum(error_array > threshold_3sigma)/len(error_array), sum(error_array > threshold_4sigma)/len(error_array)]
    
    return alarms_triggered

if __name__ == '__main__':
    predictions = model_prediction(data_path='data/processed/testing_cycles.npz')

    steady_errors = []

    for cycle_name, error_array in predictions.items():
        plt.figure(figsize=(10, 4))
        plt.plot(error_array, label='Reconstruction Error', color='red')
        plt.title(f'Reconstruction Error for {cycle_name}')
        plt.xlabel('Window Index')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join('data/plots/predictions', f'{cycle_name}_MAE.png')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()

        if len(error_array) > 30:
            steady_errors.extend(error_array[30:])
        else:
            steady_errors.extend(error_array)
    
    print(f"Reconstruction error plots saved to {os.path.join('data/plots/predictions')}/")

    alarms_default = alarm_triggering(predictions, alarm_params_path='data/processed/cloud_alarm_params.json')
    alarms_new = alarm_triggering(predictions, alarm_params_path='data/processed/cloud_alarm_params_new.json')
    print("\n--- Alarm Triggering Results with Default Params ---")
    for cycle_name, alarm_values in alarms_default.items():
        print(f"{cycle_name}: {alarm_values[0]*100:.2f}% windows > 3-sigma, {alarm_values[1]*100:.2f}% windows > 4-sigma")

    print("\n--- Alarm Triggering Results with New Params ---")
    for cycle_name, alarm_values in alarms_new.items():
        print(f"{cycle_name}: {alarm_values[0]*100:.2f}% windows > 3-sigma, {alarm_values[1]*100:.2f}% windows > 4-sigma")