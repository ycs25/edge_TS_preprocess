import os
import json
import torch
import numpy as np
from matplotlib import pyplot as plt

from cnnlstm_autoencoder import CNNLSTMAutoencoder
from utility import load_cycle, slice_good_cycle, slice_bad_cycle


def model_validation(dataset_path, model_path='data/models/cnnlstm_autoencoder_op07.pth', validation_plots_folder='data/plots/validation', alarm_params_path='data/processed/cloud_alarm_params.json'):
    """Validate the model against a saved validation dataset.

    Args:
        dataset_path (str): Path to the validation .npz archive.
    """
    val_archive = np.load(dataset_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = CNNLSTMAutoencoder(seq_len=500, n_features=3, lstm_hidden_dim=64).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    os.makedirs(validation_plots_folder, exist_ok=True)

    all_good_steady_errors = []
    all_bad_steady_errors = []

    for cycle_name in val_archive.files:
        cycle_numpy = val_archive[cycle_name]
        cycle_tensor = torch.tensor(cycle_numpy, dtype=torch.float32).to(device)

        with torch.no_grad():
            reconstructed_tensor = model(cycle_tensor)

        error_array = torch.mean(
            torch.abs(cycle_tensor - reconstructed_tensor), dim=(1, 2)
        ).cpu().numpy()

        plt.figure(figsize=(10, 4))
        plt.plot(error_array, label='Reconstruction Error', color='blue')
        plt.title(f'Reconstruction Error for {cycle_name}')
        plt.xlabel('Window Index')
        plt.ylabel('Mean Absolute Error')
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(validation_plots_folder, f'{cycle_name}_MAE.png')
        plt.savefig(save_path, dpi=150)
        plt.close()

        if 'good' in cycle_name:
            if len(error_array) > 60:
                steady_errors = error_array[30:-30]
            else:
                steady_errors = error_array
            all_good_steady_errors.extend(steady_errors)
        else:
            all_bad_steady_errors.extend(error_array)

    print(f"Validation reconstruction error plots saved to {validation_plots_folder}/")

    summary = {}
    threshold_3sigma = None
    threshold_4sigma = None

    if all_good_steady_errors:
        all_good_steady_errors = np.array(all_good_steady_errors)
        mu = np.mean(all_good_steady_errors)
        sigma = np.std(all_good_steady_errors)
        threshold_3sigma = mu + 3 * sigma
        threshold_4sigma = mu + 4 * sigma

        false_positives_3sigma = np.sum(all_good_steady_errors > threshold_3sigma)
        false_positives_4sigma = np.sum(all_good_steady_errors > threshold_4sigma)

        print("\n" + "=" * 50)
        print("=== Alarm Baseline Statistics (Steady-State) ===")
        print(f"Mean (μ): {mu:.6f}")
        print(f"Standard Deviation (σ): {sigma:.6f}")
        print("-" * 50)
        print(
            f"Suggested Threshold 3 sigma: {threshold_3sigma:.6f} (Covers 99.7% of normal), False Positives: {false_positives_3sigma}"
        )
        print(
            f"Suggested Threshold 4 sigma: {threshold_4sigma:.6f} (Stricter, fewer False Positives), False Positives: {false_positives_4sigma}"
        )
        print("=" * 50 + "\n")

        alarm_params = {
            "version": "1.0",
            "description": "Alarm thresholds based on steady-state good cycles",
            "mu": float(mu),
            "sigma": float(sigma),
            "threshold_3sigma": float(threshold_3sigma),
            "threshold_4sigma": float(threshold_4sigma),
        }

        with open(alarm_params_path, 'w') as f:
            json.dump(alarm_params, f, indent=4)

        print(f"[*] Cloud alarm parameters saved to: {alarm_params_path}")

        summary.update(
            mu=mu,
            sigma=sigma,
            threshold_3sigma=threshold_3sigma,
            threshold_4sigma=threshold_4sigma,
            false_positives_3sigma=int(false_positives_3sigma),
            false_positives_4sigma=int(false_positives_4sigma),
        )

    if all_bad_steady_errors:
        all_bad_steady_errors = np.array(all_bad_steady_errors)
        if threshold_3sigma is not None and threshold_4sigma is not None:
            above_3sigma = np.sum(all_bad_steady_errors > threshold_3sigma)
            above_4sigma = np.sum(all_bad_steady_errors > threshold_4sigma)

            print("\n" + "=" * 50)
            print("=== Anomaly Detection on Bad Cycles (Full cycles) ===")
            print(f"Total Bad Windows: {len(all_bad_steady_errors)}")
            print(
                f"Windows above 3 sigma threshold: {above_3sigma} ({above_3sigma / len(all_bad_steady_errors) * 100:.2f}%)"
            )
            print(
                f"Windows above 4 sigma threshold: {above_4sigma} ({above_4sigma / len(all_bad_steady_errors) * 100:.2f}%)"
            )
            print("=" * 50 + "\n")

            summary.update(
                bad_windows=int(len(all_bad_steady_errors)),
                bad_above_3sigma=int(above_3sigma),
                bad_above_4sigma=int(above_4sigma),
            )

    return summary

def fast_validation(cycle_path, model_path='data/models/cnnlstm_autoencoder_op07.pth', global_median=None, global_iqr=None):
    cycle_windows = slice_good_cycle(load_cycle(cycle_path))
    if not cycle_windows:
        print(f"Warning: No valid windows extracted from {cycle_path}. Skipping.")
        return None
    cycle_2d = np.vstack(cycle_windows)
    cycle_normalized = (cycle_2d - global_median) / global_iqr
    cycle_3d = cycle_normalized.reshape(-1, 500, 3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTMAutoencoder(seq_len=500, n_features=3, lstm_hidden_dim=64).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )

    model.eval()
    cycle_tensor = torch.tensor(cycle_3d, dtype=torch.float32).to(device)
    with torch.no_grad():
        reconstructed_tensor = model(cycle_tensor)
    error_array = torch.mean(
        torch.abs(cycle_tensor - reconstructed_tensor), dim=(1, 2)
    ).cpu().numpy()
    return error_array

if __name__ == '__main__':
    model_validation(dataset_path='data/processed/validation_cycles.npz')

