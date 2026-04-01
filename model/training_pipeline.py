import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

from lstm_autoencoder import SimpleLSTMAutoencoder
from cnnlstm_autoencoder import CNNLSTMAutoencoder

# 1. Load data
data_numpy = np.load('data/processed/training_tensor.npy')
x_tensor = torch.tensor(data_numpy, dtype=torch.float32)

dataset = TensorDataset(x_tensor, x_tensor)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# 2. Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNNLSTMAutoencoder(seq_len=500, n_features=3, lstm_hidden_dim=64).to(device)

# Loss and optimizer
criterion = torch.nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
epochs = 40
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")

# Visualize training loss
model.eval()

data_iter = iter(train_loader)
sample_inputs, _ = next(data_iter)

single_window_real = sample_inputs[0].unsqueeze(0).to(device)
with torch.no_grad():
    single_window_recon = model(single_window_real)

real_data = single_window_real.cpu().numpy().squeeze()
recon_data = single_window_recon.cpu().numpy().squeeze()

fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axis_names = ['X-Axis', 'Y-Axis', 'Z-Axis']

for i in range(3):
    axes[i].plot(real_data[:, i], label='Real (SymLog)', color='blue', alpha=0.7)
    axes[i].plot(recon_data[:, i], label='Reconstructed', color='red', linestyle='--', alpha=0.9)
    axes[i].set_title(f'{axis_names[i]} Reconstruction')
    axes[i].legend(loc='upper right')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()