import torch
import torch.nn as nn

class SimpleLSTMAutoencoder(nn.Module):
    def __init__(self, seq_len=500, n_features=3, hidden_dim=64):
        super(SimpleLSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,        # 1 layer as baseline
            batch_first=True     # Input shape (Batch, Window, Feature)
        )

        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        # 1. Encoding
        # shape of x: (Batch, Window, Feature) = (Batch, 500, 3)
        encoded_seq, (hidden_n, cell_n) = self.encoder_lstm(x)

        # hidden_n shape: (num_layers, Batch, hidden_dim) = (1, Batch, 64)
        latent_vector = hidden_n[-1, :, :]

        # 2. Repeat vector
        # (Batch, hidden_dim) -> (Batch, Window, hidden_dim) = (Batch, 500, 64)
        repeated_latent = latent_vector.unsqueeze(1).repeat(1, self.seq_len, 1)

        # 3. Decoding
        decoded_seq, _ = self.decoder_lstm(repeated_latent)

        # 4. Output layer
        # decoded_seq shape: (Batch, Window, hidden_dim) = (Batch, 500, 64)
        # output shape: (Batch, Window, Feature) = (Batch, 500, 3)
        reconstructed_x = self.output_layer(decoded_seq)

        return reconstructed_x
        
if __name__ == "__main__":
    model = SimpleLSTMAutoencoder(seq_len=500, n_features=3, hidden_dim=64)
    dummy_input = torch.randn(128, 500, 3)  # Batch of 128 windows
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Output shape should match input shape for autoencoder"