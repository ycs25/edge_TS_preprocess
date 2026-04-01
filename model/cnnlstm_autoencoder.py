import torch
import torch.nn as nn

class CNNLSTMAutoencoder(nn.Module):
    def __init__(self, seq_len=500, n_features=3, lstm_hidden_dim=64):
        super(CNNLSTMAutoencoder, self).__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.lstm_hidden_dim = lstm_hidden_dim

        # ==========================================
        # 1. CNN feature extractor (dimension reduction + downsampling)
        # Input shape: (Batch, Channels, SeqLen) as (Batch, 3, 500)
        # ==========================================
        self.conv_encoder = nn.Sequential(
            # First convolution: 500 -> 250 stride
            nn.Conv1d(in_channels=n_features, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            # Second convolution: 250 -> 125 stride
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        
        # ==========================================
        # 2. LSTM Bottleneck (learning temporal dependencies in compressed space)
        # Input shape: (Batch, SeqLen, Channels) as (Batch, 125, 32)
        # ==========================================
        # Here we use bidirectional=True to give the LSTM both forward and backward context
        self.lstm_encoder = nn.LSTM(input_size=32, hidden_size=lstm_hidden_dim, 
                                    batch_first=True, bidirectional=True)
        
        # Decoder receives LSTM's output (hidden_dim * 2)
        self.lstm_decoder = nn.LSTM(input_size=lstm_hidden_dim * 2, hidden_size=32, 
                                    batch_first=True)
        
        # ==========================================
        # 3. ConvTranspose1d Decoder (upsampling)
        # Input shape: (Batch, Channels, SeqLen) as (Batch, 32, 125)
        # ==========================================
        self.conv_decoder = nn.Sequential(
            # First transposed convolution: 125 -> 250 stride
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            # Second transposed convolution: 250 -> 500 stride, restoring 3 sensor features
            nn.ConvTranspose1d(in_channels=16, out_channels=n_features, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

    def forward(self, x):
        # original x shape: (Batch, 500, 3)
        
        # 1. Conv1d shape -> (Batch, 3, 500)
        x_conv_in = x.transpose(1, 2)
        
        # downsampling -> (Batch, 32, 125)
        x_encoded = self.conv_encoder(x_conv_in)
        
        # 2. LSTM shape -> (Batch, 125, 32)
        x_lstm_in = x_encoded.transpose(1, 2)
        
        # Passing LSTM bottleneck (here we pass the entire sequence, not just the last state)
        lstm_out, _ = self.lstm_encoder(x_lstm_in) # lstm_out: (Batch, 125, 128)
        lstm_dec_out, _ = self.lstm_decoder(lstm_out) # lstm_dec_out: (Batch, 125, 32)
        
        # 3. Adapt ConvTranspose1d shape requirement -> (Batch, 32, 125)
        x_conv_dec_in = lstm_dec_out.transpose(1, 2)
        
        # Restore -> (Batch, 3, 500)
        x_reconstructed = self.conv_decoder(x_conv_dec_in)
        
        # Finally convert back to original dimension -> (Batch, 500, 3)
        return x_reconstructed.transpose(1, 2)
    

if __name__ == "__main__":
    model = CNNLSTMAutoencoder(seq_len=500, n_features=3, lstm_hidden_dim=64)
    dummy_input = torch.randn(128, 500, 3)  # Batch of 128 windows
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == dummy_input.shape, "Output shape should match input shape for autoencoder"