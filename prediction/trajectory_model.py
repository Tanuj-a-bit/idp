import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, output_dim=2, num_layers=2):
        """
        input_dim: [lat, lon, speed_over_ground, course_over_ground, heading]
        hidden_dim: LSTM hidden state size
        output_dim: predicts next [lat, lon]
        """
        super(TrajectoryLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, (hn, cn) = self.lstm(x)
        # We take the output of the last time step
        last_out = out[:, -1, :]
        prediction = self.fc(last_out)
        return prediction

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=5, d_model=128, nhead=8, num_layers=3, output_dim=2):
        """
        State-of-the-art transformer for long-range dependency in trajectories
        """
        super(TrajectoryTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model)) # max seq len 100
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        b, s, d = x.shape
        x = self.embedding(x) + self.pos_encoder[:, :s, :]
        x = self.transformer_encoder(x)
        out = self.decoder(x[:, -1, :]) # Predict from the last latent vector
        return out

if __name__ == "__main__":
    # Smoke test
    model_lstm = TrajectoryLSTM()
    model_tx = TrajectoryTransformer()
    dummy_input = torch.randn(8, 20, 5) # [batch, seq_len, features]
    
    print(f"LSTM output shape: {model_lstm(dummy_input).shape}")
    print(f"Transformer output shape: {model_tx(dummy_input).shape}")
