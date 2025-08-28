import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_dim=64, latent_dim=32, num_layers=1):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, num_layers, batch_first=True)
        self.enc_fc = nn.Linear(hidden_dim, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, n_features, num_layers, batch_first=True)

    def forward(self, x):
        enc_out, _ = self.encoder(x)
        last = enc_out[:, -1, :]
        z = self.enc_fc(last)
        dec_in = self.dec_fc(z).unsqueeze(1).repeat(1, x.size(1), 1)
        recon, _ = self.decoder(dec_in)
        return recon