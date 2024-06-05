import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
        self.encoder_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))

        nn.init.kaiming_uniform_(self.encoder[0].weight)
        nn.init.kaiming_uniform_(self.decoder.weight)

    def forward(self, x):
        x = x - self.decoder_bias
        latent = self.encoder(x) + self.encoder_bias
        reconstructed = self.decoder(latent) + self.decoder_bias
        return reconstructed