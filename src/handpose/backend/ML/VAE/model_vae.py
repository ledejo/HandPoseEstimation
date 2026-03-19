import torch
import torch.nn as nn

from handpose.config_loader import get_settings


class HandPoseVae(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim: int | None = None,
        latent_dim: int | None = None,
        seq_len: int | None = None,
        dropout: float | None = None,
    ):
        """
        Initialisierung des VAE-Modells.
        Parameter können explizit übergeben werden (wichtig für Optuna).
        Falls None, werden Werte aus Config geladen.
        """
        super(HandPoseVae, self).__init__()

        vae_cfg = get_settings().vae
        hidden_dim = hidden_dim if hidden_dim is not None else vae_cfg.hidden_dim
        latent_dim = latent_dim if latent_dim is not None else vae_cfg.latent_dim
        seq_len = seq_len if seq_len is not None else vae_cfg.window_size
        dropout = dropout if dropout is not None else vae_cfg.dropout

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # --- ENCODER ---
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # --- LATENT SPACE ---
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- DECODER INPUT ---
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        # --- DECODER ---
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # --- OUTPUT ---
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        # 1. Encode
        _, (h_n, _) = self.encoder_lstm(x)
        last_hidden = h_n[-1]  # Letzter Hidden State

        # 2. Latent Distribution
        mu = self.fc_mu(last_hidden)
        logvar = self.fc_logvar(last_hidden)

        # 3. Sampling
        z = self.reparameterize(mu, logvar)

        # 4. Decode Prep
        decoder_hidden = self.decoder_input(z)
        # Wiederhole den Vektor für die gesamte Sequenzlänge
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, self.seq_len, 1)

        # 5. Decode
        decoder_output, _ = self.decoder_lstm(decoder_hidden)

        # 6. Reconstruction
        reconstruction = self.output_layer(decoder_output)

        return reconstruction, mu, logvar
